from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.graph import runtime
from runtime_fakes import patch_runtime_repo


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    patch_runtime_repo(monkeypatch, runtime)


@pytest.mark.asyncio
async def test_run_turn_creates_conversation_when_none(monkeypatch):
    async def fake_create(uid):
        return 42

    async def fake_append(cid, role, content=None, **k):
        return 1

    class FakeGraph:
        async def aget_state(self, config):
            return type("S", (), {"values": {}})()

        async def ainvoke(self, inp, config, durability=None):
            assert inp["trace"] == {}
            assert inp["resolved_query"] == "" and inp["intent_confidence"] == 0.0
            assert inp["order_id"] == "" and inp["order_data"] == {}
            return {"answer": "hi", "messages": [], "conversation_id": inp["conversation_id"]}

    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())

    out = await runtime.run_turn("u1", "你好", None)
    assert out["conversation_id"] == 42


@pytest.mark.asyncio
async def test_run_turn_raises_when_conversation_missing(monkeypatch):
    async def fake_get(cid):
        return None

    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    with pytest.raises(runtime.ConversationNotFound):
        await runtime.run_turn("u1", "hi", 999)


@pytest.mark.asyncio
async def test_stream_turn_maps_events(monkeypatch):
    async def fake_create(uid):
        return 7

    async def fake_append(cid, role, content=None, **k):
        return 1

    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)

    class FakeGraph:
        async def aget_state(self, config):
            return type("S", (), {"values": {}})()

        async def astream(self, inp, config, stream_mode=None, durability=None):
            assert inp["trace"] == {} and inp["evidence"] == ""
            assert inp["resolved_query"] == "" and inp["intent_confidence"] == 0.0
            assert inp["order_id"] == "" and inp["order_data"] == {}
            yield ("messages", (AIMessage("已"), {"langgraph_node": "agent_llm"}))
            yield ("messages", (AIMessage("发货"), {"langgraph_node": "agent_llm"}))
            yield ("messages", (AIMessage("订单"), {"langgraph_node": "classify_intent"}))
            yield ("updates", {"agent_tools": {"messages": [
                ToolMessage(content="{}", tool_call_id="1", name="query_logistics")]}})
            yield ("updates", {"forced_rag": {"citations": [{"n": 1}]}})
            yield ("updates", {"complaint_reply": {"answer": "抱歉",
                                                   "suggested_actions": [{"type": "transfer_human"}]}})

    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    events = [e async for e in runtime.stream_turn("u1", "订单1001物流", None)]
    kinds = [e["type"] for e in events]
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    tools = [e["name"] for e in events if e["type"] == "tool"]
    assert "已" in deltas and "发货" in deltas and "抱歉" in deltas
    assert "订单" not in deltas  # 非答复节点被过滤
    assert tools == ["query_logistics"]
    assert any(e["type"] == "citations" for e in events)
    assert any(e["type"] == "actions" for e in events)
    assert kinds[-1] == "done" and events[-1]["conversation_id"] == 7


@pytest.mark.asyncio
async def test_run_turn_returns_only_this_turns_messages(monkeypatch):
    old = [AIMessage("old"), AIMessage("old2")]

    async def fake_get(cid):
        return SimpleNamespace(summary="", summary_upto_msg_id=0, layer1_from_msg_id=0)

    async def fake_append(*a, **k):
        return 1

    class FakeGraph:
        async def aget_state(self, config):
            return type("S", (), {"values": {"messages": old}})()

        async def ainvoke(self, inp, config, durability=None):
            return {"messages": [*old, inp["messages"][0], AIMessage("new")]}

    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    out = await runtime.run_turn("u1", "hi", 5)
    assert [m.content for m in out["state"]["turn_messages"]] == ["hi", "new"]


async def test_run_turn_passes_context_fields_and_schedules(monkeypatch):
    conv = SimpleNamespace(summary="早前摘要", summary_upto_msg_id=0, layer1_from_msg_id=0)

    async def fake_create(uid):
        return 7

    async def fake_get(cid):
        return conv

    async def fake_rows(cid):
        return [SimpleNamespace(id=1, role="user", content="q1"),
                SimpleNamespace(id=2, role="assistant", content="a1")]

    scheduled = []

    async def fake_schedule(cid):
        scheduled.append(cid)

    class FakeGraph:
        async def aget_state(self, config):
            return type("S", (), {"values": {}})()

        async def ainvoke(self, inp, config, durability=None):
            assert inp["summary"] == "早前摘要"
            assert [m.content for m in inp["history_messages"]] == ["q1", "a1"]
            assert inp["summary_upto_msg_id"] == 0 and inp["layer1_from_msg_id"] == 0
            return {"answer": "hi", "messages": [], "conversation_id": 7}

    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    monkeypatch.setattr(runtime.repository, "list_dialog_messages", fake_rows)
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", fake_schedule)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    out = await runtime.run_turn("u1", "hi", None)
    assert out["conversation_id"] == 7
    assert scheduled == [7]


@pytest.mark.asyncio
async def test_failed_turn_writes_no_mysql_and_retry_not_duplicated(monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver

    from app.graph import nodes
    from app.graph.build import build_graph

    async def boom(state):
        raise RuntimeError("boom")

    monkeypatch.setattr(nodes, "coref", boom)
    writes = []

    async def fake_create(uid):
        return 7

    async def fake_append(cid, role, content=None, **k):
        writes.append((role, content))
        return 1

    cp = InMemorySaver()
    g = build_graph(checkpointer=cp)
    monkeypatch.setattr(runtime, "get_graph", lambda: g)
    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await runtime.run_turn("u1", "hi", None)
    snap = await g.aget_state({"configurable": {"thread_id": "7"}})
    assert len(snap.values.get("messages", [])) == 1   # 重试复用 message id,不叠重复
    assert writes == []                                # 失败轮不落 MySQL 半条记录


class _Intr:
    def __init__(self, value):
        self.value = value


@pytest.mark.asyncio
async def test_stream_turn_emits_interrupt_event(monkeypatch):
    async def fake_create(uid):
        return 9

    class FakeGraph:
        async def aget_state(self, config):
            return type("S", (), {"values": {}, "interrupts": ()})()

        async def astream(self, inp, config, stream_mode=None):
            yield ("updates", {"fetch_order": None})
            yield ("updates", {"__interrupt__": (_Intr({
                "type": "select_order",
                "orders": [{"order_id": "1001"}]}),)})

    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    events = [e async for e in runtime.stream_turn("u1", "我要退款", None)]
    intr = [e for e in events if e["type"] == "interrupt"]
    assert intr and intr[0]["kind"] == "select_order"
    assert intr[0]["orders"] == [{"order_id": "1001"}]
    assert intr[0]["conversation_id"] == 9          # 首轮无 done,也要能续跑
    assert not [e for e in events if e["type"] == "done"]


@pytest.mark.asyncio
async def test_stream_turn_reemits_pending_interrupt_without_new_turn(monkeypatch):
    async def fake_get(cid):
        return object()

    class FakeGraph:
        async def aget_state(self, config):
            return type("S", (), {"values": {}, "interrupts": (_Intr({
                "type": "select_order",
                "orders": [{"order_id": "1002"}]}),)})()

        async def astream(self, *a, **k):
            raise AssertionError("有未决中断时不应启动新一轮")

    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    events = [e async for e in runtime.stream_turn("u1", "算了查物流", 5)]
    assert events == [{"type": "interrupt", "conversation_id": 5,
                       "kind": "select_order",
                       "orders": [{"order_id": "1002"}]}]


@pytest.mark.asyncio
async def test_resume_turn_drives_command(monkeypatch):
    from langgraph.types import Command

    seen = {}

    class FakeGraph:
        async def aget_state(self, config):
            return type("S", (), {"values": {}})()

        async def ainvoke(self, inp, config):
            seen["is_command"] = isinstance(inp, Command)
            seen["resume"] = getattr(inp, "resume", None)
            return {"answer": "这一单可以退款", "messages": []}

    async def fake_get(cid):
        return object()

    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    out = await runtime.resume_turn(5, "1001")
    assert seen["is_command"] and seen["resume"] == "1001"
    assert out["conversation_id"] == 5
    assert out["state"]["answer"] == "这一单可以退款"


@pytest.mark.asyncio
async def test_stream_resume_uses_command(monkeypatch):
    from langgraph.types import Command

    seen = {}

    class FakeGraph:
        async def astream(self, inp, config, stream_mode=None):
            seen["is_command"] = isinstance(inp, Command)
            seen["resume"] = getattr(inp, "resume", None)
            yield ("updates", {"agent_llm": {"messages": []}})

    async def fake_get(cid):
        return object()

    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    events = [e async for e in runtime.stream_resume(5, "1001")]
    assert seen["is_command"] and seen["resume"] == "1001"
    assert events[-1] == {"type": "done", "conversation_id": 5}


async def test_load_context_clamps_layer1_from_to_summary_upto(monkeypatch):
    conv = SimpleNamespace(summary="s", summary_upto_msg_id=5, layer1_from_msg_id=2)
    updated = []

    async def fake_get(cid):
        return conv

    async def fake_rows(cid):
        return []

    async def fake_update(cid, mid):
        updated.append(mid)

    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    monkeypatch.setattr(runtime.repository, "list_dialog_messages", fake_rows)
    monkeypatch.setattr(runtime.repository, "update_layer1_from", fake_update)
    ctx = await runtime._load_context(7)
    assert ctx["layer1_from_msg_id"] == 5 and updated == [5]


async def test_load_context_logs_degrade_with_layer1_tokens(monkeypatch, caplog):
    conv = SimpleNamespace(summary="", summary_upto_msg_id=0, layer1_from_msg_id=0)
    rows = [
        SimpleNamespace(id=1, role="user", content="q1" * 50),
        SimpleNamespace(id=2, role="assistant", content="a1" * 50),
        SimpleNamespace(id=3, role="user", content="q2" * 50),
        SimpleNamespace(id=4, role="assistant", content="a2" * 50),
    ]
    updated = []

    async def fake_get(cid):
        return conv

    async def fake_rows(cid):
        return rows

    async def fake_update(cid, mid):
        updated.append(mid)

    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    monkeypatch.setattr(runtime.repository, "list_dialog_messages", fake_rows)
    monkeypatch.setattr(runtime.repository, "update_layer1_from", fake_update)
    monkeypatch.setattr(runtime.context_budget, "compute_budget",
                        lambda: runtime.context_budget.ContextBudget(
                            64000, 0, 0, 30000, 1, 1))
    with caplog.at_level("INFO", logger="app.graph.runtime"):
        ctx = await runtime._load_context(7)
    assert ctx["layer1_from_msg_id"] == 2 and updated == [2]
    assert "layer1 降级" in caplog.text and "layer1_tokens=" in caplog.text

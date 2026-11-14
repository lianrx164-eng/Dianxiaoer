"""流式入口:事件映射、create_ticket 拦截为提议、会话不存在。"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import nodes, runtime
from app.graph.build import build_graph


def _aval(value):
    async def _f(*a, **k):
        return value
    return _f


@pytest.fixture(autouse=True)
def _repo(monkeypatch):
    async def fake_create(uid):
        return 7

    async def fake_append(*a, **k):
        return 1

    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)


async def _collect(agen):
    return [e async for e in agen]


async def test_stream_chitchat_emits_fixed_reply(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", _aval(
        {"intent": "闲聊", "route": "chitchat", "trace": {"route": "chitchat"}}))
    monkeypatch.setattr(runtime, "get_graph",
                        lambda: build_graph(checkpointer=InMemorySaver()))

    events = await _collect(runtime.stream_turn("u1", "你好", None))
    assert events[0] == {"type": "delta", "text": nodes.CHITCHAT_REPLY}
    assert events[-1]["type"] == "done" and events[-1]["conversation_id"] == 7


async def test_stream_agent_create_ticket_becomes_action_not_write(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", _aval(
        {"intent": "售后", "route": "business", "trace": {"route": "business"}}))
    calls = {"n": 0}

    async def fake_agent(state, config=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"messages": [AIMessage("", tool_calls=[
                {"name": "create_ticket",
                 "args": {"description": "屏幕碎了", "ticket_type": "售后"}, "id": "t1"}])],
                "steps": 1, "tokens_used": 5}
        return {"messages": [AIMessage("已为您准备好工单选项,可点击下方按钮提交")],
                "steps": 2, "tokens_used": 10}

    async def fail_write(*a, **k):
        raise AssertionError("拦截为提议时不应写 tickets")

    monkeypatch.setattr(nodes, "agent_llm", fake_agent)
    monkeypatch.setattr(nodes.repository, "create_ticket", fail_write)
    monkeypatch.setattr(runtime, "get_graph",
                        lambda: build_graph(checkpointer=InMemorySaver()))

    events = await _collect(runtime.stream_turn("u1", "屏幕碎了要售后", None))
    actions = [e for e in events if e["type"] == "actions"]
    assert actions and actions[0]["items"][0]["type"] == "create_ticket"
    assert actions[0]["items"][0]["draft"]["description"] == "屏幕碎了"
    assert not [e for e in events if e["type"] == "tool"]   # create_ticket 不出工具徽章
    assert any(e["type"] == "delta" and "工单选项" in e["text"] for e in events)


async def test_stream_unknown_conversation_raises(monkeypatch):
    async def fake_get(cid):
        return None

    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    with pytest.raises(runtime.ConversationNotFound):
        await _collect(runtime.stream_turn("u1", "hi", 999999))

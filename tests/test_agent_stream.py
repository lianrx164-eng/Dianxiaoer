from langchain_core.messages import AIMessage

from app.core import agent
from app.db import repository as repo
from fake_model import FakeModel


async def _collect(agen):
    return [ev async for ev in agen]


async def test_stream_with_tool_emits_tool_delta_done(db_session_factory, db_clean):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    model = FakeModel([first], stream_tokens=["您的订单", "正在运输中。"])
    events = await _collect(agent.stream_agent_turn("u1", "订单 1001 到哪了", None, model=model))

    assert events[0] == {"type": "tool", "name": "query_logistics"}
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert "".join(deltas) == "您的订单正在运输中。"
    assert events[-1]["type"] == "done" and isinstance(events[-1]["conversation_id"], int)
    assert model.bind_calls == 1
    assert model.astream_messages, "收敛应走 astream"

    msgs = await repo.list_messages(events[-1]["conversation_id"])
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[-1].content == "您的订单正在运输中。"


async def test_stream_without_tool_yields_direct_delta(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="你好,有什么可以帮您?")])
    events = await _collect(agent.stream_agent_turn("u1", "你好", None, model=model))

    assert all(e["type"] != "tool" for e in events)
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert "".join(deltas) == "你好,有什么可以帮您?"
    assert events[-1]["type"] == "done"
    assert model.astream_messages == []                # 无工具不走收敛流式

    msgs = await repo.list_messages(events[-1]["conversation_id"])
    assert [m.role for m in msgs] == ["user", "assistant"]


async def test_stream_new_conversation_returns_new_id(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="嗨")])
    events = await _collect(agent.stream_agent_turn("u1", "hi", None, model=model))
    cid = events[-1]["conversation_id"]
    assert isinstance(cid, int)
    assert await repo.get_conversation(cid) is not None


async def test_stream_unknown_conversation_raises(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="hi")])
    try:
        await _collect(agent.stream_agent_turn("u1", "hi", 999999, model=model))
        assert False, "应抛 ConversationNotFound"
    except agent.ConversationNotFound:
        pass


async def test_stream_convergence_carries_single_round_guard(db_session_factory, db_clean):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "邮费"}, "id": "c1"}])
    model = FakeModel([first], stream_tokens=["未找到相关信息。"])
    await _collect(agent.stream_agent_turn("u1", "邮费是多少", None, model=model))
    converged = model.astream_messages[0]
    assert "不要再" in converged[-1].content
    assert converged[-1].type == "system"


async def test_tool_frame_emitted_before_tool_executes(db_session_factory, db_clean, monkeypatch):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    model = FakeModel([first], stream_tokens=["完成"])
    executed = {"done": False}
    real_exec = agent.execute_tool_call

    async def spy(tc, cid):
        executed["done"] = True
        return await real_exec(tc, cid)

    monkeypatch.setattr(agent, "execute_tool_call", spy)

    agen = agent.stream_agent_turn("u1", "查物流", None, model=model)
    first_ev = await agen.__anext__()
    assert first_ev == {"type": "tool", "name": "query_logistics"}
    assert executed["done"] is False          # 状态帧必须先于工具执行
    async for _ in agen:
        pass
    assert executed["done"] is True

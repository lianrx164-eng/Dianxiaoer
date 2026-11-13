from langchain_core.messages import AIMessage, AIMessageChunk

from app.core import agent
from app.db import repository as repo


class FakeModel:
    def __init__(self, scripted, stream_tokens=None):
        self._scripted = list(scripted)
        self._stream_tokens = list(stream_tokens or [])
        self.bind_calls = 0
        self.invoke_messages = []
        self.astream_messages = []

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def ainvoke(self, messages):
        self.invoke_messages.append(list(messages))
        return self._scripted.pop(0)

    async def astream(self, messages):
        self.astream_messages.append(list(messages))
        for t in self._stream_tokens:
            yield AIMessageChunk(content=t)


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

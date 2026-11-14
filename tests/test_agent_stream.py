"""流式入口:事件映射、create_ticket 拦截为提议、会话不存在。"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from runtime_fakes import patch_runtime_repo

from app.graph import nodes, runtime
from app.graph.build import build_graph
from app.tools import engine, registry


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
    patch_runtime_repo(monkeypatch, runtime)


async def _collect(agen):
    return [e async for e in agen]


async def test_stream_chitchat_emits_fixed_reply(monkeypatch):
    from app.core.prompts import CHITCHAT_REPLY_TEXT

    monkeypatch.setattr(nodes, "classify_intent", _aval(
        {"intent": "闲聊", "route": "fallback_script", "trace": {"route": "fallback_script"}}))
    monkeypatch.setattr(runtime, "get_graph",
                        lambda: build_graph(checkpointer=InMemorySaver()))

    events = await _collect(runtime.stream_turn("u1", "你好", None))
    assert events[0] == {"type": "delta", "text": CHITCHAT_REPLY_TEXT}
    assert events[-1]["type"] == "done" and events[-1]["conversation_id"] == 7


async def test_stream_agent_create_ticket_interrupts_for_confirmation(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", _aval(
        {"intent": "订单", "route": "business", "trace": {"route": "business"}}))
    calls = {"n": 0}

    async def fake_agent(state, config=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"messages": [AIMessage("", tool_calls=[
                {"name": "create_ticket",
                 "args": {"description": "屏幕碎了", "ticket_type": "售后"}, "id": "t1"}])],
                "steps": 1, "tokens_used": 5}
        return {"messages": [AIMessage("请确认工单信息")], "steps": 2, "tokens_used": 10}

    async def only_builtin():
        return registry.builtin_specs()

    async def fake_audit(**kw):
        return None

    monkeypatch.setattr(nodes, "agent_llm", fake_agent)
    monkeypatch.setattr(nodes.registry, "get_all_specs", only_builtin)
    monkeypatch.setattr(engine.repository, "insert_tool_audit", fake_audit)
    monkeypatch.setattr(runtime, "get_graph",
                        lambda: build_graph(checkpointer=InMemorySaver()))

    events = await _collect(runtime.stream_turn("u1", "屏幕碎了要售后", None))
    intr = [e for e in events if e["type"] == "interrupt"]
    assert intr and intr[0]["kind"] == "confirm_ticket"
    assert intr[0]["preview"] == {"ticket_type": "售后", "description": "屏幕碎了"}
    assert not [e for e in events if e["type"] in ("actions", "tool")]


async def test_stream_unknown_conversation_raises(monkeypatch):
    async def fake_get(cid):
        return None

    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    with pytest.raises(runtime.ConversationNotFound):
        await _collect(runtime.stream_turn("u1", "hi", 999999))

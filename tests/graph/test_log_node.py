import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes


def test_resolve_answer_prefers_explicit():
    assert nodes.resolve_answer({"answer": "固定话术", "messages": []}) == "固定话术"


def test_resolve_answer_falls_back_to_last_ai():
    state = {"messages": [HumanMessage("hi"), AIMessage("模型答复")]}
    assert nodes.resolve_answer(state) == "模型答复"


@pytest.mark.asyncio
async def test_log_node_persists_and_backfills_deterministic_answer(monkeypatch):
    calls = []

    async def fake_append(cid, role, content=None, **k):
        calls.append((role, content))
        return 1

    monkeypatch.setattr(nodes.repository, "append_message", fake_append)
    out = await nodes.log_node({
        "conversation_id": 8, "intent": "闲聊", "route": "chitchat",
        "answer": "你好呀", "messages": [HumanMessage("你好")], "trace": {"forced_rag": False}})
    assert calls == [("user", "你好"), ("assistant", "你好呀")]
    assert out["messages"][0].content == "你好呀"


@pytest.mark.asyncio
async def test_log_node_does_not_duplicate_agent_answer(monkeypatch):
    async def fake_append(cid, role, content=None, **k):
        return 1

    monkeypatch.setattr(nodes.repository, "append_message", fake_append)
    out = await nodes.log_node({
        "conversation_id": 8, "intent": "订单", "route": "business",
        "messages": [HumanMessage("查订单"), AIMessage("订单已发货")]})
    assert out == {}

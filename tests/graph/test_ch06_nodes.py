import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes


@pytest.mark.asyncio
async def test_classify_intent_writes_confidence_and_route(monkeypatch):
    async def fake_classify(query, history=""):
        return {"intent": "退款退货", "confidence": 0.83}

    monkeypatch.setattr(nodes.intent_mod, "classify", fake_classify)
    out = await nodes.classify_intent({
        "messages": [HumanMessage("这个能退吗")],
        "resolved_query": "蓝牙耳机还能申请退货吗"})
    assert out["intent"] == "退款退货"
    assert out["intent_confidence"] == 0.83
    assert out["route"] == "refund_flow"
    assert out["trace"]["route"] == "refund_flow"
    assert out["trace"]["intent_confidence"] == 0.83
    assert "confidence" not in out["trace"]      # 不占用证据闸的 confidence 键


def test_get_chat_model_honors_model_override():
    from app.config import settings
    from app.core.llm import get_chat_model

    m = get_chat_model(model="glm-4-flash")
    assert m.model_name == "glm-4-flash"
    assert get_chat_model().model_name == settings.chat_model


@pytest.mark.asyncio
async def test_resolve_reference_passthrough_when_complete(monkeypatch):
    async def fake_resolve(q, history=""):
        return q

    monkeypatch.setattr(nodes.coref_mod, "resolve", fake_resolve)
    out = await nodes.coref({"messages": [HumanMessage("蓝牙耳机的保修期多久")]})
    assert out["resolved_query"] == "蓝牙耳机的保修期多久"
    assert out["trace"]["coref"] == "passthrough"


@pytest.mark.asyncio
async def test_resolve_reference_rewrites_with_history(monkeypatch):
    async def fake_resolve(q, history=""):
        assert "蓝牙耳机什么时候到" in history
        return "蓝牙耳机还能申请退货吗"

    monkeypatch.setattr(nodes.coref_mod, "resolve", fake_resolve)
    out = await nodes.coref({"messages": [
        HumanMessage("蓝牙耳机什么时候到"), AIMessage("预计明天"),
        HumanMessage("这个能退吗")]})
    assert out["resolved_query"] == "蓝牙耳机还能申请退货吗"
    assert out["trace"]["coref"] == "rewrite"

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes


@pytest.mark.asyncio
async def test_script_reply_fixed_no_actions():
    from app.core.prompts import CHITCHAT_REPLY_TEXT

    out = await nodes.script_reply({"intent": "闲聊", "messages": [HumanMessage("你好")]})
    assert out["answer"] == CHITCHAT_REPLY_TEXT
    assert out["trace"]["route"] == "fallback_script"
    assert not out.get("suggested_actions")


@pytest.mark.asyncio
async def test_complaint_reply_offers_two_actions():
    out = await nodes.complaint_reply(
        {"messages": [HumanMessage("我要投诉你们")], "conversation_id": 7})
    assert out["answer"] == nodes.COMPLAINT_REPLY
    types = [a["type"] for a in out["suggested_actions"]]
    assert types == ["transfer_human", "create_ticket"]
    ticket = next(a for a in out["suggested_actions"] if a["type"] == "create_ticket")
    assert ticket["draft"]["ticket_type"] == "投诉"
    assert ticket["draft"]["description"] == "我要投诉你们"


@pytest.mark.asyncio
async def test_fallback_reply_records_low_confidence(monkeypatch):
    calls = {}

    async def fake_insert(conversation_id, raw_question, source, reason):
        calls.update(conversation_id=conversation_id, raw=raw_question, source=source)
        return 1

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)
    out = await nodes.fallback_reply(
        {"messages": [HumanMessage("怎么注销账号")], "conversation_id": 3,
         "trace": {"gate": "retrieval_low_conf", "evidence_top": 0.1}})
    assert out["answer"] == nodes.FALLBACK_REPLY
    assert calls["source"] == "retrieval_low_conf"
    assert calls["conversation_id"] == 3
    assert calls["raw"] == "怎么注销账号"


@pytest.mark.asyncio
async def test_cap_reply_aborts_pending_tool_calls_without_pooling(monkeypatch):
    async def fail(*a, **k):
        raise AssertionError("步数封顶不应写低置信池")

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fail)
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}])
    out = await nodes.cap_reply({"messages": [HumanMessage("查订单"), ai], "conversation_id": 3})
    assert out["answer"] == nodes.FALLBACK_REPLY
    assert out["trace"]["cap"] is True
    assert out["messages"][0].tool_call_id == "c1"

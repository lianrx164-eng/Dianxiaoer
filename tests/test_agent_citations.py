import json

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import text

from app.core import agent
from fake_model import FakeModel


class _FakeToolRun:
    def __init__(self, name, payload, cid="tc1", ok=True):
        self.name = name
        self.ok = ok
        self.tool_call_id = cid

        class _M:
            content = json.dumps(payload, ensure_ascii=False)
        self.tool_message = _M()


def test_extract_faq_citations():
    runs = [_FakeToolRun("query_faq", {
        "sufficient": True, "evidence": "[1] x",
        "citations": [{"n": 1, "id": 5, "section_path": "运费政策",
                       "question": "运费", "answer": "满99包邮",
                       "content_type": "faq"}]})]
    faq = agent._faq_result(runs)
    assert faq["sufficient"] is True and faq["citations"][0]["id"] == 5


def test_extract_faq_none_when_absent():
    runs = [_FakeToolRun("query_order", {"x": 1})]
    assert agent._faq_result(runs) is None


@pytest.mark.asyncio
async def test_stream_emits_citations(db_session_factory, db_clean, monkeypatch):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "邮费"}, "id": "c1"}])
    model = FakeModel([first], stream_tokens=["满99包邮[1]。"])
    runs = [_FakeToolRun("query_faq", {
        "sufficient": True, "evidence": "[1] 运费 满99包邮",
        "citations": [{"n": 1, "id": 5, "section_path": "运费政策",
                       "question": "运费", "answer": "满99包邮",
                       "content_type": "faq"}]})]

    async def fake_exec(cid, tcs):
        return runs

    monkeypatch.setattr(agent, "_execute_tools", fake_exec)
    events = [e async for e in agent.stream_agent_turn("u1", "邮费是多少", None, model=model)]
    cites = [e for e in events if e["type"] == "citations"]
    assert cites and cites[0]["items"][0]["section_path"] == "运费政策"


@pytest.mark.asyncio
async def test_stream_low_conf_lands_in_pool(db_session_factory, db_clean, monkeypatch):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "火星车"}, "id": "c1"}])
    model = FakeModel([first], stream_tokens=["暂时没有查到相关信息。"])
    runs = [_FakeToolRun("query_faq", {
        "sufficient": False, "source": "retrieval_low_conf",
        "reason": "检索证据不足(top=0.000)", "citations": []})]

    async def fake_exec(cid, tcs):
        return runs

    monkeypatch.setattr(agent, "_execute_tools", fake_exec)
    events = [e async for e in agent.stream_agent_turn("u1", "你们卖火星车吗", None, model=model)]
    assert all(e["type"] != "citations" for e in events)
    assert any("证据不足" in m.content for m in model.astream_messages[0])

    async with db_session_factory() as s:
        row = (await s.execute(text(
            "SELECT source, raw_question FROM low_confidence_questions"))).first()
    assert row.source == "retrieval_low_conf" and row.raw_question == "你们卖火星车吗"

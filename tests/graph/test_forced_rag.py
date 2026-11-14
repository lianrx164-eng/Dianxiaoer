import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes


def _aval(value):
    async def _f(*a, **k):
        return value
    return _f


@pytest.mark.asyncio
async def test_forced_rag_strong_builds_evidence(monkeypatch):
    hits = [{"id": 1, "question": "退货政策", "answer": "7天无理由", "rerank_score": 0.9,
             "section_path": "政策/退货", "content_type": "policy"}]
    monkeypatch.setattr("app.core.query_understanding.understand",
                        _aval({"standard": "退货政策", "expanded": []}))
    monkeypatch.setattr("app.core.retrieval.search_knowledge", _aval(hits))
    monkeypatch.setattr("app.core.retrieval.arrange_head_tail", lambda h: h)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", _aval({"useful": True, "reason": ""}))

    out = await nodes.forced_rag({"messages": [HumanMessage("退货政策是什么")]})
    assert out["evidence_strong"] is True
    assert "[1]" in out["evidence"]
    assert out["citations"][0]["n"] == 1
    assert out["trace"]["forced_rag"] is True


@pytest.mark.asyncio
async def test_forced_rag_weak_when_below_threshold(monkeypatch):
    monkeypatch.setattr("app.core.query_understanding.understand",
                        _aval({"standard": "注销账号", "expanded": []}))
    monkeypatch.setattr("app.core.retrieval.search_knowledge", _aval([]))
    out = await nodes.forced_rag({"messages": [HumanMessage("怎么注销账号")]})
    assert out["evidence_strong"] is False
    assert out["trace"]["gate"] == "retrieval_low_conf"


@pytest.mark.asyncio
async def test_forced_rag_weak_when_selfcheck_rejects(monkeypatch):
    hits = [{"id": 1, "question": "运费", "answer": "满99包邮", "rerank_score": 0.8,
             "section_path": "运费", "content_type": "faq"}]
    monkeypatch.setattr("app.core.query_understanding.understand",
                        _aval({"standard": "型号功能", "expanded": []}))
    monkeypatch.setattr("app.core.retrieval.search_knowledge", _aval(hits))
    monkeypatch.setattr("app.core.selfcheck.check_sufficient",
                        _aval({"useful": False, "reason": "未涉及型号"}))
    out = await nodes.forced_rag({"messages": [HumanMessage("Pro型号功能")]})
    assert out["evidence_strong"] is False
    assert out["trace"]["gate"] == "self_check"


@pytest.mark.asyncio
async def test_classify_intent_writes_state(monkeypatch):
    monkeypatch.setattr(nodes.intent_mod, "classify",
                        _aval({"intent": "物流", "confidence": 0.9}))
    out = await nodes.classify_intent({"messages": [HumanMessage("快递到哪了")]})
    assert out["intent"] == "物流"
    assert out["intent_confidence"] == 0.9
    assert out["route"] == "business"
    assert out["trace"]["intent"] == "物流"


@pytest.mark.asyncio
async def test_forced_rag_uses_resolved_query(monkeypatch):
    seen = {}

    async def fake_gather(q, category=None):
        seen["q"] = q
        return {"sufficient": False, "source": "retrieval_low_conf",
                "reason": "", "top": 0.0}

    monkeypatch.setattr(nodes.evidence, "gather_evidence", fake_gather)
    await nodes.forced_rag({"messages": [HumanMessage("它呢")],
                            "resolved_query": "订单1001到哪了"})
    assert seen["q"] == "订单1001到哪了"


@pytest.mark.asyncio
async def test_confidence_check_traces_decision():
    assert (await nodes.confidence_check({"evidence_strong": True}))["trace"]["confidence"] == "strong"
    assert (await nodes.confidence_check({"evidence_strong": False}))["trace"]["confidence"] == "weak"

"""ch09 闸接线:forced_rag 把置信度/source/快照写进 state;fallback_reply 落池带对 source 与快照。"""
from langchain_core.messages import HumanMessage

from app.graph import nodes

SNAP = [{"question": "x", "answer": "y", "rerank_score": 0.4, "section_path": "s"}]


async def test_forced_rag_writes_low_confidence_fields(monkeypatch):
    async def fake(query_raw, category=None):
        return {"sufficient": False, "source": "retrieval_low_conf", "reason": "弱",
                "confidence": 0.21, "signals": {"top1_score": 0.15},
                "snapshot": SNAP, "top": 0.15}

    monkeypatch.setattr(nodes.evidence, "gather_evidence", fake)
    out = await nodes.forced_rag(
        {"messages": [HumanMessage("猫窝能水洗吗")], "conversation_id": 1, "trace": {}})
    assert out["evidence_strong"] is False
    assert out["evidence_confidence"] == 0.21
    assert out["fallback_source"] == "retrieval_low_conf"
    assert out["retrieved_snapshot"] == SNAP
    assert out["trace"]["confidence_signals"] == {"top1_score": 0.15}


async def test_forced_rag_writes_strong_snapshot(monkeypatch):
    async def fake(query_raw, category=None):
        return {"sufficient": True, "evidence": "e", "citations": [{"n": 1}],
                "confidence": 0.9, "signals": {"top1_score": 0.9},
                "snapshot": SNAP, "top": 0.9}

    monkeypatch.setattr(nodes.evidence, "gather_evidence", fake)
    out = await nodes.forced_rag(
        {"messages": [HumanMessage("q")], "conversation_id": 1, "trace": {}})
    assert out["evidence_strong"] is True
    assert out["evidence_confidence"] == 0.9
    assert out["retrieved_snapshot"] == SNAP
    assert out["citations"] == [{"n": 1}]


async def test_fallback_reply_persists_source_and_snapshot(monkeypatch):
    recorded = {}

    async def fake_insert(cid, q, source, reason, retrieved_chunks=None):
        recorded.update(cid=cid, q=q, source=source, reason=reason, chunks=retrieved_chunks)
        return 1

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)
    state = {"messages": [HumanMessage("猫窝能水洗吗")], "conversation_id": 7,
             "fallback_source": "self_check", "evidence_confidence": 0.41,
             "retrieved_snapshot": SNAP, "trace": {"self_check": "缺关键信息"}}
    out = await nodes.fallback_reply(state)
    assert recorded["source"] == "self_check"
    assert recorded["chunks"] == SNAP
    assert "0.41" in recorded["reason"]
    assert "缺关键信息" in recorded["reason"]
    assert out["answer"] == nodes.FALLBACK_REPLY
    assert [a["type"] for a in out["suggested_actions"]] == ["transfer_human", "create_ticket"]
    assert out["suggested_actions"][1]["draft"]["ticket_type"] == "咨询"


async def test_fallback_reply_defaults_source_and_null_snapshot(monkeypatch):
    recorded = {}

    async def fake_insert(cid, q, source, reason, retrieved_chunks=None):
        recorded.update(source=source, chunks=retrieved_chunks)
        return 1

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)
    out = await nodes.fallback_reply(
        {"messages": [HumanMessage("嗯")], "conversation_id": 1, "trace": {}})
    assert recorded["source"] == "retrieval_low_conf"
    assert recorded["chunks"] is None
    assert out["answer"] == nodes.FALLBACK_REPLY

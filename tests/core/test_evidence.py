"""ch09 证据管线:置信度闸替换机械闸,三条返回路径都带 confidence/signals/snapshot。"""
from app.core import evidence


def _hit(i, score):
    return {"id": i, "question": f"q{i}", "answer": f"a{i}", "rerank_score": score,
            "section_path": "s", "content_type": "faq"}


def _stub(monkeypatch, hits, useful=True):
    async def fake_retrieve(query_raw, category=None):
        return "标准问", hits

    async def fake_check(q, ev):
        return {"useful": useful, "reason": "" if useful else "缺关键信息"}

    monkeypatch.setattr(evidence, "retrieve_hits", fake_retrieve)
    monkeypatch.setattr(evidence.selfcheck, "check_sufficient", fake_check)


async def test_low_confidence_blocks_with_signals_and_snapshot(monkeypatch):
    monkeypatch.setattr("app.config.settings.evidence_confidence_threshold", 0.5)
    _stub(monkeypatch, [_hit(1, 0.15), _hit(2, 0.14)])
    r = await evidence.gather_evidence("猫窝能水洗吗")
    assert r["sufficient"] is False and r["source"] == "retrieval_low_conf"
    assert r["confidence"] < 0.5
    assert r["signals"]["top1_score"] == 0.15
    assert len(r["snapshot"]) == 2


async def test_confidence_pass_but_selfcheck_fails(monkeypatch):
    monkeypatch.setattr("app.config.settings.evidence_confidence_threshold", 0.3)
    _stub(monkeypatch, [_hit(1, 0.9), _hit(2, 0.3), _hit(3, 0.3)], useful=False)
    r = await evidence.gather_evidence("猫窝能水洗吗")
    assert r["sufficient"] is False and r["source"] == "self_check"
    assert r["snapshot"]                     # 自评不过同样留快照


async def test_strong_evidence_keeps_snapshot(monkeypatch):
    monkeypatch.setattr("app.config.settings.evidence_confidence_threshold", 0.3)
    _stub(monkeypatch, [_hit(1, 0.9), _hit(2, 0.4), _hit(3, 0.3)])
    r = await evidence.gather_evidence("猫窝能水洗吗")
    assert r["sufficient"] is True
    assert len(r["snapshot"]) == 3
    assert r["citations"]

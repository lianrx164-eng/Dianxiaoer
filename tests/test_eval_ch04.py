import pytest

from scripts import eval_ch04 as ev


def test_retrieval_scores_single_and_cross_doc():
    sample = {"expect_sections_all": [["运费"], ["退款"]]}
    hits = [{"section_path": "商品 FAQ / 运费怎么算"},
            {"section_path": "退货政策 / 退款时效"}]
    recall, mrr = ev._retrieval_scores(sample, hits)
    assert recall == 1.0
    assert mrr == pytest.approx((1 + 0.5) / 2)


def test_retrieval_scores_partial_when_group_missing():
    sample = {"expect_sections_all": [["运费"], ["退款"]]}
    hits = [{"section_path": "商品 FAQ / 运费怎么算"}]
    recall, mrr = ev._retrieval_scores(sample, hits)
    assert recall == 0.5
    assert mrr == pytest.approx(0.5)


def test_coverage_mech_counts_points_in_evidence():
    hits = [{"question": "运费", "answer": "满 99 元包邮,未满 10 元"}]
    assert ev._coverage_mech(["满99元包邮", "10元"], hits) == 1.0
    assert ev._coverage_mech(["不存在的要点"], hits) == 0.0


@pytest.mark.asyncio
async def test_refusal_mechanical_gate_and_self_check_gate(monkeypatch):
    d = [{"id": "D1", "query": "数字人民币"},
         {"id": "D2", "query": "货到付款"},
         {"id": "D3", "query": "线下门店"}]
    hits = {"hybrid_rerank": [{"rerank_score": 0.9, "question": "q", "answer": "a"}]}
    hits_map = {("D1", "hybrid_rerank"): [],
                ("D2", "hybrid_rerank"): hits["hybrid_rerank"],
                ("D3", "hybrid_rerank"): hits["hybrid_rerank"]}
    checks = {"货到付款": {"useful": False, "reason": "不相关"},
              "线下门店": {"useful": True, "reason": "够"}}

    async def fake_check(query, texts):
        return checks[query]

    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    out = await ev._refusal(d, hits_map, {})
    assert out["judged"] == 3 and out["refused"] == 2
    assert out["rate"] == pytest.approx(2 / 3, abs=1e-3)


@pytest.mark.asyncio
async def test_refusal_self_check_failure_is_not_judged(monkeypatch):
    d = [{"id": "D1", "query": "数字人民币"}]
    hits_map = {("D1", "hybrid_rerank"): [
        {"rerank_score": 0.9, "question": "q", "answer": "a"}]}

    async def fake_check(query, texts):
        raise RuntimeError("upstream down")

    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    out = await ev._refusal(d, hits_map, {})
    assert out["judged"] == 0 and out["rate"] is None

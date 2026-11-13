import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _report():
    return {
        "meta": {"eval_count": 300, "kb_count": 51, "judge_model": "deepseek-flash"},
        "retrieval": {
            "vector": {"overall": {"mrr": 0.5, "recall@5": 0.7},
                       "A_policy": {"mrr": 0.5, "recall@5": 0.7, "n": 60}},
            "hybrid_rerank": {"overall": {"mrr": 0.8, "recall@5": 0.9},
                              "A_policy": {"mrr": 0.8, "recall@5": 0.9, "n": 60}},
        },
        "evidence_coverage": {"vector": {"overall": 0.6}, "hybrid_rerank": {"overall": 0.9}},
        "generation": {
            "answer_coverage": {"hybrid_rerank": {"overall": 0.8}},
            "faithfulness": {"hybrid_rerank": {"overall": 1.0, "cases": []}},
            "refusal": {"rate": 0.95, "refused": 57, "total": 60},
        },
    }


async def test_missing_report_is_a_state(client, monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.rageval.REPORT_PATH", tmp_path / "nope.json")
    r = await client.get("/api/rag-eval/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["present"] is False
    assert body["job"]["name"] == "eval-rag"
    assert body["retrieval"] == {}


async def test_report_numbers_are_verbatim_and_best_is_max_mrr(client, monkeypatch, tmp_path):
    path = tmp_path / "rag_eval.json"
    data = _report()
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr("app.api.rageval.REPORT_PATH", path)
    r = await client.get("/api/rag-eval/overview")
    body = r.json()
    assert body["present"] is True
    assert body["retrieval"] == data["retrieval"]
    assert body["evidence_coverage"] == data["evidence_coverage"]
    assert body["best"]["strategy"] == "hybrid_rerank"
    assert body["generation_done"] is True


async def test_generation_null_still_serves_retrieval(client, monkeypatch, tmp_path):
    path = tmp_path / "rag_eval.json"
    data = _report()
    data["generation"] = None
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr("app.api.rageval.REPORT_PATH", path)
    body = (await client.get("/api/rag-eval/overview")).json()
    assert body["present"] is True and body["retrieval"]
    assert body["generation"] is None and body["generation_done"] is False


async def test_corrupt_report_is_treated_as_missing(client, monkeypatch, tmp_path):
    path = tmp_path / "rag_eval.json"
    path.write_text('{"meta": ', encoding="utf-8")
    monkeypatch.setattr("app.api.rageval.REPORT_PATH", path)
    body = (await client.get("/api/rag-eval/overview")).json()
    assert body["present"] is False

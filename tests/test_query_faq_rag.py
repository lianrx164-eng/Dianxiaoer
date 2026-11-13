import pytest

from app.tools import business


def _hit(i, score=0.9):
    return {"id": i, "rerank_score": score, "question": f"q{i}", "answer": f"a{i}",
            "section_path": f"path{i}", "content_type": "faq", "category": "运费"}


def _patch_understand(monkeypatch, standard="标准问法", expanded=None):
    async def fake_understand(q):
        return {"standard": standard, "expanded": expanded or []}
    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)


@pytest.mark.asyncio
async def test_query_faq_sufficient(monkeypatch):
    _patch_understand(monkeypatch)
    seen = {}

    async def fake_search(query, **kw):
        seen["query"] = query
        seen.update(kw)
        return [_hit(1, 0.9), _hit(2, 0.6)]

    async def fake_check(q, texts):
        return {"useful": True, "reason": "够"}

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)

    out = await business.query_faq.ainvoke({"keyword": "邮费多少"})
    assert out["sufficient"] is True
    assert out["citations"][0]["n"] == 1 and out["citations"][0]["section_path"]
    assert "[1]" in out["evidence"]
    assert seen["strategy"] == "hybrid_rerank"


@pytest.mark.asyncio
async def test_query_faq_retrieval_low_conf(monkeypatch):
    _patch_understand(monkeypatch)

    async def fake_search(query, **kw):
        return []

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    out = await business.query_faq.ainvoke({"keyword": "火星车怎么买"})
    assert out["sufficient"] is False and out["source"] == "retrieval_low_conf"


@pytest.mark.asyncio
async def test_query_faq_self_check_fail(monkeypatch):
    _patch_understand(monkeypatch)

    async def fake_search(query, **kw):
        return [_hit(1, 0.8)]

    async def fake_check(q, texts):
        return {"useful": False, "reason": "问型号但证据只讲运费"}

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    out = await business.query_faq.ainvoke({"keyword": "Pro型号能自动铲屎吗"})
    assert out["sufficient"] is False and out["source"] == "self_check"
    assert "型号" in out["reason"]


@pytest.mark.asyncio
async def test_query_faq_expands_bm25_only(monkeypatch):
    _patch_understand(monkeypatch, standard="标准问法", expanded=["运费", "邮费"])
    seen = {}

    async def fake_search(query, **kw):
        seen["query"] = query
        seen["bm25_query"] = kw.get("bm25_query")
        return []

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    await business.query_faq.ainvoke({"keyword": "邮费多少"})
    assert seen["query"] == "标准问法"
    assert "运费" in seen["bm25_query"] and "邮费" in seen["bm25_query"]

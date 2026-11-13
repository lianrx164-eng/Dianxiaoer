from app.tools.business import query_faq


def _hit():
    return {"id": 1, "rerank_score": 0.8, "question": "运费怎么算",
            "answer": "满99包邮", "section_path": "运费政策",
            "content_type": "faq", "category": "运费"}


async def test_query_faq_maps_hits_to_citations(monkeypatch):
    async def fake_understand(q):
        return {"standard": q, "expanded": []}

    async def fake_search(query, **kw):
        return [_hit()]

    async def fake_check(q, texts):
        return {"useful": True, "reason": "够"}

    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    out = await query_faq.ainvoke({"keyword": "邮费是多少"})
    assert out["sufficient"] is True
    assert out["citations"][0]["question"] == "运费怎么算"
    assert out["citations"][0]["section_path"] == "运费政策"


async def test_query_faq_empty_returns_refusal(monkeypatch):
    async def fake_understand(q):
        return {"standard": q, "expanded": []}

    async def fake_search(query, **kw):
        return []

    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "无关问题"})
    assert out["sufficient"] is False
    assert "检索证据不足" in out["reason"]

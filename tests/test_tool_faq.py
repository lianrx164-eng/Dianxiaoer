from app.tools.business import query_faq


async def test_query_faq_passes_keyword_through(monkeypatch):
    seen = {}

    async def fake_understand(q):
        return {"standard": q, "expanded": []}

    async def fake_search(query, **kw):
        seen["query"] = query
        return []

    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "退货"})
    assert seen["query"] == "退货"
    assert out["sufficient"] is False and out["source"] == "retrieval_low_conf"

from app.tools.business import query_faq


async def test_query_faq_passes_keyword_through(monkeypatch):
    seen = {}

    async def fake_search(keyword, **kw):
        seen["keyword"] = keyword
        return []

    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "退货"})
    assert seen["keyword"] == "退货"
    assert out["hits"] == []
    assert "退货" in out["message"]

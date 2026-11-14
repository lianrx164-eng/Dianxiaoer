from httpx import ASGITransport, AsyncClient

from app.main import app


async def test_admin_overview_shows_six_cards_even_when_deps_down(monkeypatch):
    async def db_down(*a, **k):
        raise RuntimeError("mysql down")

    def milvus_down(*a, **k):
        raise RuntimeError("milvus down")

    def jobs_down(*a, **k):
        raise RuntimeError("jobs down")

    monkeypatch.setattr("app.db.repository.knowledge_stats", db_down)
    monkeypatch.setattr("app.db.repository.staging_stats", db_down)
    monkeypatch.setattr("app.db.repository.list_eval_runs", db_down)
    monkeypatch.setattr("app.api.admin.milvus_client.get_client", milvus_down)
    monkeypatch.setattr("app.core.jobs.status_all", jobs_down)
    monkeypatch.setattr("app.api.rageval.load_report", lambda: None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/admin/overview")
    assert r.status_code == 200
    cards = r.json()["cards"]
    assert len(cards) == 6
    assert all(c["status"] == "读数失败" for c in cards[:5])
    assert cards[5]["key"] == "rag-eval" and cards[5]["status"] == "没跑过"

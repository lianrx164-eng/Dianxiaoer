from httpx import ASGITransport, AsyncClient

from app.main import app


async def test_admin_overview_shows_all_cards_even_when_deps_down(monkeypatch, tmp_path):
    async def db_down(*a, **k):
        raise RuntimeError("mysql down")

    def milvus_down(*a, **k):
        raise RuntimeError("milvus down")

    def jobs_down(*a, **k):
        raise RuntimeError("jobs down")

    async def probe():
        return {"ok": False, "url": "http://127.0.0.1:9"}

    monkeypatch.setattr("app.db.repository.knowledge_stats", db_down)
    monkeypatch.setattr("app.db.repository.staging_stats", db_down)
    monkeypatch.setattr("app.db.repository.list_eval_runs", db_down)
    monkeypatch.setattr("app.db.repository.list_review_queue", db_down)
    monkeypatch.setattr("app.db.repository.topic_distribution", db_down)
    monkeypatch.setattr("app.api.admin.milvus_client.get_client", milvus_down)
    monkeypatch.setattr("app.core.jobs.status_all", jobs_down)
    monkeypatch.setattr("app.api.rageval.load_report", lambda: None)
    monkeypatch.setattr("app.api.acceptance.DATA", tmp_path)
    monkeypatch.setattr("app.api.acceptance.REPORTS", tmp_path / "reports")
    monkeypatch.setattr("app.api.acceptance._probe", probe)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/admin/overview")
    assert r.status_code == 200
    cards = r.json()["cards"]
    assert [c["key"] for c in cards] == [
        "inventory", "dualwrite", "staging", "jobs", "observability", "rag-eval",
        "review", "topics", "acceptance"]
    assert all(c["status"] == "读数失败" for c in cards[:5])
    assert cards[5]["status"] == "没跑过"
    assert cards[6]["status"] == "读数失败"      # 飞轮待审:MySQL 读不到
    assert cards[7]["status"] == "读数失败"      # 主题分布:MySQL 读不到
    assert cards[8]["status"] == "要干活"        # 分类器验收:0/9 且服务未起
    assert "0/9" in cards[8]["conclusion"] and "未起" in cards[8]["conclusion"]

from httpx import ASGITransport, AsyncClient

from app.main import app


async def test_admin_pages_and_shared_nav():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        for path in ("/kb", "/admin", "/rag-eval", "/review", "/observability"):
            r = await c.get(path)
            assert r.status_code == 200
            assert "/static/admin.js" in r.text
        r = await c.get("/")
        assert r.status_code == 200 and "/admin" in r.text
        assert 'id="sidebar"' in r.text and "/api/conversations" in r.text
        for asset in ("/static/admin.js", "/static/acceptance.js", "/static/acceptance.css"):
            assert (await c.get(asset)).status_code == 200

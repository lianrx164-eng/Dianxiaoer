import pytest
from httpx import ASGITransport, AsyncClient

from app.db import repository
from app.main import app

MD = "# 售后政策\n\n## 运费说明\n\n单笔订单满99元包邮,未满收取10元运费。"


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_preview_is_dry_run(client, db_session_factory, db_clean):
    r = await client.post("/api/kb/preview", json={"markdown": MD, "content_type": "policy"})
    assert r.status_code == 200
    body = r.json()
    assert body["counts"]["chunks"] >= 1
    stats = await repository.knowledge_stats()
    assert stats["total"] == 0  # 预览只切不写


async def test_ingest_then_reingest_is_idempotent(client, db_session_factory, db_clean):
    payload = {"markdown": MD, "content_type": "policy", "vectorize": False}
    r1 = await client.post("/api/kb/ingest", json=payload)
    assert r1.status_code == 200 and r1.json()["inserted"] == 1
    r2 = await client.post("/api/kb/ingest", json=payload)
    assert r2.status_code == 200
    assert r2.json()["inserted"] == 0 and r2.json()["skipped"] == 1
    assert (await repository.knowledge_stats())["total"] == 1


async def test_ingest_keeps_multiple_pieces_of_one_section(client, db_session_factory, db_clean):
    rows = "\n".join(f"| 商品{i} | {i} |" for i in range(1, 15))
    md = f"# 价格表\n\n## 价目\n\n| 商品 | 价格 |\n| --- | --- |\n{rows}"
    r = await client.post("/api/kb/ingest", json={
        "markdown": md, "content_type": "manual", "table_max_rows": 5, "vectorize": False,
    })
    assert r.status_code == 200
    assert r.json()["inserted"] >= 2  # 同节多块共用问法,只按问法查重会误杀


async def test_preview_rejects_bad_input(client, db_session_factory, db_clean):
    bad_payloads = [
        {"markdown": "   ", "content_type": "policy"},
        {"markdown": MD, "content_type": "nope"},
        {"filename": "../secret.md"},
        {"filename": "nope.md"},
    ]
    for payload in bad_payloads:
        r = await client.post("/api/kb/preview", json=payload)
        assert r.status_code == 400, payload


async def test_reset_clears_mysql_and_milvus(client, db_session_factory, db_clean, monkeypatch):
    await repository.insert_knowledge_chunk("c", "q", "a")

    class FakeMilvus:
        def __init__(self):
            self.dropped = []

        def has_collection(self, name):
            return True

        def drop_collection(self, name):
            self.dropped.append(name)

        def close(self):
            pass

    fake = FakeMilvus()
    monkeypatch.setattr("app.api.kb.milvus_client.get_client", lambda: fake)
    r = await client.post("/api/kb/reset")
    assert r.status_code == 200
    assert fake.dropped == ["knowledge"]
    assert (await repository.knowledge_stats())["total"] == 0

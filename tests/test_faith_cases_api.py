import pytest
from httpx import ASGITransport, AsyncClient

from app.db import repository
from app.main import app


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _seed(eval_id="A43", status=None, resolution=None):
    row = await repository.upsert_faith_case(
        eval_id, "A_policy", "问题 " + eval_id, "hybrid_rerank", "答案", "理由",
        [{"n": 1, "chunk_id": 7, "section_path": "运费政策",
          "question": "运费", "answer": "满99包邮"}], "deepseek-flash")
    if status:
        row = await repository.set_faith_case_status(row["id"], status, resolution)
    return row


@pytest.mark.asyncio
async def test_one_row_per_eval_id_and_seen_count(client, db_session_factory, db_clean):
    await _seed()
    row = await _seed()
    assert row["seen_count"] == 2
    body = (await client.get("/api/rag-eval/faith-cases")).json()
    assert body["total"] == 1
    assert body["items"][0]["eval_id"] == "A43"


@pytest.mark.asyncio
async def test_recurrence_reopens_disposed_case(client, db_session_factory, db_clean):
    await _seed(status="已解决", resolution="补了知识库条款")
    row = await _seed()
    assert row["recurrence"] is True and row["status"] == "未解决"
    body = (await client.get("/api/rag-eval/faith-cases")).json()
    assert body["items"][0]["resolution"] is None


@pytest.mark.asyncio
async def test_status_filter_and_pagination(client, db_session_factory, db_clean):
    await _seed("A1", status="已解决", resolution="已修")
    await _seed("A2")
    await _seed("A3")
    body = (await client.get("/api/rag-eval/faith-cases?status=未解决&page=1&size=1")).json()
    assert body["total"] == 2 and len(body["items"]) == 1
    assert body["counts"] == {"未解决": 2, "已解决": 1, "无需解决": 0}


@pytest.mark.asyncio
async def test_citations_snapshot_is_served(client, db_session_factory, db_clean):
    await _seed()
    body = (await client.get("/api/rag-eval/faith-cases")).json()
    cits = body["items"][0]["citations"]
    assert cits[0]["chunk_id"] == 7 and cits[0]["section_path"] == "运费政策"


@pytest.mark.asyncio
async def test_resolution_required_for_dispose(client, db_session_factory, db_clean):
    row = await _seed()
    r = await client.post(f"/api/rag-eval/faith-cases/{row['id']}/status",
                          json={"status": "已解决", "resolution": "   "})
    assert r.status_code == 400
    r = await client.post(f"/api/rag-eval/faith-cases/{row['id']}/status",
                          json={"status": "已解决", "resolution": "补了条款"})
    assert r.status_code == 200 and r.json()["resolution"] == "补了条款"


@pytest.mark.asyncio
async def test_reopen_clears_resolution(client, db_session_factory, db_clean):
    row = await _seed(status="无需解决", resolution="不是知识库问题")
    r = await client.post(f"/api/rag-eval/faith-cases/{row['id']}/status",
                          json={"status": "未解决"})
    assert r.status_code == 200
    assert r.json()["resolution"] is None and r.json()["resolved_at"] is None


@pytest.mark.asyncio
async def test_invalid_status_is_422(client, db_session_factory, db_clean):
    row = await _seed()
    r = await client.post(f"/api/rag-eval/faith-cases/{row['id']}/status",
                          json={"status": "随便"})
    assert r.status_code == 422

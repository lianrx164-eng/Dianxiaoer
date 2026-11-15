"""ch10 主题分布与类目问题列表 API:17 类全出、多标签在每类都出现、未知类目 400。"""
import httpx
import pytest
from fastapi import FastAPI

from app.api import topics as topics_api
from app.db import repository
from app.db.models import LowConfidenceQuestion, ReviewQueue


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(topics_api.router)
    return a


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _seed(session_factory, raw, norm, labels, count=1):
    async with session_factory() as s:
        rq = ReviewQueue(normalized_question=norm, occurrence_count=count)
        s.add(rq)
        await s.flush()
        q = LowConfidenceQuestion(raw_question=raw, source="retrieval_low_conf",
                                  matched_review_id=rq.id)
        s.add(q)
        await s.commit()
        qid = q.id
    await repository.insert_topic_classifications([{"question_id": qid, "labels": labels}])
    return qid


async def test_distribution_empty_returns_all_17(client, db_session_factory, db_clean):
    resp = await client.get("/api/topics/distribution")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert len(body["classes"]) == 17


async def test_distribution_counts(client, db_session_factory, db_clean):
    await _seed(db_session_factory, "猫窝买大了想退", "猫窝买大了能不能退", ["尺码", "退换货"])
    body = (await client.get("/api/topics/distribution")).json()
    by_label = {c["label"]: c for c in body["classes"]}
    assert body["total"] == 1
    assert by_label["尺码"]["count"] == 1 and by_label["退换货"]["count"] == 1
    assert by_label["物流"]["count"] == 0


async def test_questions_list_by_label(client, db_session_factory, db_clean):
    await _seed(db_session_factory, "猫窝买大了", "猫窝买大了能不能退", ["尺码"],
                count=3)
    body = (await client.get("/api/topics/questions", params={"label": "尺码"})).json()
    assert body["label"] == "尺码" and body["total"] == 1 and body["page"] == 1
    item = body["items"][0]
    assert item["text"] == "猫窝买大了能不能退"
    assert item["normalized"] is True
    assert item["raw_question"] == "猫窝买大了"
    assert item["labels"] == ["尺码"]
    assert item["source"] == "retrieval_low_conf"
    assert item["occurrence_count"] == 3
    assert item["review_status"] == "待审"


async def test_multilabel_appears_under_each_label(client, db_session_factory, db_clean):
    await _seed(db_session_factory, "买大了想退", "猫窝买大了能不能退", ["尺码", "退换货"])
    for label in ("尺码", "退换货"):
        body = (await client.get("/api/topics/questions", params={"label": label})).json()
        assert body["total"] == 1
        assert body["items"][0]["labels"] == ["尺码", "退换货"]


async def test_unknown_label_is_400(client, db_clean):
    r = await client.get("/api/topics/questions", params={"label": "瞎编类目"})
    assert r.status_code == 400


async def test_pagination(client, db_session_factory, db_clean):
    for i in range(3):
        await _seed(db_session_factory, f"退货问题{i}", f"退货问题标准化{i}", ["退换货"])
    body = (await client.get("/api/topics/questions",
                             params={"label": "退换货", "page": 2, "size": 2})).json()
    assert body["total"] == 3 and body["pages"] == 2 and len(body["items"]) == 1


async def test_unclassified_absent(client, db_session_factory, db_clean):
    body = (await client.get("/api/topics/questions", params={"label": "退换货"})).json()
    assert body["total"] == 0 and body["items"] == []

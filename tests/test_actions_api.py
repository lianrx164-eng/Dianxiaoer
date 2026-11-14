import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.api import actions
from app.main import app


def test_create_ticket_action_writes(monkeypatch):
    async def fake_create(cid, desc, ttype):
        assert cid == 5 and ttype == "投诉" and desc == "东西坏了"
        return "T20260715001"

    monkeypatch.setattr(actions.repository, "create_ticket", fake_create)
    r = TestClient(app).post("/api/actions/create-ticket", json={
        "conversation_id": 5, "description": "东西坏了", "ticket_type": "投诉"})
    assert r.status_code == 200
    assert r.json()["ticket_no"] == "T20260715001"


def test_create_ticket_rejects_bad_type():
    r = TestClient(app).post("/api/actions/create-ticket", json={
        "conversation_id": 5, "description": "x", "ticket_type": "乱填"})
    assert r.status_code == 422


def test_create_ticket_rejects_empty_description():
    r = TestClient(app).post("/api/actions/create-ticket", json={
        "conversation_id": 5, "description": "", "ticket_type": "咨询"})
    assert r.status_code == 422


def test_create_ticket_db_error_returns_503(monkeypatch):
    async def fake_create(*a, **k):
        raise SQLAlchemyError("db down")

    monkeypatch.setattr(actions.repository, "create_ticket", fake_create)
    r = TestClient(app).post("/api/actions/create-ticket", json={
        "conversation_id": 5, "description": "x", "ticket_type": "咨询"})
    assert r.status_code == 503


def test_resume_requires_order_id_or_confirmed():
    r = TestClient(app).post("/api/actions/resume", json={"conversation_id": 5})
    assert r.status_code == 400


@pytest.fixture()
def pending(monkeypatch):
    state = {"value": {"type": "confirm_ticket", "preview": {}}}

    async def fake_pending(cid):
        return state["value"]

    monkeypatch.setattr(actions.runtime, "pending_interrupt", fake_pending)
    return state


def test_resume_rejects_when_nothing_pending(monkeypatch):
    async def none_pending(cid):
        return None

    monkeypatch.setattr(actions.runtime, "pending_interrupt", none_pending)
    r = TestClient(app).post("/api/actions/resume", json={
        "conversation_id": 5, "confirmed": True})
    assert r.status_code == 400


def test_resume_rejects_mismatched_payload(pending):
    # 当前在等订单选择,却传 confirmed → 不能把 dict 当订单号
    pending["value"] = {"type": "select_order", "orders": []}
    r = TestClient(app).post("/api/actions/resume", json={
        "conversation_id": 5, "confirmed": True})
    assert r.status_code == 400

    # 当前在等工单确认,却传 order_id
    pending["value"] = {"type": "confirm_ticket", "preview": {}}
    r = TestClient(app).post("/api/actions/resume", json={
        "conversation_id": 5, "order_id": "1001"})
    assert r.status_code == 400


def test_resume_confirmed_builds_confirmation_payload(monkeypatch, pending):
    seen = {}

    async def fake_stream(cid, value):
        seen["cid"] = cid
        seen["value"] = value
        if False:
            yield {}

    monkeypatch.setattr(actions.runtime, "stream_resume", fake_stream)
    r = TestClient(app).post("/api/actions/resume", json={
        "conversation_id": 5, "confirmed": True})
    assert r.status_code == 200
    assert seen == {"cid": 5, "value": {"confirmed": True}}


def test_resume_order_id_stays_string(monkeypatch, pending):
    pending["value"] = {"type": "select_order", "orders": []}
    seen = {}

    async def fake_stream(cid, value):
        seen["value"] = value
        if False:
            yield {}

    monkeypatch.setattr(actions.runtime, "stream_resume", fake_stream)
    r = TestClient(app).post("/api/actions/resume", json={
        "conversation_id": 5, "order_id": "1001"})
    assert r.status_code == 200 and seen["value"] == "1001"

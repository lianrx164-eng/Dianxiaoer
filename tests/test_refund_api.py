from fastapi.testclient import TestClient

from app.api import actions
from app.main import app


def test_create_refund_writes_ticket(monkeypatch):
    async def fake_create(cid, desc, ttype):
        assert cid == 7 and ttype == "退款"
        assert "1001" in desc and "质量问题" in desc
        return "T20260716001"

    monkeypatch.setattr(actions.repository, "create_ticket", fake_create)
    r = TestClient(app).post("/api/actions/create-refund", json={
        "conversation_id": 7, "order_id": "1001", "reason": "质量问题"})
    assert r.status_code == 200
    assert r.json() == {"ticket_no": "T20260716001", "status": "退款申请已提交"}


def test_create_refund_rejects_bad_reason():
    r = TestClient(app).post("/api/actions/create-refund", json={
        "conversation_id": 7, "order_id": "1001", "reason": "乱填"})
    assert r.status_code == 422

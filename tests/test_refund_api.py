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


def test_resume_endpoint_streams(monkeypatch):
    async def fake_pending(cid):
        return {"type": "select_order", "orders": []}

    async def fake_stream_resume(cid, resume_value):
        assert cid == 3 and resume_value == "1001"
        yield {"type": "delta", "text": "这一单可以退款"}
        yield {"type": "actions",
               "items": [{"type": "refund_form", "draft": {"order_id": "1001"}}]}
        yield {"type": "done", "conversation_id": 3}

    monkeypatch.setattr(actions.runtime, "pending_interrupt", fake_pending)
    monkeypatch.setattr(actions.runtime, "stream_resume", fake_stream_resume)
    r = TestClient(app).post("/api/actions/resume",
                             json={"conversation_id": 3, "order_id": "1001"})
    assert r.status_code == 200
    body = r.text
    assert "这一单可以退款" in body
    assert "refund_form" in body and "[DONE]" in body

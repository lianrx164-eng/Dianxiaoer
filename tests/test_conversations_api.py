from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import conversations
from app.main import app


def test_list_conversations_passes_user_id(monkeypatch):
    async def fake_list(uid):
        assert uid == "u1"
        return [{"id": 2, "status": "进行中", "preview": "第二问",
                 "has_summary": True, "created_at": "2026-11-14T10:00:00"}]
    monkeypatch.setattr(conversations.repository, "list_conversations", fake_list)
    r = TestClient(app).get("/api/conversations?user_id=u1")
    assert r.status_code == 200
    assert r.json()["items"][0]["id"] == 2


def test_messages_404(monkeypatch):
    async def fake_get(cid):
        return None
    monkeypatch.setattr(conversations.repository, "get_conversation", fake_get)
    r = TestClient(app).get("/api/conversations/9/messages")
    assert r.status_code == 404


def test_messages_returns_user_assistant(monkeypatch):
    async def fake_get(cid):
        return SimpleNamespace(id=cid)

    async def fake_rows(cid):
        return [SimpleNamespace(role="user", content="q", created_at=None),
                SimpleNamespace(role="assistant", content="a", created_at=None)]
    monkeypatch.setattr(conversations.repository, "get_conversation", fake_get)
    monkeypatch.setattr(conversations.repository, "list_dialog_messages", fake_rows)
    r = TestClient(app).get("/api/conversations/3/messages")
    assert r.status_code == 200
    assert [i["role"] for i in r.json()["items"]] == ["user", "assistant"]

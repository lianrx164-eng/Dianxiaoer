from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime
from app.main import app


def _fake_state():
    ai = AIMessage("", tool_calls=[
        {"name": "query_logistics", "args": {"tracking_no": "SF1"}, "id": "c1"}])
    tm = ToolMessage(content='{"status":"运输中"}', tool_call_id="c1", name="query_logistics")
    return {
        "conversation_id": 12,
        "answer": "",
        "messages": [HumanMessage("订单 1001 到哪了"), ai, tm, AIMessage("订单 1001 正在运输中。")],
        "suggested_actions": [{"type": "transfer_human"}],
        "steps": 2,
        "trace": {"route": "business", "forced_rag": False},
    }


def test_agent_endpoint_returns_tool_trace(monkeypatch):
    async def fake_run(user_id, message, conversation_id):
        return {"conversation_id": 12, "state": _fake_state()}

    monkeypatch.setattr(runtime, "run_turn", fake_run)
    r = TestClient(app).post("/api/agent", json={"user_id": "u1", "message": "订单 1001 到哪了"})
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == 12
    assert body["answer"] == "订单 1001 正在运输中。"
    assert body["tool_calls"][0]["name"] == "query_logistics"
    assert body["tool_results"][0]["ok"] is True
    assert body["suggested_actions"] == [{"type": "transfer_human"}]
    assert body["steps"] == 2
    assert body["trace"]["route"] == "business"


def test_agent_endpoint_404_on_unknown_conversation(monkeypatch):
    async def fake_run(*a, **k):
        raise runtime.ConversationNotFound(999)

    monkeypatch.setattr(runtime, "run_turn", fake_run)
    r = TestClient(app).post("/api/agent", json={"user_id": "u1", "message": "hi", "conversation_id": 999})
    assert r.status_code == 404


def test_agent_endpoint_422_on_missing_fields():
    assert TestClient(app).post("/api/agent", json={"message": "hi"}).status_code == 422


def test_agent_endpoint_503_on_db_error(monkeypatch):
    async def fake_run(*a, **k):
        raise SQLAlchemyError("db down")

    monkeypatch.setattr(runtime, "run_turn", fake_run)
    r = TestClient(app).post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.status_code == 503
    assert "数据库" in r.json()["detail"]


def test_agent_endpoint_502_on_upstream_error(monkeypatch):
    async def fake_run(*a, **k):
        raise RuntimeError("upstream boom")

    monkeypatch.setattr(runtime, "run_turn", fake_run)
    r = TestClient(app).post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.status_code == 502
    assert "上游" in r.json()["detail"]


def test_agent_endpoint_surfaces_interrupt(monkeypatch):
    async def fake_run(*a, **k):
        return {"conversation_id": 3, "state": _fake_state(),
                "interrupt": [{"order_id": "1001"}]}

    monkeypatch.setattr(runtime, "run_turn", fake_run)
    r = TestClient(app).post("/api/agent", json={"user_id": "u1", "message": "我要退款"})
    assert r.status_code == 200
    assert r.json()["interrupt"] == [{"order_id": "1001"}]

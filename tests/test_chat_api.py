import json

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from sqlalchemy.exc import SQLAlchemyError

from app.api import chat as chat_api
from app.core import agent
from app.db import repository as repo
from app.main import app
from fake_model import FakeModel


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def use_model(model: FakeModel) -> None:
    app.dependency_overrides[chat_api.get_model] = lambda: model


def parse_sse(text: str) -> dict:
    """把 SSE 文本解析成 {tools: [...], deltas: str, done: dict|None, error: str|None, finished: bool}。"""
    out = {"tools": [], "deltas": "", "done": None, "error": None, "finished": False}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("event: error"):
            payload = lines[i + 1]
            out["error"] = json.loads(payload[len("data: "):])["message"]
            i += 2
            continue
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload == "[DONE]":
                out["finished"] = True
            else:
                ev = json.loads(payload)
                if ev.get("event") == "tool":
                    out["tools"].append(ev["name"])
                elif ev.get("event") == "done":
                    out["done"] = ev
                elif "delta" in ev:
                    out["deltas"] += ev["delta"]
        i += 1
    return out


async def _post_chat(payload: dict) -> tuple[int, str]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/chat", json=payload)
        return resp.status_code, resp.text


async def test_chat_with_tool_streams_tool_delta_done(db_session_factory, db_clean):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    use_model(FakeModel([first], stream_tokens=["您的订单", "正在运输中。"]))

    status, text = await _post_chat({"user_id": "u1", "message": "订单 1001 到哪了"})
    sse = parse_sse(text)
    assert status == 200
    assert sse["tools"] == ["query_logistics"]
    assert sse["deltas"] == "您的订单正在运输中。"
    assert sse["done"] is not None and isinstance(sse["done"]["conversation_id"], int)
    assert sse["finished"] is True

    msgs = await repo.list_messages(sse["done"]["conversation_id"])
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]


async def test_chat_without_tool_has_no_tool_frame(db_session_factory, db_clean):
    use_model(FakeModel([AIMessage(content="你好,有什么可以帮您?")]))

    status, text = await _post_chat({"user_id": "u1", "message": "你好"})
    sse = parse_sse(text)
    assert status == 200
    assert sse["tools"] == []
    assert sse["deltas"] == "你好,有什么可以帮您?"
    assert sse["done"] is not None


async def test_chat_unknown_conversation_emits_error_frame(db_session_factory, db_clean):
    use_model(FakeModel([AIMessage(content="hi")]))

    status, text = await _post_chat(
        {"user_id": "u1", "message": "hi", "conversation_id": 999999})
    sse = parse_sse(text)
    assert status == 200                      # 流已发 200,错误只能走帧
    assert sse["error"] == "会话不存在"
    assert sse["finished"] is False           # 错误路径不发 [DONE]


async def test_chat_empty_message_422():
    status, _ = await _post_chat({"user_id": "u1", "message": ""})
    assert status == 422


async def test_chat_db_error_emits_error_frame(db_session_factory, db_clean, monkeypatch):
    async def fake_stream(*a, **k):
        raise SQLAlchemyError("db down")
        yield  # pragma: no cover - 使其成为 async generator
    monkeypatch.setattr(agent, "stream_agent_turn", fake_stream)
    use_model(FakeModel([]))

    status, text = await _post_chat({"user_id": "u1", "message": "hi"})
    sse = parse_sse(text)
    assert status == 200
    assert sse["error"] == "数据库暂时不可用,请稍后重试"
    assert sse["finished"] is False


async def test_chat_upstream_error_emits_error_frame(db_session_factory, db_clean, monkeypatch):
    async def fake_stream(*a, **k):
        raise RuntimeError("upstream boom")
        yield  # pragma: no cover
    monkeypatch.setattr(agent, "stream_agent_turn", fake_stream)
    use_model(FakeModel([]))

    status, text = await _post_chat({"user_id": "u1", "message": "hi"})
    sse = parse_sse(text)
    assert status == 200
    assert sse["error"] == "上游模型暂时不可用,请稍后重试"
    assert sse["finished"] is False

import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime
from app.main import app


def parse_sse(text: str) -> dict:
    """把 SSE 文本解析成 {tools, deltas, actions, done, error, finished}。"""
    out = {"tools": [], "deltas": "", "actions": [], "done": None,
           "error": None, "finished": False, "interrupts": []}
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
                elif ev.get("event") == "actions":
                    out["actions"] = ev["items"]
                elif ev.get("event") == "interrupt":
                    out["interrupts"].append(ev)
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


async def test_chat_streams_delta_and_actions(monkeypatch):
    async def fake_stream(user_id, message, conversation_id):
        yield {"type": "tool", "name": "query_logistics"}
        yield {"type": "citations", "items": [{"n": 1}]}
        yield {"type": "delta", "text": "抱歉,"}
        yield {"type": "delta", "text": "我理解您的心情。"}
        yield {"type": "actions", "items": [{"type": "transfer_human"},
                                            {"type": "create_ticket", "draft": {}}]}
        yield {"type": "done", "conversation_id": 5}

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)
    status, text = await _post_chat({"user_id": "u1", "message": "我要投诉"})
    sse = parse_sse(text)
    assert status == 200
    assert sse["tools"] == ["query_logistics"]
    assert sse["deltas"] == "抱歉,我理解您的心情。"
    assert [a["type"] for a in sse["actions"]] == ["transfer_human", "create_ticket"]
    assert sse["done"] == {"event": "done", "conversation_id": 5}
    assert sse["finished"] is True


async def test_chat_unknown_conversation_emits_error_frame(monkeypatch):
    async def fake_stream(*a, **k):
        raise runtime.ConversationNotFound(999)
        yield  # pragma: no cover

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)
    status, text = await _post_chat(
        {"user_id": "u1", "message": "hi", "conversation_id": 999999})
    sse = parse_sse(text)
    assert status == 200                      # 流已发 200,错误只能走帧
    assert sse["error"] == "会话不存在"
    assert sse["finished"] is False           # 错误路径不发 [DONE]


async def test_chat_empty_message_422():
    status, _ = await _post_chat({"user_id": "u1", "message": ""})
    assert status == 422


async def test_chat_db_error_emits_error_frame(monkeypatch):
    async def fake_stream(*a, **k):
        raise SQLAlchemyError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)
    status, text = await _post_chat({"user_id": "u1", "message": "hi"})
    sse = parse_sse(text)
    assert status == 200
    assert sse["error"] == "数据库暂时不可用,请稍后重试"
    assert sse["finished"] is False


async def test_chat_upstream_error_emits_error_frame(monkeypatch):
    async def fake_stream(*a, **k):
        raise RuntimeError("upstream boom")
        yield  # pragma: no cover

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)
    status, text = await _post_chat({"user_id": "u1", "message": "hi"})
    sse = parse_sse(text)
    assert status == 200
    assert sse["error"] == "上游模型暂时不可用,请稍后重试"
    assert sse["finished"] is False


async def test_chat_maps_interrupt_frame(monkeypatch):
    async def fake_stream(*a, **k):
        yield {"type": "interrupt", "conversation_id": 5,
               "kind": "select_order", "orders": [{"order_id": "1001"}]}

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)
    status, text = await _post_chat({"user_id": "u1", "message": "我要退款"})
    sse = parse_sse(text)
    assert status == 200
    assert sse["interrupts"] == [{
        "event": "interrupt", "conversation_id": 5,
        "kind": "select_order", "orders": [{"order_id": "1001"}]}]
    assert sse["done"] is None           # 图暂停,没有 done 事件([DONE] 只是流终止)


async def test_chat_maps_confirm_ticket_preview_frame(monkeypatch):
    async def fake_stream(*a, **k):
        yield {"type": "interrupt", "conversation_id": 5,
               "kind": "confirm_ticket",
               "preview": {"ticket_type": "售后", "description": "猫砂盆漏电"}}

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)
    status, text = await _post_chat({"user_id": "u1", "message": "帮我建个工单"})
    sse = parse_sse(text)
    assert status == 200
    assert sse["interrupts"] == [{
        "event": "interrupt", "conversation_id": 5, "kind": "confirm_ticket",
        "preview": {"ticket_type": "售后", "description": "猫砂盆漏电"}}]
    assert sse["done"] is None

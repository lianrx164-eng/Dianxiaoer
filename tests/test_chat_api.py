import json

from fastapi.testclient import TestClient
from langchain_core.language_models import FakeListChatModel

from app.api import chat as chat_api
from app.main import app


def make_client(responses: list[str]) -> TestClient:
    fake = FakeListChatModel(responses=responses)
    app.dependency_overrides[chat_api.get_model] = lambda: fake
    return TestClient(app)


def collect_sse(resp) -> tuple[list[str], str]:
    """返回 (delta 列表, 终止帧)。"""
    deltas, last = [], ""
    for line in resp.iter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            last = payload
        else:
            deltas.append(json.loads(payload)["delta"])
    return deltas, last


def teardown_function():
    app.dependency_overrides.clear()
    chat_api.store._sessions.clear()


def test_chat_streams_tokens_and_done():
    client = make_client(["你好呀"])
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s1", "message": "在吗"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        deltas, last = collect_sse(resp)
    assert "".join(deltas) == "你好呀"
    assert len(deltas) > 1  # 逐 token,不是一次性整段
    assert last == "[DONE]"


def test_chat_persists_history_for_next_turn():
    client = make_client(["第一轮答", "第二轮答"])
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s2", "message": "第一问"}
    ) as r:
        collect_sse(r)
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s2", "message": "第二问"}
    ) as r:
        collect_sse(r)
    history = chat_api.store.get("s2")
    assert [m.content for m in history] == ["第一问", "第一轮答", "第二问", "第二轮答"]


def test_chat_validates_empty_message():
    client = make_client(["x"])
    assert (
        client.post("/api/chat", json={"session_id": "s3", "message": ""}).status_code
        == 422
    )


class ErrorModel:
    async def astream(self, messages):
        raise RuntimeError("upstream down")
        yield  # pragma: no cover


class RecordingModel:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[list] = []

    async def astream(self, messages):
        self.calls.append(messages)
        from langchain_core.messages import AIMessageChunk

        yield AIMessageChunk(content=self.reply)


def test_chat_emits_error_frame_and_keeps_store_clean():
    app.dependency_overrides[chat_api.get_model] = lambda: ErrorModel()
    client = TestClient(app)
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s4", "message": "在吗"}
    ) as r:
        lines = list(r.iter_lines())
    assert "event: error" in lines
    assert any(line.startswith("data: ") and "上游" in line for line in lines)
    assert "data: [DONE]" not in lines
    assert chat_api.store.get("s4") == []


def test_chat_second_turn_receives_first_turn_context():
    model = RecordingModel("好的")
    app.dependency_overrides[chat_api.get_model] = lambda: model
    client = TestClient(app)
    for msg in ("第一问", "第二问"):
        with client.stream(
            "POST", "/api/chat", json={"session_id": "s5", "message": msg}
        ) as r:
            list(r.iter_lines())
    second_call_contents = [m.content for m in model.calls[1]]
    assert second_call_contents[0] != "第一问"  # 首条是 system prompt
    assert "第一问" in second_call_contents
    assert "好的" in second_call_contents
    assert second_call_contents[-1] == "第二问"

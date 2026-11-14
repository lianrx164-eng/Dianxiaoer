import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api import errors
from app.graph import runtime
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error(message: str) -> AsyncIterator[str]:
    yield "event: error\n"
    yield _sse({"message": message})


@router.post("/api/chat")
async def chat(req: ChatRequest):
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for ev in runtime.stream_turn(req.user_id, req.message, req.conversation_id):
                if ev["type"] == "tool":
                    yield _sse({"event": "tool", "name": ev["name"]})
                elif ev["type"] == "delta":
                    yield _sse({"delta": ev["text"]})
                elif ev["type"] == "citations":
                    yield _sse({"event": "citations", "items": ev["items"]})
                elif ev["type"] == "actions":
                    yield _sse({"event": "actions", "items": ev["items"]})
                elif ev["type"] == "interrupt":
                    yield _sse({"event": "interrupt",
                                "conversation_id": ev["conversation_id"],
                                "kind": ev["kind"], "orders": ev["orders"]})
                elif ev["type"] == "done":
                    yield _sse({"event": "done", "conversation_id": ev["conversation_id"]})
        except Exception as exc:  # noqa: BLE001 - 统一映射,流已发 200 错误只能走帧
            kind = errors.classify(exc)
            if kind != errors.NOT_FOUND:
                logger.exception("聊天编排失败 kind=%s user_id=%s", kind, req.user_id)
            for f in _sse_error(errors.ERROR_MESSAGES[kind]):
                yield f
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

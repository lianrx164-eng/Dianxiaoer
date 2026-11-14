import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api import errors
from app.api.sse import event_frames, sse_error
from app.graph import runtime
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/chat")
async def chat(req: ChatRequest):
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for frame in event_frames(
                    runtime.stream_turn(req.user_id, req.message, req.conversation_id)):
                yield frame
        except Exception as exc:  # noqa: BLE001 - 统一映射,流已发 200 错误只能走帧
            kind = errors.classify(exc)
            if kind != errors.NOT_FOUND:
                logger.exception("聊天编排失败 kind=%s user_id=%s", kind, req.user_id)
            for f in sse_error(errors.ERROR_MESSAGES[kind]):
                yield f
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

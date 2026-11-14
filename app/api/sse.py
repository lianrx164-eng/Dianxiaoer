"""SSE 帧拼装与 runtime 事件映射:chat 与 actions/resume 共用,避免两套映射漂移。"""
import json
from collections.abc import AsyncIterator


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_error(message: str) -> tuple[str, str]:
    return "event: error\n", sse({"message": message})


async def event_frames(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    """把 runtime 的事件 dict 映射成 SSE data 帧(事件名走 payload.event)。"""
    async for ev in events:
        kind = ev["type"]
        if kind == "tool":
            yield sse({"event": "tool", "name": ev["name"]})
        elif kind == "delta":
            yield sse({"delta": ev["text"]})
        elif kind == "citations":
            yield sse({"event": "citations", "items": ev["items"]})
        elif kind == "actions":
            yield sse({"event": "actions", "items": ev["items"]})
        elif kind == "interrupt":
            yield sse({"event": "interrupt",
                       "conversation_id": ev["conversation_id"],
                       "kind": ev["kind"], "orders": ev["orders"]})
        elif kind == "done":
            yield sse({"event": "done", "conversation_id": ev["conversation_id"]})

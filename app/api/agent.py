import logging

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, ToolMessage

from app.api import errors
from app.graph import runtime
from app.graph.nodes import resolve_answer
from app.schemas.agent import AgentRequest, AgentResponse, ToolCallView, ToolResultView

logger = logging.getLogger(__name__)
router = APIRouter()


def _views_from_state(state):
    """从终态重建本轮工具调用/结果轨迹(供评估/单测看模型选了什么工具)。
    runtime 有 turn_messages 时只看本轮,避免续接会话把历史轮轨迹重复返回。"""
    calls, results = [], []
    messages = state.get("turn_messages") or state.get("messages", [])
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                calls.append(ToolCallView(id=tc["id"], name=tc["name"], args=tc["args"]))
        elif isinstance(m, ToolMessage):
            results.append(ToolResultView(
                tool_call_id=m.tool_call_id, name=m.name or "",
                ok=(m.status != "error"), content=m.content))
    return calls, results


@router.post("/api/agent", response_model=AgentResponse)
async def run_agent(req: AgentRequest) -> AgentResponse:
    try:
        out = await runtime.run_turn(req.user_id, req.message, req.conversation_id)
    except Exception as exc:  # noqa: BLE001 - 统一映射为 HTTP 错误
        kind = errors.classify(exc)
        if kind != errors.NOT_FOUND:
            logger.exception("agent 编排失败 kind=%s user_id=%s", kind, req.user_id)
        raise HTTPException(status_code=errors.HTTP_STATUS[kind],
                            detail=errors.ERROR_MESSAGES[kind])

    state = out["state"]
    calls, results = _views_from_state(state)
    return AgentResponse(
        conversation_id=out["conversation_id"],
        answer=resolve_answer(state),
        tool_calls=calls,
        tool_results=results,
        suggested_actions=state.get("suggested_actions", []),
        interrupt=out.get("interrupt"),
        steps=state.get("steps", 0),
        trace=state.get("trace", {}),
    )

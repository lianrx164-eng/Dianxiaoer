import logging

from fastapi import APIRouter, HTTPException

from app.api import errors
from app.core import agent
from app.schemas.agent import AgentRequest, AgentResponse, ToolCallView, ToolResultView

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/agent", response_model=AgentResponse)
async def run_agent(req: AgentRequest) -> AgentResponse:
    try:
        result = await agent.run_agent_turn(req.user_id, req.message, req.conversation_id)
    except Exception as exc:  # noqa: BLE001 - 统一映射为 HTTP 错误
        kind = errors.classify(exc)
        if kind != errors.NOT_FOUND:
            logger.exception("agent 编排失败 kind=%s user_id=%s", kind, req.user_id)
        raise HTTPException(status_code=errors.HTTP_STATUS[kind],
                            detail=errors.ERROR_MESSAGES[kind])

    return AgentResponse(
        conversation_id=result.conversation_id,
        answer=result.answer,
        tool_calls=[ToolCallView(id=tc["id"], name=tc["name"], args=tc["args"])
                    for tc in result.tool_calls],
        tool_results=[ToolResultView(tool_call_id=r.tool_call_id, name=r.name, ok=r.ok,
                                     content=r.tool_message.content)
                      for r in result.tool_runs],
    )

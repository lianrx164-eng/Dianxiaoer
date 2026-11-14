import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api import errors
from app.api.sse import event_frames, sse_error
from app.db import repository
from app.graph import runtime
from app.schemas.actions import (
    CreateRefundRequest,
    CreateRefundResponse,
    CreateTicketRequest,
    CreateTicketResponse,
    ResumeRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/actions/create-ticket", response_model=CreateTicketResponse)
async def create_ticket_action(req: CreateTicketRequest) -> CreateTicketResponse:
    """建工单按钮:用户点了才写 tickets 表(复用 ch02 工单能力)。转人工是另一件事,纯前端。"""
    try:
        ticket_no = await repository.create_ticket(
            req.conversation_id, req.description, req.ticket_type)
    except SQLAlchemyError:
        logger.exception("建工单失败 conv=%s", req.conversation_id)
        raise HTTPException(status_code=503, detail="工单系统暂时不可用,请稍后重试")
    return CreateTicketResponse(ticket_no=ticket_no)


@router.post("/api/actions/create-refund", response_model=CreateRefundResponse)
async def create_refund_action(req: CreateRefundRequest) -> CreateRefundResponse:
    """退款表单提交:写 tickets(ticket_type='退款'),描述带订单号 + 固定类目原因。"""
    desc = f"退款申请 订单号={req.order_id} 原因={req.reason}"
    try:
        ticket_no = await repository.create_ticket(req.conversation_id, desc, "退款")
    except SQLAlchemyError:
        logger.exception("退款单创建失败 conv=%s", req.conversation_id)
        raise HTTPException(status_code=503, detail="退款系统暂时不可用,请稍后重试")
    return CreateRefundResponse(ticket_no=ticket_no)


@router.post("/api/actions/resume")
async def resume_action(req: ResumeRequest):
    """订单选择器点选后续跑暂停的退款子流程(SSE,事件与 /api/chat 同构)。"""
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for frame in event_frames(
                    runtime.stream_resume(req.conversation_id, req.order_id)):
                yield frame
        except Exception as exc:  # noqa: BLE001 - 流已发 200,错误只能走帧
            kind = errors.classify(exc)
            if kind != errors.NOT_FOUND:
                logger.exception("续跑失败 conv=%s", req.conversation_id)
            for f in sse_error(errors.ERROR_MESSAGES[kind]):
                yield f
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

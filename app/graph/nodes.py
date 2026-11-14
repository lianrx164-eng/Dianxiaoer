import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.prompts import (
    CHITCHAT_REPLY_TEXT,
    COMPLAINT_REPLY_TEXT,
    FALLBACK_REPLY_TEXT,
)
from app.db import repository
from app.graph.state import trace_of

logger = logging.getLogger(__name__)

CHITCHAT_REPLY = CHITCHAT_REPLY_TEXT
COMPLAINT_REPLY = COMPLAINT_REPLY_TEXT
FALLBACK_REPLY = FALLBACK_REPLY_TEXT


def _user_text(state) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


async def chitchat_reply(state) -> dict:
    """闲聊:固定话术,零模型调用。"""
    return {"answer": CHITCHAT_REPLY,
            "trace": trace_of(state, route="chitchat")}


async def complaint_reply(state) -> dict:
    """投诉:安抚话术 + 转人工/建工单两个可选项(后端不执行,交前端自选)。"""
    actions = [
        {"type": "transfer_human"},
        {"type": "create_ticket",
         "draft": {"description": _user_text(state), "ticket_type": "投诉"}},
    ]
    return {"answer": COMPLAINT_REPLY, "suggested_actions": actions,
            "trace": trace_of(state, route="complaint")}


async def fallback_reply(state) -> dict:
    """置信度兜底:证据弱回兜底话术,并把问题落低置信池留给数据飞轮。"""
    trace = state.get("trace") or {}
    source = trace.get("gate") or "retrieval_low_conf"
    if source == "self_check":
        reason = trace.get("self_check") or "证据自评不足"
    else:
        reason = f"检索证据不足(top={trace.get('evidence_top', 0):.3f})"
    await repository.insert_low_confidence(
        state.get("conversation_id"), _user_text(state), source, reason)
    return {"answer": FALLBACK_REPLY,
            "trace": trace_of(state, route="fallback")}


async def cap_reply(state) -> dict:
    """步数封顶:不把半截推理/工具结果当答复;为未执行的 tool_calls 补应答,
    避免悬空调用污染 checkpointer 里的下一轮历史。"""
    last = state["messages"][-1]
    aborted = [
        ToolMessage(content="已中止:达到本轮最大推理步数。",
                    tool_call_id=tc["id"], name=tc["name"])
        for tc in getattr(last, "tool_calls", [])
    ]
    return {"answer": FALLBACK_REPLY, "messages": aborted,
            "trace": trace_of(state, route="cap", cap=True)}

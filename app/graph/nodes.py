import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import settings
from app.core import intent as intent_mod
from app.core import query_understanding, retrieval, selfcheck
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


async def coref(state) -> dict:
    """指代消解:本章最简,原样透传(正式版留 ch06)。"""
    return {"trace": trace_of(state, coref="passthrough")}


async def classify_intent(state) -> dict:
    """意图识别:单标签七类。"""
    intent = await intent_mod.classify(_user_text(state))
    return {"intent": intent, "trace": trace_of(state, intent=intent)}


async def forced_rag(state) -> dict:
    """知识类强制检索(复用 ch03/04 检索器):产出编号证据 + 证据强弱信号。
    复刻 query_faq 的两道生成前证据闸(检索分闸 + 自评闸),把结果落进 State。"""
    query_raw = _user_text(state)
    u = await query_understanding.understand(query_raw)
    query = u["standard"]
    bm25_query = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    hits = await retrieval.search_knowledge(
        query, strategy="hybrid_rerank", bm25_query=bm25_query)
    top = hits[0].get("rerank_score", 0.0) if hits else 0.0

    # 机械闸:无召回 or 最高分低于阈值
    if not hits or top < settings.rerank_min_score:
        return {"evidence_strong": False,
                "trace": trace_of(state, forced_rag=True, evidence_top=top,
                                  gate="retrieval_low_conf")}

    # 语义闸:生成前自评证据够不够
    ev_texts = [f"{h['question']} {h['answer']}" for h in hits]
    chk = await selfcheck.check_sufficient(query, ev_texts)
    if not chk["useful"]:
        return {"evidence_strong": False,
                "trace": trace_of(state, forced_rag=True, evidence_top=top,
                                  gate="self_check", self_check=chk["reason"])}

    arranged = retrieval.arrange_head_tail(hits)
    citations = [
        {"n": i + 1, "id": h["id"], "chunk_id": h["id"],
         "section_path": h["section_path"], "question": h["question"],
         "answer": h["answer"], "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {"evidence_strong": True, "evidence": evidence, "citations": citations,
            "trace": trace_of(state, forced_rag=True, evidence_top=top)}


async def confidence_check(state) -> dict:
    """置信度兜底(生成前证据闸)的实体节点:留痕门控决策供可观测;
    强/弱的实际分流在其后的条件边 confidence_gate(读 evidence_strong)。"""
    decision = "strong" if state.get("evidence_strong") else "weak"
    return {"trace": trace_of(state, confidence=decision)}

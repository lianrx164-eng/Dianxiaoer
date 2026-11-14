import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import settings
from app.core import intent as intent_mod
from app.core import query_understanding, retrieval, selfcheck
from app.core.llm import get_chat_model
from app.core.prompts import (
    AGENT_SYSTEM,
    CHITCHAT_REPLY_TEXT,
    COMPLAINT_REPLY_TEXT,
    FALLBACK_REPLY_TEXT,
)
from app.db import repository
from app.graph.state import trace_of
from app.tools.infra import execute_tool_call
from app.tools.registry import get_all_tools

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


_KNOWLEDGE_EVIDENCE_HINT = (
    "\n\n## 已检索到的知识证据(请据此作答,每个关键结论后标注来源编号如[1];"
    "证据已给,不要再调用 query_faq;仍可按需调用订单/物流等工具)\n"
)


def _agent_messages(state) -> list:
    """system(知识路拼证据) + 跨轮历史。"""
    sys = AGENT_SYSTEM
    if state.get("route") == "knowledge" and state.get("evidence"):
        sys = AGENT_SYSTEM + _KNOWLEDGE_EVIDENCE_HINT + state["evidence"]
    return [SystemMessage(sys), *state.get("messages", [])]


async def agent_llm(state, config=None) -> dict:
    """ReAct 推理步:调模型(带工具),累加 steps 与 token 消耗。"""
    model = get_chat_model(streaming=True).bind_tools(get_all_tools())
    ai: AIMessage = await model.ainvoke(_agent_messages(state), config)
    usage = ai.usage_metadata or {}
    return {"messages": [ai],
            "steps": state.get("steps", 0) + 1,
            "tokens_used": state.get("tokens_used", 0) + usage.get("total_tokens", 0)}


async def agent_tools(state) -> dict:
    """ReAct 行动步:执行工具并回灌结果。create_ticket 拦截为『提议』——不写库,
    转成前端可选项,并回合成 ToolMessage 让模型收敛(真正写库在按钮端点)。"""
    last = state["messages"][-1]
    tool_msgs = []
    actions = list(state.get("suggested_actions") or [])
    for tc in last.tool_calls:
        if tc["name"] == "create_ticket":
            args = tc.get("args") or {}
            actions.append({"type": "create_ticket", "draft": {
                "description": args.get("description", ""),
                "ticket_type": args.get("ticket_type", "咨询"),
            }})
            tool_msgs.append(ToolMessage(
                content="已把『建工单』选项交给用户自行确认。请用一句话简要说明并停止,不要再调用任何工具。",
                tool_call_id=tc["id"], name="create_ticket"))
        else:
            run = await execute_tool_call(tc, state.get("conversation_id", 0))
            tool_msgs.append(run.tool_message)
    out = {"messages": tool_msgs}
    if actions:
        out["suggested_actions"] = actions
    return out


def resolve_answer(state) -> str:
    """最终答复:确定性节点写在 state['answer'];Agent 答复取最后一条 AIMessage 文本。"""
    if state.get("answer"):
        return state["answer"]
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            content = m.content
            if isinstance(content, str):
                return content
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


async def log_node(state) -> dict:
    """日志记录:留痕 intent/route/trace,并落 MySQL 一条 assistant 消息(审计)。
    确定性出口的答复不经模型,这里回填一条 AIMessage,保证 checkpointer 下一轮历史完整。"""
    logger.info(
        "ch05 turn conv=%s intent=%s route=%s trace=%s",
        state.get("conversation_id"), state.get("intent"), state.get("route"),
        state.get("trace", {}),
    )
    answer = resolve_answer(state)
    if state.get("conversation_id"):
        await repository.append_message(state["conversation_id"], "assistant",
                                        content=answer or None)
    if state.get("answer"):
        return {"messages": [AIMessage(answer)]}
    return {}

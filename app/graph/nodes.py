import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import settings
from app.core import coref as coref_mod
from app.core import evidence
from app.core import intent as intent_mod
from app.core.llm import get_chat_model
from app.core.memory import trim_history
from app.core.prompts import (
    AGENT_SYSTEM,
    CHITCHAT_REPLY_TEXT,
    COMPLAINT_REPLY_TEXT,
    FALLBACK_REPLY_TEXT,
)
from app.db import repository
from app.graph.routing import INTENT_TO_ROUTE
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


def _history_text(state, max_messages: int = 6) -> str:
    """把本轮之前的 human/ai 消息压成紧凑文本,供 coref / 意图识别读上下文。
    不含本轮最后一条 human(那是待处理的话本身)。"""
    msgs = state.get("messages", [])
    prior = msgs[:-1] if msgs else []
    lines = []
    for m in prior[-max_messages:]:
        role = "用户" if isinstance(m, HumanMessage) else "客服"
        text = m.content if isinstance(m.content, str) else ""
        if text:
            lines.append(f"{role}:{text}")
    return "\n".join(lines)


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
    """指代消解 + Query 改写(合一):吃最近几轮历史把半截话补成完整问句;已完整则透传。
    写 resolved_query,供 classify_intent / forced_rag / retrieve_policy 共用。"""
    query = _user_text(state)
    resolved = await coref_mod.resolve(query, _history_text(state))
    mode = "rewrite" if resolved != query else "passthrough"
    return {"resolved_query": resolved, "trace": trace_of(state, coref=mode)}


async def classify_intent(state) -> dict:
    """意图四件套:八类 + confidence + 「其他」兜底。吃 resolved_query(空则回落原话)+
    最近历史判当前意图(应对物流→退款→物流漂移)。把归属出口 route 落进 State(供
    _agent_messages 判注入、log 留痕;route_by_intent 条件边按同一 INTENT_TO_ROUTE 分流)。"""
    query = state.get("resolved_query") or _user_text(state)
    r = await intent_mod.classify(query, _history_text(state))
    intent, conf = r["intent"], r["confidence"]
    route = INTENT_TO_ROUTE.get(intent, "business")
    return {"intent": intent, "intent_confidence": conf, "route": route,
            "trace": trace_of(state, intent=intent,
                              intent_confidence=conf, route=route)}


async def forced_rag(state) -> dict:
    """知识类强制检索:复用统一证据管线(检索分闸 + 自评闸),结果落进 State。
    用 resolved_query 检索(指代已补全,避免拿半截话去查)。"""
    r = await evidence.gather_evidence(
        state.get("resolved_query") or _user_text(state))
    if not r["sufficient"]:
        extra = {"self_check": r["reason"]} if r["source"] == "self_check" else {}
        return {"evidence_strong": False,
                "trace": trace_of(state, forced_rag=True, evidence_top=r["top"],
                                  gate=r["source"], **extra)}
    return {"evidence_strong": True, "evidence": r["evidence"],
            "citations": r["citations"],
            "trace": trace_of(state, forced_rag=True, evidence_top=r["top"])}


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
    """system(知识路拼证据) + 按 token 预算裁剪过的跨轮历史。"""
    sys = AGENT_SYSTEM
    if state.get("route") == "knowledge" and state.get("evidence"):
        sys = AGENT_SYSTEM + _KNOWLEDGE_EVIDENCE_HINT + state["evidence"]
    history = trim_history(state.get("messages", []), max_tokens=settings.token_budget)
    return [SystemMessage(sys), *history]


async def agent_llm(state, config=None) -> dict:
    """ReAct 推理步:调模型(带工具),累加 steps 与 token 消耗。"""
    model = get_chat_model(streaming=True).bind_tools(get_all_tools())
    ai: AIMessage = await model.ainvoke(_agent_messages(state), config)
    usage = ai.usage_metadata or {}
    steps = state.get("steps", 0) + 1
    return {"messages": [ai], "steps": steps,
            "tokens_used": state.get("tokens_used", 0) + usage.get("total_tokens", 0),
            "trace": trace_of(state, steps=steps,
                              tokens_used=state.get("tokens_used", 0) + usage.get("total_tokens", 0))}


async def agent_tools(state) -> dict:
    """ReAct 行动步:执行工具并回灌结果。create_ticket 拦截为『提议』——不写库,
    转成前端可选项,并回合成 ToolMessage 让模型收敛(真正写库在按钮端点)。
    模型若重复调 create_ticket,只保留一份可选项。"""
    last = state["messages"][-1]
    tool_msgs = []
    actions = list(state.get("suggested_actions") or [])
    seen = {(a["type"], str(a.get("draft"))) for a in actions}
    for tc in last.tool_calls:
        if tc["name"] == "create_ticket":
            args = tc.get("args") or {}
            draft = {"description": args.get("description", ""),
                     "ticket_type": args.get("ticket_type", "咨询")}
            key = ("create_ticket", str(draft))
            if key not in seen:
                actions.append({"type": "create_ticket", "draft": draft})
                seen.add(key)
            tool_msgs.append(ToolMessage(
                content=("已把『建工单』选项交给用户自行确认,尚未创建工单。"
                         "请用一句话请用户点击下方『建工单』按钮提交;"
                         "不要声称已提交、已登记或已创建,也不要再调用任何工具。"),
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
        "ch05 turn conv=%s intent=%s route=%s steps=%s trace=%s",
        state.get("conversation_id"), state.get("intent"), state.get("route"),
        state.get("steps", 0), state.get("trace", {}),
    )
    answer = resolve_answer(state)
    if state.get("conversation_id"):
        await repository.append_message(state["conversation_id"], "user",
                                        content=_user_text(state))
        await repository.append_message(state["conversation_id"], "assistant",
                                        content=answer or None)
    if state.get("answer"):
        return {"messages": [AIMessage(answer)]}
    return {}

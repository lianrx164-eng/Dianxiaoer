import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt

from app.config import settings
from app.core import coref as coref_mod
from app.core import context_budget, memory
from app.core import evidence
from app.core import intent as intent_mod
from app.core import query_understanding, retrieval
from app.core.llm import get_chat_model
from app.core.confidence import snapshot_from_hits
from app.core.observability import tag_intent
from app.core.prompts import (
    AGENT_SYSTEM,
    CHITCHAT_REPLY_TEXT,
    COMPLAINT_REPLY_TEXT,
    FALLBACK_REPLY_TEXT,
    REFUND_JUDGE_HINT,
    SCRIPT_REPLY_OTHER,
)
from app.db import repository
from app.graph.routing import INTENT_TO_ROUTE
from app.graph.state import trace_of
from app.tools import business
from app.tools import engine, registry

logger = logging.getLogger(__name__)

COMPLAINT_REPLY = COMPLAINT_REPLY_TEXT
FALLBACK_REPLY = FALLBACK_REPLY_TEXT


def _user_text(state) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _history_text(state, max_messages: int = 6) -> str:
    """摘要行 + 最近几轮历史,供 coref / 意图识别读上下文。
    优先用本轮从 DB 构造的 history_messages;旧测试/直接调用没给时回落
    到 state.messages[:-1](去掉当前这句)。工具 JSON 不进(它不是客服话术)。"""
    if "history_messages" in state:
        prior = state.get("history_messages") or []
    else:
        prior = state.get("messages", [])[:-1]
    lines = []
    for m in prior[-max_messages:]:
        if isinstance(m, HumanMessage):
            role = "用户"
        elif isinstance(m, AIMessage):
            role = "客服"
        else:
            continue
        text = m.content if isinstance(m.content, str) else ""
        if text:
            lines.append(f"{role}:{text}")
    body = "\n".join(lines)
    head = memory.summary_line(state.get("summary"))
    return f"{head}\n{body}".strip() if head else body


def _in_refund_flow(state) -> bool:
    """退款子流程判定:证据注入、submit_refund 可见性与拦截都以此为准。"""
    return state.get("route") == "refund_flow"


async def script_reply(state) -> dict:
    """闲聊/其他 兜底话术(零模型):按 intent 分文案——闲聊把话题引回产品,其他请用户说具体些。
    分流后立即命中、不进 Agent。(与知识路证据弱的 fallback_reply 是两码事,勿混。)"""
    text = SCRIPT_REPLY_OTHER if state.get("intent") == "其他" else CHITCHAT_REPLY_TEXT
    return {"answer": text, "trace": trace_of(state, handler="script")}


async def complaint_reply(state) -> dict:
    """投诉:安抚话术 + 转人工/建工单两个可选项(后端不执行,交前端自选)。"""
    actions = [
        {"type": "transfer_human"},
        {"type": "create_ticket",
         "draft": {"description": _user_text(state), "ticket_type": "投诉"}},
    ]
    return {"answer": COMPLAINT_REPLY, "suggested_actions": actions,
            "trace": trace_of(state, handler="complaint")}


async def fallback_reply(state) -> dict:
    """置信度兜底:证据弱回兜底话术,并把问题落低置信池留给数据飞轮。
    ch09:source 用 fallback_source(缺失回落 trace.gate),随条存召回快照给审核页。"""
    trace = state.get("trace") or {}
    source = state.get("fallback_source") or trace.get("gate") or "retrieval_low_conf"
    signals = trace.get("confidence_signals") or {}
    reason = (f"evidence_confidence={state.get('evidence_confidence', 0.0):.3f} "
              f"signals={json.dumps(signals, ensure_ascii=False)}")
    if source == "self_check":
        reason += f" self_check={trace.get('self_check', '')}"
    snapshot = state.get("retrieved_snapshot") or None   # 没走检索存 NULL,不存 []
    await repository.insert_low_confidence(
        state.get("conversation_id"), _user_text(state), source, reason,
        retrieved_chunks=snapshot,
    )
    return {"answer": FALLBACK_REPLY,
            "suggested_actions": [
                {"type": "transfer_human"},
                {"type": "create_ticket",
                 "draft": {"description": _user_text(state), "ticket_type": "咨询"}},
            ],
            "trace": trace_of(state, handler="fallback")}


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
            "trace": trace_of(state, handler="cap", cap=True)}


async def coref(state) -> dict:
    """指代消解 + Query 改写(合一):吃最近几轮历史把半截话补成完整问句;已完整则透传。
    写 resolved_query,供 classify_intent / forced_rag / retrieve_policy 共用。"""
    query = _user_text(state)
    history = _history_text(state)
    logger.info("history_ctx conv=%s\n%s", state.get("conversation_id"), history)
    resolved = await coref_mod.resolve(query, history)
    mode = "rewrite" if resolved != query else "passthrough"
    return {"resolved_query": resolved,
            "trace": trace_of(state, coref=mode, resolved_query=resolved)}


async def classify_intent(state) -> dict:
    """意图四件套:九类 + confidence + 「其他」兜底。吃 resolved_query(空则回落原话)+
    最近历史判当前意图(应对物流→退款→物流漂移)。把归属出口 route 落进 State(供
    _agent_messages 判注入、log 留痕;route_by_intent 条件边按同一 INTENT_TO_ROUTE 分流)。
    建单追问后的补充句单看会被判售后/商品咨询,靠 ticket_followup 粘一轮、直接走 business。"""
    if state.get("ticket_followup"):
        tag_intent("人工", 1.0)
        return {"intent": "人工", "intent_confidence": 1.0, "route": "business",
                "ticket_followup": False,
                "trace": trace_of(state, intent="人工", intent_confidence=1.0,
                                  route="business", ticket_followup=True)}
    query = state.get("resolved_query") or _user_text(state)
    r = await intent_mod.classify(query, _history_text(state))
    intent, conf = r["intent"], r["confidence"]
    route = INTENT_TO_ROUTE.get(intent, "business")
    tag_intent(intent, conf)   # ch09:意图进当前 trace 的 metadata+tag,Cost Control 按意图分堆
    return {"intent": intent, "intent_confidence": conf, "route": route,
            # 人工意图的下一轮直接粘一轮 business:模型这一轮可能只是口头追问、
            # 没有发 create_ticket,不能靠 agent_tools 的失败调用来置标记。
            "ticket_followup": intent == "人工",
            "trace": trace_of(state, intent=intent,
                              intent_confidence=conf, route=route)}


_ORDER_RE = re.compile(r"(?<!\d)(\d{4,})(?!\d)")
_ORDER_CUES = ("订单", "单号", "尾号", "order", "Order", "ORDER")


def _extract_order_id(text: str, known_orders: set[str] | None = None) -> str | None:
    """抽订单号,只认两种候选:①紧邻订单线索(订单/单号/尾号/order,且线索与数字之间
    没有别的数字);②是该用户名下的订单。手机号、金额、年份一律不认——宁可弹选择器,
    也不拿假单号去编造订单。中文与数字之间没有 \\b 边界,故用前后非数字断言。"""
    text = text or ""
    known = known_orders or set()
    for m in _ORDER_RE.finditer(text):
        cand = m.group(1)
        if cand in known:
            return cand
        before = text[max(0, m.start() - 12):m.start()]
        cue_pos = max((before.rfind(c) for c in _ORDER_CUES), default=-1)
        if cue_pos != -1 and not any(ch.isdigit() for ch in before[cue_pos:]):
            return cand
    return None


async def fetch_order(state) -> dict:
    """退款子流程第一步:抽订单号;缺则 interrupt 弹订单选择器等前端点选(resume 回填);
    拿到后 order_snapshot 取订单数据。interrupt 之前只做只读(resume 时本节点从头重跑)。"""
    orders = business.list_user_orders(state.get("user_id", ""))   # 只读,可安全重跑
    known = {str(o["order_id"]) for o in orders}
    text = state.get("resolved_query") or _user_text(state)
    oid = state.get("order_id") or _extract_order_id(text, known)
    if not oid:
        oid = interrupt({"type": "select_order", "orders": orders})    # resume 回填订单号
    data = business.order_snapshot(oid)
    return {"order_id": oid, "order_data": data,
            "trace": trace_of(state, fetch_order={"order_id": oid})}


async def retrieve_policy(state) -> dict:
    """退款子流程强制检索政策:Query 扩写 3 条 → 多 query 各检索一次 → 按 chunk id 去重合并
    (保留每 id 最高分)→ 按分降序拼编号证据。产出注入 agent_llm 作「能不能退」的判据
    (不让模型凭记忆答)。库里知识一份,扩写只在检索侧现查现用。"""
    base = state.get("resolved_query") or _user_text(state)
    od = state.get("order_data") or {}
    seed = f"{base} {od.get('status', '')}".strip()
    queries = await query_understanding.expand_queries(seed)

    merged: dict = {}                       # chunk id -> 最高分 hit
    for q in queries:
        for h in await retrieval.search_knowledge(
                q, strategy="hybrid_rerank", bm25_query=q):
            cur = merged.get(h["id"])
            if cur is None or h["rerank_score"] > cur["rerank_score"]:
                merged[h["id"]] = h
    ranked = sorted(merged.values(), key=lambda h: h["rerank_score"], reverse=True)
    arranged = retrieval.arrange_head_tail(ranked)
    citations, evidence_text = evidence.build_citations(arranged)
    return {"evidence": evidence_text, "citations": citations,
            # ch09:退款路也走了检索,快照留给 👎 回捞
            "retrieved_snapshot": snapshot_from_hits(ranked),
            "trace": trace_of(state, retrieve_policy={
                "queries": queries, "hits": len(ranked)})}


async def forced_rag(state) -> dict:
    """知识类强制检索:复用统一证据管线(置信度闸 + 自评闸),结果落进 State。
    用 resolved_query 检索(指代已补全,避免拿半截话去查)。
    ch09:把置信度/source/召回快照写进 state——闸不过随兜底落池,闸过了留给 👎 回捞。"""
    r = await evidence.gather_evidence(
        state.get("resolved_query") or _user_text(state))
    base = {"evidence_confidence": r["confidence"],
            "retrieved_snapshot": r["snapshot"]}
    if not r["sufficient"]:
        extra = {"self_check": r["reason"]} if r["source"] == "self_check" else {}
        return {**base, "evidence_strong": False, "fallback_source": r["source"],
                "trace": trace_of(state, forced_rag=True, evidence_top=r["top"],
                                  confidence_signals=r["signals"],
                                  gate=r["source"], **extra)}
    return {**base, "evidence_strong": True, "evidence": r["evidence"],
            "citations": r["citations"],
            "trace": trace_of(state, forced_rag=True, evidence_top=r["top"],
                              confidence_signals=r["signals"])}


async def confidence_check(state) -> dict:
    """置信度兜底(生成前证据闸)的实体节点:留痕门控决策供可观测;
    强/弱的实际分流在其后的条件边 confidence_gate(读 evidence_strong)。"""
    decision = "strong" if state.get("evidence_strong") else "weak"
    return {"trace": trace_of(state, confidence=decision)}


_KNOWLEDGE_EVIDENCE_HINT = (
    "\n\n## 已检索到的知识证据(请据此作答,每个关键结论后标注来源编号如[1];"
    "证据已给,不要再调用 query_faq;仍可按需调用订单/物流等工具)\n"
)

_TICKET_FOLLOWUP_HINT = (
    "\n\n## 建单进行中(仅本轮)\n"
    "用户上一轮明确要求建工单,这一轮是在补充问题。只要已能从对话概括问题描述,"
    "就直接调用 create_ticket(description 用用户说的问题,ticket_type 按性质选),"
    "不要再索要订单号、联系方式等非必填信息;确实仍缺问题描述时才追问。\n"
)

TURN_CTX_ID = "turn-ctx"


def _agent_messages(state) -> list:
    """固定顺序(ch07):system 人设/红线(恒定)→ 层2 截短 → 层1 原文 → 当前用户 →
    梗概+证据+退款订单数据(挂当前用户之后)→ 本轮 ReAct 后续。

    整条列表只有一条 SystemMessage,内容恒为 AGENT_SYSTEM;上游 chat template 会把
    所有 system 上提合并,可变内容一旦进 system,工具定义被挤后、前缀缓存整段作废。"""
    budget = context_budget.compute_budget()
    layer2, layer1 = memory.build_layers(
        state.get("history_messages", []),
        state.get("summary_upto_msg_id", 0) or 0,
        state.get("layer1_from_msg_id", 0) or 0,
        budget.layer1, budget.layer2,
        settings.layer2_assistant_chars, settings.layer2_tool_keep_chars)
    current = _current_turn_messages(state)
    ctx = _turn_context(state)
    msgs = [SystemMessage(AGENT_SYSTEM), *layer2, *layer1]
    if current:
        msgs.append(current[0])                       # 当前用户
        if ctx:
            msgs.append(HumanMessage(ctx, id=TURN_CTX_ID))
        msgs.extend(current[1:])                      # 本轮 ReAct 后续
    elif ctx:
        msgs.append(HumanMessage(ctx, id=TURN_CTX_ID))
    return msgs


def _current_turn_messages(state) -> list:
    """本轮新增消息:从最后一条 HumanMessage 起到尾(当前用户 + ReAct 轨迹)。"""
    msgs = state.get("messages", [])
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], HumanMessage):
            return msgs[i:]
    return msgs[-1:] if msgs else []


def _turn_context(state) -> str:
    """本轮材料:梗概 + 知识证据 + 退款订单数据/判定指令;没有则空串。"""
    parts = []
    if state.get("summary"):
        parts.append("\n\n## 早前对话摘要(更早轮次已压缩,其中事实可信)\n"
                     + state["summary"])
    if ((state.get("trace") or {}).get("ticket_followup")
            and state.get("messages")
            and isinstance(state["messages"][-1], HumanMessage)):
        # 只在用户补充信息的那一轮注入;resume 后最后一条是 AIMessage,不能再提示建单
        parts.append(_TICKET_FOLLOWUP_HINT)
    if state.get("evidence"):
        parts.append(_KNOWLEDGE_EVIDENCE_HINT + state["evidence"])
    if _in_refund_flow(state):
        parts.append(REFUND_JUDGE_HINT
                     + json.dumps(state.get("order_data") or {}, ensure_ascii=False))
    return "".join(parts)


def _log_model_context(state, msgs) -> None:
    """调模型前把实际上下文打进 log/app.log(验收可观测)。"""
    window = msgs[1:]                                 # 去掉静态 system
    desc = [f"[{m.type}] '{(m.content if isinstance(m.content, str) else '')[:40]}'"
            for m in window]
    logger.info(
        "model_ctx conv=%s step=%s summary=%r window_msgs=%d tokens=%d window=%s",
        state.get("conversation_id"), state.get("steps", 0),
        state.get("summary") or "", len(window),
        context_budget.estimate_messages_tokens(msgs), " | ".join(desc))


def _tools_for(state, specs) -> list:
    """submit_refund 只在退款子流程里可见;其他路不让模型看见退款工具,
    避免误分类时产出没有订单号、没有政策依据的退款表单。"""
    tools = [s.tool for s in specs]
    if _in_refund_flow(state):
        return tools
    return [t for t in tools if t.name != "submit_refund"]


async def agent_llm(state, config=None) -> dict:
    """ReAct 推理步:调模型(带工具),累加 steps 与 token 消耗。"""
    specs = await registry.get_all_specs()
    model = get_chat_model(streaming=True).bind_tools(_tools_for(state, specs))
    msgs = _agent_messages(state)
    _log_model_context(state, msgs)
    ai: AIMessage = await model.ainvoke(msgs, config)
    usage = ai.usage_metadata or {}
    steps = state.get("steps", 0) + 1
    return {"messages": [ai], "steps": steps,
            "tokens_used": state.get("tokens_used", 0) + usage.get("total_tokens", 0),
            "trace": trace_of(state, steps=steps,
                              tokens_used=state.get("tokens_used", 0) + usage.get("total_tokens", 0))}


async def agent_tools(state) -> dict:
    """ReAct 行动步(ch08):一切工具经统一执行引擎。create_ticket(唯一写操作)走确认流——
    参数齐则节点顶部 interrupt 推工单预览(interrupt 前只做纯计算,resume 重跑安全,照 fetch_order 范式);
    参数缺则交引擎按「校验拦下」回灌,模型自然向用户追问。submit_refund 仍拦成前端退款表单(ch06)。"""
    last = state["messages"][-1]
    cid = state.get("conversation_id", 0)
    specs = {s.name: s for s in await registry.get_all_specs()}

    ticket_calls = [tc for tc in last.tool_calls if tc["name"] == "create_ticket"]
    tspec = specs.get("create_ticket")
    ticket_call = None
    if tspec is not None:
        ticket_call = next(
            (tc for tc in ticket_calls
             if engine.validate_args(tspec, dict(tc.get("args") or {})) is None), None)
    decision = None
    if ticket_call is not None:
        args = ticket_call.get("args") or {}
        decision = interrupt({"type": "confirm_ticket",
                              "preview": {"ticket_type": args.get("ticket_type", "咨询"),
                                          "description": args.get("description", "")}})
        if not isinstance(decision, dict):        # 前端传错/旧客户端:按取消处理,不崩
            decision = {"confirmed": False}

    tool_msgs = []
    actions = list(state.get("suggested_actions") or [])
    seen = {(a["type"], str(a.get("draft"))) for a in actions}
    for tc in last.tool_calls:
        if tc["name"] == "submit_refund":
            args = tc.get("args") or {}
            order_id = args.get("order_id") or state.get("order_id", "")
            if not _in_refund_flow(state) or not order_id:
                tool_msgs.append(ToolMessage(
                    content=("当前不在退款流程或缺少订单号,不能发起退款申请;"
                             "请按当前问题继续作答,不要声称已提交。"),
                    tool_call_id=tc["id"], name="submit_refund"))
                continue
            draft = {"order_id": order_id, "reason": args.get("reason")}
            key = ("submit_refund", str(draft))
            if key not in seen:
                actions.append({"type": "refund_form", "draft": draft})
                seen.add(key)
            tool_msgs.append(ToolMessage(
                content=("已把『提交退款工单』选项交给用户自行确认,尚未提交退款。"
                         "请用一句话说明这一单可以退款并停止,不要声称已提交或已退款,"
                         "也不要再调用任何工具。"),
                tool_call_id=tc["id"], name="submit_refund"))
        elif tc["name"] == "create_ticket":
            if decision is not None and tc is ticket_call:
                if decision.get("confirmed"):
                    run = await engine.execute_tool_call(tc, cid, specs, confirmed=True)
                else:
                    run = await engine.execute_tool_call(
                        tc, cid, specs, confirmed=False,
                        deny_note="用户在工单预览卡片上点了取消,本次不建单。"
                                  "请勿再发起,除非用户再次明确要求。")
            else:
                run = await engine.execute_tool_call(
                    tc, cid, specs, confirmed=False,
                    deny_note="一次只处理一个建工单请求,本次调用未执行。")
            tool_msgs.append(run.tool_message)
        else:
            run = await engine.execute_tool_call(tc, cid, specs)
            tool_msgs.append(run.tool_message)
    out = {"messages": tool_msgs}
    if ticket_calls and decision is None:
        out["ticket_followup"] = True     # 参数不全、模型要追问:下一轮粘一轮 business
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

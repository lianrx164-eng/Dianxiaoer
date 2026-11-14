import logging
import uuid
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.config import settings
from app.core import context_budget, memory, summarizer
from app.db import repository
from app.graph.build import build_graph

logger = logging.getLogger(__name__)

# 会流式吐答复 token 的节点(其余节点的模型调用 token 不进 delta)
ANSWER_NODES = {"agent_llm"}
# 确定性节点把答复写在 state['answer'],整块作为 delta 吐出
DETERMINISTIC_ANSWER_NODES = {"script_reply", "complaint_reply", "fallback_reply", "cap_reply"}
# 产出 citations 的节点(知识路预检索 / 退款路政策检索)
CITATION_NODES = {"forced_rag", "retrieve_policy"}
# 被 agent_tools 拦截成前端动作、不该出工具徽章的工具
ACTION_TOOLS = {"create_ticket", "submit_refund"}

_graph = None
_checkpointer = None
_cm = None  # AsyncSqliteSaver context manager,持有以免被 GC


class ConversationNotFound(Exception):
    pass


async def init_graph() -> None:
    """lifespan 起:开持久 checkpointer + setup + 编译图。"""
    global _graph, _checkpointer, _cm
    _cm = AsyncSqliteSaver.from_conn_string(settings.checkpointer_db_path)
    _checkpointer = await _cm.__aenter__()
    await _checkpointer.setup()
    _graph = build_graph(checkpointer=_checkpointer)
    budget = context_budget.self_check()
    logger.info("上下文预算 window=%s fixed=%s peak=%s history=%s layer1=%s layer2=%s",
                budget.window, budget.fixed, budget.react_peak,
                budget.history, budget.layer1, budget.layer2)
    logger.info("ch05 图已编译,checkpointer=%s", settings.checkpointer_db_path)


async def close_graph() -> None:
    global _graph, _checkpointer, _cm
    if _cm is not None:
        await _cm.__aexit__(None, None, None)
        _cm = None
    _graph = None
    _checkpointer = None


def get_graph():
    if _graph is None:
        raise RuntimeError("图未初始化,应在 FastAPI lifespan 调 init_graph()")
    return _graph


async def _ensure_conversation(user_id: str, conversation_id: int | None) -> int:
    if conversation_id is None:
        return await repository.create_conversation(user_id)
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    return conversation_id


def _human_id(message: str, prev_messages: list) -> str:
    """失败轮的人类消息已写进 checkpoint;同样的内容重试时复用 id,
    add_messages 会替换而不是再追加一条,避免历史里叠出两条相同提问。"""
    if (prev_messages and isinstance(prev_messages[-1], HumanMessage)
            and prev_messages[-1].content == message):
        return prev_messages[-1].id or str(uuid.uuid4())
    return str(uuid.uuid4())


async def _load_context(cid: int) -> dict:
    """读已落库对话,拼本轮精简历史;层 1 超预算就整轮移入层 2 并写回边界。"""
    conv = await repository.get_conversation(cid)
    if conv is None:
        raise ConversationNotFound(cid)
    rows = await repository.list_dialog_messages(cid)
    history = [memory.row_to_message(r) for r in rows]
    summary = conv.summary or ""
    upto = conv.summary_upto_msg_id or 0
    layer1_from = conv.layer1_from_msg_id or 0
    if layer1_from < upto:                 # 边界不变量:层 1 起点不越过梗概覆盖
        logger.info("layer1 边界夹紧 %s->%s conv=%s", layer1_from, upto, cid)
        layer1_from = upto
        await repository.update_layer1_from(cid, layer1_from)
    budget = context_budget.compute_budget()
    layer1_msgs = [m for m in history if (memory.db_id(m) or 0) > layer1_from]
    layer1_tokens = context_budget.estimate_messages_tokens(layer1_msgs)
    new_from = memory.degrade_layer1(history, layer1_from, budget.layer1)
    if new_from != layer1_from:
        await repository.update_layer1_from(cid, new_from)
        logger.info("layer1 降级 %s->%s conv=%s layer1_tokens=%s layer1_budget=%s",
                    layer1_from, new_from, cid, layer1_tokens, budget.layer1)
        layer1_from = new_from
    kept = [m for m in history if (memory.db_id(m) or 0) > upto]
    return {"history_messages": kept, "summary": summary,
            "summary_upto_msg_id": upto, "layer1_from_msg_id": layer1_from}


def _graph_input(user_id: str, message: str, cid: int,
                 prev_messages: list | None, ctx: dict) -> dict:
    return {
        "messages": [HumanMessage(message, id=_human_id(message, prev_messages or []))],
        "user_id": user_id, "conversation_id": cid,
        "history_messages": ctx["history_messages"],
        "summary": ctx["summary"],
        "summary_upto_msg_id": ctx["summary_upto_msg_id"],
        "layer1_from_msg_id": ctx["layer1_from_msg_id"],
        # 每轮显式重置:这些是普通字段,不重置会残留上一轮的键/答案/按钮。
        # ticket_followup 故意不在此列:建单追问的粘滞标记要跨轮留在 checkpoint 里,
        # 由 classify_intent 消费后清零。
        "intent": "", "route": "", "evidence": "", "citations": [],
        "evidence_strong": False, "answer": "", "steps": 0, "tokens_used": 0,
        "suggested_actions": [], "trace": {},
        "resolved_query": "", "intent_confidence": 0.0,
        "order_id": "", "order_data": {},
    }


def _interrupt_payload(interrupts) -> dict | None:
    """从 Interrupt 序列里取第一个 dict 载荷(非流式 __interrupt__ / 状态快照 .interrupts)。"""
    for intr in interrupts or ():
        value = getattr(intr, "value", None)
        if isinstance(value, dict):
            return value
    return None


def _pending_payload(snapshot) -> dict | None:
    return _interrupt_payload(getattr(snapshot, "interrupts", ())) if snapshot else None


def _interrupt_event(cid: int, payload: dict) -> dict:
    ev = {"type": "interrupt", "conversation_id": cid,
          "kind": payload.get("type", "")}
    for k in ("orders", "preview"):                 # 按中断类型透传负载
        if payload.get(k) is not None:
            ev[k] = payload[k]
    return ev


def _merge_actions(actions: list, seen: set, incoming: list) -> None:
    for a in incoming or []:
        key = (a.get("type"), str(a.get("draft")))
        if key not in seen:
            actions.append(a)
            seen.add(key)


def _turn_result(cid: int, final: dict, prev_messages: list) -> dict:
    """本轮新增消息切片 + 统一返回形状(run_turn / resume_turn 共用)。"""
    turn = final.get("messages", [])[len(prev_messages):]
    return {"conversation_id": cid,
            "state": {**final, "turn_messages": turn},
            "interrupt": _interrupt_payload(final.get("__interrupt__"))}


async def run_turn(user_id, message, conversation_id) -> dict:
    """非流式:ainvoke → 返回终态(供 /api/agent、eval)。

    用户/助手消息由 log 节点统一落 MySQL,失败轮不留半条记录;
    重试同文时复用 message id 去重,turn_messages 只含本轮新增消息。
    上轮停在订单选择器时,不启动新一轮(否则新消息不处理、只再次触发中断),
    直接把未决中断带回去,由前端点选续跑。"""
    cid = await _ensure_conversation(user_id, conversation_id)
    config = {"configurable": {"thread_id": str(cid)}}
    graph = get_graph()
    prev = await graph.aget_state(config)
    pending = _pending_payload(prev)
    if pending:
        return {"conversation_id": cid,
                "state": {**(prev.values or {}), "answer": "", "turn_messages": []},
                "interrupt": pending}
    prev_messages = (prev.values or {}).get("messages", []) if prev else []
    ctx = await _load_context(cid)
    final = await graph.ainvoke(
        _graph_input(user_id, message, cid, prev_messages, ctx), config)
    await summarizer.maybe_schedule_summary(cid)
    return _turn_result(cid, final, prev_messages)


async def _stream_events(cid: int, stream_source) -> AsyncIterator[dict]:
    """把 astream 多模式映射成事件 dict;遇中断吐 interrupt 并结束本次流。"""
    config = {"configurable": {"thread_id": str(cid)}}
    actions: list = []
    seen_actions: set = set()
    async for mode, chunk in get_graph().astream(
            stream_source, config, stream_mode=["messages", "updates"]):
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") in ANSWER_NODES:
                text = msg.content if isinstance(msg.content, str) else ""
                if text:
                    yield {"type": "delta", "text": text}
        elif mode == "updates":
            payload = _interrupt_payload(chunk.get("__interrupt__"))
            if payload is not None:
                yield _interrupt_event(cid, payload)
                return                  # 图已暂停,前端点选后走 /resume 续流
            for node, upd in chunk.items():
                if not isinstance(upd, dict):
                    continue
                if node in DETERMINISTIC_ANSWER_NODES and upd.get("answer"):
                    yield {"type": "delta", "text": upd["answer"]}
                if node in CITATION_NODES and upd.get("citations"):
                    yield {"type": "citations", "items": upd["citations"]}
                if node == "agent_tools":
                    for m in upd.get("messages", []):
                        name = getattr(m, "name", None)
                        if name and name not in ACTION_TOOLS:
                            yield {"type": "tool", "name": name}
                _merge_actions(actions, seen_actions, upd.get("suggested_actions"))
    if actions:
        yield {"type": "actions", "items": actions}
    yield {"type": "done", "conversation_id": cid}


async def stream_turn(user_id, message, conversation_id) -> AsyncIterator[dict]:
    """流式:astream 多模式 → 映射成事件 dict(供 /api/chat 转 SSE)。"""
    cid = await _ensure_conversation(user_id, conversation_id)
    config = {"configurable": {"thread_id": str(cid)}}
    graph = get_graph()
    prev = await graph.aget_state(config)
    pending = _pending_payload(prev)
    if pending:
        # 上一轮还停在订单选择器:重吐中断,不启动新一轮(新消息不落库、不处理)
        yield _interrupt_event(cid, pending)
        return
    prev_messages = (prev.values or {}).get("messages", []) if prev else []
    ctx = await _load_context(cid)
    async for ev in _stream_events(
            cid, _graph_input(user_id, message, cid, prev_messages, ctx)):
        yield ev
    await summarizer.maybe_schedule_summary(cid)


async def stream_resume(conversation_id: int, resume_value) -> AsyncIterator[dict]:
    """前端点选订单后续跑同一会话的暂停流。"""
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    async for ev in _stream_events(conversation_id, Command(resume=resume_value)):
        yield ev
    await summarizer.maybe_schedule_summary(conversation_id)


async def resume_turn(conversation_id: int, resume_value) -> dict:
    """非流式续跑(供 eval/脚本);SSE 入口用 stream_resume。"""
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    config = {"configurable": {"thread_id": str(conversation_id)}}
    graph = get_graph()
    prev = await graph.aget_state(config)
    prev_messages = (prev.values or {}).get("messages", []) if prev else []
    final = await graph.ainvoke(Command(resume=resume_value), config)
    await summarizer.maybe_schedule_summary(conversation_id)
    return _turn_result(conversation_id, final, prev_messages)

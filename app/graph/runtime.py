import logging
import uuid
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings
from app.db import repository
from app.graph.build import build_graph

logger = logging.getLogger(__name__)

# 会流式吐答复 token 的节点(其余节点的模型调用 token 不进 delta)
ANSWER_NODES = {"agent_llm"}
# 确定性节点把答复写在 state['answer'],整块作为 delta 吐出
DETERMINISTIC_ANSWER_NODES = {"chitchat_reply", "complaint_reply", "fallback_reply", "cap_reply"}

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


def _graph_input(user_id: str, message: str, cid: int, prev_messages: list | None = None) -> dict:
    return {
        "messages": [HumanMessage(message, id=_human_id(message, prev_messages or []))],
        "user_id": user_id, "conversation_id": cid,
        # 每轮显式重置:这些是普通字段,不重置会残留上一轮的键/答案/按钮
        "intent": "", "route": "", "evidence": "", "citations": [],
        "evidence_strong": False, "answer": "", "steps": 0, "tokens_used": 0,
        "suggested_actions": [], "trace": {},
        "resolved_query": "", "intent_confidence": 0.0,
        "order_id": "", "order_data": {},
    }


async def run_turn(user_id, message, conversation_id) -> dict:
    """非流式:ainvoke → 返回终态(供 /api/agent、eval)。

    用户/助手消息由 log 节点统一落 MySQL,失败轮不留半条记录;
    重试同文时复用 message id 去重,turn_messages 只含本轮新增消息。"""
    cid = await _ensure_conversation(user_id, conversation_id)
    config = {"configurable": {"thread_id": str(cid)}}
    graph = get_graph()
    prev = await graph.aget_state(config)
    prev_messages = (prev.values or {}).get("messages", []) if prev else []
    final = await graph.ainvoke(
        _graph_input(user_id, message, cid, prev_messages), config)
    turn = final.get("messages", [])[len(prev_messages):]
    return {"conversation_id": cid, "state": {**final, "turn_messages": turn}}


async def stream_turn(user_id, message, conversation_id) -> AsyncIterator[dict]:
    """流式:astream 多模式 → 映射成事件 dict(供 /api/chat 转 SSE)。"""
    cid = await _ensure_conversation(user_id, conversation_id)
    config = {"configurable": {"thread_id": str(cid)}}
    graph = get_graph()
    prev = await graph.aget_state(config)
    prev_messages = (prev.values or {}).get("messages", []) if prev else []

    actions: list = []
    async for mode, chunk in graph.astream(
        _graph_input(user_id, message, cid, prev_messages), config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") in ANSWER_NODES:
                text = msg.content if isinstance(msg.content, str) else ""
                if text:
                    yield {"type": "delta", "text": text}
        elif mode == "updates":
            for node, upd in chunk.items():
                if not isinstance(upd, dict):
                    continue
                if node in DETERMINISTIC_ANSWER_NODES and upd.get("answer"):
                    yield {"type": "delta", "text": upd["answer"]}
                if node == "forced_rag" and upd.get("citations"):
                    yield {"type": "citations", "items": upd["citations"]}
                if node == "agent_tools":
                    for m in upd.get("messages", []):
                        name = getattr(m, "name", None)
                        if name and name != "create_ticket":
                            yield {"type": "tool", "name": name}
                if upd.get("suggested_actions"):
                    actions = upd["suggested_actions"]
    if actions:
        yield {"type": "actions", "items": actions}
    yield {"type": "done", "conversation_id": cid}

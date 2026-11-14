import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from app.config import settings
from app.core.context_budget import estimate_messages_tokens, keep_newest_fitting

logger = logging.getLogger(__name__)


class SessionStore:
    """内存会话存储,session_id -> 消息列表。本章不做持久化,也不淘汰,

    会话数多了内存只增,跨章做持久化时一并处理。
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}

    def get(self, session_id: str) -> list[BaseMessage]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, *messages: BaseMessage) -> None:
        self._sessions.setdefault(session_id, []).extend(messages)

    def clear(self) -> None:
        self._sessions.clear()


def trim_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        allow_partial=False,
    )
    if not trimmed and messages:
        # 预算放不下任何完整的轮次时,至少保留最新一条,不能把当前问题裁没
        logger.warning(
            "token 预算 %s 小于单条消息,仅保留最新一条(size=%s)", max_tokens, len(messages)
        )
        trimmed = [messages[-1]]
    return trimmed


# ---------------- ch07 三层上下文 ----------------

_DB_ID_PREFIX = "db-"


def db_id(m: BaseMessage) -> int | None:
    """从消息 id 解出 MySQL 消息 id(约定 `db-{id}`);非锚点返回 None。"""
    mid = getattr(m, "id", None)
    if isinstance(mid, str) and mid.startswith(_DB_ID_PREFIX):
        try:
            return int(mid[len(_DB_ID_PREFIX):])
        except ValueError:
            return None
    return None


def has_db_anchors(messages: list[BaseMessage]) -> bool:
    """消息列表里是否至少有一条带 `db-{id}` 锚点。"""
    return any(db_id(m) is not None for m in messages)


def row_to_message(row) -> BaseMessage:
    """DB 行 → 带锚点的 Human/AI 消息;runtime/summarizer 共用。"""
    cls = HumanMessage if row.role == "user" else AIMessage
    return cls(content=row.content or "", id=f"db-{row.id}")


def _group_turns(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """按 user 消息切轮;层 1/层 2 的 DB 行天然 user 开头,这里兜底旧消息。"""
    turns: list[list[BaseMessage]] = []
    cur: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, HumanMessage) and cur:
            turns.append(cur)
            cur = []
        cur.append(m)
    if cur:
        turns.append(cur)
    return turns


def split_layers(history_messages: list[BaseMessage], summary_upto: int,
                 layer1_from: int) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """按两个 DB 锚点切层:`summary_upto < id ≤ layer1_from` 层 2,`> layer1_from` 层 1。
    无锚点消息列表整体归层 1。"""
    if not has_db_anchors(history_messages):
        return [], list(history_messages)
    ids = [db_id(m) for m in history_messages]
    layer2: list[BaseMessage] = []
    layer1: list[BaseMessage] = []
    for m, did in zip(history_messages, ids):
        if did is None:                             # 无锚点消息跟随前一段
            (layer1 if layer1 else layer2).append(m)
        elif did <= summary_upto:
            continue
        elif did <= layer1_from:
            layer2.append(m)
        else:
            layer1.append(m)
    return layer2, layer1


def degrade_layer1(history_messages: list[BaseMessage], layer1_from: int,
                   budget_tokens: int, s=settings) -> int:
    """层 1 超预算:从最老的整轮开始移入层 2,只返回新边界 id(不搬数据)。
    最新一轮永远保留;无锚点消息列表不挪边界。"""
    if not has_db_anchors(history_messages):
        return layer1_from
    ids = [db_id(m) for m in history_messages]
    start = next((i for i, did in enumerate(ids)
                  if did is not None and did > layer1_from), len(history_messages))
    turns = _group_turns(history_messages[start:])
    if not turns:
        return layer1_from
    if estimate_messages_tokens([m for t in turns for m in t], s) <= budget_tokens:
        return layer1_from
    newest_tokens = estimate_messages_tokens(turns[-1], s)
    if newest_tokens > budget_tokens:
        logger.warning("layer1 单轮超预算 tokens=%s budget=%s", newest_tokens, budget_tokens)
    keep = keep_newest_fitting(
        turns, budget_tokens, lambda turn: estimate_messages_tokens(turn, s))
    if len(keep) == len(turns):
        return layer1_from
    dropped = turns[:len(turns) - len(keep)]
    new_id = db_id(dropped[-1][-1])
    if new_id is None:
        return layer1_from
    return max(layer1_from, new_id)                 # 只向前


def render_layer2(messages: list[BaseMessage], assistant_chars: int,
                  tool_keep_chars: int) -> list[BaseMessage]:
    """层 2 半压:用户原话不动;客服答复留开头;大块工具结果换一行标识。
    工具调用链折叠成一条 AI 文本,避免留下没有 tool_call 的孤儿 ToolMessage。"""
    out: list[BaseMessage] = []
    skip_tool = False
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append(HumanMessage(m.content if isinstance(m.content, str) else ""))
            skip_tool = False
        elif isinstance(m, AIMessage):
            if getattr(m, "tool_calls", None):
                names = ",".join(tc.get("name", "") for tc in m.tool_calls)
                out.append(AIMessage(f"[调用工具:{names},结果已省略]"))
                skip_tool = True
            else:
                text = m.content if isinstance(m.content, str) else ""
                out.append(AIMessage(
                    text[:assistant_chars] + ("…" if len(text) > assistant_chars else "")))
                skip_tool = False
        elif isinstance(m, ToolMessage):
            if skip_tool:
                continue
            text = m.content if isinstance(m.content, str) else ""
            if len(text) <= tool_keep_chars:
                out.append(AIMessage(text))
            else:
                out.append(AIMessage(f"[工具结果 {m.name or ''} 已省略,{len(text)}字]"))
    return out


def trim_turns_to_budget(messages: list[BaseMessage], budget_tokens: int,
                         s=settings) -> list[BaseMessage]:
    """按整轮从新到旧攒预算;最新一轮永远保留。不拆轮,避免孤立客服答复。"""
    if not messages:
        return []
    turns = _group_turns(messages)
    keep = keep_newest_fitting(
        turns, budget_tokens, lambda turn: estimate_messages_tokens(turn, s))
    return [m for turn in keep for m in turn]


def build_layers(history_messages: list[BaseMessage], summary_upto: int,
                 layer1_from: int, layer1_budget: int, layer2_budget: int,
                 assistant_chars: int, tool_keep_chars: int,
                 s=settings) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """切层 → 渲染层 2 → 返回(层2裁剪后, 层1裁剪后)。
    无锚点消息列表走 `trim_history` 纯 token 兜底,不改 DB 边界。"""
    layer2, layer1 = split_layers(history_messages, summary_upto, layer1_from)
    if not layer2 and not has_db_anchors(layer1):
        return [], trim_history(layer1, layer1_budget)
    rendered = render_layer2(layer2, assistant_chars, tool_keep_chars)
    return (trim_turns_to_budget(rendered, layer2_budget, s),
            trim_turns_to_budget(layer1, layer1_budget, s))


def summary_line(summary: str | None) -> str:
    """coref/意图的历史前缀:多段梗概压成一行;无摘要空串。"""
    if not summary:
        return ""
    return f"(早前对话摘要:{' '.join(str(summary).split())})"

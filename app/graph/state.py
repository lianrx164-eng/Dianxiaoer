from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def trace_of(state: dict, **updates) -> dict:
    """本轮 trace 合并:在同轮多个节点之间累加,不修改入参。"""
    return {**(state.get("trace") or {}), **updates}


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # 跨轮历史,checkpointer 续接
    user_id: str
    conversation_id: int
    intent: str            # 七类之一
    route: str             # knowledge | business | complaint | chitchat
    evidence: str          # 知识路编号证据文本
    citations: list        # 引用 chunk(前端可点)
    evidence_strong: bool  # 生成前证据闸信号
    answer: str            # 确定性节点产出的答复(Agent 答复走 messages)
    steps: int             # ReAct 步数(停止条件)
    tokens_used: int       # 逐步累加,供排障与 ch09 成本统计,不当停止条件
    suggested_actions: list  # [{"type": "transfer_human"} | {"type": "create_ticket", "draft": {...}}]
    trace: dict            # 留痕;每轮输入显式重置,节点内用 trace_of 合并

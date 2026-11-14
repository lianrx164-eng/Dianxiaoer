from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def trace_of(state: dict, **updates) -> dict:
    """本轮 trace 合并:在同轮多个节点之间累加,不修改入参。"""
    return {**(state.get("trace") or {}), **updates}


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # 跨轮历史,checkpointer 续接
    history_messages: list[AnyMessage]  # 本轮精简历史(DB 已落层2+层1,带 db-id 锚点)
    summary: str                        # conversations.summary 投影
    summary_upto_msg_id: int            # 梗概覆盖到哪条
    layer1_from_msg_id: int             # 层1起点;id > 此值为原文
    user_id: str
    conversation_id: int
    intent: str            # 九类之一(含「其他」)
    intent_confidence: float  # 意图 JSON 的 confidence(0-1),不参与分流
    ticket_followup: bool  # ch08 建单追问后的下一轮粘滞标记,classify_intent 消费即清
    route: str             # escalate | fallback_script | knowledge | refund_flow | business
    resolved_query: str    # 指代消解 + 改写后的完整问句,下游检索/判意图都用它
    order_id: str          # refund_flow:抽到/点选回填的订单号
    order_data: dict       # refund_flow:query_order 查到的订单数据
    evidence: str          # 知识路编号证据文本
    citations: list        # 引用 chunk(前端可点)
    evidence_strong: bool  # 生成前证据闸信号
    evidence_confidence: float  # ch09 置信度闸:四信号加权总分(0-1)
    fallback_source: str        # ch09 兜底落池 source:retrieval_low_conf | self_check
    retrieved_snapshot: list    # ch09 召回快照(Top3 原文+得分):落池/👎 回捞共用
    answer: str            # 确定性节点产出的答复(Agent 答复走 messages)
    steps: int             # ReAct 步数(停止条件)
    tokens_used: int       # 逐步累加,供排障与 ch09 成本统计,不当停止条件
    suggested_actions: list  # [{"type": "transfer_human"} | {"type": "create_ticket", "draft": {...}}]
    trace: dict            # 留痕;每轮输入显式重置,节点内用 trace_of 合并

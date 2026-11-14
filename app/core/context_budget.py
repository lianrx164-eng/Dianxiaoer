"""ch07 上下文预算:从模型窗口倒推历史/层 1/层 2 预算,并提供统一的中文 token 估算。

口径见 spec §7:预算常量与 chars_per_token 一起校准;只算已落库历史,当前用户消息
由 max_user_input_tokens 单独预留。本模块只依赖 app.config 与 langchain_core,
不 import DB/工具,避免与 repository/summarizer 形成循环。
"""
import json
import logging
import math
from dataclasses import dataclass

from langchain_core.messages import BaseMessage

from app.config import Settings, settings

logger = logging.getLogger(__name__)

# 每条消息的 role/分隔符等包装开销(粗估)
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_text_tokens(text: str | None, s: Settings = settings) -> int:
    """中文按字数折 token:ceil(len / chars_per_token);空串 0。"""
    if not text:
        return 0
    return math.ceil(len(text) / s.chars_per_token)


def estimate_message_tokens(m: BaseMessage, s: Settings = settings) -> int:
    """单条消息:包装开销 + 文本 + tool_calls 的 name/args。"""
    total = MESSAGE_OVERHEAD_TOKENS
    content = getattr(m, "content", "")
    if isinstance(content, str):
        total += estimate_text_tokens(content, s)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                total += estimate_text_tokens(part["text"], s)
    for tc in getattr(m, "tool_calls", None) or []:
        if not isinstance(tc, dict):
            continue
        total += estimate_text_tokens(tc.get("name", ""), s)
        total += estimate_text_tokens(
            json.dumps(tc.get("args", {}), ensure_ascii=False, default=str), s)
    return total


def estimate_messages_tokens(msgs: list[BaseMessage], s: Settings = settings) -> int:
    return sum(estimate_message_tokens(m, s) for m in msgs)


def keep_newest_fitting(items, budget_tokens: int, token_of):
    """从新到旧攒预算,最新一项永远保留;返回原顺序(旧→新)。

    memory 的整轮降级/层 2 裁剪与 repository 的摘要投影共用这一条规则。"""
    keep = []
    acc = 0
    for item in reversed(items):
        t = token_of(item)
        if keep and acc + t > budget_tokens:
            break
        keep.append(item)
        acc += t
    keep.reverse()
    return keep


@dataclass(frozen=True)
class ContextBudget:
    window: int
    fixed: int
    react_peak: int
    history: int
    layer1: int
    layer2: int


def compute_budget(s: Settings = settings) -> ContextBudget:
    """按 spec §7.1 公式算预算;layer1+layer2 比 history 少 1,是边界余量。"""
    fixed = (
        s.system_prompt_tokens + s.tool_schema_tokens + s.system_overhead_tokens
        + s.max_output_tokens + s.max_user_input_tokens
        + s.rerank_top_k * s.evidence_chunk_tokens
        + s.summary_budget_tokens + s.safety_margin_tokens
    )
    react_peak = s.max_agent_steps * s.tool_result_max_tokens
    leftover = s.model_context_window - fixed - react_peak
    history = max(0, min(s.desired_history_turns * s.turn_token_estimate, leftover))
    layer1 = max(0, history * 7 // 10 - 1)
    layer2 = max(0, history - layer1 - 1)
    return ContextBudget(window=s.model_context_window, fixed=fixed,
                         react_peak=react_peak, history=history,
                         layer1=layer1, layer2=layer2)


def self_check(s: Settings = settings) -> ContextBudget:
    """启动自检:连一轮都装不下就报警;不抛异常,应用仍可起。"""
    b = compute_budget(s)
    if b.history <= 0 or b.layer1 < s.turn_token_estimate:
        logger.error(
            "上下文预算不足: history=%s layer1=%s 单轮估算=%s "
            "window=%s fixed=%s react_peak=%s",
            b.history, b.layer1, s.turn_token_estimate,
            b.window, b.fixed, b.react_peak)
    return b

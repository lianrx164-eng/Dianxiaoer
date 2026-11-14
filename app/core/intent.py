from typing import Literal

from pydantic import BaseModel, Field

from app.core.llm import structured_model
from app.core.prompts import INTENT_CLASSIFY_PROMPT

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")


class _Intent(BaseModel):
    intent: Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"] = Field(
        description="七类意图之一"
    )


async def classify(query: str) -> str:
    """单标签七类意图。越界/异常回退闲聊(最保守出口)。"""
    try:
        r: _Intent = await structured_model(_Intent).ainvoke(
            INTENT_CLASSIFY_PROMPT.format_messages(query=query))
    except Exception:  # noqa: BLE001 - 分类失败不应打断整轮
        return "闲聊"
    return r.intent if r.intent in INTENTS else "闲聊"

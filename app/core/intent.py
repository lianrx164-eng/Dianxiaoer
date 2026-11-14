import logging
from typing import Literal, get_args

from pydantic import BaseModel, Field

from app.core.llm import structured_model
from app.core.prompts import INTENT_CLASSIFY_PROMPT

logger = logging.getLogger(__name__)

Intent = Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"]
INTENTS: tuple[str, ...] = get_args(Intent)


class _Intent(BaseModel):
    intent: Intent = Field(description="七类意图之一")


async def classify(query: str) -> str:
    """单标签七类意图。分类失败返回「未知」,由路由按 business 交 Agent 兜底,
    不要把业务问题伪装成闲聊固定话术。"""
    try:
        r: _Intent = await structured_model(_Intent).ainvoke(
            INTENT_CLASSIFY_PROMPT.format_messages(query=query))
    except Exception:  # noqa: BLE001 - 分类失败不应打断整轮
        logger.warning("意图分类失败,回退 business", exc_info=True)
        return "未知"
    return r.intent if r.intent in INTENTS else "未知"

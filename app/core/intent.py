import logging
from typing import Literal, get_args

from pydantic import BaseModel, Field

from app.config import settings
from app.core.llm import get_chat_model, structured_model, thinking_body
from app.core.prompts import INTENT_CLASSIFY_PROMPT

logger = logging.getLogger(__name__)

Intent = Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他"]
INTENTS: tuple[str, ...] = get_args(Intent)


class _Intent(BaseModel):
    intent: Intent = Field(description="八类意图之一")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="判断把握 0-1")


async def classify(query: str, history: str = "") -> dict:
    """八类意图 + confidence。扁平字段避开 glm 嵌套 502;解析失败/越界 → 归「其他」
    (最保守兜底,替换 ch05 的回退闲聊)。模型默认走 settings.intent_model(空则回落
    chat_model)——先求准。"""
    model = get_chat_model(model=settings.intent_model or None,
                           extra_body=thinking_body())
    try:
        r: _Intent = await structured_model(_Intent, model=model).ainvoke(
            INTENT_CLASSIFY_PROMPT.format_messages(
                query=query, history=history or "(无)"))
    except Exception:  # noqa: BLE001 - 分类失败不应打断整轮
        logger.warning("意图分类失败,归其他", exc_info=True)
        return {"intent": "其他", "confidence": 0.0}
    intent = r.intent if r.intent in INTENTS else "其他"
    return {"intent": intent, "confidence": float(r.confidence)}

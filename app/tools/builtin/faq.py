from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core import evidence
from app.tools import registry


class FaqInput(BaseModel):
    keyword: str = Field(description="用户咨询的政策/规则/操作类问题(可用原话)")
    category: str | None = Field(
        default=None,
        description="可选:知识库一级分类,如『商品与购物 FAQ』『退货退款政策』『售后手册』『商品规格手册』")


@tool(args_schema=FaqInput)
async def query_faq(keyword: str, category: str | None = None) -> dict:
    """查询常见问题/政策知识库(混合检索+重排)。用于政策、规则、时效、费用、商品手册等通用问题。
    返回带编号证据供作答引用;证据不足时返回 sufficient=False,请据此向用户拒答。"""
    result = await evidence.gather_evidence(keyword, category=category)
    return {k: v for k, v in result.items() if k != "top"}


# query_faq 走 RAG 管线(改写+混合检索+重排+自评,多次上游往返),远慢于默认 5s,
# 且重试会叠加分钟级等待,故显式 retryable=False。
registry.register(registry.spec_from_langchain_tool(
    query_faq, source="builtin", timeout=45.0, retryable=False))

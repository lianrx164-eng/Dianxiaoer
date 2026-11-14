import random
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from app.core import evidence
from app.db import repository


class OrderInput(BaseModel):
    order_id: str = Field(description="订单号,例如 1001")


class ProductInput(BaseModel):
    product_name: str = Field(description="商品名称或关键词,例如 猫粮")


class LogisticsInput(BaseModel):
    tracking_no: str = Field(
        description="物流单号(如 SF1234567890),只能通过 query_order 查询订单后从返回的 tracking_no 字段获得")


@tool(args_schema=OrderInput)
async def query_order(order_id: str) -> dict:
    """查询订单的状态、金额、下单时间、商品名和物流单号。用于用户询问某个订单情况时。"""
    rng = random.Random(f"order:{order_id}")
    return {
        "order_id": order_id,
        "status": rng.choice(["待付款", "已付款", "已发货", "已签收"]),
        "amount": rng.randint(50, 2000),
        "created_at": f"2026-07-{rng.randint(1, 12):02d} 10:00",
        "product": rng.choice(["智能猫砂盆", "猫粮 5kg", "猫爬架", "自动饮水机"]),
        "tracking_no": f"SF{rng.randint(10**9, 10**10 - 1)}",
    }


@tool(args_schema=ProductInput)
async def query_product(product_name: str) -> dict:
    """查询商品的价格、库存和规格。用于用户咨询某商品是否有货、多少钱时。"""
    rng = random.Random(f"product:{product_name}")
    return {
        "product_name": product_name,
        "price": rng.randint(20, 999),
        "stock": rng.randint(0, 500),
        "spec": rng.choice(["标准装", "家庭装", "试用装"]),
    }


@tool(args_schema=LogisticsInput)
async def query_logistics(tracking_no: str) -> dict:
    """查询物流状态、当前位置和轨迹。用于用户询问物流/快递到哪了时。
    入参是物流单号,需要先用 query_order 查到订单的 tracking_no。"""
    rng = random.Random(f"logistics:{tracking_no}")
    status = rng.choice(["已揽件", "运输中", "派送中", "已签收"])
    city = rng.choice(["深圳", "广州", "杭州", "上海", "成都"])
    return {
        "tracking_no": tracking_no,
        "status": status,
        "location": f"{city}分拨中心",
        "timeline": [f"{city}分拨中心 已发出", f"当前状态:{status}"],
    }


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


@tool
async def create_ticket(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    conversation_id: Annotated[int, InjectedToolArg],
) -> dict:
    """当用户问题需要人工介入(投诉、无法自助解决、明确要求人工)时,创建人工工单。
    description 填用户问题描述,ticket_type 从 售后/投诉/咨询 中选。"""
    ticket_no = await repository.create_ticket(conversation_id, description, ticket_type)
    return {"ticket_no": ticket_no, "status": "已创建"}

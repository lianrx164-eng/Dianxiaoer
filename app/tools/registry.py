from langchain_core.tools import BaseTool

from app.tools.business import (
    create_ticket,
    query_faq,
    query_logistics,
    query_order,
    query_product,
    submit_refund,
)

_ALL: list[BaseTool] = [query_order, query_product, query_logistics, query_faq,
                        create_ticket, submit_refund]
_BY_NAME: dict[str, BaseTool] = {t.name: t for t in _ALL}

NO_RETRY: set[str] = {"create_ticket", "query_faq", "submit_refund"}  # 写类不重试;query_faq 是重 RAG,重试会叠加分钟级等待
INJECT_CONVERSATION: set[str] = {"create_ticket"}  # 需注入会话主键的工具
# RAG 工具要跑改写 + 混合检索 + 重排 + 自评,默认 5s 不够
TOOL_TIMEOUTS: dict[str, float] = {"query_faq": 45.0}


def get_all_tools() -> list[BaseTool]:
    return _ALL


def get_tool(name: str) -> BaseTool | None:
    return _BY_NAME.get(name)

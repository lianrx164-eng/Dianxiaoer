from pydantic import BaseModel, Field

from app.core.llm import structured_model
from app.core.prompts import EXPAND_QUERIES_PROMPT, QUERY_REWRITE_PROMPT


class _Rewrite(BaseModel):
    standard: str = Field(description="标准问法")
    expanded: list[str] = Field(default_factory=list, description="同义/近义扩展词")


class _Expanded(BaseModel):
    queries: list[str] = Field(default_factory=list, description="严格 3 条检索友好查询")


def _chain(model=None):
    return QUERY_REWRITE_PROMPT | structured_model(_Rewrite, model=model)


async def understand(query: str, model=None) -> dict:
    """口语→标准问法 + 同义词扩展。扁平字段(list[str],非嵌套对象)。"""
    r: _Rewrite = await _chain(model).ainvoke({"query": query})
    return {"standard": r.standard or query, "expanded": list(r.expanded or [])}


async def expand_queries(query: str) -> list[str]:
    """把一句问题泛化成检索友好查询(强制 JSON 单字段,扁平 list[str] 避开 glm 502)。
    严格取前 3 条非空;模型异常/全空则回落 [query],保证 retrieve_policy 至少有一条可查。"""
    try:
        r: _Expanded = await (EXPAND_QUERIES_PROMPT |
                              structured_model(_Expanded)).ainvoke({"query": query})
        qs = [q.strip() for q in (r.queries or []) if q and q.strip()]
    except Exception:  # noqa: BLE001 - 扩写失败不应阻断检索
        qs = []
    return qs[:3] or [query]

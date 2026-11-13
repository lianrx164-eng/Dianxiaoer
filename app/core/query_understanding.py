from pydantic import BaseModel, Field

from app.core.llm import structured_model
from app.core.prompts import QUERY_REWRITE_PROMPT


class _Rewrite(BaseModel):
    standard: str = Field(description="标准问法")
    expanded: list[str] = Field(default_factory=list, description="同义/近义扩展词")


def _chain(model=None):
    return QUERY_REWRITE_PROMPT | structured_model(_Rewrite, model=model)


async def understand(query: str, model=None) -> dict:
    """口语→标准问法 + 同义词扩展。扁平字段(list[str],非嵌套对象)。"""
    r: _Rewrite = await _chain(model).ainvoke({"query": query})
    return {"standard": r.standard or query, "expanded": list(r.expanded or [])}

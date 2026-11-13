from pydantic import BaseModel, Field

from app.config import settings
from app.core.llm import get_chat_model
from app.core.prompts import QUERY_REWRITE_PROMPT


class _Rewrite(BaseModel):
    standard: str = Field(description="标准问法")
    expanded: list[str] = Field(default_factory=list, description="同义/近义扩展词")


def _chain(model=None):
    if model is None:
        extra = ({"thinking": {"type": settings.extract_thinking}}
                 if settings.extract_thinking else None)
        model = get_chat_model(extra_body=extra)
    return QUERY_REWRITE_PROMPT | model.with_structured_output(
        _Rewrite, method="function_calling")


async def understand(query: str, model=None) -> dict:
    """口语→标准问法 + 同义词扩展。扁平字段(list[str],非嵌套对象)。"""
    r: _Rewrite = await _chain(model).ainvoke({"query": query})
    return {"standard": r.standard or query, "expanded": list(r.expanded or [])}

from pydantic import BaseModel, Field

from app.core.llm import structured_model
from app.core.prompts import SELF_CHECK_PROMPT


class _Check(BaseModel):
    useful: bool = Field(description="证据是否足以回答")
    reason: str = Field(default="", description="判断依据")


def _chain(model=None):
    return SELF_CHECK_PROMPT | structured_model(_Check, model=model)


async def check_sufficient(query: str, evidence_texts: list[str], model=None) -> dict:
    """生成前证据充分性自评。扁平字段 useful/reason。"""
    evidence = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(evidence_texts)) or "(无证据)"
    r: _Check = await _chain(model).ainvoke({"query": query, "evidence": evidence})
    return {"useful": bool(r.useful), "reason": r.reason or ""}

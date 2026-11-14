from pydantic import BaseModel, Field

from app.core.llm import get_chat_model, structured_model, thinking_body
from app.core.prompts import FAITHFULNESS_PROMPT


class FaithVerdict(BaseModel):
    faithful: bool = Field(description="回答是否被证据支撑")
    reason: str = Field(default="", description="判断依据")


def judge_model(temperature: float = 0.0, model=None):
    if model is not None:
        return model
    return get_chat_model(extra_body=thinking_body(), temperature=temperature)


async def judge_faithfulness(evidence: str, answer: str,
                             model=None, temperature: float = 0.0) -> tuple[bool, str]:
    """忠实度裁判。temperature 默认 0,减少同一输入的判分抖动。"""
    chain = FAITHFULNESS_PROMPT | structured_model(
        FaithVerdict, temperature=temperature, model=model)
    r: FaithVerdict = await chain.ainvoke({"evidence": evidence, "answer": answer})
    return bool(r.faithful), r.reason or ""


def rebuild_evidence(citations: list[dict] | None) -> str:
    """把台账里的 Top-K 快照拼回裁判当时的证据格式;老快照缺 n 时按序补。"""
    lines = []
    for i, c in enumerate(citations or [], 1):
        n = c.get("n") or i
        lines.append(f"[{n}] {c.get('question', '')}: {c.get('answer', '')}")
    return "\n".join(lines) or "(无证据)"


def expected_faithful(status: str) -> bool | None:
    """人工处置翻译成标准答案:已解决 = 当时确实编造;无需解决 = 当时其实忠实。"""
    return {"已解决": False, "无需解决": True}.get(status)


def summarize_check(rows: list[dict]) -> dict:
    """rows: [{id, status, expected, verdict, error}]。调用失败不计入一致率。"""
    judged = [r for r in rows if r.get("error") is None and r.get("verdict") is not None]
    agree = sum(1 for r in judged if r["verdict"] == r["expected"])
    return {"total": len(rows), "judged": len(judged),
            "agree": agree, "disagree": len(judged) - agree,
            "failed": len(rows) - len(judged),
            "agreement": round(agree / len(judged), 4) if judged else None}


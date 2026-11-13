from pydantic import BaseModel, Field

from app.config import settings
from app.core.llm import get_chat_model
from app.core.prompts import MINING_PROMPT
from app.db import repository
from app.kb import dedup, dualwrite
from app.kb.documents import Chunk


class QaPair(BaseModel):
    question: str = Field(description="通用问法,去掉具体订单号/人名")
    answer: str = Field(description="客服原答,不编造")


class QaExtraction(BaseModel):
    pairs: list[QaPair] = Field(default_factory=list, description="抽出的问答对,可为空")


def _mining_model():
    # 与 ch01 提取任务同款:DeepSeek 思考模式不支持强制 tool_choice,需显式关闭
    extra_body = (
        {"thinking": {"type": settings.extract_thinking}}
        if settings.extract_thinking else None
    )
    return get_chat_model(extra_body=extra_body)


async def extract_qa(conversation_texts: list[str], model=None) -> list[QaPair]:
    model = model or _mining_model()
    # 上游 DeepSeek 不支持 json_schema 的 response_format,走 function_calling(ch01 已验证)
    chain = MINING_PROMPT | model.with_structured_output(
        QaExtraction, method="function_calling"
    )
    result: QaExtraction = await chain.ainvoke({"conversations": "\n---\n".join(conversation_texts)})
    return result.pairs if result else []


async def _load_conversation_texts() -> list[tuple[str, str]]:
    """返回 [(source_ref, 对话文本)];只取 user/assistant 消息,跳过工具结果与已知上游泄漏文本。"""
    convs = await repository.list_conversations_with_messages()
    out = []
    for conv_id, msgs in convs:
        lines = [
            f"{m.role}: {m.content}"
            for m in msgs
            if m.role in ("user", "assistant") and m.content and "DSML" not in m.content
        ]
        if lines:
            out.append((f"conv:{conv_id}", "\n".join(lines)))
    return out


async def mine(batch_size: int = 20, model=None) -> dict:
    sources = await _load_conversation_texts()
    batch_no = f"mine-{len(sources)}"  # 以来源数标批;重跑去重靠 dedupe,不靠批号
    # 1) 分批抽取 → 写 staging(extracted)
    for start in range(0, len(sources), batch_size):
        batch = sources[start:start + batch_size]
        pairs = await extract_qa([t for _, t in batch], model=model)
        for p in pairs:
            await repository.insert_staging(batch_no, batch[0][0], p.question, p.answer)
    # 2) 整体去重(staging extracted 内 + 对已有 knowledge)
    staged = await repository.list_staging_by_status("extracted")
    existing = await repository.list_all_questions()
    kept, discarded = dedup.dedupe(staged, existing)
    await repository.set_staging_status([s.id for s in kept], "kept")
    await repository.set_staging_status([s.id for s in discarded], "discarded")
    # 3) kept 写 knowledge_chunks(pending, content_type=mined, questions=真实问法)
    chunks = [Chunk(category="历史对话", questions=s.question, answer=s.answer,
                    section_path="mined", content_type="mined") for s in kept]
    if chunks:
        await dualwrite.write_pending(chunks)
    return {"sources": len(sources), "extracted": len(staged),
            "kept": len(kept), "discarded": len(discarded)}

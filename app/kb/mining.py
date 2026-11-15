from datetime import datetime

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model, thinking_body
from app.core.prompts import MINING_PROMPT
from app.db import repository
from app.kb import dedup, dualwrite
from app.kb.documents import Chunk


class QaPair(BaseModel):
    question: str = Field(description="通用问法,去掉具体订单号/人名")
    answer: str = Field(description="客服原答,不编造")
    source_ref: str = Field(default="", description="该问答对来自的对话来源标识,如 conv:12")


class QaExtraction(BaseModel):
    pairs: list[QaPair] = Field(default_factory=list, description="抽出的问答对,可为空")


def _mining_model():
    # 与 ch01 提取任务同款:DeepSeek 思考模式不支持强制 tool_choice,需显式关闭
    return get_chat_model(extra_body=thinking_body())


async def extract_qa_with_sources(
    sources: list[tuple[str, str]], model=None
) -> list[QaPair]:
    """sources = [(source_ref, 对话文本)];每段对话带来源标识喂给模型,出处回填到 source_ref。"""
    model = model or _mining_model()
    # 上游 DeepSeek 不支持 json_schema 的 response_format,走 function_calling(ch01 已验证)
    chain = MINING_PROMPT | model.with_structured_output(
        QaExtraction, method="function_calling"
    )
    conversations = "\n---\n".join(f"[{ref}]\n{text}" for ref, text in sources)
    result: QaExtraction = await chain.ainvoke({"conversations": conversations})
    return result.pairs if result else []


async def extract_qa(conversation_texts: list[str], model=None) -> list[QaPair]:
    """eval / 调用方只有纯文本、不关心溯源时用;来源编号是就地生成的。"""
    sources = [(f"conv:{i}", text) for i, text in enumerate(conversation_texts)]
    return await extract_qa_with_sources(sources, model=model)


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
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")  # 每次运行一批独立编号,重跑不撞
    # 1) 分批抽取 → 写 staging(extracted);每条问答对记自己的来源
    for idx, start in enumerate(range(0, len(sources), batch_size), start=1):
        batch = sources[start:start + batch_size]
        batch_no = f"mine-{run_id}-{idx:02d}"
        pairs = await extract_qa_with_sources(batch, model=model)
        known = {ref for ref, _ in batch}
        for p in pairs:
            source_ref = p.source_ref if p.source_ref in known else None
            await repository.insert_staging(batch_no, source_ref, p.question, p.answer)
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

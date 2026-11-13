from datetime import datetime

from sqlalchemy import func, select, text

import app.db.base as db          # 用模块属性引用,便于测试 monkeypatch async_session
from app.db.models import (
    Conversation,
    FaithCase,
    Faq,
    KnowledgeChunk,
    LowConfidenceQuestion,
    Message,
    QaExtractionStaging,
    Ticket,
)

_TICKET_SEQ = 0


def _gen_ticket_no() -> str:
    global _TICKET_SEQ
    _TICKET_SEQ += 1
    return f"T{datetime.now():%Y%m%d%H%M%S}{_TICKET_SEQ:03d}"


async def create_conversation(user_id: str) -> int:
    async with db.async_session() as s:
        conv = Conversation(user_id=user_id)
        s.add(conv)
        await s.commit()
        return conv.id


async def get_conversation(conversation_id: int) -> Conversation | None:
    async with db.async_session() as s:
        return await s.get(Conversation, conversation_id)


async def append_message(
    conversation_id: int,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> int:
    async with db.async_session() as s:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        s.add(msg)
        await s.commit()
        return msg.id


async def list_messages(conversation_id: int) -> list[Message]:
    async with db.async_session() as s:
        result = await s.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list(result.scalars())


async def search_faq(keyword: str) -> list[Faq]:
    async with db.async_session() as s:
        result = await s.execute(
            select(Faq).where(Faq.question.like(f"%{keyword}%"))
        )
        return list(result.scalars())


async def create_ticket(conversation_id: int, description: str, ticket_type: str) -> str:
    ticket_no = _gen_ticket_no()
    async with db.async_session() as s:
        s.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            )
        )
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.status = "已转人工"
        await s.commit()
    return ticket_no


async def insert_knowledge_chunk(
    category: str, questions: str, answer: str,
    section_path: str | None = None, content_type: str | None = None,
    is_key_clause: int = 0,
) -> int:
    async with db.async_session() as s:
        row = KnowledgeChunk(
            category=category, questions=questions, answer=answer,
            section_path=section_path, content_type=content_type,
            is_key_clause=is_key_clause,
        )
        s.add(row)
        await s.commit()
        return row.id


async def list_pending_chunks() -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
        return list(result.scalars())


async def mark_chunk_vectorized(chunk_id: int, vector_id: str) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.vector_id = vector_id
            row.vectorize_status = "done"
            await s.commit()


async def set_chunk_neighbors(chunk_id: int, prev_id: int | None, next_id: int | None) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.prev_chunk_id = prev_id
            row.next_chunk_id = next_id
            await s.commit()


async def count_chunks_by_status(status: str) -> int:
    async with db.async_session() as s:
        result = await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == status)
        )
        return int(result.scalar_one())


async def list_all_questions() -> list[str]:
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions))
        return list(result.scalars())


async def insert_staging(batch_no: str, source_ref: str | None, question: str, answer: str) -> int:
    async with db.async_session() as s:
        row = QaExtractionStaging(
            batch_no=batch_no, source_ref=source_ref, question=question, answer=answer
        )
        s.add(row)
        await s.commit()
        return row.id


async def list_staging_by_status(status: str) -> list[QaExtractionStaging]:
    async with db.async_session() as s:
        result = await s.execute(
            select(QaExtractionStaging)
            .where(QaExtractionStaging.status == status)
            .order_by(QaExtractionStaging.id)
        )
        return list(result.scalars())


async def set_staging_status(ids: list[int], status: str) -> None:
    if not ids:
        return
    async with db.async_session() as s:
        for i in ids:
            row = await s.get(QaExtractionStaging, i)
            if row is not None:
                row.status = status
        await s.commit()


async def list_conversations_with_messages() -> list[tuple[int, list[Message]]]:
    async with db.async_session() as s:
        conv_ids = list((await s.execute(select(Conversation.id).order_by(Conversation.id))).scalars())
        out = []
        for cid in conv_ids:
            msgs = list((await s.execute(
                select(Message).where(Message.conversation_id == cid).order_by(Message.id)
            )).scalars())
            out.append((cid, msgs))
        return out


async def knowledge_stats() -> dict:
    async with db.async_session() as s:
        async def _count(*where):
            q = select(func.count()).select_from(KnowledgeChunk)
            for w in where:
                q = q.where(w)
            return int((await s.execute(q)).scalar_one())

        return {
            "total": await _count(),
            "pending": await _count(KnowledgeChunk.vectorize_status == "pending"),
            "done": await _count(KnowledgeChunk.vectorize_status == "done"),
            "key_clause": await _count(KnowledgeChunk.is_key_clause == 1),
        }


async def list_recent_chunks(limit: int = 10) -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk).order_by(KnowledgeChunk.id.desc()).limit(limit)
        )
        return list(result.scalars())


async def list_chunk_pairs() -> list[tuple[str, str]]:
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions, KnowledgeChunk.answer))
        return list(result.all())


async def staging_stats() -> dict:
    async with db.async_session() as s:
        rows = (
            await s.execute(
                select(QaExtractionStaging.status, func.count())
                .group_by(QaExtractionStaging.status)
            )
        ).all()
        counts = {"extracted": 0, "kept": 0, "discarded": 0}
        for status, n in rows:
            counts[status] = int(n)
        batches = (
            await s.execute(
                select(func.count(func.distinct(QaExtractionStaging.batch_no)))
            )
        ).scalar_one()
        return {"counts": counts, "batches": int(batches)}


async def list_staging(limit: int = 200) -> list[QaExtractionStaging]:
    async with db.async_session() as s:
        result = await s.execute(
            select(QaExtractionStaging)
            .order_by(QaExtractionStaging.id.desc())
            .limit(limit)
        )
        return list(result.scalars())


async def truncate_knowledge_chunks() -> None:
    async with db.async_session() as s:
        await s.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        await s.execute(text("TRUNCATE TABLE knowledge_chunks"))
        await s.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        await s.commit()


async def insert_low_confidence(
    conversation_id: int | None, raw_question: str, source: str, reason: str | None
) -> int:
    async with db.async_session() as s:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id, raw_question=raw_question,
            source=source, reason=reason,
        )
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return row.id


DISPOSED = ("已解决", "无需解决")


async def upsert_faith_case(
    eval_id: str, bucket: str, query: str, strategy: str, answer: str,
    reason: str, citations: list | None, judge_model: str | None,
) -> dict:
    """一题一行:同题再判只更新答案/理由/角标快照并累加 seen_count;
    已处置的题又被判出 → 退回未解决(复发),清空处置说明。"""
    async with db.async_session() as s:
        row = (await s.execute(
            select(FaithCase).where(FaithCase.eval_id == eval_id)
        )).scalar_one_or_none()
        recurrence = False
        if row is None:
            row = FaithCase(
                eval_id=eval_id, bucket=bucket, query=query, strategy=strategy,
                answer=answer, reason=reason, citations=citations,
                judge_model=judge_model,
            )
            s.add(row)
        else:
            recurrence = row.status in DISPOSED
            row.bucket, row.query, row.strategy = bucket, query, strategy
            row.answer, row.reason = answer, reason
            row.citations, row.judge_model = citations, judge_model
            row.seen_count += 1
            row.last_seen_at = datetime.now()
            if recurrence:
                row.status = "未解决"
                row.resolution = None
        await s.commit()
        await s.refresh(row)
        return {"id": row.id, "eval_id": row.eval_id, "status": row.status,
                "seen_count": row.seen_count, "recurrence": recurrence}


def _faith_case(row: FaithCase) -> dict:
    return {
        "id": row.id, "eval_id": row.eval_id, "bucket": row.bucket,
        "query": row.query, "strategy": row.strategy, "answer": row.answer,
        "reason": row.reason, "citations": row.citations,
        "judge_model": row.judge_model, "status": row.status,
        "seen_count": row.seen_count,
        "first_seen_at": row.first_seen_at.isoformat() if row.first_seen_at else None,
        "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
        "resolution": row.resolution,
        "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
    }


async def list_faith_cases(status: str | None = None,
                           page: int = 1, size: int = 20) -> dict:
    async with db.async_session() as s:
        counts = {"未解决": 0, "已解决": 0, "无需解决": 0}
        rows = (await s.execute(
            select(FaithCase.status, func.count()).group_by(FaithCase.status)
        )).all()
        for st, n in rows:
            counts[st] = int(n)
        q = select(FaithCase)
        if status:
            q = q.where(FaithCase.status == status)
        total = int((await s.execute(
            select(func.count()).select_from(q.subquery())
        )).scalar_one())
        page = max(1, page)
        size = max(1, min(size, 100))
        rows = (await s.execute(
            q.order_by(FaithCase.status, FaithCase.last_seen_at.desc())
            .offset((page - 1) * size).limit(size)
        )).scalars()
        return {"items": [_faith_case(r) for r in rows], "counts": counts,
                "total": total, "page": page, "size": size}


async def set_faith_case_status(case_id: int, status: str,
                                resolution: str | None) -> dict | None:
    if status not in ("未解决", "已解决", "无需解决"):
        raise ValueError("status 不合法")
    text_res = (resolution or "").strip()
    if status in DISPOSED and not text_res:
        raise ValueError("标已解决/无需解决时必须写一句处置说明")
    async with db.async_session() as s:
        row = await s.get(FaithCase, case_id)
        if row is None:
            return None
        row.status = status
        if status == "未解决":
            row.resolution = None
            row.resolved_at = None
        else:
            row.resolution = text_res
            row.resolved_at = datetime.now()
        await s.commit()
        await s.refresh(row)
        return _faith_case(row)


async def faith_case_status_map() -> dict[str, str]:
    async with db.async_session() as s:
        rows = (await s.execute(select(FaithCase.eval_id, FaithCase.status))).all()
        return {eval_id: status for eval_id, status in rows}


async def faith_case_count() -> int:
    async with db.async_session() as s:
        return int((await s.execute(
            select(func.count()).select_from(FaithCase)
        )).scalar_one())


async def list_disposed_faith_cases() -> list[FaithCase]:
    """已处置且带证据快照的个案:judge-check 的回归集。"""
    async with db.async_session() as s:
        rows = (await s.execute(
            select(FaithCase)
            .where(FaithCase.status.in_(DISPOSED), FaithCase.citations.isnot(None))
            .order_by(FaithCase.id)
        )).scalars()
        return list(rows)

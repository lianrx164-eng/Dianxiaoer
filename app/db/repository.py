from datetime import datetime

from sqlalchemy import func, select, text

from app.config import settings
from app.core.context_budget import estimate_text_tokens, keep_newest_fitting

import app.db.base as db          # 用模块属性引用,便于测试 monkeypatch async_session
from app.db.models import (
    Conversation,
    ConversationSummary,
    EvalRun,
    FaithCase,
    Faq,
    KnowledgeChunk,
    LowConfidenceQuestion,
    Message,
    QaExtractionStaging,
    ReviewQueue,
    Ticket,
    ToolAuditLog,
    TopicClassification,
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
    conversation_id: int | None, raw_question: str, source: str, reason: str | None,
    retrieved_chunks: list | None = None,
) -> int:
    async with db.async_session() as s:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id, raw_question=raw_question,
            source=source, reason=reason, retrieved_chunks=retrieved_chunks,
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


# ---------------- ch07 会话上下文 ----------------


async def list_dialog_messages(conversation_id: int) -> list[Message]:
    """user/assistant 消息按 id 升序(摘要与历史回载共用;tool 行不落库,过滤一道保险)。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id,
                   Message.role.in_(("user", "assistant")))
            .order_by(Message.id)
        )
        return list(result.scalars())


async def update_layer1_from(conversation_id: int, msg_id: int) -> None:
    """层 1 降级:只挪边界 id,不搬数据。"""
    async with db.async_session() as s:
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.layer1_from_msg_id = msg_id
            await s.commit()


async def list_summaries(conversation_id: int) -> list[ConversationSummary]:
    async with db.async_session() as s:
        result = await s.execute(
            select(ConversationSummary)
            .where(ConversationSummary.conversation_id == conversation_id)
            .order_by(ConversationSummary.seq)
        )
        return list(result.scalars())


async def append_summary(
    conversation_id: int, from_msg_id: int, upto_msg_id: int, content: str,
) -> int:
    """插一段 + 重建投影 + 追 summary_upto,单事务;旧段永不改。返回新段 seq。"""
    budget = settings.summary_budget_tokens
    async with db.async_session() as s:
        conv = await s.get(Conversation, conversation_id)
        if conv is None:
            return 0
        if (conv.summary_upto_msg_id or 0) >= upto_msg_id:
            return 0                      # 另一进程已推进更远,不重复段/不回退边界
        max_seq = (await s.execute(
            select(func.max(ConversationSummary.seq))
            .where(ConversationSummary.conversation_id == conversation_id)
        )).scalar_one()
        seq = int(max_seq or 0) + 1
        s.add(ConversationSummary(
            conversation_id=conversation_id, seq=seq,
            from_msg_id=from_msg_id, upto_msg_id=upto_msg_id, content=content))
        await s.flush()
        rows = list((await s.execute(
            select(ConversationSummary)
            .where(ConversationSummary.conversation_id == conversation_id)
            .order_by(ConversationSummary.seq))).scalars())
        kept = keep_newest_fitting(
            rows, budget, lambda row: estimate_text_tokens(row.content))
        conv.summary = "\n".join(row.content for row in kept)
        conv.summary_upto_msg_id = upto_msg_id
        await s.commit()
        return seq


async def list_conversations(user_id: str, limit: int = 50) -> list[dict]:
    """某用户的会话列表:id desc 新在前,带首问预览与是否已摘要。"""
    async with db.async_session() as s:
        convs = list((await s.execute(
            select(Conversation).where(Conversation.user_id == user_id)
            .order_by(Conversation.id.desc()).limit(limit))).scalars())
        out = []
        for conv in convs:
            preview = (await s.execute(
                select(Message.content)
                .where(Message.conversation_id == conv.id, Message.role == "user")
                .order_by(Message.id).limit(1))).scalar_one_or_none()
            out.append({
                "id": conv.id, "status": conv.status,
                "preview": (preview or "").strip()[:40],
                "has_summary": bool(conv.summary_upto_msg_id),
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
            })
        return out


# ---------------- ch08 工具审计 ----------------


async def insert_tool_audit(
    conversation_id: int | None, tool_call_id: str | None, tool_name: str,
    tool_source: str, mcp_server: str | None, arguments: dict | None,
    result_summary: str | None, status: str, error_message: str | None,
    retry_count: int, duration_ms: int | None,
) -> None:
    """工具调用审计落一条;调用方(engine)自行 try/except——审计失败不许反拦工具执行。"""
    async with db.async_session() as s:
        s.add(ToolAuditLog(
            conversation_id=conversation_id, tool_call_id=tool_call_id,
            tool_name=tool_name, tool_source=tool_source, mcp_server=mcp_server,
            arguments=arguments, result_summary=result_summary, status=status,
            error_message=error_message, retry_count=retry_count, duration_ms=duration_ms))
        await s.commit()


# ---------------- ch09 数据飞轮(待审队列 + 评估轮次) ----------------


async def fetch_unmatched_low_conf(limit: int) -> list[LowConfidenceQuestion]:
    """飞轮处理游标:尚未归并(matched_review_id IS NULL)的池内问题,按 id 升序。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id.is_(None))
            .order_by(LowConfidenceQuestion.id).limit(limit)
        )
        return list(result.scalars())


async def list_review_candidates(limit: int = 200) -> list[dict]:
    """查重候选:全部状态的缺口行(驳回即终审,只累加),updated_at 倒序截断防 token 爆。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(ReviewQueue.id, ReviewQueue.normalized_question)
            .order_by(ReviewQueue.updated_at.desc(), ReviewQueue.id.desc())
            .limit(limit)
        )
        return [{"id": r.id, "normalized_question": r.normalized_question} for r in result]


async def insert_review_item(normalized_question: str, ai_suggested_answer: str | None) -> int:
    async with db.async_session() as s:
        row = ReviewQueue(normalized_question=normalized_question,
                          ai_suggested_answer=ai_suggested_answer)
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return row.id


async def increment_occurrence(review_id: int) -> None:
    """查重命中:只累加次数,状态不动(命中已驳回/已通过也一样——驳回即终审)。"""
    async with db.async_session() as s:
        row = await s.get(ReviewQueue, review_id)
        if row is not None:
            row.occurrence_count = row.occurrence_count + 1
            await s.commit()


async def set_matched_review(lcq_id: int, review_id: int) -> None:
    async with db.async_session() as s:
        row = await s.get(LowConfidenceQuestion, lcq_id)
        if row is not None:
            row.matched_review_id = review_id
            await s.commit()


async def list_review_queue(status: str | None) -> list[ReviewQueue]:
    """审核页列表:按出现次数降序(频次=优先级),同频新的在前。"""
    async with db.async_session() as s:
        q = select(ReviewQueue).order_by(
            ReviewQueue.occurrence_count.desc(), ReviewQueue.id.desc())
        if status:
            q = q.where(ReviewQueue.review_status == status)
        return list((await s.execute(q)).scalars())


async def get_review_detail(
    review_id: int,
) -> tuple[ReviewQueue, list[LowConfidenceQuestion]] | None:
    """详情:缺口行 + 归并进来的原话流水(带 source/快照,审核人判断真缺还是没检到)。"""
    async with db.async_session() as s:
        item = await s.get(ReviewQueue, review_id)
        if item is None:
            return None
        raws = list((await s.execute(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id == review_id)
            .order_by(LowConfidenceQuestion.id)
        )).scalars())
        return item, raws


async def update_review_status(
    review_id: int, status: str, approved_answer: str | None = None
) -> bool:
    """仅「待审」可流转(通过/驳回都是终态);返回是否真的更新了。"""
    async with db.async_session() as s:
        row = await s.get(ReviewQueue, review_id)
        if row is None or row.review_status != "待审":
            return False
        row.review_status = status
        if approved_answer is not None:
            row.approved_answer = approved_answer
        await s.commit()
        return True


async def insert_eval_run(triggered_by: str, dataset_size: int, metrics: dict) -> int:
    async with db.async_session() as s:
        row = EvalRun(triggered_by=triggered_by, dataset_size=dataset_size, metrics=metrics)
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return row.id


async def list_eval_runs(limit: int = 10) -> list[EvalRun]:
    async with db.async_session() as s:
        result = await s.execute(select(EvalRun).order_by(EvalRun.id.desc()).limit(limit))
        return list(result.scalars())


# ---------- ch10 主题分类 ----------

def _pool_text_stmt():
    """池问题取文本的公共查询:优先归并后的标准化问法,没归并退回原话。"""
    return (
        select(LowConfidenceQuestion.id, LowConfidenceQuestion.raw_question,
               ReviewQueue.normalized_question)
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
        .order_by(LowConfidenceQuestion.id)
    )


async def list_pool_texts() -> list[dict]:
    """ch10 训练语料捞取:全量低置信度问题(未归并的退回原话)。"""
    async with db.async_session() as s:
        rows = (await s.execute(_pool_text_stmt())).all()
    return [{"question_id": qid, "text": norm or raw} for qid, raw, norm in rows]


async def list_unclassified_questions(limit: int = 500) -> list[dict]:
    """ch10 旁路批处理捞取:尚未归类、且已归并的低置信度问题。

    只归有 matched_review_id 的问题——分类器只吃归并阶段产出的标准化问法,
    未归并的留到下一轮归并后再归,不喂原话(spec §14.1 方案 A)。
    """
    stmt = (
        _pool_text_stmt()
        .outerjoin(TopicClassification,
                   TopicClassification.question_id == LowConfidenceQuestion.id)
        .where(TopicClassification.id.is_(None))
        .where(LowConfidenceQuestion.matched_review_id.is_not(None))
        .limit(limit)
    )
    async with db.async_session() as s:
        rows = (await s.execute(stmt)).all()
    return [{"question_id": qid, "text": norm or raw} for qid, raw, norm in rows]


async def insert_topic_classifications(rows: list[dict]) -> int:
    """批量写归类结果;rows: [{question_id, labels}]。"""
    async with db.async_session() as s:
        s.add_all([TopicClassification(question_id=r["question_id"], labels=r["labels"])
                   for r in rows])
        await s.commit()
    return len(rows)


async def topic_distribution(samples_per_class: int = 3) -> dict:
    """主题分布:17 类各自问题量 + 每类样例;labels JSON 在 Python 侧聚合(量级小)。

    计数按归类行算;样例按文本去重后再截断——同一句标准化问法常对应池里好几行。
    """
    from app.core.taxonomy import TOPIC_NAMES

    stmt = (
        select(TopicClassification.labels, LowConfidenceQuestion.raw_question,
               ReviewQueue.normalized_question, TopicClassification.classified_at)
        .join(LowConfidenceQuestion,
              TopicClassification.question_id == LowConfidenceQuestion.id)
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
    )
    async with db.async_session() as s:
        rows = (await s.execute(stmt)).all()
    counts = {name: 0 for name in TOPIC_NAMES}
    samples: dict[str, list[str]] = {name: [] for name in TOPIC_NAMES}
    latest = None
    for labels, raw, norm, ts in rows:
        text = norm or raw
        latest = ts if latest is None or ts > latest else latest
        for lb in labels or []:
            if lb in counts:
                counts[lb] += 1
                if len(samples[lb]) < samples_per_class and text not in samples[lb]:
                    samples[lb].append(text)
    return {
        "total": len(rows),
        "latest": latest.isoformat() if latest else None,
        "classes": [{"label": n, "count": counts[n], "samples": samples[n]}
                    for n in TOPIC_NAMES],
    }


async def topic_questions(label: str, page: int = 1, size: int = 20) -> dict:
    """某一类目下的全部问题(分页):labels 是 JSON 数组,筛选与分页在 Python 侧做。

    归类总量是问题池规模(百级),一次全捞换来不依赖 MySQL 的 JSON 函数,
    测试库上结果一致。多标签问题在它命中的每个类目下都出现。
    """
    stmt = (
        select(TopicClassification.labels, TopicClassification.classified_at,
               LowConfidenceQuestion.id, LowConfidenceQuestion.raw_question,
               LowConfidenceQuestion.source, LowConfidenceQuestion.created_at,
               ReviewQueue.normalized_question, ReviewQueue.occurrence_count,
               ReviewQueue.review_status)
        .join(LowConfidenceQuestion,
              TopicClassification.question_id == LowConfidenceQuestion.id)
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
        .order_by(TopicClassification.classified_at.desc(), TopicClassification.id.desc())
    )
    async with db.async_session() as s:
        rows = (await s.execute(stmt)).all()
    matched = [r for r in rows if label in (r.labels or [])]
    total = len(matched)
    size = max(1, min(size, 100))
    pages = max(1, (total + size - 1) // size)
    page = max(1, min(page, pages))
    chunk = matched[(page - 1) * size: page * size]
    items = [{
        "question_id": r.id,
        "text": r.normalized_question or r.raw_question,
        "raw_question": r.raw_question,
        "labels": r.labels or [],
        "source": r.source,
        "occurrence_count": r.occurrence_count,
        "review_status": r.review_status,
        "pooled_at": r.created_at.isoformat() if r.created_at else None,
        "classified_at": r.classified_at.isoformat() if r.classified_at else None,
    } for r in chunk]
    return {"label": label, "total": total, "page": page, "pages": pages,
            "size": size, "items": items}

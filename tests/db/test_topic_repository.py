"""ch10 主题归类仓储:只归已归并问题、文本优先标准化问法、写表幂等、分布聚合去重样例。"""
from app.db import repository
from app.db.models import LowConfidenceQuestion, ReviewQueue


async def _seed_question(session_factory, raw="猫窝买大了想退", norm=None):
    async with session_factory() as s:
        review_id = None
        if norm:
            rq = ReviewQueue(normalized_question=norm)
            s.add(rq)
            await s.flush()
            review_id = rq.id
        q = LowConfidenceQuestion(raw_question=raw, source="retrieval_low_conf",
                                  matched_review_id=review_id)
        s.add(q)
        await s.commit()
        return q.id


async def test_insert_and_exclude_classified(db_session_factory, db_clean):
    qid = await _seed_question(db_session_factory, norm="猫窝买大了能不能退货")
    rows = await repository.list_unclassified_questions()
    assert any(r["question_id"] == qid for r in rows)

    n = await repository.insert_topic_classifications(
        [{"question_id": qid, "labels": ["尺码", "退换货"]}])
    assert n == 1
    rows = await repository.list_unclassified_questions()
    assert not any(r["question_id"] == qid for r in rows)


async def test_unmerged_excluded_from_classify(db_session_factory, db_clean):
    qid = await _seed_question(db_session_factory, raw="猫抓板回收吗")
    rows = await repository.list_unclassified_questions()
    assert not any(r["question_id"] == qid for r in rows)
    # 但训练语料捞取仍能看到它(退回原话)
    pool = await repository.list_pool_texts()
    assert any(r["question_id"] == qid for r in pool)


async def test_text_prefers_normalized(db_session_factory, db_clean):
    qid = await _seed_question(db_session_factory, raw="这个还能退吗",
                               norm="智能猫砂盆能否七天无理由退货")
    rows = await repository.list_pool_texts()
    row = next(r for r in rows if r["question_id"] == qid)
    assert row["text"] == "智能猫砂盆能否七天无理由退货"


async def test_topic_distribution_counts_all_17(db_session_factory, db_clean):
    qid = await _seed_question(db_session_factory)
    await repository.insert_topic_classifications(
        [{"question_id": qid, "labels": ["尺码", "退换货"]}])
    dist = await repository.topic_distribution()
    assert len(dist["classes"]) == 17
    by_label = {c["label"]: c for c in dist["classes"]}
    assert by_label["尺码"]["count"] >= 1 and by_label["退换货"]["count"] >= 1
    assert by_label["尺码"]["samples"]


async def test_distribution_samples_deduped_by_text(db_session_factory, db_clean):
    # 同一句标准化问法对应池里两行:计数按条,样例按文本去重
    q1 = await _seed_question(db_session_factory, raw="猫窝洗了能退吗",
                              norm="猫窝是否支持水洗退货")
    q2 = await _seed_question(db_session_factory, raw="猫窝水洗后退不退",
                              norm="猫窝是否支持水洗退货")
    await repository.insert_topic_classifications([
        {"question_id": q1, "labels": ["退换货"]},
        {"question_id": q2, "labels": ["退换货"]},
    ])
    dist = await repository.topic_distribution(samples_per_class=3)
    by_label = {c["label"]: c for c in dist["classes"]}
    assert by_label["退换货"]["count"] == 2
    assert by_label["退换货"]["samples"] == ["猫窝是否支持水洗退货"]

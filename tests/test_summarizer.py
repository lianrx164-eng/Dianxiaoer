import asyncio

from sqlalchemy.exc import SQLAlchemyError

from app.core import context_budget, summarizer
from app.db import repository as repo


async def _seed(cid: int, turns: int) -> list[int]:
    ids = []
    for i in range(turns):
        ids.append(await repo.append_message(cid, "user", content=f"问题{i} 订单1001"))
        ids.append(await repo.append_message(cid, "assistant", content=f"回答{i}"))
    return ids


async def test_run_summary_appends_segment_and_skips_second_time(
        db_session_factory, db_clean, monkeypatch, caplog):
    cid = await repo.create_conversation("u1")
    ids = await _seed(cid, 12)                         # ids[0..23]
    await repo.update_layer1_from(cid, ids[15])        # 第 8 轮 assistant
    calls = []

    async def fake(old, dialog):
        calls.append((old, dialog))
        return "第一段:订单1001/手机13800138000"
    monkeypatch.setattr(summarizer, "summarize_dialog", fake)

    with caplog.at_level("INFO", logger="app.core.summarizer"):
        await summarizer.run_summary(cid)
    conv = await repo.get_conversation(cid)
    assert conv.summary_upto_msg_id == ids[15]
    assert conv.layer1_from_msg_id == ids[15]
    assert "第一段" in conv.summary
    rows = await repo.list_summaries(cid)
    assert len(rows) == 1 and rows[0].from_msg_id == ids[0] and rows[0].upto_msg_id == ids[15]
    assert "summary done" in caplog.text

    await summarizer.run_summary(cid)                  # 边界没前进 → 跳过
    assert len(await repo.list_summaries(cid)) == 1
    assert len(calls) == 1

    await repo.update_layer1_from(cid, ids[19])        # 层 1 又前移

    async def fake2(old, dialog):
        calls.append((old, dialog))
        return "第二段:补发立柱"
    monkeypatch.setattr(summarizer, "summarize_dialog", fake2)
    await summarizer.run_summary(cid)
    rows = await repo.list_summaries(cid)
    assert [r.seq for r in rows] == [1, 2]
    assert rows[0].content == "第一段:订单1001/手机13800138000"   # 旧段不变
    conv = await repo.get_conversation(cid)
    assert "第一段" in conv.summary and "第二段" in conv.summary
    assert conv.summary_upto_msg_id == ids[19]


async def test_run_summary_failure_or_empty_writes_nothing(
        db_session_factory, db_clean, monkeypatch):
    cid = await repo.create_conversation("u1")
    ids = await _seed(cid, 4)
    await repo.update_layer1_from(cid, ids[3])

    async def boom(old, dialog):
        raise RuntimeError("upstream")
    monkeypatch.setattr(summarizer, "summarize_dialog", boom)
    await summarizer.run_summary(cid)
    conv = await repo.get_conversation(cid)
    assert conv.summary is None and conv.summary_upto_msg_id is None
    assert await repo.list_summaries(cid) == []

    async def empty(old, dialog):
        return "   "
    monkeypatch.setattr(summarizer, "summarize_dialog", empty)
    await summarizer.run_summary(cid)
    assert (await repo.get_conversation(cid)).summary is None
    assert await repo.list_summaries(cid) == []


async def test_maybe_schedule_skips_empty_layer2(db_session_factory, db_clean, monkeypatch):
    cid = await repo.create_conversation("u1")
    await _seed(cid, 2)
    called = []

    async def fake(c):
        called.append(c)
    monkeypatch.setattr(summarizer, "run_summary", fake)
    await summarizer.maybe_schedule_summary(cid)       # layer1_from=0 → 层 2 空
    assert called == [] and summarizer._running.get(cid) is None


async def test_maybe_schedule_skips_below_budget(db_session_factory, db_clean, monkeypatch):
    cid = await repo.create_conversation("u1")
    ids = await _seed(cid, 4)
    await repo.update_layer1_from(cid, ids[5])
    monkeypatch.setattr(summarizer, "compute_budget",
                        lambda: context_budget.ContextBudget(
                            64000, 0, 0, 30000, 20000, 10**9))
    called = []

    async def fake(c):
        called.append(c)
    monkeypatch.setattr(summarizer, "run_summary", fake)
    await summarizer.maybe_schedule_summary(cid)
    assert called == []


async def test_skip_below_budget_wins_over_running(
        db_session_factory, db_clean, monkeypatch, caplog):
    """§8 顺序:空层2/低于预算先判,running 在后;只影响日志。"""
    cid = await repo.create_conversation("u1")
    ids = await _seed(cid, 4)
    await repo.update_layer1_from(cid, ids[5])
    monkeypatch.setattr(summarizer, "compute_budget",
                        lambda: context_budget.ContextBudget(
                            64000, 0, 0, 30000, 20000, 10**9))

    async def noop():
        return None
    task = asyncio.create_task(noop())
    summarizer._running[cid] = task
    try:
        with caplog.at_level("INFO", logger="app.core.summarizer"):
            await summarizer.maybe_schedule_summary(cid)
        assert "reason=below_budget" in caplog.text
        assert "reason=running" not in caplog.text
    finally:
        summarizer._running.pop(cid, None)
        await task


async def test_maybe_schedule_fires_and_debounces(db_session_factory, db_clean, monkeypatch):
    cid = await repo.create_conversation("u1")
    ids = await _seed(cid, 4)
    await repo.update_layer1_from(cid, ids[5])
    monkeypatch.setattr(summarizer, "compute_budget",
                        lambda: context_budget.ContextBudget(
                            64000, 0, 0, 30000, 20000, 1))   # 层 2 预算 1,必触发
    gate = asyncio.Event()
    ran = []

    async def slow(c):
        ran.append(c)
        await gate.wait()
    monkeypatch.setattr(summarizer, "run_summary", slow)

    await summarizer.maybe_schedule_summary(cid)
    await summarizer.maybe_schedule_summary(cid)       # 在跑 → 防抖
    await asyncio.sleep(0)
    assert ran == [cid]
    gate.set()
    await summarizer._running[cid]
    assert cid not in summarizer._running


async def test_maybe_schedule_db_error_does_not_raise(
        db_session_factory, db_clean, monkeypatch, caplog):
    cid = await repo.create_conversation("u1")

    async def boom(cid):
        raise SQLAlchemyError("db down")

    monkeypatch.setattr(summarizer.repository, "get_conversation", boom)
    with caplog.at_level("ERROR", logger="app.core.summarizer"):
        await summarizer.maybe_schedule_summary(cid)   # 不应抛到回复路径
    assert "summary schedule failed" in caplog.text

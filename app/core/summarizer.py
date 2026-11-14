"""ch07 后台分段摘要:层 2 超预算 → 轮后 create_task 压成新段;不阻塞当轮回复。
失败只 log 不重试;新段插入 + 投影重建 + summary_upto 更新在一个事务里。"""
import asyncio
import logging
import time

from pydantic import BaseModel, Field

from app.config import settings
from app.core import memory
from app.core.context_budget import compute_budget, estimate_messages_tokens
from app.core.llm import get_chat_model, structured_model, thinking_body
from app.core.prompts import SUMMARY_PROMPT
from app.db import repository

logger = logging.getLogger(__name__)
_running: dict[int, asyncio.Task] = {}


class _Summary(BaseModel):
    summary: str = Field(description="本次新增对话的分段摘要,只含事实与诉求")


def _layer2_rows(rows, upto: int, boundary: int):
    """层 2 的 DB 行区间:`summary_upto < id ≤ layer1_from`。"""
    return [r for r in rows if upto < r.id <= boundary]


async def summarize_dialog(old_summary: str, dialog: str) -> str:
    """LLM 背景 + 只总结本次新增对话;严格 JSON(structured output)。"""
    model = get_chat_model(extra_body=thinking_body())
    r: _Summary = await structured_model(_Summary, model=model).ainvoke(
        SUMMARY_PROMPT.format_messages(
            old_summary=old_summary or "(无)", dialog=dialog))
    return (r.summary or "").strip()


async def run_summary(conversation_id: int) -> None:
    """任务体:读消息 → 算边界 → 压缩本批 → 原子追加。异常只 log。"""
    t0 = time.monotonic()
    boundary = old_upto = 0
    try:
        conv = await repository.get_conversation(conversation_id)
        if conv is None:
            return
        boundary = conv.layer1_from_msg_id or 0
        old_upto = conv.summary_upto_msg_id or 0
        if boundary <= old_upto:
            logger.info("summary skip conv=%s reason=no_boundary upto=%s layer1_from=%s",
                        conversation_id, old_upto, boundary)
            return
        rows = await repository.list_dialog_messages(conversation_id)
        seg = _layer2_rows(rows, old_upto, boundary)
        if not seg:
            logger.info("summary skip conv=%s reason=empty_batch", conversation_id)
            return
        dialog = "\n".join(
            f"{'用户' if r.role == 'user' else '客服'}:{r.content}"
            for r in seg if r.content)
        logger.info("summary start conv=%s msgs=%s upto %s->%s",
                    conversation_id, len(seg), old_upto, boundary)
        summary = (await summarize_dialog(conv.summary or "", dialog)).strip()
        if not summary:
            raise ValueError("摘要为空,放弃更新")
        seq = await repository.append_summary(
            conversation_id, seg[0].id, boundary, summary)
        if not seq:
            logger.info("summary skip conv=%s reason=boundary_advanced boundary=%s upto=%s",
                        conversation_id, boundary, old_upto)
            return
        logger.info("summary done conv=%s 第%s段 from=%s upto=%s len=%s cost=%.0fms",
                    conversation_id, seq, seg[0].id, boundary, len(summary),
                    (time.monotonic() - t0) * 1000)
    except Exception:  # noqa: BLE001 - 后台任务失败不拖垮主链路
        logger.exception("summary failed conv=%s boundary=%s upto=%s cost=%.0fms",
                         conversation_id, boundary, old_upto,
                         (time.monotonic() - t0) * 1000)


async def maybe_schedule_summary(conversation_id: int) -> None:
    """轮结束后调用:层 2 渲染 token 超预算且无在跑任务 → 后台跑,不 await。
    触发检查自身失败只 log,不把已生成的回复打成失败。"""
    try:
        conv = await repository.get_conversation(conversation_id)
        if conv is None:
            return
        upto = conv.summary_upto_msg_id or 0
        boundary = conv.layer1_from_msg_id or 0
        if boundary <= upto:
            logger.info("summary skip conv=%s reason=layer2_empty upto=%s layer1_from=%s",
                        conversation_id, upto, boundary)
            return
        rows = await repository.list_dialog_messages(conversation_id)
        layer2 = [memory.row_to_message(r) for r in _layer2_rows(rows, upto, boundary)]
        rendered = memory.render_layer2(
            layer2, settings.layer2_assistant_chars, settings.layer2_tool_keep_chars)
        tokens = estimate_messages_tokens(rendered)
        budget = compute_budget()
        if tokens <= budget.layer2:
            logger.info("summary skip conv=%s reason=below_budget tokens=%s budget=%s "
                        "upto=%s layer1_from=%s",
                        conversation_id, tokens, budget.layer2, upto, boundary)
            return
        if conversation_id in _running:
            logger.info("summary skip conv=%s reason=running upto=%s layer1_from=%s",
                        conversation_id, upto, boundary)
            return
        logger.info("summary trigger conv=%s layer2_tokens=%s budget=%s upto=%s layer1_from=%s",
                    conversation_id, tokens, budget.layer2, upto, boundary)
        task = asyncio.create_task(run_summary(conversation_id))
        _running[conversation_id] = task

        def _done(_t):
            _running.pop(conversation_id, None)

        task.add_done_callback(_done)
    except Exception:  # noqa: BLE001 - 触发检查失败不影响已返回的答复
        logger.exception("summary schedule failed conv=%s", conversation_id)

"""ch08 统一执行引擎:所有工具调用的唯一通道。
管道:查工具 → JSON Schema 校验 → 权限门 → 执行(超时/重试)→ 分诊 → 格式化 + 审计。
坏消息如实回灌模型;审计写失败只 log,绝不反拦工具执行。"""
import asyncio
import contextvars
import json
import logging
import time
from dataclasses import dataclass

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from langchain_core.messages import ToolMessage

from app.config import settings
from app.db import repository
from app.tools.registry import ToolSpec

logger = logging.getLogger(__name__)

# 暂时性故障(值得重试):超时、网络抖动。业务异常不在此列——重试不解决问题。
_TRANSIENT = (asyncio.TimeoutError, TimeoutError, ConnectionError, httpx.TransportError)
_SUMMARY_LIMIT = 500


@dataclass
class ToolRun:
    tool_call_id: str
    name: str
    ok: bool
    tool_message: ToolMessage
    status: str               # 成功 / 失败 / 超时 / 校验拦下 / 权限拒绝
    retry_count: int = 0
    duration_ms: int = 0


def validate_args(spec: ToolSpec, args: dict) -> str | None:
    """按 JSON Schema 校验模型给的参数;返回人类可读错误(None=通过)。
    agent_tools 复用它判「create_ticket 参数齐不齐、要不要弹确认卡」。"""
    err = best_match(Draft202012Validator(spec.json_schema).iter_errors(args))
    if err is None:
        return None
    where = f"(字段 {err.json_path})" if err.json_path != "$" else ""
    return f"{err.message}{where}"


def _timeout_of(spec: ToolSpec) -> float:
    if spec.timeout is not None:
        return spec.timeout
    return settings.mcp_tool_timeout if spec.source == "mcp" else settings.tool_default_timeout


def _summarize(content: str) -> str:
    return content if len(content) <= _SUMMARY_LIMIT else content[:_SUMMARY_LIMIT] + "…(截断)"


# ch09:工具自报的内部快照(如 query_faq)由引擎剥走,不透给模型;
# agent_tools 在执行完工具后读走写进 retrieved_snapshot。
LAST_TOOL_SNAPSHOT: contextvars.ContextVar[list] = contextvars.ContextVar(
    "mewhelp_tool_snapshot", default=[])


def _flatten_content(result):
    """adapters 0.3.x 的工具调用返回 MCP content block 列表([{type,text,id}...]),
    先拼成文本再按 JSON 解析;普通 tuple/str/dict 原样返回。"""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list) and all(
            isinstance(b, dict) and "type" in b for b in result):
        texts = [b.get("text", "") for b in result if b.get("type") == "text"]
        result = "\n".join(t for t in texts if t)
    return result


def _format_content(spec: ToolSpec, result) -> str:
    """结果格式化:先归一 MCP content block / tuple / JSON 文本;
    format_result 钩子挑字段/枚举翻人话;统一 ensure_ascii=False 序列化(中文不转义)。"""
    result = _flatten_content(result)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return result                      # 纯文本结果原样回灌
    if spec.format_result is not None and isinstance(result, dict):
        result = spec.format_result(result)
    return json.dumps(result, ensure_ascii=False, default=str)


async def _audit(conversation_id, tool_call_id, name, spec: ToolSpec | None, args,
                 result_summary, status, error_message, retry_count, duration_ms) -> None:
    try:
        await repository.insert_tool_audit(
            conversation_id=conversation_id or None, tool_call_id=tool_call_id or None,
            tool_name=name,
            tool_source=(spec.source if spec else "builtin"),   # 未知工具无来源,按缺省 builtin 落
            mcp_server=(spec.mcp_server if spec else None),
            arguments=args or None, result_summary=result_summary, status=status,
            error_message=error_message, retry_count=retry_count, duration_ms=duration_ms,
        )
    except Exception:  # noqa: BLE001 —— 审计失败不许反拦工具执行
        logger.exception("审计写入失败(不影响工具执行) tool=%s status=%s", name, status)


async def execute_tool_call(
    tool_call: dict, conversation_id: int, specs: dict[str, ToolSpec],
    *, confirmed: bool = False, deny_note: str | None = None,
) -> ToolRun:
    name = tool_call.get("name") or ""
    tc_id = tool_call.get("id") or ""
    args = dict(tool_call.get("args") or {})
    model_args = dict(args)          # 审计只记模型实际发来的参数,不含系统注入字段
    started = time.monotonic()
    spec = specs.get(name)

    async def _finish(content: str, status: str, *, ok: bool,
                      summary: str | None = None, error: str | None = None,
                      retry_count: int = 0) -> ToolRun:
        """收尾只此一处:组 ToolRun、写审计(审计失败不反拦)、返回。"""
        run = ToolRun(
            tool_call_id=tc_id, name=name, ok=ok, status=status,
            retry_count=retry_count,
            duration_ms=int((time.monotonic() - started) * 1000),
            tool_message=ToolMessage(
                content=content, tool_call_id=tc_id, name=name or "unknown",
                status="success" if ok else "error"))
        await _audit(conversation_id, tc_id, name or "unknown", spec, model_args, summary,
                     status, error, retry_count, run.duration_ms)
        return run

    # ① 查工具
    if spec is None:
        return await _finish(f"工具执行失败:未知工具 {name}", "失败",
                             ok=False, error=f"未知工具 {name}")

    # ② JSON Schema 校验:拦下不抛异常,错误说明回灌模型(让它追问用户或重组调用)
    verr = validate_args(spec, args)
    if verr is not None:
        return await _finish(
            f"参数校验未通过:{verr}。请修正参数重新调用;缺少的信息请先向用户追问,不要编造。",
            "校验拦下", ok=False, error=verr)

    # ③ 权限门:写操作必须带确认令牌(interrupt 确认后由 agent_tools 传入),模型绕不过
    if spec.permission == "write" and not confirmed:
        note = deny_note or "该写操作需要用户确认,未确认前拒绝执行。请勿再次发起,除非用户明确要求。"
        return await _finish(f"建工单未执行:{note}", "权限拒绝", ok=False, error=note)

    # ④ 执行:超时 + 暂时性故障重试(写操作/retryable=False 恒不重试)
    if spec.inject_conversation:
        args["conversation_id"] = conversation_id      # 校验后注入(schema 不含注入参数)
    retries = 0 if (spec.permission == "write" or not spec.retryable) else settings.tool_max_retries
    tool_timeout = _timeout_of(spec)
    attempt = 0
    while True:
        try:
            result = await asyncio.wait_for(spec.tool.ainvoke(args), timeout=tool_timeout)
            result = _flatten_content(result)
            if isinstance(result, dict) and "_snapshot" in result:
                LAST_TOOL_SNAPSHOT.set(result.pop("_snapshot") or [])
            content = _format_content(spec, result)
            return await _finish(content, "成功", ok=True,
                                 summary=_summarize(content), retry_count=attempt)
        except _TRANSIENT as e:
            if attempt < retries:
                attempt += 1
                logger.warning("工具暂时性故障,第 %s 次重试 name=%s err=%s",
                               attempt, name, type(e).__name__)
                await asyncio.sleep(0.2 * attempt)
                continue
            is_timeout = isinstance(e, (asyncio.TimeoutError, TimeoutError))
            status = "超时" if is_timeout else "失败"
            # ⑤ 分诊:真故障如实回灌,不装没事
            return await _finish(
                f"工具暂时不可用:{'执行超时' if is_timeout else type(e).__name__},"
                "请稍后再试或如实告知用户。",
                status, ok=False, error=type(e).__name__, retry_count=attempt)
        except Exception as e:  # noqa: BLE001 业务/未知异常:不重试,如实回灌
            logger.exception("工具执行失败 name=%s", name)
            return await _finish(
                f"工具暂时不可用:{type(e).__name__}。请如实告知用户,不要编造结果。",
                "失败", ok=False, error=f"{type(e).__name__}: {e}"[:500],
                retry_count=attempt)

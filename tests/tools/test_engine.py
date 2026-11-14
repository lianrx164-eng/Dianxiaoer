import asyncio

import pytest

from app.tools import engine
from app.tools.registry import ToolSpec

SCHEMA = {"type": "object", "properties": {"order_id": {"type": "string"}},
          "required": ["order_id"]}


def _spec(name="query_order", *, permission="read", source="builtin", tool=None,
          schema=SCHEMA, timeout=None, inject=False, retryable=True, fmt=None):
    return ToolSpec(name=name, description="测试工具", json_schema=schema, tool=tool,
                    permission=permission, source=source, timeout=timeout,
                    inject_conversation=inject, retryable=retryable, format_result=fmt)


def _tool(fn, name="query_order"):
    return type("T", (), {"name": name, "ainvoke": staticmethod(fn)})()


@pytest.fixture()
def audits(monkeypatch):
    rows = []

    async def fake_audit(**kw):
        rows.append(kw)

    monkeypatch.setattr(engine.repository, "insert_tool_audit", fake_audit)
    return rows


async def test_unknown_tool(audits):
    run = await engine.execute_tool_call(
        {"name": "nope", "args": {}, "id": "c1"}, 1, {})
    assert run.ok is False and "未知工具" in run.tool_message.content
    assert audits[-1]["status"] == "失败"
    assert run.tool_message.status == "error"


async def test_validation_blocks_and_feeds_back(audits):
    async def boom(_):
        raise AssertionError("校验不过不许执行")

    spec = _spec(tool=_tool(boom))
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {}, "id": "c1"}, 1, {"query_order": spec})
    assert run.ok is False and run.status == "校验拦下"
    assert "参数校验未通过" in run.tool_message.content
    assert run.tool_message.status == "error"
    assert audits[-1]["status"] == "校验拦下" and audits[-1]["retry_count"] == 0


async def test_write_without_confirmation_denied(audits):
    async def boom(_):
        raise AssertionError("未确认不许执行")

    schema = {"type": "object", "properties": {"description": {"type": "string"}},
              "required": ["description"]}
    spec = _spec("create_ticket", permission="write", tool=_tool(boom, "create_ticket"),
                 schema=schema)
    run = await engine.execute_tool_call(
        {"name": "create_ticket", "args": {"description": "x"}, "id": "c1"},
        1, {"create_ticket": spec})
    assert run.ok is False and run.status == "权限拒绝"
    assert audits[-1]["status"] == "权限拒绝"


async def test_write_confirmed_executes_and_not_retried(audits):
    calls = {"n": 0}

    async def flaky(_):
        calls["n"] += 1
        raise RuntimeError("boom")

    schema = {"type": "object", "properties": {"description": {"type": "string"}},
              "required": ["description"]}
    spec = _spec("create_ticket", permission="write", tool=_tool(flaky, "create_ticket"),
                 schema=schema)
    run = await engine.execute_tool_call(
        {"name": "create_ticket", "args": {"description": "x"}, "id": "c1"},
        1, {"create_ticket": spec}, confirmed=True)
    assert run.ok is False and calls["n"] == 1 and run.retry_count == 0


async def test_transient_timeout_retries_then_gives_up(audits):
    async def slow(_):
        await asyncio.sleep(1)

    spec = _spec(tool=_tool(slow), timeout=0.05)
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
        1, {"query_order": spec})
    assert run.ok is False and run.status == "超时"
    assert run.retry_count == 2
    assert audits[-1]["status"] == "超时" and audits[-1]["retry_count"] == 2
    assert audits[-1]["duration_ms"] is not None


async def test_business_error_not_retried(audits):
    calls = {"n": 0}

    async def fail(_):
        calls["n"] += 1
        raise ValueError("业务错")

    spec = _spec(tool=_tool(fail))
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
        1, {"query_order": spec})
    assert run.ok is False and calls["n"] == 1
    assert "工具暂时不可用" in run.tool_message.content


async def test_retry_succeeds_second_attempt(audits):
    calls = {"n": 0}

    async def flaky(_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("网络抖动")
        return {"order_id": "1", "status": "已发货"}

    spec = _spec(tool=_tool(flaky))
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
        1, {"query_order": spec})
    assert run.ok is True and calls["n"] == 2 and run.retry_count == 1
    assert "已发货" in run.tool_message.content
    assert audits[-1]["status"] == "成功" and audits[-1]["retry_count"] == 1


async def test_retryable_false_never_retries(audits):
    calls = {"n": 0}

    async def slow(_):
        calls["n"] += 1
        await asyncio.sleep(1)

    spec = _spec(tool=_tool(slow), timeout=0.05, retryable=False)
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
        1, {"query_order": spec})
    assert run.ok is False and calls["n"] == 1 and run.retry_count == 0


async def test_format_result_hook_translates_enum(audits):
    async def ok(_):
        return {"tracking_no": "SF1", "status_code": "IN_TRANSIT", "internal_ref": "x9"}

    def fmt(d):
        return {"tracking_no": d["tracking_no"],
                "status": {"IN_TRANSIT": "运输中"}.get(d.get("status_code"), d.get("status_code"))}

    spec = _spec("query_logistics", source="mcp", tool=_tool(ok, "query_logistics"),
                 schema={"type": "object", "properties": {"tracking_no": {"type": "string"}},
                         "required": ["tracking_no"]}, fmt=fmt)
    run = await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"tracking_no": "SF1"}, "id": "c1"},
        1, {"query_logistics": spec})
    assert "运输中" in run.tool_message.content
    assert "internal_ref" not in run.tool_message.content
    assert audits[-1]["tool_source"] == "mcp"


async def test_format_handles_mcp_content_blocks(audits):
    """adapters 0.3.x 返回 [{type:text,text:<json>,id:<随机>}];应解成 dict 并丢掉内部 id。"""
    async def ok(_):
        return [{"type": "text", "id": "lc-random",
                 "text": '{"tracking_no": "SF1", "status_code": "IN_TRANSIT"}'}]

    spec = _spec("query_logistics", source="mcp", tool=_tool(ok, "query_logistics"),
                 schema={"type": "object", "properties": {"tracking_no": {"type": "string"}},
                         "required": ["tracking_no"]})
    run = await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"tracking_no": "SF1"}, "id": "c1"},
        1, {"query_logistics": spec})
    assert run.ok is True
    assert "IN_TRANSIT" in run.tool_message.content
    assert "lc-random" not in run.tool_message.content


async def test_audit_failure_never_blocks_execution(monkeypatch):
    async def audit_boom(**kw):
        raise RuntimeError("审计库挂了")

    monkeypatch.setattr(engine.repository, "insert_tool_audit", audit_boom)

    async def ok(_):
        return {"order_id": "1"}

    spec = _spec(tool=_tool(ok))
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
        1, {"query_order": spec})
    assert run.ok is True


async def test_inject_conversation_after_validation(audits):
    seen = {}

    async def ok(args):
        seen.update(args)
        return {"ticket_no": "T1"}

    schema = {"type": "object", "properties": {"description": {"type": "string"}},
              "required": ["description"], "additionalProperties": False}
    spec = _spec("create_ticket", permission="write", tool=_tool(ok, "create_ticket"),
                 schema=schema, inject=True)
    run = await engine.execute_tool_call(
        {"name": "create_ticket", "args": {"description": "x"}, "id": "c1"},
        7, {"create_ticket": spec}, confirmed=True)
    assert run.ok is True and seen.get("conversation_id") == 7

"""ch08 验收样例端到端(需 make dev + make mcp-up;BASE_URL 可覆盖端口)。
用法:BASE_URL=http://localhost:8002 PYTHONUTF8=1 uv run python scripts/eval_ch08.py
"""
import asyncio
import json
import os
import pathlib
import re
import sys

import httpx
from sqlalchemy import text

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.environ.get("BASE_URL", "http://localhost:8000")
USER = "eval-ch08"

_TICKET_RE = re.compile(r"T\d{10,}")


async def agent(c, msg, cid=None):
    r = await c.post(f"{BASE}/api/agent",
                     json={"user_id": USER, "message": msg, "conversation_id": cid})
    r.raise_for_status()
    return r.json()


async def resume_stream(c, cid, payload):
    """POST /api/actions/resume,收集 delta / interrupt / done / error。"""
    out = {"delta": "", "interrupt": None, "done": None, "finished": False, "error": None}
    async with c.stream("POST", f"{BASE}/api/actions/resume",
                        json={"conversation_id": cid, **payload}) as r:
        event = None
        async for line in r.aiter_lines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].lstrip()
                if data == "[DONE]":
                    out["finished"] = True
                    event = None
                    continue
                ev = json.loads(data)
                if event == "error":
                    out["error"] = ev.get("message")
                elif ev.get("event") == "interrupt":
                    out["interrupt"] = ev
                elif ev.get("event") == "done":
                    out["done"] = ev
                elif "delta" in ev:
                    out["delta"] += ev["delta"]
                event = None
    return out


async def _ticket_count(cid):
    from app.db import base

    async with base.async_session() as s:
        return int((await s.execute(
            text("SELECT COUNT(*) FROM tickets WHERE conversation_id = :cid"),
            {"cid": cid})).scalar_one())


async def _latest_audit(tool_name):
    from app.db import base

    async with base.async_session() as s:
        row = (await s.execute(
            text("SELECT status, retry_count, duration_ms, tool_source, mcp_server "
                 "FROM tool_audit_logs WHERE tool_name = :n ORDER BY id DESC LIMIT 1"),
            {"n": tool_name})).first()
    return row


async def main():
    results = []

    def check(name, ok, detail=None):
        results.append((name, bool(ok), detail or {}))

    async with httpx.AsyncClient(timeout=240, trust_env=False) as c:
        # 验收 4:缺信息追问 → 补充 → 预览 → 确认落表 + 工单号
        r1 = await agent(c, "帮我建个工单")
        cid = r1["conversation_id"]
        check("4a 缺描述先追问,不弹卡不建单",
              not r1.get("interrupt") and await _ticket_count(cid) == 0,
              {"answer": r1.get("answer", "")[:40]})

        r2 = await agent(c, "猫砂盆漏电", cid)
        intr = r2.get("interrupt") or {}
        preview = intr.get("preview") or {}
        check("4b 补充后弹工单预览(粘滞路由生效)",
              intr.get("type") == "confirm_ticket"
              and preview.get("description") and preview.get("ticket_type"),
              {"interrupt": intr})

        done = await resume_stream(c, cid, {"confirmed": True})
        ticket_no = _TICKET_RE.search(done["delta"] or "")
        check("4c 确认提交落 tickets 且回复带工单号",
              done["error"] is None and await _ticket_count(cid) == 1 and bool(ticket_no),
              {"delta": (done["delta"] or "")[:80],
               "tickets": await _ticket_count(cid)})

        # 验收 5:取消 → 不建单 + 审计「权限拒绝」
        r3 = await agent(c, "帮我建个工单")
        cid2 = r3["conversation_id"]
        await agent(c, "猫砂盆漏电", cid2)
        await resume_stream(c, cid2, {"confirmed": False})
        audit = await _latest_audit("create_ticket")
        check("5 取消不建单,审计权限拒绝",
              await _ticket_count(cid2) == 0 and audit and audit[0] == "权限拒绝",
              {"tickets": await _ticket_count(cid2),
               "audit": tuple(audit) if audit else None})

        # 验收 2:物流经 MCP(先查订单拿单号,再查轨迹)
        r4 = await agent(c, "订单1001的物流到哪了")
        names = {tc["name"] for tc in r4.get("tool_calls", [])}
        audit = await _latest_audit("query_logistics")
        check("2 物流走 MCP 并答出中文状态",
              "query_logistics" in names and audit and audit[0] == "成功"
              and audit[3] == "mcp" and bool(r4.get("answer")),
              {"tools": sorted(names), "audit": tuple(audit) if audit else None,
               "answer": r4.get("answer", "")[:60]})

        # 验收 1(演示物料已放入时):新工具只靠丢文件即注册
        if pathlib.Path("app/tools/builtin/promotions.py").exists():
            r5 = await agent(c, "最近有什么优惠活动")
            names = {tc["name"] for tc in r5.get("tool_calls", [])}
            check("1 promotions 丢文件即注册并被调用",
                  "query_promotions" in names, {"tools": sorted(names)})
        else:
            results.append(("1 promotions 丢文件即注册(未放演示文件,SKIP)", None, {}))

    passed = sum(1 for _, ok, _ in results if ok)
    failed = [r for r in results if r[1] is False]
    for name, ok, detail in results:
        mark = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print(f"{mark} {name} {detail}")
    print(f"\n通过 {passed}/{sum(1 for _, ok, _ in results if ok is not None)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

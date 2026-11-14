"""ch06 四验收端到端评估。需服务已起(BASE_URL 可覆盖端口)+ 已应用 sql/ch06-ticket-type.sql。
用法:BASE_URL=http://localhost:8002 PYTHONUTF8=1 uv run python -m scripts.eval_ch06
"""
import asyncio
import json
import os
import sys

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.environ.get("BASE_URL", "http://localhost:8000")
USER = "eval-ch06"


async def agent(c, msg, cid=None):
    r = await c.post(f"{BASE}/api/agent",
                     json={"user_id": USER, "message": msg, "conversation_id": cid})
    r.raise_for_status()
    return r.json()


async def resume_stream(c, cid, order_id):
    """POST /api/actions/resume,收集 delta/actions/done/[DONE]。"""
    out = {"delta": "", "actions": [], "done": None, "finished": False, "error": None}
    async with c.stream("POST", f"{BASE}/api/actions/resume",
                        json={"conversation_id": cid, "order_id": order_id}) as r:
        event = None
        async for line in r.aiter_lines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].lstrip()
                if data == "[DONE]":
                    out["finished"] = True
                    continue
                ev = json.loads(data)
                if event == "error":
                    out["error"] = ev.get("message")
                elif ev.get("event") == "actions":
                    out["actions"] = ev["items"]
                elif ev.get("event") == "done":
                    out["done"] = ev
                elif "delta" in ev:
                    out["delta"] += ev["delta"]
                event = None
    return out


async def main():
    from app.core.prompts import SCRIPT_REPLY_OTHER

    results = []
    async with httpx.AsyncClient(timeout=240, trust_env=False) as c:
        # 验收1:多轮 物流→退款→物流,每轮意图/指代随上下文
        r1 = await agent(c, "订单1001的物流到哪了")
        cid = r1["conversation_id"]
        r2 = await agent(c, "那它能退吗", cid)
        r3 = await agent(c, "算了,它现在到哪了", cid)
        routes = [r.get("trace", {}).get("route") for r in (r1, r2, r3)]
        results.append((
            "验收1 多轮意图漂移 物流→退款→物流",
            routes == ["business", "refund_flow", "business"],
            {"routes": routes,
             "coref": [r.get("trace", {}).get("coref") for r in (r1, r2, r3)],
             "resolved": [r.get("trace", {}).get("resolved_query") for r in (r1, r2, r3)]}))

        # 验收2:怪问题落「其他」→ fallback_script 固定话术
        b = await agent(c, "帮我写一首关于宇宙的诗")
        tr = b.get("trace", {})
        results.append((
            "验收2 怪问题落其他 + 固定话术",
            tr.get("route") == "fallback_script" and b.get("answer") == SCRIPT_REPLY_OTHER,
            {"intent": tr.get("intent"), "route": tr.get("route"),
             "answer": b.get("answer", "")[:30]}))

        # 验收3:带订单号退款,先补指代再走退款子流程
        b = await agent(c, "订单2002这个能退吗")
        tr = b.get("trace", {})
        acts = {a["type"] for a in b.get("suggested_actions", [])}
        ok3 = (tr.get("route") == "refund_flow" and "retrieve_policy" in tr
               and "2002" in (tr.get("resolved_query") or "") and bool(b.get("answer")))
        results.append((
            "验收3 退款子流程(补指代→取单→检索政策→判定)",
            ok3,
            {"route": tr.get("route"),
             "resolved_query": tr.get("resolved_query"),
             "retrieve_policy": tr.get("retrieve_policy"),
             "actions": sorted(acts), "answer": b.get("answer", "")[:40]}))

        # 验收4 非流式:缺单 → interrupt
        b = await agent(c, "我要退款")
        cid4 = b["conversation_id"]
        orders = b.get("interrupt") or []
        results.append(("验收4a 缺单弹订单选择器(interrupt)", bool(orders), orders))

        # 验收4 流式续跑:点第一单 → resume SSE
        if orders:
            sse = await resume_stream(c, cid4, orders[0]["order_id"])
        else:
            sse = {"error": "no orders"}
        ok4 = (not sse.get("error")
               and (bool(sse.get("delta")) or bool(sse.get("actions")))
               and bool(sse.get("done")) and sse.get("finished"))
        results.append((
            "验收4b 点选后续流(政策→判定→退款按钮/说明)", ok4,
            {"delta": sse.get("delta", "")[:50],
             "actions": [a["type"] for a in sse.get("actions", [])],
             "done": sse.get("done"), "finished": sse.get("finished"),
             "error": sse.get("error")}))

    for name, ok, detail in results:
        print(f"{'✅' if ok else '❌'} {name} -> {detail}")
    print("\n验收2 的 JSON 稳定性另见 `make eval-intent`;验收4 前端卡片点击见浏览器验收。")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)

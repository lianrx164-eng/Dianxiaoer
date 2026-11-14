"""ch05 五验收端到端评估。需服务已起(BASE_URL 可覆盖端口)。
用法:BASE_URL=http://localhost:8002 uv run python -m scripts.eval_ch05
"""
import asyncio
import os
import sys

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.environ.get("BASE_URL", "http://localhost:8000")


async def agent(client, msg):
    r = await client.post(f"{BASE}/api/agent",
                          json={"user_id": "eval-ch05", "message": msg})
    r.raise_for_status()
    return r.json()


async def main():
    results = []
    async with httpx.AsyncClient(timeout=180, trust_env=False) as c:
        # 验收1:政策类走知识路,forced_rag 留痕(trace 由 /api/agent 暴露)
        b = await agent(c, "退货政策是什么")
        tr = b.get("trace", {})
        results.append(("验收1 强制检索(trace)",
                        b.get("route") == "knowledge" and tr.get("forced_rag") is True,
                        {"route": b.get("route"), "forced_rag": tr.get("forced_rag")}))

        # 验收2:业务数据类,Agent 自调 query_logistics
        b = await agent(c, "订单1001的物流到哪了")
        names = [tc["name"] for tc in b["tool_calls"]]
        results.append(("验收2 物流自调工具", "query_logistics" in names, names))

        # 验收3:投诉出两可选项(转人工/建工单),后端不自动建单
        b = await agent(c, "我要投诉,你们太差了")
        types = {a["type"] for a in b.get("suggested_actions", [])}
        results.append(("验收3 投诉出两选项",
                        {"transfer_human", "create_ticket"} <= types, sorted(types)))

        # 验收4:闲聊固定话术,零工具
        b = await agent(c, "你好呀")
        results.append(("验收4 闲聊固定话术",
                        not b["tool_calls"] and b["answer"].startswith("你好呀"),
                        b["answer"][:18]))

        # 验收5:复合问 ReAct 多步,且先查订单再查物流;trace 步数 > 1
        b = await agent(c, "订单1001买的是什么?顺便看看物流到哪了")
        names = [tc["name"] for tc in b["tool_calls"]]
        ordered = ("query_order" in names and "query_logistics" in names
                   and names.index("query_order") < names.index("query_logistics"))
        results.append(("验收5 ReAct 多步(订单→物流)",
                        ordered and b.get("steps", 0) > 1,
                        {"tools": names, "steps": b.get("steps")}))

    for name, ok, detail in results:
        print(f"{'✅' if ok else '❌'} {name} -> {detail}")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)

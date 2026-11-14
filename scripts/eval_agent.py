"""标注样例评估:核对新图骨架下的路由/工具/动作。需服务运行中(BASE_URL 可覆盖端口)。

知识类(ch05 起)由 forced_rag 预检索,Agent 不再自己调 query_faq;投诉走 complaint_reply,
只产出可选项、不建单。
"""
import asyncio
import os
import re
import sys

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # GBK 控制台下模型回复含 emoji 时不崩

BASE = os.environ.get("BASE_URL", "http://localhost:8000")


def _names(body):
    return {tc["name"] for tc in body.get("tool_calls", [])}


def _cites(body):
    return bool(re.search(r"[\[【]\d{1,2}[\]】]", body.get("answer", "")))


def _actions(body):
    return {a["type"] for a in body.get("suggested_actions", [])}


# (用户问法, 判定函数, 期望说明)
SAMPLES = [
    ("订单 1001 的物流到哪了",
     lambda b: "query_logistics" in _names(b), "query_logistics"),
    ("退货政策是什么",
     lambda b: not _names(b) and _cites(b), "预检索 + 引用角标"),
    ("邮费是多少",
     lambda b: not _names(b) and _cites(b), "预检索 + 引用角标"),
    ("iPhone 还有货吗",
     lambda b: not _names(b) and bool(b.get("answer")), "库外商品:强制 RAG 证据弱 → 拒答,不调工具"),
    ("订单 2002 多少钱",
     lambda b: "query_order" in _names(b), "query_order"),
    ("我要投诉,给我登记一下",
     lambda b: {"transfer_human", "create_ticket"} <= _actions(b), "两个可选项,不建单"),
    ("今天天气怎么样",
     lambda b: not _names(b) and bool(b.get("answer")), "不调工具"),
]


async def main() -> None:
    passed = 0
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        # trust_env=False:只连本机服务,避免被 Windows 系统代理(如 Clash)劫持成 502
        for msg, check, expect in SAMPLES:
            r = await client.post(f"{BASE}/api/agent", json={"user_id": "eval", "message": msg})
            if r.status_code != 200:
                print(f"❌ {msg!r} -> HTTP {r.status_code} {r.text[:80]}")
                continue
            body = r.json()
            try:
                ok = bool(check(body))
            except Exception:  # noqa: BLE001 - 判定函数不该让评估崩
                ok = False
            passed += ok
            detail = _names(body) or (_actions(body) or "(未调用/未出选项)")
            print(f"{'✅' if ok else '❌'} {msg!r} -> {detail} 期望={expect}")
            print(f"    answer: {body['answer'][:60]}")
    print(f"\n通过 {passed}/{len(SAMPLES)}")


if __name__ == "__main__":
    asyncio.run(main())

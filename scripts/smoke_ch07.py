"""ch07 端到端冒烟:
  default: 默认窗口 22 轮短消息,断言不触发层 1 降级/摘要(装得下就不压);
  demo:    演示配置 25 轮长消息,断言层 1 降级 → 层 2 超预算 → 摘要落库。

用法:
  PYTHONUTF8=1 uv run python -m scripts.smoke_ch07 default
  MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 \
  MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5 \
  PYTHONUTF8=1 uv run python -m scripts.smoke_ch07 demo
"""
import asyncio
import logging
import sys
import uuid

import app.main  # noqa: F401  配置 app logger(含 log/app.log FileHandler)

from app.config import settings
from app.core import summarizer
from app.db import repository
from app.graph import runtime
from app.graph.nodes import resolve_answer

DEFAULT_QUESTIONS = [
    "订单1001的物流到哪了",
    "订单1001发货了吗",
    "订单2002现在什么状态",
    "订单2002是什么时候下的单",
    "订单1001买的是什么",
    "订单1001的快递单号是多少",
    "订单2002的金额是多少",
    "订单1001到哪个分拨中心了",
]

DEMO_QUESTIONS = [
    "我上周在你们这买的猫爬架订单1001到现在还没到,我手机13800138000,"
    "麻烦帮我查一下物流到底到哪了,到了让快递员提前打电话给我。另外我想确认一下,"
    "如果实在太慢我能不能拒收后退款,退货运费谁承担,我下单时用了优惠券退款金额怎么算,"
    "请一条一条说清楚。",
    "还是订单1001这个猫爬架,我刚收到货发现少了一根立柱,包装也破了。"
    "我想申请补发立柱,如果补发要等太久就整单退货,退货原因算质量问题还是发错货,"
    "运费和退款时效怎么算?我的手机还是13800138000,请帮我登记。",
    "订单1001这个猫爬架我决定退了,原因是质量问题:立柱缺失、包装破损。"
    "我想确认退款金额怎么算、下单时的优惠券和满减怎么处理、退货运费谁承担、"
    "退款原路退回大概多久,以及我是先寄回还是你们上门取件?请一条一条说明。",
    "订单2002的猫粮我买错了口味,还没拆封,想退货。订单2002是上周签收的,"
    "我想知道7天无理由还来不来得及,退货运费谁出,退款是退到原支付方式吗,"
    "需要我提供什么材料?如果超过7天还有没有别的办法?",
    "我手机13800138000,之前留过这个号码。麻烦把我名下最近的订单都看一下:"
    "订单1001猫爬架、订单2002猫粮,分别是什么状态、多少钱、什么时候下的单?"
    "如果我要对其中一单申请退款,流程是什么,需要先联系人工吗?",
]


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


async def _run_rounds(user: str, questions: list[str], rounds: int) -> int:
    cid = None
    for i in range(rounds):
        out = await runtime.run_turn(user, questions[i % len(questions)], cid)
        cid = out["conversation_id"]
        if out.get("interrupt"):
            orders = out["interrupt"] or []
            if orders:
                await runtime.resume_turn(cid, orders[0]["order_id"])
        if (i + 1) % 5 == 0:
            print(f"  round {i + 1}/{rounds} done (conv={cid})", flush=True)
    return cid


async def _drain_summaries() -> None:
    tasks = list(summarizer._running.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def main(mode: str) -> int:
    if mode not in ("default", "demo"):
        print("usage: python -m scripts.smoke_ch07 [default|demo]", file=sys.stderr)
        return 2
    if mode == "demo" and settings.model_context_window != 18000:
        print(f"WARN: demo 模式需注入演示配置;当前 MODEL_CONTEXT_WINDOW="
              f"{settings.model_context_window}", flush=True)

    await runtime.init_graph()
    capture = Capture()
    app_logger = logging.getLogger("app")
    app_logger.addHandler(capture)
    user = f"smoke-ch07-{mode}-{uuid.uuid4().hex[:8]}"
    ok = True
    try:
        if mode == "default":
            print(f"[default] user={user} window={settings.model_context_window}",
                  flush=True)
            cid = await _run_rounds(user, DEFAULT_QUESTIONS, 22)
            await _drain_summaries()
            degraded = [l for l in capture.lines if "layer1 降级" in l]
            triggers = [l for l in capture.lines if "summary trigger" in l]
            print(f"  conv={cid} degraded={len(degraded)} triggers={len(triggers)}",
                  flush=True)
            if degraded or triggers:
                print("  ❌ 默认窗口不应触发降级/摘要", flush=True)
                ok = False
            else:
                print("  ✅ 默认窗口 22 轮未压缩", flush=True)
        else:
            print(f"[demo] user={user} window={settings.model_context_window} "
                  f"steps={settings.max_agent_steps} top_k={settings.rerank_top_k}",
                  flush=True)
            cid = await _run_rounds(user, DEMO_QUESTIONS, 30)
            await _drain_summaries()
            degraded = [l for l in capture.lines if "layer1 降级" in l]
            triggers = [l for l in capture.lines if "summary trigger" in l]
            dones = [l for l in capture.lines if "summary done" in l]
            conv = await repository.get_conversation(cid)
            segs = await repository.list_summaries(cid)
            summary = (conv.summary or "") if conv else ""
            print(f"  conv={cid} degraded={len(degraded)} triggers={len(triggers)} "
                  f"done={len(dones)} segments={len(segs)}", flush=True)
            print(f"  summary={summary[:120]!r}", flush=True)
            if not degraded:
                print("  ❌ 未出现层 1 降级", flush=True)
                ok = False
            if not triggers or not dones:
                print("  ❌ 未出现摘要触发/完成", flush=True)
                ok = False
            if "1001" not in summary:
                print("  ❌ 摘要未保留早期订单号 1001", flush=True)
                ok = False

            out = await runtime.run_turn(user, "最开始那个订单后来怎么说", cid)
            answer = resolve_answer(out["state"])
            print(f"  late answer: {answer[:200]!r}", flush=True)
            if "1001" not in (answer or ""):
                print("  (软断言)晚问答复未含 1001,以摘要库为准", flush=True)
    finally:
        app_logger.removeHandler(capture)
        await runtime.close_graph()
    if ok:
        print(f"\n✅ smoke {mode} 通过", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "default")))

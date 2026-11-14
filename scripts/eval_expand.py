"""Query 扩写标注评估:核心场景出 3 条 + JSON 稳定 + 保持关键实体。需上游可用。
用法:PYTHONUTF8=1 uv run python -m scripts.eval_expand
"""
import asyncio

from app.core.query_understanding import expand_queries

CASES = [
    "蓝牙耳机还能申请退货吗",
    "订单1001的猫粮质量问题能退款吗",
    "智能猫砂盆坏了怎么保修",
]


async def main():
    passed = 0
    for q in CASES:
        qs = await expand_queries(q)
        ok = len(qs) == 3 and len(set(qs)) == 3
        passed += ok
        print(f"{'✅' if ok else '⚠️'} {q!r} -> {qs}(条数={len(qs)})")
    print(f"\n通过 {passed}/{len(CASES)};核心场景应出 3 条、彼此侧重点不同、含原问题关键实体;"
          f"JSON 不稳时条数会 <3。抖动如实重跑记录")


if __name__ == "__main__":
    asyncio.run(main())

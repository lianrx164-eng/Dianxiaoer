"""裁判回归:拿台账里已处置、带证据快照的个案原样重放,量裁判与人工的一致率。

运行:make judge-check  或  PYTHONPATH=. uv run python scripts/judge_check.py
退出码:有不一致 → 1;全部一致(调用失败不计入) → 0。
"""
import asyncio
import sys

from app.core.judges import (
    expected_faithful, judge_faithfulness, rebuild_evidence, summarize_check,
)
from app.db import repository


async def _judge_once(evidence: str, answer: str):
    return await asyncio.wait_for(judge_faithfulness(evidence, answer), 60)


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cases = await repository.list_disposed_faith_cases()
    if not cases:
        print("台账里没有已处置且带证据快照的个案,先跑评估并处置几条。")
        return

    rows = []
    for c in cases:
        expected = expected_faithful(c.status)
        verdict, reason, error = None, "", None
        for attempt in range(2):
            try:
                verdict, reason = await _judge_once(
                    rebuild_evidence(c.citations), c.answer)
                error = None
                break
            except Exception as e:  # noqa: BLE001 - 空 completion 等偶发失败重试一次
                error = f"{type(e).__name__}: {str(e)[:100]}"
                if attempt == 0:
                    await asyncio.sleep(0.5)
        rows.append({"id": c.eval_id, "status": c.status, "expected": expected,
                     "verdict": verdict, "reason": reason, "error": error})
        mark = "FAIL" if error else ("OK" if verdict == expected else "MISMATCH")
        print(f"{mark:8s} {c.eval_id:5s} 处置={c.status} 应判忠实={expected} "
              f"裁判={verdict} {('错误:' + error) if error else reason}")

    summary = summarize_check(rows)
    print(f"\n一致 {summary['agree']}/{summary['judged']} "
          f"(一致率 {summary['agreement']}),不一致 {summary['disagree']},"
          f"调用失败 {summary['failed']}")
    if summary["disagree"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

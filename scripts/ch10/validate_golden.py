"""ch10 预标质量闸:黄金样例上标签集合完全一致率 ≥ 80% 才放行批量预标。

运行:make ch10-golden(需上游可用)。不过线就改 prompt,不改黄金样例来凑分。
"""
import asyncio
import json
import pathlib
import sys

from scripts.ch10.prelabel import prelabel_batch

GOLDEN = pathlib.Path(__file__).parent / "golden_samples.jsonl"
PASS_RATE = 0.8


async def main() -> int:
    samples = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines()
               if l.strip()]
    predicted = await prelabel_batch([s["text"] for s in samples])
    hits = 0
    failures = []
    for s, pred in zip(samples, predicted):
        ok = set(pred) == set(s["labels"])
        hits += ok
        if not ok:
            failures.append({"text": s["text"], "gold": s["labels"], "pred": pred})
            print(f"✗ {s['text']}\n    标准: {s['labels']}  预标: {pred}")
    rate = hits / len(samples)
    report = {
        "ran_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "total": len(samples), "hits": hits, "rate": round(rate, 4),
        "pass_line": PASS_RATE, "passed": rate >= PASS_RATE, "failures": failures,
    }
    reports = pathlib.Path("data/ch10/reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "golden_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n黄金样例 {len(samples)} 条,集合全对 {hits} 条,通过率 {rate:.0%}(闸线 {PASS_RATE:.0%})")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

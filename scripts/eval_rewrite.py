"""Query 改写标注样例验证(纯 Prompt 任务的验收方式)。

运行:make eval-rewrite  或  PYTHONPATH=. uv run python scripts/eval_rewrite.py
判定:标准问法或扩展词里至少出现一个 expect_any 词;≥4/5 通过。
"""
import asyncio
import json
import sys
from pathlib import Path

from app.core.query_understanding import understand

SAMPLES = Path(__file__).resolve().parent.parent / "tests" / "data" / "query_rewrite_samples.jsonl"


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ok = total = 0
    for line in SAMPLES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        total += 1
        result = await understand(sample["query"])
        blob = result["standard"] + " " + " ".join(result["expanded"])
        hit = any(word in blob for word in sample["expect_any"])
        ok += hit
        print(("OK  " if hit else "MISS") + f" {sample['query']} -> {result}")
    print(f"{ok}/{total}")
    if ok < 4:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

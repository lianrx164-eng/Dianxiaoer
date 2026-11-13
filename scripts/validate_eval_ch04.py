"""评估集自检:题数/去重/章节可命中/要点逐字在库。

运行:make eval-check  或  PYTHONPATH=. uv run python scripts/validate_eval_ch04.py
"""
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import select

import app.db.base as db
from app.db.models import KnowledgeChunk

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET = ROOT / "tests" / "data" / "eval_ch04.jsonl"
BUCKETS = {"A_policy": 60, "B_model": 60, "C_colloquial": 60, "D_absent": 60, "E_multi": 60}


def _norm(s: str) -> str:
    return "".join((s or "").split()).lower()


async def _chunks() -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        return list((await s.execute(select(KnowledgeChunk))).scalars())


def _check(samples: list[dict], chunks: list[KnowledgeChunk]) -> list[str]:
    errors: list[str] = []
    ids = Counter(s["id"] for s in samples)
    queries = Counter(s["query"] for s in samples)
    for k, n in ids.items():
        if n > 1:
            errors.append(f"id 重复:{k} ×{n}")
    for k, n in queries.items():
        if n > 1:
            errors.append(f"query 重复:{k} ×{n}")
    counts = Counter(s["bucket"] for s in samples)
    if counts != Counter(BUCKETS):
        errors.append(f"桶题数不符:{dict(counts)} != {BUCKETS}")

    by_path = defaultdict(list)
    for c in chunks:
        by_path[c.section_path or ""].append(c)

    for s in samples:
        if s["bucket"] == "D_absent":
            if s["expect_sections_all"] or s["expect_points"]:
                errors.append(f"{s['id']}: D 桶不应带 ground truth")
            continue
        if not s["expect_points"]:
            errors.append(f"{s['id']}: 缺少 expect_points")
        for group in s["expect_sections_all"]:
            hit_rows = [c for path, rows in by_path.items()
                        if any(_norm(a) in _norm(path) for a in group) for c in rows]
            if not hit_rows:
                errors.append(f"{s['id']}: 章节不可命中 {group}")
                continue
            blob = _norm("\n".join(c.answer for c in hit_rows))
            for p in s["expect_points"]:
                if _norm(p) not in blob:
                    # E 桶要点可能落在另一组章节,最后统一按全库再查一次
                    all_blob = _norm("\n".join(c.answer for c in chunks))
                    if s["bucket"] != "E_multi" or _norm(p) not in all_blob:
                        errors.append(f"{s['id']}: 要点不在目标章节 {p!r}")
    return errors


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    samples = [json.loads(l) for l in EVAL_SET.read_text(encoding="utf-8").splitlines()
               if l.strip()]
    chunks = await _chunks()
    errors = _check(samples, chunks)
    if errors:
        print(f"❌ {len(errors)} 个问题:")
        for e in errors[:80]:
            print("  -", e)
        raise SystemExit(1)
    print(f"✅ {len(samples)} 题、{len(chunks)} 块:题数/去重/章节/要点全部通过")


if __name__ == "__main__":
    asyncio.run(main())

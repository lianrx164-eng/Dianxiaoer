"""CLI:把 data/kb/*.md 切块写入 knowledge_chunks(pending)。之后跑 vectorize_kb.py。
运行:PYTHONPATH=. uv run python scripts/build_kb.py"""
import asyncio
import sys

from app.db import repository
from app.kb import dedup, documents, dualwrite, sources


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    total = skipped = 0
    for fname, ctype in sources.SOURCE_TYPES.items():
        md = (sources.KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        existing = await repository.list_chunk_pairs()
        fresh = dedup.filter_new_chunks(chunks, existing)
        skipped += len(chunks) - len(fresh)
        ids = await dualwrite.write_pending(fresh) if fresh else []
        total += len(ids)
        print(f"  {fname}: 新增 {len(ids)} 块,跳过重复 {len(chunks) - len(ids)} 块")
    print(f"✅ 建库(pending):新增 {total} 块,跳过重复 {skipped} 块。下一步:scripts/vectorize_kb.py")


if __name__ == "__main__":
    asyncio.run(main())

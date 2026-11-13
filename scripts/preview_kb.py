"""dry-run:按建库材料清单切块并打印各文件结果,不写库。
运行:PYTHONPATH=. uv run python scripts/preview_kb.py"""
import sys

from app.kb import chunking, documents, sources


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    total = 0
    for fname, ctype in sources.SOURCE_TYPES.items():
        md = (sources.KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, ctype)
        tables = sum(1 for c in chunks if chunking.is_table_block(c.answer))
        keys = sum(c.is_key_clause for c in chunks)
        total += len(chunks)
        print(f"{fname} [{ctype}]: {len(chunks)} 块,表格块 {tables},关键条款 {keys}")
    print(f"✅ 预览完成:共 {total} 块(dry-run,未写库)")


if __name__ == "__main__":
    main()

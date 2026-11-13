"""清空知识库:TRUNCATE knowledge_chunks + 删除 Milvus 集合(重建用)。
运行:PYTHONPATH=. uv run python scripts/reset_kb.py"""
import asyncio
import sys

from sqlalchemy import text

from app.db.base import engine
from app.kb import milvus_client


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = milvus_client.get_client()
    if client.has_collection(milvus_client.COLLECTION):
        client.drop_collection(milvus_client.COLLECTION)
    async with engine.begin() as conn:
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        await conn.execute(text("TRUNCATE TABLE knowledge_chunks"))
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
    await engine.dispose()
    print("✅ 已清空 knowledge_chunks 与 Milvus 集合")


if __name__ == "__main__":
    asyncio.run(main())

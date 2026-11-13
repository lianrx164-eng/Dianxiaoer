"""把 sql/ch02-seed.sql 灌入 DATABASE_URL 指向的库。幂等,可重复执行。"""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.db.base import engine
from app.db.sql_script import split_statements

SEED = pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch02-seed.sql"


def _statements() -> list[str]:
    return split_statements(SEED.read_text(encoding="utf-8"))


async def main() -> None:
    async with engine.begin() as conn:
        for s in _statements():
            await conn.execute(text(s))
    await engine.dispose()
    print(f"seeded: {len(_statements())} statement(s) from {SEED.name}")


if __name__ == "__main__":
    asyncio.run(main())

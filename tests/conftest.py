"""测试不依赖仓库根的 .env:给缺省的 dummy 值,真实值只在运行时需要。"""

import os
import pathlib

os.environ.setdefault("CHAT_MODEL", "test-model")
os.environ.setdefault("CHAT_BASE_URL", "http://localhost:9/v1")
os.environ.setdefault("CHAT_API_KEY", "test-key")
os.environ.setdefault("EMBED_API_KEY", "test-key")

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.db.sql_script import split_statements

_DDL = pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch02-ddl.sql"
_TABLES = ["messages", "tickets", "conversations", "faq"]  # 删除顺序:先子表后父表


def _ddl_statements() -> list[str]:
    return split_statements(_DDL.read_text(encoding="utf-8"))


@pytest_asyncio.fixture(scope="session")
async def _test_engine():
    # 库名从 TEST_DATABASE_URL 解析,并强制含 test:fixture 会 DROP 该库,配错代价太大
    url = make_url(settings.test_database_url)
    db_name = url.database or ""
    assert db_name.isidentifier() and "test" in db_name.lower(), (
        f"TEST_DATABASE_URL 必须指向名称含 test 的库(会被 DROP/重建),当前:{db_name!r}"
    )
    admin = create_async_engine(url.set(database=None), isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f"DROP DATABASE IF EXISTS `{db_name}`"))
        await conn.execute(
            text(f"CREATE DATABASE `{db_name}` CHARACTER SET utf8mb4")
        )
    await admin.dispose()

    engine = create_async_engine(settings.test_database_url, pool_pre_ping=True)
    # 跑 DDL 建表(以 ch02-ddl.sql 为权威)
    async with engine.begin() as conn:
        for s in _ddl_statements():
            await conn.execute(text(s))
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def db_session_factory(_test_engine, monkeypatch):
    """把 repository 用到的 async_session 指向测试库。"""
    factory = async_sessionmaker(_test_engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.base.async_session", factory)
    return factory


@pytest_asyncio.fixture()
async def db_clean(_test_engine):
    """每个测试后清空 4 张表(关外键检查以便 TRUNCATE)。"""
    yield
    async with _test_engine.begin() as conn:
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for t in _TABLES:
            await conn.execute(text(f"TRUNCATE TABLE {t}"))
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))

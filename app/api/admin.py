from fastapi import APIRouter

from app.core import jobs as jobs_core
from app.db import repository
from app.kb import dualwrite, milvus_client

router = APIRouter()


def _card(key: str, title: str, href: str, status: str, conclusion: str, metrics: dict) -> dict:
    return {"key": key, "title": title, "href": href,
            "status": status, "conclusion": conclusion, "metrics": metrics}


async def _inventory_card() -> dict:
    try:
        stats = await repository.knowledge_stats()
        status = "没数据" if stats["total"] == 0 else ("要干活" if stats["pending"] else "正常")
        conclusion = f"共 {stats['total']} 块,待向量化 {stats['pending']} 块"
        return _card("inventory", "知识库存", "/kb", status, conclusion, stats)
    except Exception:
        return _card("inventory", "知识库存", "/kb", "读数失败", "MySQL 读不到", {})


async def _dualwrite_card() -> dict:
    try:
        done = await repository.count_chunks_by_status("done")
        pending = await repository.count_chunks_by_status("pending")
    except Exception:
        return _card("dualwrite", "双写一致性", "/kb", "读数失败", "MySQL 读不到", {})
    try:
        with milvus_client.session() as client:
            n = milvus_client.count(client)
    except Exception:
        return _card("dualwrite", "双写一致性", "/kb", "读数失败", "Milvus 读不到",
                     {"done": done, "pending": pending})
    total = done + pending
    ok = dualwrite.consistency(total, n)
    conclusion = "知识块数与 Milvus 条数一致" if ok else (
        f"还有 {pending} 块待向量化" if pending else "知识块数与 Milvus 条数对不上"
    )
    return _card(
        "dualwrite", "双写一致性", "/kb",
        "正常" if ok else "要干活",
        conclusion,
        {"total": total, "done": done, "pending": pending, "milvus": n},
    )


async def _staging_card() -> dict:
    try:
        stats = await repository.staging_stats()
        total = sum(stats["counts"].values())
        status = "没数据" if total == 0 else "正常"
        return _card("staging", "对话挖知识暂存", "/kb", status,
                     f"共 {total} 条,保留 {stats['counts']['kept']} 条",
                     {"extracted": stats["counts"]["extracted"],
                      "kept": stats["counts"]["kept"],
                      "discarded": stats["counts"]["discarded"],
                      "batches": stats["batches"]})
    except Exception:
        return _card("staging", "对话挖知识暂存", "/kb", "读数失败", "MySQL 读不到", {})


async def _jobs_card() -> dict:
    try:
        jobs = jobs_core.status_all()
        running = sum(1 for j in jobs if j["running"])
        return _card("jobs", "作业", "/kb", "要干活" if running else "正常",
                     f"{len(jobs)} 个作业,运行中 {running}", {"total": len(jobs), "running": running})
    except Exception:
        return _card("jobs", "作业", "/kb", "读数失败", "作业运行器读不到", {})


@router.get("/api/admin/overview")
async def overview() -> dict:
    return {"cards": [
        await _inventory_card(),
        await _dualwrite_card(),
        await _staging_card(),
        await _jobs_card(),
    ]}

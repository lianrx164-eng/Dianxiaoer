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


async def _rageval_card() -> dict:
    from app.api import rageval
    report = rageval.load_report()
    if report is None:
        return _card("rag-eval", "RAG 评估", "/rag-eval", "没跑过",
                     "还没有评估报告,先跑一次 make eval-rag", {})
    meta = report.get("meta") or {}
    best = rageval.best_strategy(report.get("retrieval") or {})
    refusal = ((report.get("generation") or {}).get("refusal") or {}).get("rate")
    return _card(
        "rag-eval", "RAG 评估", "/rag-eval", "正常",
        f"最佳 MRR {best['mrr'] if best else '—'}({best['strategy'] if best else '—'}),"
        f"评估集 {meta.get('eval_count', '—')} 题",
        {"best_mrr": best["mrr"] if best else None,
         "best_strategy": best["strategy"] if best else None,
         "eval_count": meta.get("eval_count"), "refusal_rate": refusal},
    )


async def _observability_card() -> dict:
    from app.api import observability as obs
    try:
        d = await obs.overview()
    except Exception:
        return _card("observability", "观测与成本", "/observability", "读数失败", "读不到", {})
    cost, ev, cal = d["cost"], d["eval"], d["calibration"]
    latest = ev["runs"][0]["metrics"] if ev.get("runs") else {}
    if ev["status"] == "error":
        return _card("observability", "观测与成本", "/observability", "读数失败",
                     "评估趋势读不到(MySQL)", {})
    if cal.get("present") and not cal.get("in_sync"):
        status, conclusion = "要干活", cal.get("hint", "校准阈值未回填")
    elif cost["present"] and ev["present"] and cal["present"]:
        status, conclusion = "正常", "成本账、趋势、校准三块都有数"
    else:
        status, conclusion = "没跑过", "有报表还没跑,进页面按一次"
    return _card("observability", "观测与成本", "/observability", status, conclusion, {
        "top_intent": cost.get("top_intent"),
        "eval_runs": len(ev.get("runs", [])),
        "faithfulness": latest.get("faithfulness"),
        "threshold": cal.get("current_threshold"),
    })


async def _review_card() -> dict:
    try:
        rows = await repository.list_review_queue(None)
    except Exception:
        return _card("review", "飞轮待审", "/review", "读数失败", "MySQL 读不到", {})
    counts = {"待审": 0, "通过": 0, "驳回": 0}
    for row in rows:
        counts[row.review_status] = counts.get(row.review_status, 0) + 1
    if not rows:
        status, conclusion = "没数据", "还没有低置信度问题落池"
    elif counts["待审"]:
        status, conclusion = "要干活", f"{counts['待审']} 条待审,审完进知识库"
    else:
        status, conclusion = "正常", "待审已清零"
    return _card("review", "飞轮待审", "/review", status, conclusion, counts)


async def _topics_card() -> dict:
    try:
        dist = await repository.topic_distribution(samples_per_class=0)
    except Exception:
        return _card("topics", "主题分布", "/topics", "读数失败", "MySQL 读不到", {})
    if not dist["total"]:
        status, conclusion = "没数据", "池里还没有已归类的问题"
    else:
        top = max(dist["classes"], key=lambda c: c["count"])
        status = "正常"
        conclusion = (f"已归类 {dist['total']} 条,最多是「{top['label']}」{top['count']} 条")
    return _card("topics", "主题分布", "/topics", status, conclusion,
                 {"total": dist["total"], "latest": dist["latest"]})


async def _acceptance_card() -> dict:
    from app.api import acceptance
    try:
        ctx = await acceptance._load_ctx()
        gates = acceptance._gates(ctx)
    except Exception:
        return _card("acceptance", "分类器验收", "/acceptance", "读数失败",
                     "产物判定出错", {})
    passed = sum(1 for g in gates if g["status"] == "pass")
    online = bool(ctx["service"]["ok"])
    return _card("acceptance", "分类器验收", "/acceptance",
                 "正常" if passed == len(gates) else "要干活",
                 f"{passed}/{len(gates)} 闸通过,分类器{'在线' if online else '未起'}",
                 {"passed": passed, "total": len(gates), "online": online,
                  "classified": ctx["classified"]})


@router.get("/api/admin/overview")
async def overview() -> dict:
    return {"cards": [
        await _inventory_card(),
        await _dualwrite_card(),
        await _staging_card(),
        await _jobs_card(),
        await _observability_card(),
        await _rageval_card(),
        await _review_card(),
        await _topics_card(),
        await _acceptance_card(),
    ]}

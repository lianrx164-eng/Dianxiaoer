import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core import jobs as jobs_core
from app.db import repository

router = APIRouter()

REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "ch04" / "reports" / "rag_eval.json"


class StatusRequest(BaseModel):
    status: Literal["未解决", "已解决", "无需解决"]
    resolution: str | None = None


def load_report() -> dict | None:
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def best_strategy(retrieval: dict) -> dict | None:
    scored = [(name, cell.get("overall", {}).get("mrr"))
              for name, cell in retrieval.items() if isinstance(cell, dict)]
    scored = [(n, m) for n, m in scored if m is not None]
    if not scored:
        return None
    name, mrr = max(scored, key=lambda x: x[1])
    return {"strategy": name, "mrr": mrr}


async def _hallucination(report: dict) -> dict | None:
    gen = report.get("generation") or {}
    faith = ((gen.get("faithfulness") or {}).get("hybrid_rerank") or {})
    cases = faith.get("cases") or []
    eval_count = (report.get("meta") or {}).get("eval_count")
    try:
        ledger_total = await repository.faith_case_count()
    except Exception:
        ledger_total = None
    if not cases:
        return {"judged": 0, "confirmed": 0, "eval_count": eval_count,
                "judged_rate": 0.0, "confirmed_rate": 0.0,
                "ledger_total": ledger_total}
    try:
        status_map = await repository.faith_case_status_map()
    except Exception:
        status_map = {}
    current = {c["id"] for c in cases}
    confirmed = sum(1 for i in current if status_map.get(i) == "已解决")
    return {
        "judged": len(current), "confirmed": confirmed,
        "eval_count": eval_count, "ledger_total": ledger_total,
        "judged_rate": round(len(current) / eval_count, 4) if eval_count else None,
        "confirmed_rate": round(confirmed / eval_count, 4) if eval_count else None,
    }


@router.get("/api/rag-eval/overview")
async def overview() -> dict:
    report = load_report()
    job = jobs_core.status("eval-rag") if "eval-rag" in jobs_core.JOBS else {
        "name": "eval-rag", "running": False}
    if report is None:
        return {"present": False, "meta": None, "retrieval": {},
                "evidence_coverage": {}, "generation": None,
                "generation_done": False, "best": None, "hallucination": None,
                "job": {"name": job["name"], "running": job["running"]}}
    retrieval = report.get("retrieval") or {}
    generation = report.get("generation")
    return {
        "present": True,
        "meta": report.get("meta"),
        "retrieval": retrieval,
        "evidence_coverage": report.get("evidence_coverage") or {},
        "generation": generation,
        "generation_done": generation is not None,
        "best": best_strategy(retrieval),
        "hallucination": await _hallucination(report),
        "job": {"name": job["name"], "running": job["running"]},
        "errors": report.get("errors") or [],
    }


@router.get("/api/rag-eval/faith-cases")
async def faith_cases(
    status: Literal["未解决", "已解决", "无需解决"] | None = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
) -> dict:
    return await repository.list_faith_cases(status, page, size)


@router.post("/api/rag-eval/faith-cases/{case_id}/status")
async def set_faith_case_status(case_id: int, req: StatusRequest) -> dict:
    try:
        row = await repository.set_faith_case_status(
            case_id, req.status, req.resolution)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if row is None:
        raise HTTPException(status_code=404, detail="个案不存在")
    return row

import json
from pathlib import Path

from fastapi import APIRouter

from app.core import jobs as jobs_core

router = APIRouter()

REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "ch04" / "reports" / "rag_eval.json"


def _load() -> dict | None:
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _best(retrieval: dict) -> dict | None:
    scored = [(name, cell.get("overall", {}).get("mrr"))
              for name, cell in retrieval.items() if isinstance(cell, dict)]
    scored = [(n, m) for n, m in scored if m is not None]
    if not scored:
        return None
    name, mrr = max(scored, key=lambda x: x[1])
    return {"strategy": name, "mrr": mrr}


@router.get("/api/rag-eval/overview")
async def overview() -> dict:
    report = _load()
    job = jobs_core.status("eval-rag") if "eval-rag" in jobs_core.JOBS else {
        "name": "eval-rag", "running": False}
    if report is None:
        return {"present": False, "meta": None, "retrieval": {},
                "evidence_coverage": {}, "generation": None,
                "generation_done": False, "best": None,
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
        "best": _best(retrieval),
        "job": {"name": job["name"], "running": job["running"]},
        "errors": report.get("errors") or [],
    }

from fastapi import APIRouter, HTTPException

from app.core import jobs as jobs_core

router = APIRouter()


@router.get("/api/jobs")
def list_jobs() -> dict:
    return {"jobs": jobs_core.status_all()}


@router.post("/api/jobs/{name}")
def start_job(name: str) -> dict:
    try:
        return jobs_core.start(name)
    except KeyError:
        raise HTTPException(status_code=404, detail="未知作业")
    except jobs_core.JobAlreadyRunning:
        raise HTTPException(status_code=409, detail="同名作业还在跑,稍候再试")


@router.get("/api/jobs/{name}")
def job_status(name: str) -> dict:
    try:
        return jobs_core.status(name)
    except KeyError:
        raise HTTPException(status_code=404, detail="未知作业")


@router.post("/api/jobs/{name}/stop")
def stop_job(name: str) -> dict:
    try:
        return jobs_core.stop(name)
    except KeyError:
        raise HTTPException(status_code=404, detail="未知作业")

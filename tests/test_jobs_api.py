import pathlib
import threading

import pytest
from httpx import ASGITransport, AsyncClient

from app.core import jobs as jobs_core
from app.main import app


class FakeProc:
    def __init__(self, argv, **kw):
        self.argv = argv
        self.kw = kw
        self.returncode = None

    def poll(self):
        return self.returncode


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _clean_jobs():
    jobs_core._procs.clear()
    yield
    for proc in jobs_core._procs.values():
        handle = proc.kw.get("stdout")
        if handle is not None:
            handle.close()
    jobs_core._procs.clear()


async def test_unknown_job_is_rejected(client):
    r = await client.post("/api/jobs/rm-rf")
    assert r.status_code == 404


async def test_running_job_refuses_reentry(client, monkeypatch):
    captured = {}

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        return FakeProc(argv, **kw)

    monkeypatch.setattr(jobs_core.subprocess, "Popen", fake_popen)
    r1 = await client.post("/api/jobs/kb-build")
    assert r1.status_code == 200
    r2 = await client.post("/api/jobs/kb-build")
    assert r2.status_code == 409
    assert captured["argv"] == ["make", "kb-build"]  # 前端只能传作业名,命令 argv 写死


def test_every_job_target_exists_in_makefile():
    text = pathlib.Path("Makefile").read_text(encoding="utf-8")
    for spec in jobs_core.JOBS.values():
        assert f"{spec.target}:" in text, spec


async def test_force_job_argv_carries_make_vars(client, monkeypatch):
    captured = {}

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        return FakeProc(argv, **kw)

    monkeypatch.setattr(jobs_core.subprocess, "Popen", fake_popen)
    r = await client.post("/api/jobs/classify-pool-force")
    assert r.status_code == 200
    assert captured["argv"] == ["make", "classify-pool", "FORCE=1"]


def test_concurrent_start_only_one_wins(monkeypatch):
    monkeypatch.setattr(
        jobs_core.subprocess, "Popen", lambda argv, **kw: FakeProc(argv, **kw)
    )
    barrier = threading.Barrier(4)
    results = []

    def worker():
        barrier.wait()
        try:
            jobs_core.start("kb-build")
            results.append("ok")
        except jobs_core.JobAlreadyRunning:
            results.append("dup")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 1 and results.count("dup") == 3

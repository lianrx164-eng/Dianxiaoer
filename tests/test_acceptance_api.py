"""ch10 验收 API:九道闸只读产物;缺失/坏 json/服务不可达都不 500。"""
import json

import httpx
import pytest
from fastapi import FastAPI

from app.api import acceptance as acc


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(acc.router)
    return a


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.setattr(acc, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(acc, "DATA", tmp_path)
    (tmp_path / "reports").mkdir()
    return tmp_path


@pytest.fixture
def service_down(monkeypatch):
    async def probe():
        return {"ok": False}

    monkeypatch.setattr(acc, "_probe", probe)
    return {"ok": False}


async def test_overview_all_missing(client, reports, service_down):
    r = await client.get("/api/acceptance/overview")
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 9 and d["passed"] == 0 and d["all_pass"] is False
    statuses = {b["key"]: b["status"] for b in d["blocks"]}
    assert set(statuses.values()) == {"missing"}
    for b in d["blocks"]:
        assert "不达标" not in b["note"] and "没过线" not in b["note"]


async def test_broken_json_is_missing(client, reports, service_down):
    (reports / "reports" / "eval_report.json").write_text("{", encoding="utf-8")
    d = (await client.get("/api/acceptance/eval")).json()
    assert d["eval"]["present"] is False


async def test_eval_passthrough(client, reports, service_down):
    (reports / "reports" / "eval_report.json").write_text(json.dumps({
        "present": True, "test_size": 3, "threshold": 0.5,
        "micro": {"p": 0.9, "r": 0.9, "f1": 0.9},
        "macro": {"p": 0.8, "r": 0.8, "f1": 0.8},
        "classes": [{"name": "退换货", "severity": "严", "p": 1, "r": 1, "f1": 1,
                     "support": 1, "red_line": 0.9, "passed": True,
                     "tn": 1, "fp": 0, "fn": 0, "tp": 2}],
        "errors": [], "total_fp": 0, "total_fn": 0, "red_line_passed": True,
    }), encoding="utf-8")
    d = (await client.get("/api/acceptance/eval")).json()
    assert d["eval"]["present"] is True
    assert d["eval"]["micro"]["f1"] == 0.9


async def test_errors_missing_returns_empty(client, reports, service_down):
    d = (await client.get("/api/acceptance/errors")).json()
    assert d["errors"] == [] and d["kinds"] == {}


async def test_service_probe_failure(client, reports, monkeypatch):
    monkeypatch.setattr(acc, "CLASSIFIER_URL", "http://127.0.0.1:9")   # 关闭的端口
    d = (await client.get("/api/acceptance/service")).json()
    assert d["ok"] is False


async def test_data_missing_dataset(client, reports, service_down):
    d = (await client.get("/api/acceptance/data")).json()
    assert d["dataset"]["present"] is False
    assert d["lineage"]["raw"] is None
    assert d["sample_review"] is False


async def test_half_written_dataset_is_missing(client, reports, service_down):
    ds = reports / "dataset"
    ds.mkdir()
    for name in ("val", "test"):
        (ds / f"{name}.jsonl").write_text('{"text": "a", "labels": ["尺码"]}\n',
                                          encoding="utf-8")
    # 作业写到一半被杀:最后一行没有换行也没有闭合括号
    (ds / "train.jsonl").write_text('{"text": "a", "labels": ["尺码"]}\n{"text": "半',
                                    encoding="utf-8")
    r = await client.get("/api/acceptance/data")
    assert r.status_code == 200
    assert r.json()["dataset"]["present"] is False
    overview = (await client.get("/api/acceptance/overview")).json()
    assert next(b for b in overview["blocks"] if b["key"] == "data")["status"] == "missing"

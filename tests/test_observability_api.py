"""ch09 观测与成本只读 API:三块互不连坐,缺产物/坏 json 不 500。"""
import json

import httpx
import pytest
from fastapi import FastAPI

from app.api import observability as obs_api


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(obs_api.router)
    return a


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.setattr(obs_api, "COST_JSON", tmp_path / "cost.json")
    monkeypatch.setattr(obs_api, "CALIB_JSON", tmp_path / "calib.json")
    return tmp_path


@pytest.fixture
def no_eval(monkeypatch):
    async def empty(limit=10):
        return []

    monkeypatch.setattr(obs_api.repository, "list_eval_runs", empty)


async def test_all_missing(client, reports, no_eval):
    r = await client.get("/api/observability/overview")
    assert r.status_code == 200
    d = r.json()
    assert d["cost"]["present"] is False
    assert d["eval"]["present"] is False
    assert d["calibration"]["present"] is False


async def test_cost_passthrough(client, reports, no_eval):
    (reports / "cost.json").write_text(json.dumps({
        "present": True, "days": 7, "total_tokens": 100,
        "rows": [{"intent": "商品咨询", "count": 2, "tokens": 100,
                  "avg_tokens": 50, "share": 1.0}],
        "top_intent": "商品咨询"}), encoding="utf-8")
    d = (await client.get("/api/observability/overview")).json()
    assert d["cost"]["present"] is True
    assert d["cost"]["rows"][0]["intent"] == "商品咨询"
    assert d["cost"]["top_intent"] == "商品咨询"


async def test_cost_present_but_no_rows_hints(client, reports, no_eval):
    (reports / "cost.json").write_text(json.dumps({
        "present": False, "rows": [], "top_intent": None}), encoding="utf-8")
    d = (await client.get("/api/observability/overview")).json()
    assert d["cost"]["present"] is False
    assert "聊几句" in d["cost"]["hint"]


async def test_eval_runs_newest_first(client, reports, monkeypatch):
    class Run:
        id = 2
        triggered_by = "手动"
        dataset_size = 300
        metrics = {"recall_at_10": 0.9}
        created_at = __import__("datetime").datetime(2026, 11, 14, 10, 0)

    async def runs(limit=10):
        return [Run()]

    monkeypatch.setattr(obs_api.repository, "list_eval_runs", runs)
    d = (await client.get("/api/observability/overview")).json()
    assert d["eval"]["present"] is True
    assert d["eval"]["runs"][0]["id"] == 2


async def test_calibration_out_of_sync(client, reports, no_eval, monkeypatch):
    (reports / "calib.json").write_text(json.dumps({
        "recommended_threshold": 0.59, "youden_j": 0.874, "separable": True,
        "answerable": {"n": 1}, "absent": {"n": 1}, "curve": []}), encoding="utf-8")
    monkeypatch.setattr("app.config.settings.evidence_confidence_threshold", 0.5)
    d = (await client.get("/api/observability/overview")).json()
    assert d["calibration"]["present"] is True
    assert d["calibration"]["in_sync"] is False
    assert "回填" in d["calibration"]["hint"]


async def test_broken_json_is_treated_as_missing(client, reports, no_eval):
    (reports / "cost.json").write_text('{"rows": [', encoding="utf-8")
    (reports / "calib.json").write_text("not json", encoding="utf-8")
    r = await client.get("/api/observability/overview")
    assert r.status_code == 200
    assert r.json()["cost"]["present"] is False
    assert r.json()["calibration"]["present"] is False


async def test_eval_error_does_not_break_other_blocks(client, reports, monkeypatch):
    (reports / "cost.json").write_text(json.dumps({
        "present": True, "rows": [{"intent": "物流", "count": 1, "tokens": 10,
                                   "avg_tokens": 10, "share": 1.0}],
        "top_intent": "物流"}), encoding="utf-8")

    async def boom(limit=10):
        raise RuntimeError("mysql down")

    monkeypatch.setattr(obs_api.repository, "list_eval_runs", boom)
    d = (await client.get("/api/observability/overview")).json()
    assert d["eval"]["status"] == "error"
    assert d["cost"]["present"] is True

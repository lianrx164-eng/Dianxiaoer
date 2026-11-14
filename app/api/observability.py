"""ch09 观测与成本只读 API:意图成本账 / 评估趋势 / 置信度校准。

三块互不连坐:某块产物缺失或依赖读不到,只让该块 present=false / status=error,
其余照常端出。数字全部由脚本算好写进产物或直接读 eval_runs 表,这里不重算。
"""
import json
import logging
import pathlib

from fastapi import APIRouter

from app.config import settings
from app.db import repository

logger = logging.getLogger(__name__)
router = APIRouter()

_ROOT = pathlib.Path(__file__).resolve().parents[2]
COST_JSON = _ROOT / "data/ch09/reports/cost_by_intent.json"
CALIB_JSON = _ROOT / "data/ch09/reports/confidence_calibration.json"


def _load_json(path: pathlib.Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cost_block() -> dict:
    data = _load_json(COST_JSON)
    if not data or not data.get("present") or not data.get("rows"):
        hint = ("先在聊天页聊几句攒 trace,再重跑「意图成本账」。"
                if data is None else "窗口内还没有带 intent tag 的 trace——先聊几句再重跑。")
        return {"present": False, "status": "missing", "rows": [],
                "days": data.get("days") if data else None, "hint": hint}
    return {"present": True, "status": "ok", "days": data.get("days"),
            "rows": data.get("rows", []), "total_tokens": data.get("total_tokens"),
            "top_intent": data.get("top_intent"), "hint": ""}


async def _eval_block() -> dict:
    try:
        runs = await repository.list_eval_runs(limit=10)
    except Exception:
        logger.warning("评估趋势读库失败", exc_info=True)
        return {"present": False, "status": "error", "runs": [],
                "hint": "评估趋势读数失败(MySQL 不可用)。"}
    if not runs:
        return {"present": False, "status": "missing", "runs": [],
                "hint": "按一次「评估流水线」就是趋势的第一个点。"}
    return {"present": True, "status": "ok", "hint": "", "runs": [
        {"id": r.id,
         "triggered_by": r.triggered_by,
         "dataset_size": r.dataset_size,
         "metrics": r.metrics,
         "created_at": r.created_at.isoformat() if r.created_at else None}
        for r in runs
    ]}


def _calibration_block() -> dict:
    data = _load_json(CALIB_JSON)
    if not data or data.get("recommended_threshold") is None:
        return {"present": False, "status": "missing",
                "hint": "按一次「置信度校准」扫一遍评估集。"}
    rec = float(data["recommended_threshold"])
    current = settings.evidence_confidence_threshold
    in_sync = abs(rec - current) < 0.005
    hint = ("" if in_sync else
            f"推荐阈值 {rec:.2f} 与在用 {current:.2f} 不一致,该回填 app/config.py。")
    return {"present": True, "status": "ok",
            "recommended_threshold": rec, "current_threshold": current,
            "in_sync": in_sync, "hint": hint,
            "youden_j": data.get("youden_j"), "separable": data.get("separable"),
            "answerable": data.get("answerable"), "absent": data.get("absent"),
            "curve": data.get("curve", [])}


@router.get("/api/observability/overview")
async def overview() -> dict:
    return {"cost": _cost_block(), "eval": await _eval_block(),
            "calibration": _calibration_block()}

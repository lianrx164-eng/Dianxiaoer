"""ch10 验收 API:九道闸只读 data/ch10/reports 的产物,不重算任何指标。

闸的元数据(编号/标题/跳转/可重跑的作业)集中在 GATES;每道闸的判定是一个小函数,
缺产物回 missing,坏 json 当没跑过,服务探活失败只降级自己那一项。
"""
import json
import pathlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.classifier import CLASSIFIER_URL
from app.core.taxonomy import SEVERITY, TOPIC_NAMES
from app.db import repository

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "ch10"
REPORTS = DATA / "reports"

MIN_MICRO_F1 = 0.8         # 测试集总闸
MIN_VAL_F1 = 0.8           # 训练产物闸

router = APIRouter(prefix="/api/acceptance")


@dataclass(frozen=True)
class Gate:
    key: str
    no: int
    title: str
    page: str
    job: str


GATES: tuple[Gate, ...] = (
    Gate("data", 1, "语料与数据集", "/acceptance/data", "ch10-dataset"),
    Gate("golden", 2, "黄金样例闸", "/acceptance/data", "ch10-golden"),
    Gate("model", 3, "训练产物", "/acceptance/data", "ch10-train"),
    Gate("onnx", 4, "ONNX 导出与服务", "/acceptance/data", "ch10-export"),
    Gate("eval", 5, "测试集评测", "/acceptance/eval", "ch10-eval"),
    Gate("scan", 6, "阈值扫描重演", "/acceptance/eval", "ch10-threshold-scan"),
    Gate("matrix", 7, "混淆矩阵", "/acceptance/eval", "ch10-eval"),
    Gate("errors", 8, "错例复核", "/acceptance/errors", "ch10-eval"),
    Gate("classify", 9, "旁路批量归类", "/acceptance", "classify-pool"),
)


def _load(name: str, make: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        return {"present": False, "make": make, "hint": f"还没跑过;先执行 make {make}"}
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"present": False, "make": make, "hint": "产物损坏,重跑该作业"}
    return {"present": True, **body}


async def _probe() -> dict:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get(f"{CLASSIFIER_URL}/healthz")
        return {"ok": r.status_code == 200, "url": CLASSIFIER_URL}
    except Exception:                                  # noqa: BLE001 - 探活失败即不在线
        return {"ok": False, "url": CLASSIFIER_URL}


async def _classified_total() -> int | None:
    """已归类条数:读库失败(如 DB 未起)返回 None,只降级这一项。"""
    try:
        dist = await repository.topic_distribution()
        return int(dist["total"])
    except Exception:                                  # noqa: BLE001 - 单独降级
        return None


def _line_count(path: pathlib.Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for l in path.read_text(encoding="utf-8").splitlines() if l.strip())


def _read_jsonl(path: pathlib.Path) -> list[dict] | None:
    """逐行解析;出现坏行(作业写到一半被杀)就整份作废,不拿半份数据装完整。"""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            return None
    return rows


def _split_stats() -> dict:
    splits: dict[str, dict] = {}
    texts: dict[str, set[str]] = {}
    per_class_total = {n: 0 for n in TOPIC_NAMES}
    for name in ("train", "val", "test"):
        path = DATA / "dataset" / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = _read_jsonl(path)
        if rows is None:
            continue
        counts = {n: 0 for n in TOPIC_NAMES}
        for s in rows:
            for lb in s.get("labels", []):
                if lb in counts:
                    counts[lb] += 1
                    per_class_total[lb] += 1
        splits[name] = {"size": len(rows), "per_class": counts}
        texts[name] = {s["text"] for s in rows}
    present = len(splits) == 3
    leaks = {
        "train_val": len(texts.get("train", set()) & texts.get("val", set())),
        "train_test": len(texts.get("train", set()) & texts.get("test", set())),
        "val_test": len(texts.get("val", set()) & texts.get("test", set())),
    }
    every_class_present = present and all(
        all(splits[n]["per_class"][c] > 0 for n in ("train", "val", "test"))
        for c in TOPIC_NAMES)
    return {"present": present, "splits": splits, "leaks": leaks,
            "clean": present and sum(leaks.values()) == 0,
            "every_class_present": every_class_present,
            "per_class_total": per_class_total}


def _threshold_in_use() -> float | None:
    path = DATA / "model" / "threshold.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))["threshold"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None


# ---------- 每道闸的判定:ctx → (status, headline, note) ----------

def _gate_data(ctx: dict) -> tuple[str, str, str]:
    ds = ctx["ds"]
    if not ds["present"]:
        return "missing", "三份考卷还没生成", "先跑 make ch10-dataset"
    ok = ds["clean"] and ds["every_class_present"]
    head = (f"训练/验证/测试 {ds['splits']['train']['size']}/"
            f"{ds['splits']['val']['size']}/{ds['splits']['test']['size']} 条")
    note = "三份两两重叠 0 条" if ds["clean"] else f"存在跨集重复:{ds['leaks']}"
    note += ";17 类在三份里都有样本" if ds["every_class_present"] else ";有类目在某份里缺席"
    return ("pass" if ok else "fail"), head, note


def _gate_golden(ctx: dict) -> tuple[str, str, str]:
    g = ctx["golden"]
    if not g["present"]:
        return "missing", "还没跑过", "先跑 make ch10-golden"
    return ("pass" if g.get("passed") else "fail",
            f"通过率 {g.get('rate', 0):.0%}({g.get('hits')}/{g.get('total')})",
            f"闸线 {g.get('pass_line', 0.8):.0%}")


def _gate_model(ctx: dict) -> tuple[str, str, str]:
    model_dir = DATA / "model"
    threshold = ctx["threshold"]
    if not (model_dir / "config.json").exists() or threshold is None:
        return "missing", "还没有训练产物", "先跑 make ch10-train"
    val_f1 = None
    try:
        val_f1 = json.loads((model_dir / "threshold.json")
                            .read_text(encoding="utf-8")).get("val_micro_f1")
    except (OSError, json.JSONDecodeError):
        pass
    ok = val_f1 is not None and val_f1 >= MIN_VAL_F1
    head = (f"阈值 {threshold},验证集 micro-F1 {val_f1:.3f}"
            if val_f1 is not None else f"阈值 {threshold}")
    return ("pass" if ok else "fail"), head, f"目标是验证集 ≥ {MIN_VAL_F1}"


def _gate_onnx(ctx: dict) -> tuple[str, str, str]:
    e = ctx["export"]
    if not e["present"]:
        return "missing", "还没导出", "先跑 make ch10-export"
    online = ctx["service"]["ok"]
    ok = bool(e.get("passed")) and online
    head = (f"一致性校验 {'通过' if e.get('passed') else '未通过'}"
            f"({e.get('checked', 0)} 条)")
    return ("pass" if ok else "fail"), head, (
        "服务在线" if online else "服务未起:make classifier-up")


def _gate_eval(ctx: dict) -> tuple[str, str, str]:
    e = ctx["eval"]
    if not e["present"]:
        return "missing", "还没评测", "先跑 make ch10-eval"
    micro = (e.get("micro") or {}).get("f1")
    ok = micro is not None and micro >= MIN_MICRO_F1
    head = f"micro-F1 {micro:.3f}" if micro is not None else "缺 micro"
    note = "红线按类目档位逐条看"
    if e.get("red_line_passed") is False:
        note += ";有类目未过自己的红线(见评测详情)"
    return ("pass" if ok else "fail"), head, note


def _gate_scan(ctx: dict) -> tuple[str, str, str]:
    s = ctx["scan"]
    if not s["present"]:
        return "missing", "还没重演", "先跑 make ch10-threshold-scan"
    return ("pass" if s.get("consistent") else "fail",
            f"在用阈值 {s.get('in_use_threshold')},重演当选 {s.get('best_threshold')}",
            f"容差口径 {s.get('tolerance')}")


def _gate_matrix(ctx: dict) -> tuple[str, str, str]:
    e = ctx["eval"]
    if not e["present"]:
        return "missing", "还没评测", "先跑 make ch10-eval"
    classes = e.get("classes") or []
    ok = len(classes) == len(TOPIC_NAMES) and all(
        all(k in c for k in ("tn", "fp", "fn", "tp")) for c in classes)
    head = (f"{len(classes)}/17 类宫格齐全" if ok else f"只有 {len(classes)} 类带矩阵")
    return ("pass" if ok else "fail"), head, ""


def _gate_errors(ctx: dict) -> tuple[str, str, str]:
    e = ctx["eval"]
    path = REPORTS / "error_samples.md"
    if not e["present"] or not path.exists():
        return "missing", "还没有错例文件", "先跑 make ch10-eval"
    file_lines = sum(1 for l in path.read_text(encoding="utf-8").splitlines()
                     if l.startswith("- "))
    reported = len(e.get("errors") or [])
    return ("pass" if file_lines == reported else "fail",
            f"错例 {file_lines} 条,报告记账 {reported} 条", "条数须与 eval_report 一致")


def _gate_classify(ctx: dict) -> tuple[str, str, str]:
    run = ctx["run"]
    if not run["present"]:
        return "missing", "还没归过类", "先跑 make classify-pool"
    status = run.get("status")
    total = ctx["classified"]
    if total is None:                       # 库不可达:只能按上次运行的写入数判
        ok = status == "done" and int(run.get("written", 0)) > 0
        note = "库不可达,按上次运行判定"
    else:
        ok = status in ("done", "empty") and total >= 1
        note = "不足一批时用 classify-pool-force"
    head = {"done": f"本次归类 {run.get('written', 0)} 条",
            "empty": "池里没有待归类问题",
            "below_batch": f"待归类 {run.get('pending', 0)} 条,不足一批"}.get(
        status, f"状态 {status}")
    if total is not None:
        head += f";库里已归类 {total} 条"
    return ("pass" if ok else "fail"), head, note


_RULES: dict[str, Callable[[dict], tuple[str, str, str]]] = {
    "data": _gate_data, "golden": _gate_golden, "model": _gate_model,
    "onnx": _gate_onnx, "eval": _gate_eval, "scan": _gate_scan,
    "matrix": _gate_matrix, "errors": _gate_errors, "classify": _gate_classify,
}


async def _load_ctx() -> dict:
    return {
        "ds": _split_stats(),
        "golden": _load("golden_report.json", "ch10-golden"),
        "export": _load("export_report.json", "ch10-export"),
        "eval": _load("eval_report.json", "ch10-eval"),
        "scan": _load("threshold_scan.json", "ch10-threshold-scan"),
        "run": _load("classify_run.json", "classify-pool"),
        "threshold": _threshold_in_use(),
        "service": await _probe(),
        "classified": await _classified_total(),
    }


def _gates(ctx: dict) -> list[dict]:
    out = []
    for gate in GATES:
        status, headline, note = _RULES[gate.key](ctx)
        out.append({"key": gate.key, "no": gate.no, "title": gate.title,
                    "page": gate.page, "job": gate.job,
                    "status": status, "headline": headline, "note": note})
    return out


@router.get("/overview")
async def overview():
    ctx = await _load_ctx()
    gates = _gates(ctx)
    passed = sum(1 for g in gates if g["status"] == "pass")
    return {"blocks": gates, "passed": passed, "total": len(gates),
            "all_pass": passed == len(gates), "classifier": ctx["service"]}


@router.get("/eval")
async def eval_detail():
    return {"eval": _load("eval_report.json", "ch10-eval"),
            "scan": _load("threshold_scan.json", "ch10-threshold-scan"),
            "threshold_in_use": _threshold_in_use(),
            "severity": SEVERITY, "classifier": await _probe()}


@router.get("/data")
def data_detail():
    ds = _split_stats()
    model_dir = DATA / "model"
    onnx_dir = DATA / "onnx"

    def stat(d: pathlib.Path, *names: str) -> dict:
        out = {}
        for n in names:
            p = d / n
            out[n] = {"exists": p.exists(),
                      "bytes": p.stat().st_size if p.exists() else None}
        return out

    return {
        "lineage": {"raw": _line_count(DATA / "corpus_raw.jsonl"),
                    "clean": _line_count(DATA / "corpus_clean.jsonl"),
                    "labeled": _line_count(DATA / "corpus_labeled.jsonl")},
        "dataset": ds,
        "leaks": ds["leaks"],
        "sample_review": (DATA / "sample_review.md").exists(),
        "model": stat(model_dir, "config.json", "threshold.json", "model.safetensors"),
        "onnx": stat(onnx_dir, "model.onnx", "tokenizer.json", "threshold.json"),
        "topic_names": list(TOPIC_NAMES),
    }


@router.get("/errors")
def errors_detail():
    evalr = _load("eval_report.json", "ch10-eval")
    rows = evalr.get("errors") or []
    kinds = dict(Counter(r.get("kind", "?") for r in rows))
    pairs = Counter()
    for r in rows:
        if r.get("kind") == "错位":
            for missed in r.get("missed", []):
                for extra in r.get("extra", []):
                    pairs[f"{missed}→{extra}"] += 1
    return {"present": evalr["present"], "eval": evalr, "errors": rows,
            "kinds": kinds, "matrix_entries": sum(r.get("matrix_entries", 0) for r in rows),
            "total_fp": evalr.get("total_fp"), "total_fn": evalr.get("total_fn"),
            "pairs": [{"pair": k, "count": v} for k, v in pairs.most_common()]}


@router.get("/service")
async def service_detail():
    return await _probe()


class ClassifyIn(BaseModel):
    texts: list[str]


@router.post("/classify")
async def classify(body: ClassifyIn):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{CLASSIFIER_URL}/classify", json={"texts": body.texts})
            r.raise_for_status()
            return r.json()
    except Exception as exc:                            # noqa: BLE001 - 服务不可达要说清楚
        raise HTTPException(status_code=503, detail=f"分类器服务不可用:{exc}")

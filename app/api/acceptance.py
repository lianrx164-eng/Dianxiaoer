"""ch10 验收 API:九道闸只读 data/ch10/reports 的产物,不重算任何指标。

产物缺失/写坏 → present=false + 该跑哪个 make;服务不可达只降级自己那一项。
"""
import json
import pathlib
from collections import Counter

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.taxonomy import SEVERITY, TOPIC_NAMES

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "ch10"
REPORTS = DATA / "reports"
SERVICE = "http://127.0.0.1:8110"

router = APIRouter(prefix="/api/acceptance")

GATES = [
    ("data", 1, "语料与数据集", "/acceptance/data", "ch10-corpus"),
    ("golden", 2, "黄金样例闸", "/acceptance/data", "ch10-golden"),
    ("model", 3, "训练产物", "/acceptance/data", "ch10-train"),
    ("onnx", 4, "ONNX 导出与服务", "/acceptance/data", "ch10-export"),
    ("eval", 5, "测试集评测", "/acceptance/eval", "ch10-eval"),
    ("scan", 6, "阈值扫描重演", "/acceptance/eval", "ch10-threshold-scan"),
    ("matrix", 7, "混淆矩阵", "/acceptance/eval", "ch10-eval"),
    ("errors", 8, "错例复核", "/acceptance/errors", "ch10-eval"),
    ("classify", 9, "旁路批量归类", "/acceptance", "classify-pool"),
]


def _load(name: str, make: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        return {"present": False, "make": make, "hint": f"还没跑过;先执行 make {make}"}
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"present": False, "make": make, "hint": "产物损坏,重跑该作业"}
    return {"present": True, **body}


def _probe() -> dict:
    try:
        r = httpx.get(f"{SERVICE}/healthz", timeout=1.5)
        return {"ok": r.status_code == 200, "url": SERVICE}
    except Exception:                                  # noqa: BLE001 - 探活失败即不在线
        return {"ok": False, "url": SERVICE}


def _line_count(path: pathlib.Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for l in path.read_text(encoding="utf-8").splitlines() if l.strip())


def _split_stats() -> dict:
    splits: dict[str, dict] = {}
    texts: dict[str, set[str]] = {}
    per_class_total = {n: 0 for n in TOPIC_NAMES}
    for name in ("train", "val", "test"):
        path = DATA / "dataset" / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
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


def _gate(key: str, no: int, title: str, page: str, job: str,
          status: str, headline: str, note: str = "") -> dict:
    return {"key": key, "no": no, "title": title, "page": page, "job": job,
            "status": status, "headline": headline, "note": note}


def _gates() -> list[dict]:
    ds = _split_stats()
    golden = _load("golden_report.json", "ch10-golden")
    export = _load("export_report.json", "ch10-export")
    evalr = _load("eval_report.json", "ch10-eval")
    scan = _load("threshold_scan.json", "ch10-threshold-scan")
    run = _load("classify_run.json", "classify-pool")
    model_dir = DATA / "model"
    service = _probe()
    gates = []

    # 1 语料与数据集
    if not ds["present"]:
        gates.append(_gate("data", 1, "语料与数据集", "/acceptance/data", "ch10-dataset",
                           "missing", "三份考卷还没生成", "先跑 make ch10-dataset"))
    else:
        ok = ds["clean"] and ds["every_class_present"]
        head = (f"训练/验证/测试 {ds['splits']['train']['size']}/"
                f"{ds['splits']['val']['size']}/{ds['splits']['test']['size']} 条")
        note = ("三份两两重叠 0 条" if ds["clean"]
                else f"存在跨集重复:{ds['leaks']}")
        if ok:
            note += ";17 类在三份里都有样本"
        else:
            note += ";有类目在某份里缺席"
        gates.append(_gate("data", 1, "语料与数据集", "/acceptance/data", "ch10-dataset",
                           "pass" if ok else "fail", head, note))

    # 2 黄金样例闸
    if not golden["present"]:
        gates.append(_gate("golden", 2, "黄金样例闸", "/acceptance/data", "ch10-golden",
                           "missing", "还没跑过", "先跑 make ch10-golden"))
    else:
        gates.append(_gate("golden", 2, "黄金样例闸", "/acceptance/data", "ch10-golden",
                           "pass" if golden.get("passed") else "fail",
                           f"通过率 {golden.get('rate', 0):.0%}"
                           f"({golden.get('hits')}/{golden.get('total')})",
                           f"闸线 {golden.get('pass_line', 0.8):.0%}"))

    # 3 训练产物
    threshold = _threshold_in_use()
    have_model = (model_dir / "config.json").exists() and threshold is not None
    if not have_model:
        gates.append(_gate("model", 3, "训练产物", "/acceptance/data", "ch10-train",
                           "missing", "还没有训练产物", "先跑 make ch10-train"))
    else:
        val_f1 = None
        try:
            val_f1 = json.loads((model_dir / "threshold.json")
                                .read_text(encoding="utf-8")).get("val_micro_f1")
        except (OSError, json.JSONDecodeError):
            pass
        ok = val_f1 is not None and val_f1 >= 0.8
        gates.append(_gate("model", 3, "训练产物", "/acceptance/data", "ch10-train",
                           "pass" if ok else "fail",
                           f"阈值 {threshold},验证集 micro-F1 {val_f1:.3f}"
                           if val_f1 is not None else f"阈值 {threshold}",
                           "目标是验证集 ≥ 0.8"))

    # 4 ONNX 导出与服务
    if not export["present"]:
        gates.append(_gate("onnx", 4, "ONNX 导出与服务", "/acceptance/data", "ch10-export",
                           "missing", "还没导出", "先跑 make ch10-export"))
    else:
        ok = bool(export.get("passed")) and service["ok"]
        note = "服务在线" if service["ok"] else "服务未起:make classifier-up"
        gates.append(_gate("onnx", 4, "ONNX 导出与服务", "/acceptance/data", "ch10-export",
                           "pass" if ok else "fail",
                           f"一致性校验 {'通过' if export.get('passed') else '未通过'}"
                           f"({export.get('checked', 0)} 条)", note))

    # 5 测试集评测
    if not evalr["present"]:
        gates.append(_gate("eval", 5, "测试集评测", "/acceptance/eval", "ch10-eval",
                           "missing", "还没评测", "先跑 make ch10-eval"))
    else:
        micro_f1 = (evalr.get("micro") or {}).get("f1")
        ok = micro_f1 is not None and micro_f1 >= 0.8
        gates.append(_gate("eval", 5, "测试集评测", "/acceptance/eval", "ch10-eval",
                           "pass" if ok else "fail",
                           f"micro-F1 {micro_f1:.3f}" if micro_f1 is not None else "缺 micro",
                           "红线按类目档位逐条看"))

    # 6 阈值扫描
    if not scan["present"]:
        gates.append(_gate("scan", 6, "阈值扫描重演", "/acceptance/eval",
                           "ch10-threshold-scan", "missing", "还没重演",
                           "先跑 make ch10-threshold-scan"))
    else:
        gates.append(_gate("scan", 6, "阈值扫描重演", "/acceptance/eval",
                           "ch10-threshold-scan",
                           "pass" if scan.get("consistent") else "fail",
                           f"在用阈值 {scan.get('in_use_threshold')},"
                           f"重演当选 {scan.get('best_threshold')}",
                           "容差口径见报告 tolerance 字段"))

    # 7 混淆矩阵
    classes = (evalr.get("classes") or []) if evalr["present"] else []
    have_matrix = len(classes) == len(TOPIC_NAMES) and all(
        all(k in c for k in ("tn", "fp", "fn", "tp")) for c in classes)
    if not evalr["present"]:
        gates.append(_gate("matrix", 7, "混淆矩阵", "/acceptance/eval", "ch10-eval",
                           "missing", "还没评测", "先跑 make ch10-eval"))
    else:
        gates.append(_gate("matrix", 7, "混淆矩阵", "/acceptance/eval", "ch10-eval",
                           "pass" if have_matrix else "fail",
                           f"{len(classes)}/17 类宫格齐全" if have_matrix
                           else f"只有 {len(classes)} 类带矩阵"))

    # 8 错例复核
    err_path = REPORTS / "error_samples.md"
    if not evalr["present"] or not err_path.exists():
        gates.append(_gate("errors", 8, "错例复核", "/acceptance/errors", "ch10-eval",
                           "missing", "还没有错例文件", "先跑 make ch10-eval"))
    else:
        file_lines = sum(1 for l in err_path.read_text(encoding="utf-8").splitlines()
                         if l.startswith("- "))
        ok = file_lines == len(evalr.get("errors") or [])
        gates.append(_gate("errors", 8, "错例复核", "/acceptance/errors", "ch10-eval",
                           "pass" if ok else "fail",
                           f"错例 {file_lines} 条,报告记账 {len(evalr.get('errors') or [])} 条",
                           "条数须与 eval_report 一致"))

    # 9 旁路批量归类
    if not run["present"]:
        gates.append(_gate("classify", 9, "旁路批量归类", "/acceptance", "classify-pool",
                           "missing", "还没归过类", "先跑 make classify-pool"))
    else:
        status = run.get("status")
        ok = status in ("done", "empty")
        head = {"done": f"归类 {run.get('written', 0)} 条",
                "empty": "池里没有待归类问题",
                "below_batch": f"待归类 {run.get('pending', 0)} 条,不足一批"}.get(
            status, f"状态 {status}")
        gates.append(_gate("classify", 9, "旁路批量归类", "/acceptance", "classify-pool",
                           "pass" if ok else "fail", head,
                           "不足一批时用 classify-pool-force"))
    return gates


@router.get("/overview")
def overview():
    gates = _gates()
    passed = sum(1 for g in gates if g["status"] == "pass")
    return {"blocks": gates, "passed": passed, "total": len(gates),
            "all_pass": passed == len(gates), "classifier": _probe()}


@router.get("/eval")
def eval_detail():
    return {"eval": _load("eval_report.json", "ch10-eval"),
            "scan": _load("threshold_scan.json", "ch10-threshold-scan"),
            "threshold_in_use": _threshold_in_use(),
            "severity": SEVERITY, "classifier": _probe()}


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
def service_detail():
    return _probe()


class ClassifyIn(BaseModel):
    texts: list[str]


@router.post("/classify")
async def classify(body: ClassifyIn):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{SERVICE}/classify", json={"texts": body.texts})
            r.raise_for_status()
            return r.json()
    except Exception as exc:                            # noqa: BLE001 - 服务不可达要说清楚
        raise HTTPException(status_code=503, detail=f"分类器服务不可用:{exc}")

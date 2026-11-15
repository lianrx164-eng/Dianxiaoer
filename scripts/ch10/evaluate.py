"""ch10 评测:留出测试集上每类 P/R/F1 + 每类混淆矩阵 + 容错红线 + 判错样本导出。

运行:make ch10-eval。评测集扎在自家电商场景(dataset/test.jsonl),不引公开榜单。
"""
import json
import pathlib
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.taxonomy import LABEL2ID, NUM_CLASSES, SEVERITY, TOPIC_NAMES
from scripts.ch10.inference_lib import apply_threshold

MODEL_DIR = pathlib.Path("data/ch10/model")
TEST = pathlib.Path("data/ch10/dataset/test.jsonl")
REPORTS = pathlib.Path("data/ch10/reports")


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def predict(model, tokenizer, texts: list[str], threshold: float, device: str) -> np.ndarray:
    model.eval()
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, padding=True,
                            max_length=128, return_tensors="pt").to(device)
            probs_all.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    probs = np.concatenate(probs_all)
    return apply_threshold(probs, threshold)


def main() -> None:
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    threshold = json.loads((MODEL_DIR / "threshold.json").read_text())["threshold"]
    samples = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [s["text"] for s in samples]
    gold = np.zeros((len(samples), NUM_CLASSES), dtype=int)
    for i, s in enumerate(samples):
        for lb in s["labels"]:
            gold[i][LABEL2ID[lb]] = 1
    preds = predict(model, tokenizer, texts, threshold, device)

    p, r, f1, support = precision_recall_fscore_support(gold, preds, zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        gold, preds, average="micro", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gold, preds, average="macro", zero_division=0)
    cms = multilabel_confusion_matrix(gold, preds)

    REPORTS.mkdir(parents=True, exist_ok=True)
    classes, errors = [], []
    for i, name in enumerate(TOPIC_NAMES):
        sev = SEVERITY[name]
        line = {"严": 0.9, "中": 0.8}.get(sev)
        tn, fp = cms[i][0]
        fn, tp = cms[i][1]
        passed = None if line is None else bool(f1[i] >= line)
        classes.append({"name": name, "severity": sev,
                        "p": round(float(p[i]), 4), "r": round(float(r[i]), 4),
                        "f1": round(float(f1[i]), 4), "support": int(support[i]),
                        "red_line": line, "passed": passed,
                        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    for i, s in enumerate(samples):
        pred_labels = [TOPIC_NAMES[j] for j in range(NUM_CLASSES) if preds[i][j]]
        missed = [lb for lb in s["labels"] if lb not in pred_labels]
        extra = [lb for lb in pred_labels if lb not in s["labels"]]
        if not missed and not extra:
            continue
        kind = "错位" if missed and extra else ("漏打" if missed else "多打")
        errors.append({"text": s["text"], "gold": list(s["labels"]), "pred": pred_labels,
                       "missed": missed, "extra": extra, "kind": kind,
                       "matrix_entries": len(missed) + len(extra)})
    total_fp = sum(c["fp"] for c in classes)
    total_fn = sum(c["fn"] for c in classes)

    lines = ["# ch10 分类器评测报告(留出测试集)", "",
             f"测试集 {len(samples)} 条;判定阈值 {threshold}(验证集扫描所得)。", "",
             f"**micro**: P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f}  |  "
             f"**macro**: P={macro_p:.3f} R={macro_r:.3f} F1={macro_f1:.3f}", "",
             "## 每类指标(严档 F1 红线 0.9 / 中档 0.8 / 宽档不设线)", "",
             "| 类目 | 档位 | P | R | F1 | support | 红线 |",
             "|---|---|---|---|---|---|---|"]
    for c in classes:
        if c["red_line"] is None:
            flag = "—"
        elif not c["passed"]:
            flag = f"🔴 不达标(线 {c['red_line']}),先回头搞数据"
        else:
            flag = "✅"
        lines.append(f"| {c['name']} | {c['severity']} | {c['p']:.3f} | {c['r']:.3f} "
                     f"| {c['f1']:.3f} | {c['support']} | {flag} |")
    lines += ["", "## 每类混淆矩阵(TN FP / FN TP)", ""]
    for c in classes:
        lines.append(f"- **{c['name']}**: TN={c['tn']} FP={c['fp']} "
                     f"FN={c['fn']} TP={c['tp']}")
    (REPORTS / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")

    report = {
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "test_size": len(samples), "threshold": threshold,
        "micro": {"p": round(float(micro_p), 4), "r": round(float(micro_r), 4),
                  "f1": round(float(micro_f1), 4)},
        "macro": {"p": round(float(macro_p), 4), "r": round(float(macro_r), 4),
                  "f1": round(float(macro_f1), 4)},
        "classes": classes, "errors": errors,
        "total_fp": total_fp, "total_fn": total_fn,
        "red_line_passed": all(c["passed"] is not False for c in classes),
    }
    (REPORTS / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    err = ["# ch10 判错样本(人工复核:错在哪一类?标注本身有没有毛病?)", ""]
    for e in errors:
        err.append(f"- {e['text']}\n  标准: {e['gold']}  预测: {e['pred']}")
    (REPORTS / "error_samples.md").write_text("\n".join(err), encoding="utf-8")
    print(f"micro-F1 {micro_f1:.4f} / macro-F1 {macro_f1:.4f};"
          f"报告与判错样本已落 {REPORTS}/")


if __name__ == "__main__":
    main()

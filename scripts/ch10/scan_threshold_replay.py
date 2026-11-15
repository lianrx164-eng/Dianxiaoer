"""ch10 阈值扫描重演:验证集经 :8110 打分一次,九个候选线套同一张表各算一遍 micro-F1。

证明 threshold.json 里的线是扫出来的、可复算。运行:make ch10-threshold-scan(需 :8110)。
"""
import json
import pathlib

import httpx
import numpy as np
from sklearn.metrics import f1_score

from app.core.taxonomy import LABEL2ID, NUM_CLASSES, TOPIC_NAMES

SERVICE = "http://127.0.0.1:8110"
VAL = pathlib.Path("data/ch10/dataset/val.jsonl")
MODEL = pathlib.Path("data/ch10/model/threshold.json")
REPORTS = pathlib.Path("data/ch10/reports")
LINES = [round(0.30 + 0.05 * i, 2) for i in range(9)]


def main() -> None:
    samples = [json.loads(l) for l in VAL.read_text(encoding="utf-8").splitlines()
               if l.strip()]
    texts = [s["text"] for s in samples]
    with httpx.Client(timeout=300) as c:
        r = c.post(f"{SERVICE}/classify", json={"texts": texts})
        r.raise_for_status()
        results = r.json()["results"]
    probs = np.array([[res["scores"][n] for n in TOPIC_NAMES] for res in results])
    gold = np.zeros((len(samples), NUM_CLASSES), dtype=int)
    for i, s in enumerate(samples):
        for lb in s["labels"]:
            gold[i][LABEL2ID[lb]] = 1

    in_use = json.loads(MODEL.read_text())["threshold"]
    scan, best_t, best_f1 = [], None, -1.0
    for t in LINES:
        preds = (probs >= t).astype(int)
        f1 = float(f1_score(gold, preds, average="micro", zero_division=0))
        f1s = {n: float(f1_score(gold[:, i], preds[:, i], zero_division=0))
               for i, n in enumerate(TOPIC_NAMES)}
        tp = int((preds & gold).sum())
        fp = int((preds & (1 - gold)).sum())
        fn = int(((1 - preds) & gold).sum())
        scan.append({"threshold": t, "micro_f1": round(f1, 4),
                     "tp": tp, "fp": fp, "fn": fn, "per_class_f1": {
                         k: round(v, 4) for k, v in f1s.items()}})
        if f1 > best_f1:                 # 打平时先到的留任(严格大于)
            best_t, best_f1 = t, f1

    in_use_f1 = next((s["micro_f1"] for s in scan if s["threshold"] == in_use), None)
    exact = best_t == in_use
    adjacent = (not exact and in_use_f1 is not None
                and abs(best_t - in_use) <= 0.051
                and abs(best_f1 - in_use_f1) <= 0.005)
    report = {
        "ran_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "val_size": len(samples), "scan": scan,
        "best_threshold": best_t, "best_micro_f1": round(best_f1, 4),
        "in_use_threshold": in_use,
        "in_use_micro_f1": in_use_f1,
        "consistent": bool(exact or adjacent),
        "tolerance": "exact" if exact else ("adjacent" if adjacent else "none"),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "threshold_scan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"扫描 {len(LINES)} 条候选线;重演当选 {best_t},在用 {in_use},"
          f"一致={report['consistent']}({report['tolerance']})")
    if not report["consistent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

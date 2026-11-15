"""ch10 ONNX 导出:torch 模型 → onnx(变长 batch/seq),导出后跑测试集校验与 torch 预测完全一致。

torch 2.9+ 默认 dynamo=True,TorchScript 导出器(dynamo=False)已标废弃:优先走 dynamo,
失败再退 dynamo=False + dynamic_axes。运行:make ch10-export。
"""
import json
import pathlib
import shutil

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = pathlib.Path("data/ch10/model")
OUT = pathlib.Path("data/ch10/onnx")
TEST = pathlib.Path("data/ch10/dataset/test.jsonl")


class LogitsWrapper(torch.nn.Module):
    """HF 模型输出是 ModelOutput dict,导出前包一层只回 logits 张量。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits


def _export(wrapper, sample, path: pathlib.Path) -> str:
    ids, mask, ttids = sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]
    try:
        from torch.export import Dim
        b, s = Dim("batch", min=1), Dim("seq", min=1)
        torch.onnx.export(
            wrapper, (ids, mask, ttids), str(path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_shapes={"input_ids": {0: b, 1: s},
                            "attention_mask": {0: b, 1: s},
                            "token_type_ids": {0: b, 1: s}},
            dynamo=True,
            opset_version=17,
        )
        return "dynamo"
    except Exception as exc:                       # noqa: BLE001 - 退路要看得见原因
        print(f"dynamo 导出失败({exc!r}),退 TorchScript 导出器")
        dyn = {0: "batch", 1: "seq"}
        torch.onnx.export(
            wrapper, (ids, mask, ttids), str(path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": dyn, "attention_mask": dyn,
                          "token_type_ids": dyn, "logits": {0: "batch"}},
            dynamo=False,
            opset_version=17,
        )
        return "torchscript"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    wrapper = LogitsWrapper(model)
    sample = tokenizer(["买大了想退", "快递到哪了"], padding=True, return_tensors="pt")
    route = _export(wrapper, sample, OUT / "model.onnx")
    tokenizer.save_pretrained(OUT)          # 带出 tokenizer.json 给轻运行时用
    shutil.copy(MODEL_DIR / "threshold.json", OUT / "threshold.json")

    # 一致性校验:测试集全量,ONNX 与 torch 的过线标签必须完全一致
    threshold = json.loads((OUT / "threshold.json").read_text())["threshold"]
    texts = [json.loads(l)["text"]
             for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    sess = ort.InferenceSession(str(OUT / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    mismatch = 0
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, padding=True,
                            max_length=128, return_tensors="pt")
            t_logits = wrapper(**enc).numpy()
            (o_logits,) = sess.run(["logits"], {k: v.numpy() for k, v in enc.items()})
            assert np.allclose(t_logits, o_logits, atol=1e-3), "logits 超容差"
            t_pred = (1 / (1 + np.exp(-t_logits)) >= threshold)
            o_pred = (1 / (1 + np.exp(-o_logits)) >= threshold)
            mismatch += int((t_pred != o_pred).any(axis=1).sum())
    if mismatch:
        _write_report(len(texts), mismatch, route, OUT / "model.onnx")
        raise SystemExit(f"ONNX 与 torch 预测不一致 {mismatch} 条,导出失败")
    _write_report(len(texts), 0, route, OUT / "model.onnx")
    print(f"ONNX 导出({route})并校验通过({len(texts)} 条预测完全一致):{OUT}/model.onnx")


def _write_report(checked: int, mismatch: int, route: str, path: pathlib.Path) -> None:
    reports = pathlib.Path("data/ch10/reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "export_report.json").write_text(json.dumps({
        "ran_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "checked": checked, "mismatch": mismatch, "passed": mismatch == 0,
        "route": route, "onnx_path": str(path),
        "onnx_bytes": path.stat().st_size if path.exists() else None, "opset": 17,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

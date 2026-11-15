"""ch09 自动化评估流水线:复用 ch04 评估集与指标,定期跑、落 eval_runs、连趋势。
指标:检索段 Recall@10 / MRR(hybrid_rerank,可答桶),生成段 Faithfulness + D 桶拒答率。
D 桶拒答判定与线上一致:置信度闸(evidence_confidence) + 自评闸。

运行:make eval-flywheel(TRIGGER=手动|定时,默认手动)。cron 示例:
  0 6 * * * cd /path/to/dianxiaoer && make eval-flywheel TRIGGER=定时 >> log/eval.log 2>&1
趋势:读最近 10 轮,对比上一轮涨跌;任何指标下滑标 ⚠。
产物:data/ch09/reports/eval_trend.txt
"""
import argparse
import asyncio
import pathlib
import sys

from app.config import settings
from app.core import evidence, judges, selfcheck
from app.core.confidence import compute_evidence_confidence
from app.core.llm import get_chat_model
from app.core.prompts import RAG_ANSWER_PROMPT
from app.db import repository
from scripts.eval_ch04 import (
    format_evidence, load_absent, load_samples, mean, ranks, try_call,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT = _ROOT / "data/ch09/reports/eval_trend.txt"
_RECALL_K = 10
_SEM = asyncio.Semaphore(3)
_LINES: list[str] = []


def _log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(msg)


async def _eval_sample(sample: dict) -> tuple[float, float, bool | None] | None:
    """一条样本一次检索:同时算检索指标与生成忠实度(避免重复嵌入/重排)。"""
    async with _SEM:
        try:
            _, hits = await asyncio.wait_for(evidence.retrieve_hits(sample["query"]), 60)
        except Exception:
            return None
        groups = sample.get("expect_sections_all") or []
        if groups:
            hit_ranks = ranks(hits, groups)
            recall = sum(1 for r in hit_ranks if r and r <= _RECALL_K) / len(hit_ranks)
            mrr = sum((1 / r if r else 0.0) for r in hit_ranks) / len(hit_ranks)
        else:
            recall = mrr = 0.0
        ans = await try_call(
            (RAG_ANSWER_PROMPT | get_chat_model()).ainvoke(
                {"query": sample["query"], "evidence": format_evidence(hits)}),
            f"gen:{sample['id']}")
    if ans is None:
        return recall, mrr, None
    text = ans.content if isinstance(ans.content, str) else str(ans.content)
    verdict = await try_call(judges.judge_faithfulness(format_evidence(hits), text),
                             f"faith:{sample['id']}")
    return recall, mrr, (None if verdict is None else verdict[0])


async def _refused(sample: dict) -> bool | None:
    pair = await try_call(evidence.retrieve_hits(sample["query"]), f"ret:{sample['id']}")
    if pair is None:
        return None
    _, hits = pair
    if compute_evidence_confidence(hits).score < settings.evidence_confidence_threshold:
        return True
    chk = await try_call(selfcheck.check_sufficient(
        sample["query"], [f"{h['question']} {h['answer']}" for h in hits]),
        f"refusal:{sample['id']}")
    return None if chk is None else not chk["useful"]


async def _refusal_guarded(sample: dict):
    async with _SEM:
        try:
            return await _refused(sample)
        except Exception:
            return None


async def run_once() -> dict:
    graded = load_samples()
    absent = load_absent()

    results = [r for r in await asyncio.gather(*(_eval_sample(s) for s in graded))
               if r is not None]
    recall = mean([r[0] for r in results]) or 0.0
    mrr = mean([r[1] for r in results]) or 0.0
    faiths = [r[2] for r in results if r[2] is not None]
    faithfulness = mean([1.0 if f else 0.0 for f in faiths]) or 0.0

    refusals = [r for r in await asyncio.gather(*(_refusal_guarded(s) for s in absent))
                if r is not None]
    refusal_rate = (sum(1 for r in refusals if r) / len(refusals)) if refusals else 0.0

    return {"dataset_size": len(graded) + len(absent),
            "metrics": {"recall_at_10": round(recall, 3), "mrr": round(mrr, 3),
                        "faithfulness": round(faithfulness, 3),
                        "refusal_rate": round(refusal_rate, 3)}}


def _trend(runs) -> None:
    """runs 按新→旧;打印每轮各指标,并与上一轮(更旧一行)对比涨跌,下滑标 ⚠。"""
    _log("\n=== 评估趋势(新在上)===")
    names = ["recall_at_10", "mrr", "faithfulness", "refusal_rate"]
    _log(f"{'轮次':>4s} {'时间':16s} {'触发':4s} " + "".join(f"{n:>16s}" for n in names))
    for i, r in enumerate(runs):
        prev = runs[i + 1].metrics if i + 1 < len(runs) else None
        row = f"#{r.id:>3d} {r.created_at:%m-%d %H:%M}   {r.triggered_by:4s} "
        for n in names:
            v = r.metrics.get(n)
            cell = f"{v:.3f}" if v is not None else "—"
            if prev and v is not None and prev.get(n) is not None:
                d = v - prev[n]
                cell += " ↑" if d > 0.005 else (" ⚠↓" if d < -0.005 else " →")
            row += f"{cell:>16s}"
        _log(row)
    latest = runs[0].metrics
    older = runs[1].metrics if len(runs) > 1 else None
    if older:
        drops = [n for n in names if latest.get(n, 0) < older.get(n, 0) - 0.005]
        _log(f"\n⚠ 下滑指标:{', '.join(drops)}" if drops else "\n所有指标持平或上涨。")


async def main():
    ap = argparse.ArgumentParser()
    # Windows make 会把非 ASCII 参数按本地码页转坏,命令行只用 ASCII,入库再映射中文
    ap.add_argument("--triggered-by", default="manual", choices=["manual", "scheduled"])
    args = ap.parse_args()
    triggered_by = {"manual": "手动", "scheduled": "定时"}[args.triggered_by]

    _log(f"=== ch09 评估流水线(触发:{triggered_by})===")
    result = await run_once()
    m = result["metrics"]
    _log(f"本轮:Recall@10={m['recall_at_10']:.3f} MRR={m['mrr']:.3f} "
         f"Faithfulness={m['faithfulness']:.3f} 拒答率={m['refusal_rate']:.3f}")
    await repository.insert_eval_run(triggered_by, result["dataset_size"], m)
    _trend(await repository.list_eval_runs(limit=10))
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    _log(f"\n趋势报告已落 {_OUT.relative_to(_ROOT)}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)

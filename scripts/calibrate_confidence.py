"""ch09 置信度阈值校准:拿 ch04 评估集,A/B/C/E 桶(应可答)与 D 桶(应拒答)分别算
evidence_confidence 分布,扫阈值取 Youden J(=可答通过率 - 应拒放行率)最大的分离点。
输出分布表 + 推荐阈值 → 人工回填 settings.evidence_confidence_threshold(不拍脑袋)。

运行:make calibrate-confidence(需 Milvus + 上游可用 + 知识库已建)。
产物:data/ch09/reports/confidence_calibration.{txt,json}
"""
import asyncio
import json
import pathlib
import sys

from app.core import evidence
from app.core.confidence import compute_evidence_confidence

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT_DIR = _ROOT / "data/ch09/reports"
ANSWERABLE = {"A_policy", "B_model", "C_colloquial", "E_multi"}
ABSENT = "D_absent"
_SEM = asyncio.Semaphore(3)   # 上游对并发敏感;低并发 + 重试比跑得快更重要
_RETRIES = 2
_CALL_TIMEOUT = 45.0
_LINES: list[str] = []


def _log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(msg)


def _percentiles(xs: list[float]) -> dict:
    xs = sorted(xs)
    if not xs:
        return {}
    p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]
    return {"n": len(xs), "min": round(xs[0], 3), "p25": round(p(.25), 3),
            "p50": round(p(.5), 3), "p75": round(p(.75), 3), "max": round(xs[-1], 3)}


def _dist_line(name: str, d: dict) -> str:
    return (f"{name:12s} n={d['n']:3d} min={d['min']:.3f} p25={d['p25']:.3f} "
            f"p50={d['p50']:.3f} p75={d['p75']:.3f} max={d['max']:.3f}")


async def _conf(sample: dict) -> tuple[str, float | None]:
    async with _SEM:
        for attempt in range(_RETRIES):
            try:
                _, hits = await asyncio.wait_for(
                    evidence.retrieve_hits(sample["query"]), _CALL_TIMEOUT)
                return sample["bucket"], compute_evidence_confidence(hits).score
            except Exception as e:
                if attempt == _RETRIES - 1:
                    _log(f"[跳过] {sample['id']} 检索失败:{type(e).__name__}")
                    return sample["bucket"], None
                await asyncio.sleep(1 + attempt)
    return sample["bucket"], None


async def main() -> int:
    samples = [json.loads(ln) for ln in
               open(_ROOT / "tests/data/eval_ch04.jsonl", encoding="utf-8")]
    results = await asyncio.gather(*(_conf(s) for s in samples))
    answerable = [c for b, c in results if b in ANSWERABLE and c is not None]
    absent = [c for b, c in results if b == ABSENT and c is not None]
    if not answerable or not absent:
        _log("评估集为空或全部检索失败,无法校准。")
        return 1

    _log("=== evidence_confidence 分布(hybrid_rerank)===")
    dist_a, dist_d = _percentiles(answerable), _percentiles(absent)
    _log(_dist_line("可答(ABCE)", dist_a))
    _log(_dist_line("应拒(D)", dist_d))

    _log("\n=== 阈值扫描(通过率=conf>=t 的可答占比;放行率=conf>=t 的应拒占比)===")
    _log(f"{'t':>6s} {'可答通过率':>10s} {'应拒放行率':>10s} {'YoudenJ':>8s}")
    curve, best_t, best_j = [], 0.0, -1.0
    for i in range(5, 96):
        t = i / 100
        tpr = sum(1 for c in answerable if c >= t) / len(answerable)
        fpr = sum(1 for c in absent if c >= t) / len(absent)
        j = tpr - fpr
        curve.append({"t": t, "tpr": round(tpr, 4), "fpr": round(fpr, 4), "j": round(j, 4)})
        if i % 5 == 0 or j > best_j:
            _log(f"{t:6.2f} {tpr:10.3f} {fpr:10.3f} {j:8.3f}")
        if j > best_j:
            best_t, best_j = t, j

    _log(f"\n推荐阈值 evidence_confidence_threshold = {best_t:.2f}(Youden J={best_j:.3f})")
    if best_j < 0.3:
        _log("⚠ 两桶分离度不足(J<0.3):别硬选,把分布表交给人工定夺。")
    _log("请回填 app/config.py 默认值,并在 dev-notes/ch09.md 记录本次校准。")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "confidence_calibration.txt").write_text(
        "\n".join(_LINES) + "\n", encoding="utf-8")
    (_OUT_DIR / "confidence_calibration.json").write_text(json.dumps({
        "dataset_size": len(samples),
        "answerable": dist_a, "absent": dist_d,
        "curve": curve,
        "recommended_threshold": best_t, "youden_j": round(best_j, 4),
        "separable": best_j >= 0.3,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"报告已落 {_OUT_DIR.relative_to(_ROOT)}/confidence_calibration.txt")
    return 0 if best_j >= 0.3 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

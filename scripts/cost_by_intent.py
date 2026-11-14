"""ch09 Cost Control:按意图统计 token 花销(意图写在每轮根 span 的 intent 属性上)。
数据源 = Phoenix 本地服务(`phoenix serve`,SQLite):拉取窗口内 span 明细,按
trace 根 span 的 intent 归组,把 LLM span 的 token 汇总到意图上。
运行:make cost-report(DAYS=N 窗口天数,默认 7;需 Phoenix 在跑且 PHOENIX_ENDPOINT 已配)。
产物:data/ch09/reports/cost_by_intent.{txt,json}
说明:自定义模型名在 Phoenix 无内置单价,统计以 token 数为准。
"""
import argparse
import json
import pathlib
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

from app.config import settings

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT_DIR = _ROOT / "data/ch09/reports"


def _pick_col(df, *suffixes):
    for suffix in suffixes:
        for col in df.columns:
            if str(col) == suffix or str(col).endswith(suffix):
                return col
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    if not settings.phoenix_endpoint:
        print("Phoenix 未配置(PHOENIX_ENDPOINT),无账可查。")
        return 1
    from phoenix.client import Client

    client = Client(base_url=settings.phoenix_base_url)
    start = datetime.now(timezone.utc) - timedelta(days=args.days)
    try:
        df = client.spans.get_spans_dataframe(
            project_name=settings.phoenix_project, start_time=start, limit=10000)
    except Exception as e:  # noqa: BLE001 - 服务没起时给人话
        print(f"读取 Phoenix 失败({type(e).__name__}):{e}")
        return 1
    if df is None or df.empty:
        print("窗口内还没有 trace——先聊几句再来。")
        return 0

    intent_col = _pick_col(df, "attributes.intent")
    trace_col = _pick_col(df, "context.trace_id")
    token_col = _pick_col(df, "attributes.llm.token_count.total", "llm.token_count.total")
    kind_col = _pick_col(df, "span_kind")
    if intent_col is None or trace_col is None or token_col is None:
        print(f"Phoenix 返回缺少预期列(intent={intent_col} trace={trace_col} "
              f"tokens={token_col})——按实际 schema 调整脚本。")
        return 1

    # 意图可能落在根 span 或 classify_intent 节点 span,按 trace 取任意一个非空值
    intent_rows = (df[df[intent_col].notna()][[trace_col, intent_col]]
                   .drop_duplicates(trace_col))
    trace_intent = dict(zip(intent_rows[trace_col], intent_rows[intent_col]))
    counts = Counter(trace_intent.values())

    llm = df[df[kind_col] == "LLM"] if kind_col is not None else df
    tokens_by_trace = llm.groupby(trace_col)[token_col].sum().to_dict()
    tokens = Counter()
    for trace_id, n in tokens_by_trace.items():
        intent = trace_intent.get(trace_id)
        if intent:
            tokens[intent] += int(n or 0)

    rows = [{"intent": i, "count": counts[i], "tokens": tokens[i]}
            for i in counts]
    rows.sort(key=lambda r: r["tokens"], reverse=True)
    total = sum(r["tokens"] for r in rows) or 1

    lines = [f"=== 按意图 token 花销(近 {args.days} 天,数据源 Phoenix)===",
             f"{'意图':6s} {'请求数':>8s} {'总tokens':>12s} {'平均tokens':>12s} {'占比':>7s}"]
    for i, r in enumerate(rows):
        mark = "  ← 最烧钱" if i == 0 else ""
        lines.append(f"{r['intent']:6s} {r['count']:>8d} {r['tokens']:>12,d} "
                     f"{r['tokens'] // max(r['count'], 1):>12,d} {r['tokens'] / total:>6.0%}{mark}")
    if not rows:
        lines.append("(窗口内没有带 intent 属性的 trace——先聊几句再来)")
    out = "\n".join(lines)
    print(out)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "cost_by_intent.txt").write_text(out + "\n", encoding="utf-8")
    (_OUT_DIR / "cost_by_intent.json").write_text(json.dumps({
        "days": args.days,
        "from": start.isoformat(),
        "rows": [{"intent": r["intent"], "count": r["count"], "tokens": r["tokens"],
                  "avg_tokens": r["tokens"] // max(r["count"], 1),
                  "share": round(r["tokens"] / total, 4)} for r in rows],
        "total_tokens": sum(r["tokens"] for r in rows),
        "top_intent": rows[0]["intent"] if rows else None,
        "present": bool(rows),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已落 {_OUT_DIR.relative_to(_ROOT)}/cost_by_intent.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

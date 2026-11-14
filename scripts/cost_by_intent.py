"""ch09 Cost Control:按意图统计 token 花销(意图已打成 intent:xxx tag)。
数据源 = Langfuse v3 的 Metrics v1 端点(observations 视图);自部署 v3.218 的 traces 视图
不可用,故九类意图逐个 tag 过滤后客户端求和,不依赖服务端分组。
运行:make cost-report(DAYS=N 窗口天数,默认 7;需 Langfuse 在跑且 .env 配好三变量)。
产物:data/ch09/reports/cost_by_intent.{txt,json}
说明:自定义模型名在 Langfuse 无内置单价,统计以 token 数为准。
"""
import argparse
import json
import pathlib
import sys
from datetime import datetime, timedelta, timezone

from app.core.observability import get_langfuse

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT_DIR = _ROOT / "data/ch09/reports"
INTENTS = ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "人工", "闲聊", "其他"]


def _window(days: int) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (now - timedelta(days=days)).strftime(fmt), now.strftime(fmt)


def _field(row, name: str) -> float:
    if isinstance(row, dict):
        return float(row.get(name) or 0)
    return float(getattr(row, name, 0) or 0)


def _query_intent(client, intent: str, frm: str, to: str) -> tuple[int, int]:
    """一次 v1 Metrics 查询:observations 视图,tags 过滤到单意图,聚合 token/count。"""
    query = json.dumps({
        "view": "observations",
        "metrics": [{"measure": "totalTokens", "aggregation": "sum"},
                    {"measure": "count", "aggregation": "count"}],
        "filters": [{"column": "tags", "operator": "any of",
                     "value": [f"intent:{intent}"], "type": "arrayOptions"}],
        "fromTimestamp": frm, "toTimestamp": to,
    }, ensure_ascii=False)
    resp = client.api.metrics.metrics(query=query)
    data = getattr(resp, "data", None) or []
    tokens = sum(_field(r, "sum_totalTokens") for r in data)
    count = sum(_field(r, "count_count") for r in data)
    return int(tokens), int(count)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    client = get_langfuse()
    if client is None:
        print("Langfuse 未配置(.env 三变量),无账可查。")
        return 1
    frm, to = _window(args.days)
    rows, failed = [], 0
    for intent in INTENTS:
        try:
            tokens, count = _query_intent(client, intent, frm, to)
        except Exception as e:  # noqa: BLE001 - 单类失败不拖垮整张表
            failed += 1
            print(f"[{intent}] 查询失败:{type(e).__name__}: {e}")
            continue
        if count:
            rows.append({"intent": intent, "tokens": tokens, "count": count})
    if failed == len(INTENTS):
        print("九类意图全部查询失败——先确认 Langfuse 在跑、key 正确。")
        return 1
    rows.sort(key=lambda r: r["tokens"], reverse=True)

    total = sum(r["tokens"] for r in rows) or 1
    lines = [f"=== 按意图 token 花销(近 {args.days} 天,数据源 Langfuse)===",
             f"{'意图':6s} {'请求数':>8s} {'总tokens':>12s} {'平均tokens':>12s} {'占比':>7s}"]
    for i, r in enumerate(rows):
        mark = "  ← 最烧钱" if i == 0 else ""
        lines.append(f"{r['intent']:6s} {r['count']:>8d} {r['tokens']:>12,d} "
                     f"{r['tokens'] // max(r['count'], 1):>12,d} {r['tokens'] / total:>6.0%}{mark}")
    if not rows:
        lines.append("(窗口内没有带 intent tag 的 trace——先聊几句再来)")
    out = "\n".join(lines)
    print(out)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "cost_by_intent.txt").write_text(out + "\n", encoding="utf-8")
    (_OUT_DIR / "cost_by_intent.json").write_text(json.dumps({
        "days": args.days,
        "from": frm, "to": to,
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

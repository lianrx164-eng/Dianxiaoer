"""ch04 四策略分桶评估:检索排序 + 证据覆盖度 + 生成质量。

产物:
  data/ch04/reports/rag_eval.txt   运行日志(给人读)
  data/ch04/reports/rag_eval.json  页面读的同一份数字(web 层不重算)

用法:
  make eval-rag                 # 全量 300 题 × 4 策略
  uv run python scripts/eval_ch04.py --no-generation   # 只跑确定性两段
  uv run python scripts/eval_ch04.py --limit 5         # 每桶 5 题冒烟
"""
import argparse
import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import settings
from app.core import embeddings, judges, retrieval
from app.core.llm import get_chat_model
from app.core.prompts import ANSWER_COVERAGE_PROMPT, RAG_ANSWER_PROMPT
from app.db import repository
from app.kb import milvus_client

STRATEGIES = ["vector", "bm25", "hybrid", "hybrid_rerank"]
BUCKETS = ["A_policy", "B_model", "C_colloquial", "E_multi"]
D_BUCKET = "D_absent"
RECALL_K = 5
TOP_K = 10
GEN_CONCURRENCY = 3
CALL_TIMEOUT = 45.0

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET = ROOT / "tests" / "data" / "eval_ch04.jsonl"
OUT_DIR = ROOT / "data" / "ch04" / "reports"
OUT_TXT = OUT_DIR / "rag_eval.txt"
OUT_JSON = OUT_DIR / "rag_eval.json"

_ERRORS: list[str] = []
_LINES: list[str] = []


class _Coverage(BaseModel):
    covered: list[int] = Field(default_factory=list, description="被覆盖要点序号")


def _norm(s: str) -> str:
    return "".join((s or "").split()).lower()


def _log(line: str = "") -> None:
    print(line)
    _LINES.append(line)


def _load(limit: int = 0) -> list[dict]:
    samples = [json.loads(l) for l in EVAL_SET.read_text(encoding="utf-8").splitlines()
               if l.strip()]
    samples = [s for s in samples if s["bucket"] != D_BUCKET]
    if limit:
        kept, seen = [], defaultdict(int)
        for s in samples:
            if seen[s["bucket"]] < limit:
                kept.append(s)
                seen[s["bucket"]] += 1
        samples = kept
    return samples


def _load_absent() -> list[dict]:
    return [json.loads(l) for l in EVAL_SET.read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l)["bucket"] == D_BUCKET]


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _match_section(hit: dict, aliases: list[str]) -> bool:
    path = _norm(hit.get("section_path", ""))
    return any(_norm(a) in path for a in aliases)


def _ranks(hits: list[dict], groups: list[list[str]]) -> list[int | None]:
    ranks: list[int | None] = []
    for aliases in groups:
        rank = next((i for i, h in enumerate(hits, 1) if _match_section(h, aliases)), None)
        ranks.append(rank)
    return ranks


def _retrieval_scores(sample: dict, hits: list[dict]) -> tuple[float, float] | None:
    groups = sample.get("expect_sections_all") or []
    if not groups:
        return None
    ranks = _ranks(hits, groups)
    recall = sum(1 for r in ranks if r and r <= RECALL_K) / len(ranks)
    mrr = sum((1 / r if r else 0.0) for r in ranks) / len(ranks)
    return recall, mrr


def _coverage_mech(points: list[str], hits: list[dict]) -> float | None:
    if not points:
        return None
    text = _norm("\n".join(f"{h.get('question', '')} {h.get('answer', '')}"
                           for h in hits[:TOP_K]))
    return sum(1 for p in points if _norm(p) in text) / len(points)


def _answer_coverage_mech(points: list[str], answer: str) -> float | None:
    if not points:
        return None
    text = _norm(answer)
    return sum(1 for p in points if _norm(p) in text) / len(points)


def _evidence(hits: list[dict]) -> str:
    if not hits:
        return "(无证据)"
    return "\n".join(f"[{i}] {h['question']}: {h['answer']}"
                     for i, h in enumerate(hits[:TOP_K], 1))


def _citations(hits: list[dict]) -> list[dict]:
    return [{"n": i, "chunk_id": h["id"], "section_path": h["section_path"],
             "question": h["question"], "answer": h["answer"]}
            for i, h in enumerate(hits[:TOP_K], 1)]


async def _try(coro, label: str):
    try:
        return await asyncio.wait_for(coro, CALL_TIMEOUT)
    except Exception as e:  # noqa: BLE001 - 单点失败不拖垮整轮
        if len(_ERRORS) < 50:
            _ERRORS.append(f"{label}: {type(e).__name__}: {str(e)[:120]}")
        return None


async def _gen_answer(query: str, hits: list[dict]) -> str:
    msg = await (RAG_ANSWER_PROMPT | get_chat_model()).ainvoke(
        {"query": query, "evidence": _evidence(hits)})
    return msg.content if isinstance(msg.content, str) else ""


async def _judge_coverage(points: list[str], answer: str) -> float:
    model = judges.judge_model().with_structured_output(
        _Coverage, method="function_calling")
    prompt = "\n".join(f"{i}. {p}" for i, p in enumerate(points, 1))
    r: _Coverage = await (ANSWER_COVERAGE_PROMPT | model).ainvoke(
        {"points": prompt, "answer": answer})
    return len(set(r.covered)) / len(points) if points else 0.0


def _is_refusal(answer: str) -> bool:
    markers = ["没有查到", "未查到", "未找到", "暂无相关", "无相关信息",
               "无法回答", "不能回答", "建议联系", "转人工", "联系人工",
               "暂时没有", "不清楚"]
    return any(m in answer for m in markers)


async def _retrieve_all(samples: list[dict], strategies: list[str], client) -> dict:
    jobs = [(s, st) for s in samples for st in strategies
            if not (s["bucket"] == D_BUCKET and st != "hybrid_rerank")]
    total = len(jobs)
    out: dict = {}
    _log(f"retrieval start: {total} calls")
    for i, (sample, strategy) in enumerate(jobs, 1):
        if i <= 3:
            _log(f"  try {i}: {sample['id']} {strategy}")
        hits = await _try(
            retrieval.search_knowledge(sample["query"], strategy=strategy,
                                       top_k=TOP_K, client=client),
            f"retrieval:{sample['id']}:{strategy}")
        if i <= 3:
            _log(f"  done {i}: hits={len(hits or [])}")
        out[(sample["id"], strategy)] = hits or []
        if i % 20 == 0 or i == total:
            _log(f"  retrieval {i}/{total}")
        if i % 50 == 0:
            await asyncio.sleep(0.05)
    return out


async def _generation(samples: list[dict], d_samples: list[dict],
                      hits_map: dict, strategies: list[str]) -> dict | None:
    sem = asyncio.Semaphore(GEN_CONCURRENCY)
    progress = [0, len(samples) * len(strategies)]

    async def one(sample: dict, strategy: str):
        async with sem:
            hits = hits_map[(sample["id"], strategy)]
            answer = await _try(_gen_answer(sample["query"], hits),
                                f"gen:{sample['id']}:{strategy}")
            if answer is None:
                progress[0] += 1
                return None
            coverage = await _try(_judge_coverage(sample["expect_points"], answer),
                                  f"cover:{sample['id']}:{strategy}")
            if coverage is None:
                coverage = _answer_coverage_mech(sample["expect_points"], answer)
            row = {"id": sample["id"], "bucket": sample["bucket"], "strategy": strategy,
                   "coverage": coverage, "answer": answer}
            if strategy == "hybrid_rerank":
                faith = await _try(judges.judge_faithfulness(_evidence(hits), answer),
                                   f"faith:{sample['id']}")
                if faith is not None:
                    row["faithful"], row["reason"] = faith
                else:
                    row["faithful"], row["reason"] = None, ""
                row["citations"] = _citations(hits)
            progress[0] += 1
            if progress[0] % 40 == 0 or progress[0] == progress[1]:
                _log(f"  generation {progress[0]}/{progress[1]}")
            return row

    tasks = [one(s, st) for s in samples for st in strategies]
    rows = [r for r in await asyncio.gather(*tasks) if r]
    if not rows:
        return None

    coverage = defaultdict(lambda: defaultdict(list))
    faith = defaultdict(lambda: defaultdict(list))
    cases = []
    for r in rows:
        coverage[r["strategy"]][r["bucket"]].append(r["coverage"])
        if r["strategy"] == "hybrid_rerank":
            if r.get("faithful") is not None:
                faith["hybrid_rerank"][r["bucket"]].append(1 if r["faithful"] else 0)
            if r.get("faithful") is False:
                cases.append({"id": r["id"], "bucket": r["bucket"],
                              "query": next(s["query"] for s in samples if s["id"] == r["id"]),
                              "answer": r["answer"], "reason": r["reason"],
                              "citations": r["citations"],
                              "judge_model": settings.chat_model})

    answer_cov = {st: {**{b: _mean(v) for b, v in coverage[st].items()},
                       "overall": _mean([x for v in coverage[st].values() for x in v])}
                  for st in coverage}
    for c in cases:
        try:
            await repository.upsert_faith_case(
                c["id"], c["bucket"], c["query"], "hybrid_rerank",
                c["answer"], c["reason"], c["citations"], settings.chat_model)
        except Exception as e:  # noqa: BLE001 - 台账失败不影响本轮报告
            _log(f"[台账写入失败] {c['id']}: {type(e).__name__}: {str(e)[:100]}")
    faith_out = {"hybrid_rerank": {
        **{b: _mean(v) for b, v in faith["hybrid_rerank"].items()},
        "overall": _mean([x for v in faith["hybrid_rerank"].values() for x in v]),
        "cases": cases,
    }} if faith["hybrid_rerank"] else {}
    return {"answer_coverage": answer_cov, "faithfulness": faith_out,
            "refusal": await _refusal(d_samples, hits_map), "errors": list(_ERRORS)}


async def _refusal(samples: list[dict], hits_map: dict) -> dict:
    d_samples = [s for s in samples if s["bucket"] == D_BUCKET]
    if not d_samples:
        return {"total": 0, "refused": 0, "rate": None}
    sem = asyncio.Semaphore(GEN_CONCURRENCY)

    async def one(sample):
        async with sem:
            hits = hits_map.get((sample["id"], "hybrid_rerank"), [])
            answer = await _try(_gen_answer(sample["query"], hits),
                                f"refusal:{sample['id']}")
            return answer is not None and _is_refusal(answer)

    results = await asyncio.gather(*(one(s) for s in d_samples))
    refused = sum(1 for r in results if r)
    return {"total": len(d_samples), "refused": refused,
            "rate": round(refused / len(d_samples), 4)}


def _write_report(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def _print_retrieval(retrieval_d: dict, coverage_d: dict) -> None:
    for st in STRATEGIES:
        if st not in retrieval_d:
            continue
        parts = []
        for b in BUCKETS:
            cell = retrieval_d[st].get(b, {})
            parts.append(f"{b}: R@5={cell.get('recall@5')} MRR={cell.get('mrr')}")
        _log(f"[检索] {st:14s} " + " | ".join(parts))
        cov = coverage_d.get(st, {}).get("overall")
        _log(f"        证据覆盖度 overall={cov}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-generation", action="store_true")
    args = parser.parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    started = time.time()
    samples = _load(args.limit)
    d_samples = _load_absent()
    if args.limit:
        d_samples = d_samples[:args.limit]
    all_samples = samples + d_samples

    client = milvus_client.get_client()
    milvus_client.ensure_collection(client)
    kb_count = milvus_client.count(client)

    # 同一 query 的四策略复用一份向量,少打 3/4 的嵌入调用
    embed_cache: dict[str, list[float]] = {}
    real_embed = embeddings.embed_query

    async def cached_embed(q: str) -> list[float]:
        if q not in embed_cache:
            embed_cache[q] = await real_embed(q)
        return embed_cache[q]

    embeddings.embed_query = cached_embed

    _log(f"ch04 RAG eval @ {datetime.now():%Y-%m-%d %H:%M:%S}")
    _log(f"samples={len(samples)} (D={len(d_samples)}) strategies={strategies} "
         f"kb_count={kb_count} judge={settings.chat_model}")
    hits_map = await _retrieve_all(all_samples, strategies, client)

    retrieval_d = {}
    coverage_d = {}
    for st in strategies:
        per_bucket = defaultdict(lambda: {"recall": [], "mrr": []})
        cov_bucket = defaultdict(list)
        for s in samples:
            hits = hits_map[(s["id"], st)]
            scored = _retrieval_scores(s, hits)
            if scored:
                per_bucket[s["bucket"]]["recall"].append(scored[0])
                per_bucket[s["bucket"]]["mrr"].append(scored[1])
            cov_bucket[s["bucket"]].append(_coverage_mech(s["expect_points"], hits))
        retrieval_d[st] = {
            **{b: {"recall@5": _mean(v["recall"]), "mrr": _mean(v["mrr"]),
                   "n": len(v["recall"])} for b, v in per_bucket.items()},
            "overall": {"recall@5": _mean([x for v in per_bucket.values() for x in v["recall"]]),
                        "mrr": _mean([x for v in per_bucket.values() for x in v["mrr"]])},
        }
        coverage_d[st] = {**{b: _mean(v) for b, v in cov_bucket.items()},
                          "overall": _mean([x for v in cov_bucket.values() for x in v])}

    _print_retrieval(retrieval_d, coverage_d)

    report = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "eval_count": len(samples) + len(d_samples),
            "kb_count": kb_count,
            "embed_model": settings.embed_model,
            "rerank_model": settings.rerank_model,
            "judge_model": settings.chat_model,
            "strategies": strategies,
            "recall_k": RECALL_K,
            "top_k": TOP_K,
            "elapsed_sec": round(time.time() - started, 1),
        },
        "retrieval": retrieval_d,
        "evidence_coverage": coverage_d,
        "generation": None,
        "errors": list(_ERRORS),
    }
    # 先把确定性两段落盘:生成段跑挂也有报告可看
    _write_report(report)

    generation_d = None
    if not args.no_generation:
        try:
            generation_d = await _generation(samples, d_samples, hits_map, strategies)
            if generation_d:
                ac = generation_d["answer_coverage"]
                for st in strategies:
                    _log(f"[生成] {st:14s} 答案覆盖度 overall={ac.get(st, {}).get('overall')}")
                faith = generation_d["faithfulness"].get("hybrid_rerank", {})
                _log(f"[忠实度] hybrid_rerank overall={faith.get('overall')} "
                     f"cases={len(faith.get('cases', []))}")
                _log(f"[拒答] D 桶 rate={generation_d['refusal']['rate']} "
                     f"({generation_d['refusal']['refused']}/{generation_d['refusal']['total']})")
        except Exception as e:  # noqa: BLE001 - 生成段失败不拖垮检索报告
            _log(f"[生成段未完成] {type(e).__name__}: {e}")
            generation_d = None

    report["generation"] = generation_d
    report["errors"] = list(_ERRORS)
    report["meta"]["elapsed_sec"] = round(time.time() - started, 1)
    _write_report(report)
    _log(f"report -> {OUT_TXT}")
    _log(f"report -> {OUT_JSON}")


if __name__ == "__main__":
    asyncio.run(main())

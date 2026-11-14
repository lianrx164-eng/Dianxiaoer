"""检索 + 两道生成前证据闸的统一入口(工具 query_faq 与图 forced_rag 共用)。

ch09:机械分闸升级为四信号加权置信度闸(阈值由评估集校准),并统一产出召回快照
(落池与 👎 回捞共用)。
"""
from app.config import settings
from app.core import query_understanding, retrieval, selfcheck
from app.core.confidence import compute_evidence_confidence, snapshot_from_hits


def build_citations(arranged: list) -> tuple[list[dict], str]:
    """把排好序的命中转成编号 citations 与编号证据文本。
    知识路(forced_rag)与退款路(retrieve_policy)共用,避免两处拼装漂移。"""
    citations = [
        {"n": i + 1, "id": h["id"], "chunk_id": h["id"],
         "section_path": h["section_path"], "question": h["question"],
         "answer": h["answer"], "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    text = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}"
                     for c in citations)
    return citations, text


async def retrieve_hits(query_raw: str, category: str | None = None) -> tuple[str, list[dict]]:
    """理解 → 混合检索(hybrid_rerank);返回(标准问句, 命中)。
    校准脚本与在线闸共用同一条检索路径,也共用同一次理解调用。"""
    u = await query_understanding.understand(query_raw)
    query = u["standard"]
    bm25_query = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    hits = await retrieval.search_knowledge(
        query, strategy="hybrid_rerank", category=category, bm25_query=bm25_query)
    if category and not hits:
        # 模型给的分类名不是库里的精确 category 时,退回全库检索,避免误拒答
        hits = await retrieval.search_knowledge(
            query, strategy="hybrid_rerank", bm25_query=bm25_query)
    return query, hits


async def gather_evidence(query_raw: str, category: str | None = None) -> dict:
    """理解 → 混合检索 → 置信度闸 → 自评闸 → 首尾组装。

    返回字段:sufficient / source / reason / citations / evidence / top /
    confidence / signals / snapshot。
    source 仅在 insufficient 时有意义:retrieval_low_conf | self_check。
    """
    query, hits = await retrieve_hits(query_raw, category=category)
    conf = compute_evidence_confidence(hits)
    snapshot = snapshot_from_hits(hits)
    base = {"confidence": conf.score, "signals": conf.signals, "snapshot": snapshot,
            "top": hits[0].get("rerank_score", 0.0) if hits else 0.0}

    # 置信度闸(原机械闸升级):四信号加权总分低于校准阈值 → 证据弱,拒答落池
    if conf.score < settings.evidence_confidence_threshold:
        return {**base, "sufficient": False, "source": "retrieval_low_conf",
                "reason": f"证据置信度不足(conf={conf.score:.3f})", "citations": []}

    # 语义闸:生成前自评证据够不够
    ev_texts = [f"{h['question']} {h['answer']}" for h in hits]
    chk = await selfcheck.check_sufficient(query, ev_texts)
    if not chk["useful"]:
        return {**base, "sufficient": False, "source": "self_check",
                "reason": chk["reason"], "citations": []}

    arranged = retrieval.arrange_head_tail(hits)
    citations, evidence_text = build_citations(arranged)
    return {**base, "sufficient": True, "evidence": evidence_text,
            "citations": citations}

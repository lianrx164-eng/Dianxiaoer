"""检索 + 两道生成前证据闸的统一入口(工具 query_faq 与图 forced_rag 共用)。"""
from app.config import settings
from app.core import query_understanding, retrieval, selfcheck


async def gather_evidence(query_raw: str, category: str | None = None) -> dict:
    """理解 → 混合检索 → 检索分闸 → 自评闸 → 首尾组装。

    返回字段:sufficient / source / reason / citations / evidence / top。
    source 仅在 insufficient 时有意义:retrieval_low_conf | self_check。
    """
    u = await query_understanding.understand(query_raw)
    query = u["standard"]
    bm25_query = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    hits = await retrieval.search_knowledge(
        query, strategy="hybrid_rerank", category=category, bm25_query=bm25_query)
    if category and not hits:
        # 模型给的分类名不是库里的精确 category 时,退回全库检索,避免误拒答
        hits = await retrieval.search_knowledge(
            query, strategy="hybrid_rerank", bm25_query=bm25_query)

    top = hits[0].get("rerank_score", 0.0) if hits else 0.0
    if not hits or top < settings.rerank_min_score:
        return {"sufficient": False, "source": "retrieval_low_conf",
                "reason": f"检索证据不足(top={top:.3f})", "citations": [], "top": top}

    ev_texts = [f"{h['question']} {h['answer']}" for h in hits]
    chk = await selfcheck.check_sufficient(query, ev_texts)
    if not chk["useful"]:
        return {"sufficient": False, "source": "self_check",
                "reason": chk["reason"], "citations": [], "top": top}

    arranged = retrieval.arrange_head_tail(hits)
    citations = [
        {"n": i + 1, "id": h["id"], "chunk_id": h["id"],
         "section_path": h["section_path"], "question": h["question"],
         "answer": h["answer"], "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {"sufficient": True, "evidence": evidence, "citations": citations, "top": top}

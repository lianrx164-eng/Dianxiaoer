import asyncio

from app.config import settings
from app.core import embeddings, rerank
from app.kb import milvus_client


def arrange_head_tail(items: list) -> list:
    """最相关放首、次相关放尾,其余按序居中(缓解 lost-in-the-middle)。"""
    if len(items) <= 2:
        return items
    return [items[0], *items[2:], items[1]]


async def search_knowledge(
    query: str, strategy: str = "hybrid_rerank", top_k: int | None = None,
    category: str | None = None, client=None, collection: str = "knowledge",
    bm25_query: str | None = None, min_score: float | None = None,
) -> list[dict]:
    """strategy: vector | bm25 | hybrid | hybrid_rerank。

    dense 始终用标准问法 `query`;`bm25_query`(标准问法 + 同义词扩展)只作用于
    BM25 那一路。hybrid_rerank 时返回值带 `rerank_score`。`min_score` 是 ch03
    兼容过滤,按最终分数(有 rerank_score 用它)截断。"""
    if strategy not in {"vector", "bm25", "hybrid", "hybrid_rerank"}:
        raise ValueError(f"unknown strategy: {strategy}")
    top_k = top_k or settings.rerank_top_k
    text = bm25_query or query
    client = client or milvus_client.get_client()
    await asyncio.to_thread(milvus_client.ensure_collection, client, collection=collection)

    if strategy == "vector":
        vec = await embeddings.embed_query(query)
        hits = await asyncio.to_thread(
            milvus_client.dense_search,
            client, vec, top_k, category=category, collection=collection)
    elif strategy == "bm25":
        hits = await asyncio.to_thread(
            milvus_client.bm25_search,
            client, text, top_k, category=category, collection=collection)
    elif strategy == "hybrid":
        vec = await embeddings.embed_query(query)
        hits = await asyncio.to_thread(
            milvus_client.hybrid_search,
            client, vec, text, top_k, recall=settings.recall_top_k,
            category=category, collection=collection)
    else:
        vec = await embeddings.embed_query(query)
        recall_hits = await asyncio.to_thread(
            milvus_client.hybrid_search,
            client, vec, text, settings.recall_top_k,
            recall=settings.recall_top_k, category=category, collection=collection)
        docs = [f"{h['question']} {h['answer']}" for h in recall_hits]
        ranked = await rerank.rerank(query, docs, top_n=top_k)
        hits = []
        for idx, score in ranked:
            h = dict(recall_hits[idx])
            h["rerank_score"] = score
            hits.append(h)

    if min_score is not None:
        hits = [h for h in hits
                if h.get("rerank_score", h.get("score", 0.0)) >= min_score]
    return hits

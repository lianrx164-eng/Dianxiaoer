import pytest

from app.core import retrieval


def test_arrange_head_tail():
    out = retrieval.arrange_head_tail(["a", "b", "c", "d", "e"])
    assert out[0] == "a" and out[-1] == "b"
    assert set(out[1:-1]) == {"c", "d", "e"}


def test_arrange_head_tail_short():
    assert retrieval.arrange_head_tail(["a", "b"]) == ["a", "b"]


@pytest.mark.asyncio
async def test_strategy_vector(monkeypatch):
    async def fake_embed(q):
        return [0.1] * 1024

    called = {}

    def fake_dense(client, vec, top_k, category=None, collection="knowledge"):
        called["dense"] = True
        return [{"id": 1, "score": 0.9, "question": "q", "answer": "a",
                 "section_path": "p", "content_type": "faq", "category": "运费"}]

    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.dense_search", fake_dense)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    out = await retrieval.search_knowledge("运费", strategy="vector", client=object())
    assert called.get("dense") and out[0]["id"] == 1


@pytest.mark.asyncio
async def test_strategy_hybrid_rerank(monkeypatch):
    async def fake_embed(q):
        return [0.1] * 1024

    def fake_hybrid(client, vec, text, top_k, recall=50, category=None,
                    collection="knowledge"):
        return [
            {"id": 1, "score": 0.5, "question": "运费", "answer": "满99包邮",
             "section_path": "p1", "content_type": "faq", "category": "运费"},
            {"id": 2, "score": 0.4, "question": "型号", "answer": "Pro自动清理",
             "section_path": "p2", "content_type": "manual", "category": "商品"},
        ]

    async def fake_rerank(q, docs, top_n=None):
        return [(1, 0.95), (0, 0.2)]

    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr("app.core.rerank.rerank", fake_rerank)
    out = await retrieval.search_knowledge("Pro 型号", strategy="hybrid_rerank",
                                           client=object())
    assert out[0]["id"] == 2 and out[0]["rerank_score"] == 0.95


@pytest.mark.asyncio
async def test_bm25_uses_bm25_query_only(monkeypatch):
    seen = {}

    def fake_bm25(client, text, top_k, category=None, collection="knowledge"):
        seen["text"] = text
        return []

    monkeypatch.setattr("app.kb.milvus_client.bm25_search", fake_bm25)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    await retrieval.search_knowledge("标准问法", bm25_query="标准问法 同义词",
                                     strategy="bm25", client=object())
    assert seen["text"] == "标准问法 同义词"

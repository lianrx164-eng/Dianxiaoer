import httpx

from app.config import settings


async def _post(url, json, headers, timeout):
    async with httpx.AsyncClient() as c:
        return await c.post(url, json=json, headers=headers, timeout=timeout)


async def rerank(query: str, docs: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """调百炼原生 rerank(qwen3-rerank)。返回 [(原始索引, 相关分)] 按分降序,截断 top_n。

    百炼 rerank 不是 OpenAI 协议:请求体是 {model, input:{query,documents}, parameters},
    结果是 output.results[].index / relevance_score,故手写请求、不用 openai 客户端。"""
    if not docs:
        return []
    payload = {"model": settings.rerank_model,
               "input": {"query": query, "documents": docs},
               "parameters": {"top_n": top_n or len(docs), "return_documents": False}}
    key = settings.rerank_api_key or settings.embed_api_key
    resp = await _post(settings.rerank_url, payload,
                       {"Authorization": f"Bearer {key}"}, 60)
    resp.raise_for_status()
    results = resp.json()["output"]["results"]
    ranked = sorted(((r["index"], float(r["relevance_score"])) for r in results),
                    key=lambda x: x[1], reverse=True)
    return ranked[:top_n] if top_n else ranked

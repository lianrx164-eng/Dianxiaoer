from openai import AsyncOpenAI

from app.config import settings

# 阿里云百炼 text-embedding-v4 的 OpenAI 兼容接口单次 input 最多 10 条
EMBED_BATCH_SIZE = 10


def _client() -> AsyncOpenAI:
    """直连嵌入上游(阿里云百炼兼容网关,OpenAI 协议)。"""
    return AsyncOpenAI(base_url=settings.embed_base_url, api_key=settings.embed_api_key)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    client = _client()
    out: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        resp = await client.embeddings.create(model=settings.embed_model, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]

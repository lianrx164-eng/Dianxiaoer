from openai import AsyncOpenAI

from app.config import settings


def _client() -> AsyncOpenAI:
    """直连嵌入上游(阿里云百炼兼容网关,OpenAI 协议)。"""
    return AsyncOpenAI(base_url=settings.embed_base_url, api_key=settings.embed_api_key)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = await _client().embeddings.create(model=settings.embed_model, input=texts)
    return [d.embedding for d in resp.data]


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]

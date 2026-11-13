"""冒烟:直连阿里云百炼兼容网关的嵌入接口,验证连通与维度。
需 .env 里填好 EMBED_API_KEY。
运行:PYTHONPATH=. uv run python scripts/smoke_embed.py"""
import asyncio
import sys

from openai import AsyncOpenAI

from app.config import settings


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = AsyncOpenAI(base_url=settings.embed_base_url, api_key=settings.embed_api_key)
    resp = await client.embeddings.create(
        model=settings.embed_model, input=["邮费是多少", "运费怎么算"]
    )
    dims = [len(d.embedding) for d in resp.data]
    print(f"model={settings.embed_model} 返回 {len(resp.data)} 条向量,维度={dims}")
    assert dims and dims[0] == 1024, f"期望维度 1024,实际 {dims}"
    print("✅ 冒烟通过:直连百炼兼容网关,嵌入维度 1024")


if __name__ == "__main__":
    asyncio.run(main())

from app.config import settings
from app.core import embeddings
from app.kb import milvus_client


async def search_knowledge(
    query: str, top_k: int | None = None,
    min_score: float | None = None, client=None,
) -> list[dict]:
    top_k = top_k or settings.retrieval_top_k
    min_score = settings.retrieval_min_score if min_score is None else min_score
    vector = await embeddings.embed_query(query)
    if client is not None:
        milvus_client.ensure_collection(client)
        hits = milvus_client.search(client, vector, top_k)
    else:
        with milvus_client.session() as owned:
            hits = milvus_client.search(owned, vector, top_k)
    return [h for h in hits if h["score"] >= min_score]

from contextlib import contextmanager

from pymilvus import (
    AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker,
)

from app.config import settings

COLLECTION = "knowledge"
DIM = 1024

_OUTPUT = ["question", "answer", "section_path", "content_type", "category"]

_CLIENT: MilvusClient | None = None
_ensured: set[str] = set()


def get_client(uri: str | None = None) -> MilvusClient:
    """默认返回进程内单例(连 Lite 文件);传 uri(测试)返回独立客户端。"""
    global _CLIENT
    if uri is not None:
        return MilvusClient(uri=uri)
    if _CLIENT is None:
        _CLIENT = MilvusClient(uri=settings.milvus_uri)
    return _CLIENT


@contextmanager
def session(uri: str | None = None):
    """一次请求/一段逻辑的客户端生命周期:确保集合存在,退出时关闭。"""
    global _CLIENT
    client = get_client(uri)
    try:
        ensure_collection(client)
        yield client
    finally:
        client.close()
        if client is _CLIENT:
            _CLIENT = None
            _ensured.clear()


def _schema_compatible(client: MilvusClient, collection: str) -> bool:
    fields = {f.get("name") for f in client.describe_collection(collection).get("fields", [])}
    return {"id", "dense", "text", "sparse"} <= fields


def ensure_collection(client: MilvusClient, collection: str = COLLECTION) -> None:
    """幂等建集合:dense(COSINE) + text(jieba analyzer) + sparse(BM25 Function) + 标量字段。
    单例默认集合只实际执行一次;不同集合名(测试临时库)各自 ensure。
    旧版 ch03 集合(只有 vector 字段)会被识别为不兼容并重建。"""
    if client is _CLIENT and collection in _ensured:
        return
    if client.has_collection(collection):
        if not _schema_compatible(client, collection):
            client.drop_collection(collection)
        else:
            client.load_collection(collection)
            if client is _CLIENT:
                _ensured.add(collection)
            return

    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=16384,
                     enable_analyzer=True, analyzer_params={"tokenizer": "jieba"})
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("question", DataType.VARCHAR, max_length=2048)
    schema.add_field("answer", DataType.VARCHAR, max_length=8192)
    schema.add_field("section_path", DataType.VARCHAR, max_length=512)
    schema.add_field("content_type", DataType.VARCHAR, max_length=32)
    schema.add_field("category", DataType.VARCHAR, max_length=255)
    schema.add_function(Function(
        name="text_bm25", input_field_names=["text"],
        output_field_names=["sparse"], function_type=FunctionType.BM25,
    ))
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                           metric_type="BM25")
    client.create_collection(collection, schema=schema, index_params=index_params)
    client.load_collection(collection)
    if client is _CLIENT:
        _ensured.add(collection)


def _normalize_row(row: dict) -> dict:
    question = row.get("question", "")
    answer = row.get("answer", "")
    category = row.get("category", "")
    return {
        "id": row["id"],
        "dense": row.get("dense") or row["vector"],
        "text": row.get("text") or f"{category}\n{question}\n{answer}",
        "question": question,
        "answer": answer,
        "section_path": row.get("section_path", ""),
        "content_type": row.get("content_type", ""),
        "category": category,
    }


def upsert_vectors(client: MilvusClient, rows: list[dict],
                   collection: str = COLLECTION) -> None:
    if rows:
        client.upsert(collection, [_normalize_row(r) for r in rows])


def flush(client: MilvusClient, collection: str = COLLECTION) -> None:
    client.flush(collection)


def _cat_expr(category: str | None) -> str:
    return f'category == "{category}"' if category else ""


def _hit(h: dict) -> dict:
    e = h["entity"]
    return {"id": h["id"], "score": float(h["distance"]),
            "question": e.get("question", ""), "answer": e.get("answer", ""),
            "section_path": e.get("section_path", ""),
            "content_type": e.get("content_type", ""),
            "category": e.get("category", "")}


def dense_search(client: MilvusClient, vector: list[float], top_k: int,
                 category: str | None = None,
                 collection: str = COLLECTION) -> list[dict]:
    res = client.search(collection, data=[vector], anns_field="dense", limit=top_k,
                        output_fields=_OUTPUT, search_params={"metric_type": "COSINE"},
                        filter=_cat_expr(category))
    # 实测 Lite COSINE 的 distance 即相似度(identical 向量=1.0),直接当分数用
    return [_hit(h) for h in res[0]]


def bm25_search(client: MilvusClient, text: str, top_k: int,
                category: str | None = None,
                collection: str = COLLECTION) -> list[dict]:
    res = client.search(collection, data=[text], anns_field="sparse", limit=top_k,
                        output_fields=_OUTPUT, search_params={"metric_type": "BM25"},
                        filter=_cat_expr(category))
    return [_hit(h) for h in res[0]]


def hybrid_search(client: MilvusClient, vector: list[float], text: str, top_k: int,
                  recall: int = 50, category: str | None = None,
                  collection: str = COLLECTION) -> list[dict]:
    expr = _cat_expr(category)
    dense_req = AnnSearchRequest(data=[vector], anns_field="dense",
                                 param={"metric_type": "COSINE"}, limit=recall, expr=expr)
    sparse_req = AnnSearchRequest(data=[text], anns_field="sparse",
                                  param={"metric_type": "BM25"}, limit=recall, expr=expr)
    res = client.hybrid_search(collection, reqs=[dense_req, sparse_req], ranker=RRFRanker(),
                               limit=top_k, output_fields=_OUTPUT)
    return [_hit(h) for h in res[0]]


def search(client: MilvusClient, vector: list[float], top_k: int,
           collection: str = COLLECTION) -> list[dict]:
    """ch03 兼容出口:dense 单路。新代码用 dense_search/hybrid_search。"""
    return dense_search(client, vector, top_k, collection=collection)


def count(client: MilvusClient, collection: str = COLLECTION) -> int:
    return client.query(collection, filter="id >= 0", output_fields=["count(*)"])[0]["count(*)"]


def drop(client: MilvusClient, collection: str) -> None:
    if client.has_collection(collection):
        client.drop_collection(collection)

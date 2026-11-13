from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core import jobs as jobs_core
from app.core import retrieval
from app.db import repository
from app.kb import chunking, dedup, documents, dualwrite, milvus_client, sources

router = APIRouter(prefix="/api/kb")


class PreviewRequest(BaseModel):
    markdown: str | None = None
    filename: str | None = None
    content_type: str | None = None
    chunk_size: int = 400
    overlap: int = 60
    table_max_rows: int = 10


class IngestRequest(PreviewRequest):
    vectorize: bool = True


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


def _resolve(req: PreviewRequest) -> tuple[str, str]:
    if req.filename is not None:
        if req.filename not in sources.SOURCE_TYPES:
            raise HTTPException(status_code=400, detail="文件名不在建库材料清单里")
        md = (sources.KB_DIR / req.filename).read_text(encoding="utf-8")
        return md, sources.SOURCE_TYPES[req.filename]
    if not req.markdown or not req.markdown.strip():
        raise HTTPException(status_code=400, detail="正文不能为空")
    if req.content_type not in sources.CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="content_type 不合法")
    return req.markdown, req.content_type


def _chunk_view(c: documents.Chunk, duplicate: bool) -> dict:
    return {
        "category": c.category,
        "questions": c.questions,
        "answer": c.answer,
        "section_path": c.section_path,
        "content_type": c.content_type,
        "is_key_clause": c.is_key_clause,
        "is_table": chunking.is_table_block(c.answer),
        "duplicate": duplicate,
    }


def _build(req: PreviewRequest, md: str, ctype: str) -> list[documents.Chunk]:
    return documents.build_chunks(
        md, ctype,
        chunk_size=req.chunk_size, overlap=req.overlap, table_max_rows=req.table_max_rows,
    )


@router.get("/overview")
async def overview() -> dict:
    try:
        stats = await repository.knowledge_stats()
    except Exception:
        stats = None
    try:
        staging = await repository.staging_stats()
    except Exception:
        staging = None

    milvus_count = None
    try:
        with milvus_client.session() as client:
            milvus_count = milvus_client.count(client)
    except Exception:
        pass
    consistent = dualwrite.consistency(
        stats["total"] if stats is not None else None, milvus_count
    )

    materials = []
    for fname, ctype in sources.SOURCE_TYPES.items():
        try:
            md = (sources.KB_DIR / fname).read_text(encoding="utf-8")
            chunks = documents.build_chunks(md, ctype)
            materials.append({
                "filename": fname, "content_type": ctype, "chunks": len(chunks),
                "key_clauses": sum(c.is_key_clause for c in chunks),
                "table_blocks": sum(1 for c in chunks if chunking.is_table_block(c.answer)),
            })
        except Exception as exc:
            materials.append({"filename": fname, "content_type": ctype, "error": str(exc)})

    try:
        jobs = jobs_core.status_all()
    except Exception:
        jobs = None

    try:
        recent = [
            {"id": c.id, "questions": c.questions, "answer": c.answer[:80],
             "content_type": c.content_type, "vectorize_status": c.vectorize_status}
            for c in await repository.list_recent_chunks(10)
        ]
    except Exception:
        recent = []

    return {
        "knowledge": stats, "staging": staging, "milvus_count": milvus_count,
        "consistent": consistent, "materials": materials, "jobs": jobs, "recent": recent,
    }


@router.post("/preview")
async def preview(req: PreviewRequest) -> dict:
    md, ctype = _resolve(req)
    chunks = _build(req, md, ctype)
    existing = {dedup.fingerprint(q, a) for q, a in await repository.list_chunk_pairs()}
    seen = set(existing)
    views = []
    for c in chunks:
        fp = dedup.fingerprint(c.questions, c.answer)
        duplicate = fp in seen
        if not duplicate:
            seen.add(fp)
        views.append(_chunk_view(c, duplicate))
    return {
        "content_type": ctype,
        "chunks": views,
        "counts": {
            "chunks": len(views),
            "key_clauses": sum(v["is_key_clause"] for v in views),
            "table_blocks": sum(v["is_table"] for v in views),
            "duplicates": sum(v["duplicate"] for v in views),
        },
    }


@router.post("/ingest")
async def ingest(req: IngestRequest) -> dict:
    md, ctype = _resolve(req)
    chunks = _build(req, md, ctype)
    ids, skipped = await dualwrite.write_new_chunks(chunks)

    vectorized = 0
    if req.vectorize and ids:
        try:
            with milvus_client.session() as client:
                vectorized = await dualwrite.vectorize_pending(client)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"已入库 {len(ids)} 块(pending),补跑一次即可:向量化失败:{exc}",
            ) from exc
    return {"inserted": len(ids), "skipped": skipped,
            "vectorized": vectorized, "ids": ids}


@router.post("/vectorize")
async def vectorize() -> dict:
    try:
        with milvus_client.session() as client:
            n = await dualwrite.vectorize_pending(client)
            total = milvus_client.count(client)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"向量化失败,补跑一次即可:{exc}") from exc
    return {"vectorized": n, "milvus_count": total}


@router.post("/reset")
async def reset() -> dict:
    """清库重建:drop Milvus 集合 + TRUNCATE knowledge_chunks。
    Milvus Lite 单进程独占数据目录,所以清库必须在本进程内做,不能走 make 子进程。"""
    try:
        # 清库不能用 session():它会先 ensure_collection,旧 schema 会被拦下
        client = milvus_client.get_client()
        try:
            milvus_client.drop(client, milvus_client.COLLECTION)
        finally:
            milvus_client.close_client(client)
        await repository.truncate_knowledge_chunks()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"清库失败:{exc}") from exc
    return {"reset": True}


@router.post("/search")
async def search(req: SearchRequest) -> dict:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="检索问题不能为空")
    try:
        hits = await retrieval.search_knowledge(req.query, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"嵌入或向量库不可用:{exc}") from exc
    return {"hits": hits}


@router.get("/staging")
async def staging_list() -> dict:
    try:
        stats = await repository.staging_stats()
        items = [
            {"id": s.id, "batch_no": s.batch_no, "source_ref": s.source_ref,
             "question": s.question, "answer": s.answer, "status": s.status}
            for s in await repository.list_staging()
        ]
        return {"stats": stats, "items": items}
    except Exception:
        return {"stats": None, "items": []}

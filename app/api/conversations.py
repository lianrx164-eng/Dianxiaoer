"""ch07 会话侧栏只读接口:会话列表 + 历史回载。不碰写路径。"""
from fastapi import APIRouter, HTTPException, Query

from app.db import repository

router = APIRouter()


@router.get("/api/conversations")
async def list_conversations(user_id: str = Query(min_length=1)):
    return {"items": await repository.list_conversations(user_id)}


@router.get("/api/conversations/{conversation_id}/messages")
async def conversation_messages(conversation_id: int):
    conv = await repository.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    rows = await repository.list_dialog_messages(conversation_id)
    return {"items": [
        {"role": r.role, "content": r.content,
         "created_at": r.created_at.isoformat() if r.created_at else None}
        for r in rows
    ]}

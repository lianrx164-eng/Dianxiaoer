"""接口层错误映射:异常 → 错误类别 → 文案/状态码,chat(SSE)与 agent(JSON)共用。"""
from sqlalchemy.exc import SQLAlchemyError

from app.core.agent import ConversationNotFound

NOT_FOUND = "not_found"
DB = "db"
UPSTREAM = "upstream"

ERROR_MESSAGES: dict[str, str] = {
    NOT_FOUND: "会话不存在",
    DB: "数据库暂时不可用,请稍后重试",
    UPSTREAM: "上游模型暂时不可用,请稍后重试",
}
HTTP_STATUS: dict[str, int] = {NOT_FOUND: 404, DB: 503, UPSTREAM: 502}


def classify(exc: BaseException) -> str:
    if isinstance(exc, ConversationNotFound):
        return NOT_FOUND
    if isinstance(exc, SQLAlchemyError):
        return DB
    return UPSTREAM

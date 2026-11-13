import logging

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

logger = logging.getLogger(__name__)


class SessionStore:
    """内存会话存储,session_id -> 消息列表。本章不做持久化。"""

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}

    def get(self, session_id: str) -> list[BaseMessage]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, *messages: BaseMessage) -> None:
        self._sessions.setdefault(session_id, []).extend(messages)


def trim_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        allow_partial=False,
    )
    if not trimmed and messages:
        # 预算放不下任何完整的轮次时,至少保留最新一条,不能把当前问题裁没
        logger.warning(
            "token 预算 %s 小于单条消息,仅保留最新一条(size=%s)", max_tokens, len(messages)
        )
        trimmed = [messages[-1]]
    return trimmed

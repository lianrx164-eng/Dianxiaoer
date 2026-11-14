from typing import Any

from langchain_openai import ChatOpenAI

from app.config import settings


def get_chat_model(
    streaming: bool = False, extra_body: dict[str, Any] | None = None,
    temperature: float = 0.3, model: str | None = None,
) -> ChatOpenAI:
    """直连聊天上游。地址、模型名、密钥全从 .env 来,换上游不动这里。
    model 显式覆盖模型名(意图识别可配更大模型求准),None 时回落 settings.chat_model。
    ch09:stream_usage=True 让流式调用也回传 token usage(Langfuse 按意图统计不漏账)。"""
    return ChatOpenAI(
        model=model or settings.chat_model,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        streaming=streaming,
        stream_usage=True,
        temperature=temperature,
        timeout=60,
        max_retries=1,
        extra_body=extra_body,
    )


def thinking_body() -> dict[str, Any] | None:
    return ({"thinking": {"type": settings.extract_thinking}}
            if settings.extract_thinking else None)


def structured_model(schema, temperature: float = 0.3, model=None):
    """结构化输出统一入口:扁平 schema + function_calling + 关闭上游思考模式。"""
    if model is not None:
        return model.with_structured_output(schema, method="function_calling")
    return get_chat_model(
        extra_body=thinking_body(), temperature=temperature
    ).with_structured_output(schema, method="function_calling")

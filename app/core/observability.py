"""ch09 可观测:Phoenix 挂载与 trace 标注,全部可选降级。

Phoenix 本地自部署(`phoenix serve`,SQLite,不需要 Docker),OpenInference 的
LangChain 插桩在进程内挂一次,覆盖 LangGraph 全部节点、LLM、工具与检索 span。
PHOENIX_ENDPOINT 为空时完全不启用:不 import、不插桩、不导出——观测是增强,不是依赖。
"""
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar

from app.config import settings

logger = logging.getLogger(__name__)

_enabled_impl = False
_tracer_provider = None
_tracer = None
_root_span: ContextVar = ContextVar("mewhelp_root_span", default=None)


def phoenix_enabled() -> bool:
    return bool(settings.phoenix_endpoint)


def _instrument() -> None:
    """进程内挂一次 OpenInference LangChain 插桩(独立出来便于测试替换)。"""
    global _enabled_impl, _tracer_provider, _tracer
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry import trace
    from phoenix.otel import register

    _tracer_provider = register(
        project_name=settings.phoenix_project,
        endpoint=settings.phoenix_endpoint,
        batch=True,
    )
    LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
    _tracer = trace.get_tracer("mewhelp")
    _enabled_impl = True


def attach_observability(graph):
    """启动时插桩一次;未配置时原样返回,不 import Phoenix。"""
    if not phoenix_enabled():
        return graph
    if not _enabled_impl:
        _instrument()
    return graph


@asynccontextmanager
async def turn_span(conversation_id: int):
    """一轮对话的根 span:session.id 归组 + 供 tag_intent 写 trace 级属性。"""
    if not _enabled_impl:
        yield None
        return
    from phoenix.otel import using_session

    with using_session(session_id=str(conversation_id)):
        with _tracer.start_as_current_span(
                "chat_turn", attributes={"conversation.id": conversation_id}) as span:
            token = _root_span.set(span)
            try:
                yield span
            finally:
                _root_span.reset(token)


def tag_intent(intent: str, confidence: float) -> None:
    """把意图写进当前节点 span(Cost Control 按意图分堆的钩子)。
    根 span 可能在独立上下文里取不到,故运行结束时由 runtime 再用 state 补写一次。
    任何异常静默降级——观测失败不许影响业务流。"""
    if not _enabled_impl:
        return
    try:
        from opentelemetry import trace

        spans = [trace.get_current_span(), _root_span.get()]
        for span in spans:
            if span is not None and getattr(span, "is_recording", lambda: False)():
                span.set_attribute("intent", intent)
                span.set_attribute("intent.confidence", float(confidence))
    except Exception:
        logger.debug("phoenix tag_intent 失败(已忽略)", exc_info=True)


def flush() -> None:
    """进程退出前把缓冲的 span 刷给 Phoenix;未启用或失败都静默。"""
    if _tracer_provider is None:
        return
    try:
        _tracer_provider.force_flush()
    except Exception:
        logger.debug("phoenix flush 失败(已忽略)", exc_info=True)

"""ch09 可观测:Langfuse 挂载与 trace 标注,全部可选降级。

课程姿势:设三个环境变量 + 图编译时挂一次回调,节点业务代码零侵入。
项目配置经 pydantic-settings(.env),不依赖 os.environ,故显式传参初始化单例。
所有对外函数在未配置 Langfuse 时必须是安全 no-op——观测是增强,不是依赖。
"""
import logging

from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def langfuse_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key
                and settings.langfuse_base_url)


def _init_client():
    """初始化(或复用)Langfuse 单例。v3 构造参 base_url(host 是废弃别名)。"""
    global _client
    if _client is None:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
        )
    return _client


def _make_handler():
    from langfuse.langchain import CallbackHandler
    return CallbackHandler()


def get_langfuse():
    """启用时返回已初始化 client(cost 脚本用 client.api 查询);未启用返回 None。"""
    if not langfuse_enabled():
        return None
    return _init_client()


def flush() -> None:
    """进程退出前把缓冲的 span 刷给 Langfuse;未启用或失败都静默。"""
    if _client is None:
        return
    try:
        _client.flush()
    except Exception:
        logger.debug("langfuse flush 失败(已忽略)", exc_info=True)


def attach_observability(graph):
    """编译后挂一次 Langfuse 回调,全图自动 trace。
    未配置时原样返回——不挂、不 import langfuse。"""
    if not langfuse_enabled():
        return graph
    _init_client()
    return graph.with_config({"callbacks": [_make_handler()]})


def tag_intent(intent: str, confidence: float) -> None:
    """把意图写进当前 trace 的 metadata + tag(Cost Control 按意图分堆的钩子)。
    任何异常静默降级——观测失败不许影响业务流。"""
    if not langfuse_enabled():
        return
    try:
        _init_client().update_current_trace(
            metadata={"intent": intent, "intent_confidence": confidence},
            tags=[f"intent:{intent}"],
        )
    except Exception:
        logger.debug("langfuse tag_intent 失败(已忽略)", exc_info=True)

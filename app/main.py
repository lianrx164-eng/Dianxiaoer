import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.actions import router as actions_router
from app.api.admin import router as admin_router
from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.extract import router as extract_router
from app.api.feedback import router as feedback_router
from app.api.jobs import router as jobs_router
from app.api.kb import router as kb_router
from app.api.rageval import router as rageval_router
from app.api.review import router as review_router
from app.graph import runtime

STATIC_DIR = Path(__file__).parent / "static"

# 应用自己的 INFO 日志(节点 trace)不被 root 的 WARNING 挡掉;显式挂 handler,避免
# uvicorn 先配好 logging 后 basicConfig 变成空操作
_app_log = logging.getLogger("app")
_app_log.setLevel(logging.INFO)
_app_log.propagate = False
if not _app_log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    _app_log.addHandler(_h)
# ch07 验收可观测:model_ctx / history_ctx / summary 落 log/app.log(log/ 已 gitignore)
if not any(isinstance(h, logging.FileHandler) for h in _app_log.handlers):
    Path("log").mkdir(exist_ok=True)
    _fh = logging.FileHandler(Path("log") / "app.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    _app_log.addHandler(_fh)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from app.tools import registry as tool_registry
    tool_registry.scan_builtin()   # ch08:内置工具服务启动时登记
    await runtime.init_graph()
    yield
    await runtime.close_graph()


app = FastAPI(title="MewHelp", version="0.1.0", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(extract_router)
app.include_router(agent_router)
app.include_router(kb_router)
app.include_router(jobs_router)
app.include_router(admin_router)
app.include_router(rageval_router)
app.include_router(feedback_router)
app.include_router(review_router)
app.include_router(actions_router)
app.include_router(conversations_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/kb", include_in_schema=False)
def kb_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "kb.html")


@app.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/rag-eval", include_in_schema=False)
def rag_eval_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "rageval.html")

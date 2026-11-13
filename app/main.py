from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.admin import router as admin_router
from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.api.jobs import router as jobs_router
from app.api.kb import router as kb_router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="MewHelp", version="0.1.0")
app.include_router(chat_router)
app.include_router(extract_router)
app.include_router(agent_router)
app.include_router(kb_router)
app.include_router(jobs_router)
app.include_router(admin_router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

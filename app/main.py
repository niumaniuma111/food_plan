"""
FastAPI application entry point.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    settings = get_settings()
    
    # Startup: Ensure data directories exist
    settings.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_docs_dir.mkdir(parents=True, exist_ok=True)
    settings.feedback_qa_dir.mkdir(parents=True, exist_ok=True)
    Path(settings.chroma.persist_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize BM25 index if not exists
    if not settings.bm25_index_path.exists():
        with open(settings.bm25_index_path, "w", encoding="utf-8") as f:
            json.dump([], f)
    
    # Load BM25 index into memory
    from app.core.bm25_retriever import BM25Retriever
    app.state.bm25_retriever = BM25Retriever(str(settings.bm25_index_path))
    
    yield
    
    # Shutdown: cleanup if needed


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="私人食谱与饮食规划师",
        description="基于 LangChain 的 RAG 系统，支持离线文档处理、在线混合检索与流式生成",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    from app.api.router_chat import router as chat_router
    from app.api.router_documents import router as documents_router
    from app.api.router_feedback import router as feedback_router
    
    app.include_router(chat_router, prefix="/api")
    app.include_router(documents_router, prefix="/api")
    app.include_router(feedback_router, prefix="/api")
    
    # Mount static files (frontend)
    frontend_dir = settings.project_root / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

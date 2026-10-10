"""FastAPI Application factory and lifecycle for VoicePilot."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from voicepilot.adapters.stt import get_stt_adapter
from voicepilot.adapters.tts import get_tts_adapter
from voicepilot.agent.graph import VoicePilotGraph
from voicepilot.agent.memory import get_session_store
from voicepilot.config import Settings, get_settings
from voicepilot.logging_config import setup_logging
from voicepilot.server.routes import router as http_router
from voicepilot.server.websocket import ws_router

_app_graph: VoicePilotGraph = None


def get_app_graph() -> VoicePilotGraph:
    """Return the initialized LangGraph instance."""
    global _app_graph
    if _app_graph is None:
        settings = get_settings()
        stt = get_stt_adapter(settings)
        tts = get_tts_adapter(settings)
        _app_graph = VoicePilotGraph(stt, tts, settings=settings)
    return _app_graph


def set_app_graph(graph: VoicePilotGraph) -> None:
    """Override the global app graph (useful for test mocks)."""
    global _app_graph
    _app_graph = graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    settings = get_settings()
    setup_logging(level=settings.LOG_LEVEL, redact=settings.REDACT_LOGS)

    # Initialize graph & adapters
    get_app_graph()

    # In demo mode, pre-seed the demo conversation fixtures
    if settings.DEMO_MODE:
        store = get_session_store()
        store.seed_demo_fixtures()

    yield

    # Clean shutdown


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Voice-first AI agent orchestrator with defensive audio validation and low-latency turn management.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(http_router)
    app.include_router(ws_router)

    # Mount static files and frontend
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def serve_index():
            index_file = static_dir / "index.html"
            if index_file.exists():
                return FileResponse(str(index_file))
            return {"status": "ok", "app": settings.APP_NAME}

    return app


app = create_app()

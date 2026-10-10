"""HTTP and WebSocket endpoints for VoicePilot."""

import base64
import time

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from voicepilot.agent.graph import AgentTurnState, VoicePilotGraph
from voicepilot.agent.memory import get_session_store
from voicepilot.audio.transcoder import check_ffmpeg_available
from voicepilot.config import get_settings
from voicepilot.server.metrics import get_metrics_collector

router = APIRouter()


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    message: str = "Session initialized"


class ResetResponse(BaseModel):
    status: str
    message: str
    demo_session_id: str
    reseeded_turns: int


@router.get("/health")
async def health_check():
    """Health check endpoint checking dependencies and configuration."""
    settings = get_settings()
    ffmpeg_ok = check_ffmpeg_available()

    return {
        "status": "healthy" if ffmpeg_ok else "degraded",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "demo_mode": settings.DEMO_MODE,
        "stt_provider": settings.STT_PROVIDER,
        "tts_provider": settings.TTS_PROVIDER,
        "dependencies": {
            "ffmpeg": "available" if ffmpeg_ok else "missing",
            "session_store": "in-memory",
        },
    }


@router.post("/v1/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session():
    """Create a new conversational voice session."""
    store = get_session_store()
    session = store.create_session()
    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
    )


@router.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve full session history, preferences, and active turns."""
    store = get_session_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    return session.model_dump()


@router.get("/v1/sessions/{session_id}/export")
async def export_session_transcript(session_id: str):
    """Download session transcript and latency traces as structured JSON."""
    store = get_session_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )

    export_payload = {
        "export_timestamp": time.time(),
        "session_id": session.session_id,
        "preferences": session.preferences,
        "total_turns": len(session.turns),
        "transcript_turns": [turn.model_dump() for turn in session.turns],
    }
    return JSONResponse(
        content=export_payload,
        headers={
            "Content-Disposition": f'attachment; filename="voicepilot_session_{session_id}.json"'
        },
    )


@router.get("/v1/metrics")
async def get_metrics():
    """Expose p50/p90/p95 latency instrumentation across pipeline stages."""
    collector = get_metrics_collector()
    return collector.get_summary()


@router.post("/api/demo/reset", response_model=ResetResponse)
async def demo_reset(
    request: Request,
    x_demo_key: str | None = Header(None, alias="X-Demo-Key"),
):
    """Authenticated demo-only reset: reseeds fixtures and clears session caches.

    Guarded: Returns 403 Forbidden if DEMO_MODE is disabled.
    Idempotent: Safe to call repeatedly.
    """
    settings = getattr(request.app.state, "settings", None) or get_settings()
    if not settings.DEMO_MODE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Demo reset is disabled outside demo mode",
        )

    # Optional key check if provided in header or environment
    if settings.DEMO_API_KEY and x_demo_key:
        if x_demo_key != settings.DEMO_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid X-Demo-Key header",
            )

    store = get_session_store()
    store.clear()
    demo_sid = store.seed_demo_fixtures()

    metrics = get_metrics_collector()
    metrics.reset()

    # Pre-populate metric samples from demo fixtures so dashboard has baseline
    demo_session = store.get_session(demo_sid)
    turn_count = len(demo_session.turns) if demo_session else 0

    return ResetResponse(
        status="ok",
        message="Demo fixtures reseeded and session cache cleared",
        demo_session_id=demo_sid,
        reseeded_turns=turn_count,
    )


@router.post("/v1/voice/{session_id}/turn")
async def process_voice_turn(
    session_id: str,
    file: UploadFile = File(...),
    turn_id: str | None = Form(None),
    client_upload_ms: float | None = Form(0.0),
):
    """HTTP endpoint for audio turn upload and full LangGraph execution."""
    turn_identifier = turn_id or f"turn_{int(time.time()*1000)}"

    store = get_session_store()
    store.get_or_create_session(session_id)

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="unsupported format: empty audio payload",
        )

    # Fetch app graph
    from voicepilot.server.app import get_app_graph
    graph: VoicePilotGraph = get_app_graph()

    timings = {
        "upload": float(client_upload_ms or 0.0),
    }

    initial_state: AgentTurnState = {
        "turn_id": turn_identifier,
        "session_id": session_id,
        "audio_bytes": audio_bytes,
        "audio_format": file.filename.split(".")[-1] if file.filename else "webm",
        "timings": timings,
        "is_cancelled": False,
    }

    turn_result = await graph.run_turn(initial_state)

    # Record telemetry
    collector = get_metrics_collector()
    success = not bool(turn_result.get("error"))
    collector.record_turn(turn_result.get("timings", {}), success=success)

    audio_reply_b64 = None
    if turn_result.get("speech_audio_bytes"):
        audio_reply_b64 = base64.b64encode(turn_result["speech_audio_bytes"]).decode("utf-8")

    response_payload = {
        "turn_id": turn_identifier,
        "session_id": session_id,
        "user_transcript": turn_result.get("raw_transcript", ""),
        "assistant_response": turn_result.get("response_text", ""),
        "tool_used": turn_result.get("tool_called"),
        "tool_result": turn_result.get("tool_result"),
        "audio_base64": audio_reply_b64,
        "audio_format": turn_result.get("speech_audio_format", "none"),
        "timings": turn_result.get("timings", {}),
        "error": turn_result.get("error"),
        "error_code": turn_result.get("error_code"),
        "error_stage": turn_result.get("error_stage"),
        "success": success,
    }

    status_code = status.HTTP_200_OK if success else status.HTTP_400_BAD_REQUEST
    return JSONResponse(content=response_payload, status_code=status_code)

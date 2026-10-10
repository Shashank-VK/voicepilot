"""WebSocket handler for real-time voice streaming, cancellation, and event envelopes."""

import asyncio
import base64
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from voicepilot.agent.graph import AgentTurnState, VoicePilotGraph
from voicepilot.agent.memory import get_session_store
from voicepilot.server.metrics import get_metrics_collector

logger = logging.getLogger("voicepilot.server.websocket")
ws_router = APIRouter()


class WebSocketTurnSession:
    """Manages audio chunk buffering, cancellation, and turn execution for a WebSocket."""

    def __init__(self, ws: WebSocket, session_id: str):
        self.ws = ws
        self.session_id = session_id
        self.active_turn_id: str | None = None
        self.cancelled_turn_ids: set[str] = set()
        self.audio_buffer: bytearray = bytearray()
        self.turn_start_time: float = 0.0

    async def emit_envelope(
        self,
        event_type: str,
        turn_id: str,
        stage: str | None = None,
        latency_ms: float = 0.0,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Send a standardized PRD event envelope to the client."""
        if turn_id in self.cancelled_turn_ids and event_type != "turn.cancelled":
            # Discard any stale events for cancelled turns
            return

        envelope = {
            "type": event_type,
            "turn_id": turn_id,
            "latency_ms": round(latency_ms, 1),
            "payload": payload or {},
        }
        if stage:
            envelope["stage"] = stage

        try:
            await self.ws.send_text(json.dumps(envelope))
        except Exception as exc:
            logger.debug("Failed sending WS envelope (%s): %s", event_type, exc)

    async def handle_cancel(self, turn_id: str) -> None:
        """Process turn cancellation requested by client."""
        self.cancelled_turn_ids.add(turn_id)
        self.audio_buffer.clear()
        if self.active_turn_id == turn_id:
            self.active_turn_id = None

        await self.emit_envelope(
            event_type="turn.cancelled",
            turn_id=turn_id,
            payload={"message": f"Turn {turn_id} cancelled by client"},
        )

    async def handle_audio_chunk(self, turn_id: str, data_b64: str) -> None:
        """Append base64 encoded audio chunk."""
        if turn_id in self.cancelled_turn_ids:
            return
        self.active_turn_id = turn_id
        raw_bytes = base64.b64decode(data_b64)
        self.audio_buffer.extend(raw_bytes)

    async def handle_audio_end(self, turn_id: str, client_upload_ms: float = 0.0) -> None:
        """Finalize incoming audio and execute the LangGraph turn."""
        if turn_id in self.cancelled_turn_ids:
            self.audio_buffer.clear()
            return

        audio_bytes = bytes(self.audio_buffer)
        self.audio_buffer.clear()

        from voicepilot.server.app import get_app_graph
        graph: VoicePilotGraph = get_app_graph()

        t_turn_start = time.perf_counter()
        timings: dict[str, float] = {"upload": client_upload_ms}

        # 1. Validation stage
        await self.emit_envelope("stage.started", turn_id, stage="validate_audio")

        initial_state: AgentTurnState = {
            "turn_id": turn_id,
            "session_id": self.session_id,
            "audio_bytes": audio_bytes,
            "timings": timings,
            "is_cancelled": turn_id in self.cancelled_turn_ids,
        }

        # Execute turn through compiled LangGraph
        result = await graph.run_turn(initial_state)

        if turn_id in self.cancelled_turn_ids:
            # Stale / cancelled turn
            return

        # Check for validation or STT error
        if result.get("error"):
            await self.emit_envelope(
                event_type="error",
                turn_id=turn_id,
                stage=result.get("error_stage", "unknown"),
                latency_ms=result.get("timings", {}).get(result.get("error_stage", "validate_audio"), 0.0),
                payload={
                    "code": result.get("error_code", "error"),
                    "message": result.get("error", "Unknown error"),
                    "recoverable_text": result.get("response_text", ""),
                },
            )
            # Record failed turn in telemetry
            get_metrics_collector().record_turn(result.get("timings", {}), success=False)
            return

        # Emit stage completions
        turn_timings = result.get("timings", {})

        await self.emit_envelope(
            event_type="stage.completed",
            turn_id=turn_id,
            stage="validate_audio",
            latency_ms=turn_timings.get("validate_audio", 0.0),
            payload={"valid": True},
        )

        await self.emit_envelope(
            event_type="stage.completed",
            turn_id=turn_id,
            stage="transcribe",
            latency_ms=turn_timings.get("stt", 0.0),
            payload={"text": result.get("raw_transcript", "")},
        )

        # Agent thinking / tool
        if result.get("tool_called"):
            await self.emit_envelope(
                event_type="stage.completed",
                turn_id=turn_id,
                stage="execute_tool",
                latency_ms=turn_timings.get("agent", 0.0),
                payload={
                    "tool": result.get("tool_called"),
                    "result": result.get("tool_result"),
                },
            )

        await self.emit_envelope(
            event_type="stage.completed",
            turn_id=turn_id,
            stage="generate_response",
            latency_ms=turn_timings.get("agent", 0.0),
            payload={"text": result.get("response_text", "")},
        )

        # TTS speech ready
        speech_bytes = result.get("speech_audio_bytes")
        if speech_bytes:
            audio_b64 = base64.b64encode(speech_bytes).decode("utf-8")
            fmt = "audio/wav" if result.get("speech_audio_format") == "wav" else "audio/mp3"
            await self.emit_envelope(
                event_type="speech.ready",
                turn_id=turn_id,
                stage="synthesize_speech",
                latency_ms=turn_timings.get("tts", 0.0),
                payload={"audio_base64": audio_b64, "format": fmt},
            )

        # Turn completed envelope
        total_ms = (time.perf_counter() - t_turn_start) * 1000.0
        turn_timings["total"] = total_ms

        await self.emit_envelope(
            event_type="turn.completed",
            turn_id=turn_id,
            latency_ms=total_ms,
            payload={
                "transcript": result.get("raw_transcript", ""),
                "response": result.get("response_text", ""),
                "tool_used": result.get("tool_called"),
                "tool_result": result.get("tool_result"),
                "timings": turn_timings,
            },
        )

        # Record metrics
        get_metrics_collector().record_turn(turn_timings, success=True)


@ws_router.websocket("/v1/voice/{session_id}")
async def voice_websocket_endpoint(websocket: WebSocket, session_id: str):
    """Bi-directional WebSocket streaming audio and turn events."""
    await websocket.accept()
    session_manager = WebSocketTurnSession(websocket, session_id)

    # Initialize session in store if needed
    get_session_store().get_or_create_session(session_id)

    try:
        while True:
            raw_msg = await websocket.receive_text()
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid JSON format"})
                )
                continue

            msg_type = msg.get("type")
            turn_id = msg.get("turn_id", f"turn_{int(time.time()*1000)}")

            if msg_type == "turn.cancel":
                await session_manager.handle_cancel(turn_id)

            elif msg_type == "audio.chunk":
                chunk_data = msg.get("data", "")
                await session_manager.handle_audio_chunk(turn_id, chunk_data)

            elif msg_type == "audio.end":
                client_upload_ms = float(msg.get("client_upload_ms", 0.0))
                # Run turn asynchronously so client can send turn.cancel if needed
                asyncio.create_task(
                    session_manager.handle_audio_end(turn_id, client_upload_ms)
                )

            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "time": time.time()}))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
    except Exception as exc:
        logger.error("WebSocket unhandled exception: %s", exc)

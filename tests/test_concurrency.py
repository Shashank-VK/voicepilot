"""Concurrency test ensuring multiple simultaneous voice turns run without race conditions."""

import asyncio
import io

import pytest
from httpx import ASGITransport, AsyncClient

from voicepilot.adapters.stt.mock import MockSTTAdapter
from voicepilot.adapters.tts.mock import MockTTSAdapter
from voicepilot.agent.graph import VoicePilotGraph
from voicepilot.agent.memory import SessionStore
from voicepilot.audio.fixtures import generate_speech_like_wav
from voicepilot.config import Settings
from voicepilot.server.app import create_app, set_app_graph


@pytest.mark.asyncio
async def test_concurrent_voice_turns():
    settings = Settings(DEMO_MODE=True, STT_PROVIDER="mock", TTS_PROVIDER="mock")
    store = SessionStore()
    stt = MockSTTAdapter(simulated_latency_ms=20.0)
    tts = MockTTSAdapter(simulated_latency_ms=20.0)
    graph = VoicePilotGraph(stt, tts, session_store=store, settings=settings)

    app = create_app(settings)
    set_app_graph(graph)

    audio_bytes = generate_speech_like_wav(duration_s=1.5)

    num_concurrent = 8

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Create N sessions concurrently
        session_tasks = [client.post("/v1/sessions") for _ in range(num_concurrent)]
        session_responses = await asyncio.gather(*session_tasks)

        session_ids = [res.json()["session_id"] for res in session_responses]
        assert len(set(session_ids)) == num_concurrent  # All unique session IDs

        # 2. Concurrently execute a voice turn for each session
        async def submit_turn(sid: str, index: int):
            files = {"file": (f"audio_{index}.wav", io.BytesIO(audio_bytes), "audio/wav")}
            data = {"turn_id": f"conc_turn_{index}"}
            res = await client.post(f"/v1/voice/{sid}/turn", files=files, data=data)
            return res

        turn_tasks = [submit_turn(sid, i) for i, sid in enumerate(session_ids)]
        turn_responses = await asyncio.gather(*turn_tasks)

        for i, res in enumerate(turn_responses):
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["session_id"] == session_ids[i]
            assert data["turn_id"] == f"conc_turn_{i}"

        # 3. Verify each session has exactly 1 turn recorded in memory
        for sid in session_ids:
            sess = store.get_session(sid)
            assert sess is not None
            assert len(sess.turns) == 1

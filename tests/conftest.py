"""Pytest shared fixtures and synthetic audio generation for VoicePilot."""

import pytest
from fastapi.testclient import TestClient

from voicepilot.adapters.stt.mock import MockSTTAdapter
from voicepilot.adapters.tts.mock import MockTTSAdapter
from voicepilot.agent.graph import VoicePilotGraph
from voicepilot.agent.memory import SessionStore
from voicepilot.audio.fixtures import (
    generate_clipped_wav,
    generate_mp4_audio,
    generate_noisy_wav,
    generate_short_wav,
    generate_silence_wav,
    generate_speech_like_wav,
    generate_webm_audio,
)
from voicepilot.config import Settings
from voicepilot.server.app import create_app, set_app_graph


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        APP_NAME="VoicePilot-Test",
        DEMO_MODE=True,
        DEMO_API_KEY="voicepilot-demo-key",
        STT_PROVIDER="mock",
        TTS_PROVIDER="mock",
        LLM_PROVIDER="mock",
        MIN_SPEECH_DURATION_SECONDS=0.5,
        MAX_AUDIO_SECONDS=30.0,
        MIN_SNR_DB=10.0,
        MAX_SILENCE_RATIO=0.85,
        MAX_CLIPPING_RATIO=0.05,
    )


@pytest.fixture
def mock_stt() -> MockSTTAdapter:
    return MockSTTAdapter(simulated_latency_ms=10.0)


@pytest.fixture
def mock_tts() -> MockTTSAdapter:
    return MockTTSAdapter(simulated_latency_ms=10.0)


@pytest.fixture
def test_session_store() -> SessionStore:
    return SessionStore()


@pytest.fixture
def test_graph(mock_stt, mock_tts, test_session_store, test_settings) -> VoicePilotGraph:
    return VoicePilotGraph(
        stt_adapter=mock_stt,
        tts_adapter=mock_tts,
        session_store=test_session_store,
        settings=test_settings,
    )


@pytest.fixture
def client(test_settings, test_graph) -> TestClient:
    app = create_app(test_settings)
    set_app_graph(test_graph)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def speech_wav() -> bytes:
    return generate_speech_like_wav(duration_s=2.0)


@pytest.fixture(scope="session")
def silence_wav() -> bytes:
    return generate_silence_wav(duration_s=2.0)


@pytest.fixture(scope="session")
def noisy_wav() -> bytes:
    return generate_noisy_wav(duration_s=2.0)


@pytest.fixture(scope="session")
def clipped_wav() -> bytes:
    return generate_clipped_wav(duration_s=2.0)


@pytest.fixture(scope="session")
def short_wav() -> bytes:
    return generate_short_wav(duration_s=0.15)


@pytest.fixture(scope="session")
def webm_audio() -> bytes:
    return generate_webm_audio(duration_s=1.5)


@pytest.fixture(scope="session")
def mp4_audio() -> bytes:
    return generate_mp4_audio(duration_s=1.5)

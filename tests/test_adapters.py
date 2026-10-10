"""Tests for STT and TTS adapters: Groq, Local/Mock, retry, and fallback logic."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from voicepilot.adapters.stt import get_stt_adapter
from voicepilot.adapters.stt.base import (
    ProviderAuthError,
    TranscriptionResult,
)
from voicepilot.adapters.stt.groq import GroqSTTAdapter
from voicepilot.adapters.stt.mock import MockSTTAdapter
from voicepilot.adapters.tts import get_tts_adapter
from voicepilot.adapters.tts.edge import EdgeTTSAdapter
from voicepilot.adapters.tts.mock import MockTTSAdapter
from voicepilot.config import Settings


@pytest.mark.asyncio
async def test_mock_stt_adapter(speech_wav):
    adapter = MockSTTAdapter(default_text="Testing VoicePilot STT", simulated_latency_ms=10.0)
    result = await adapter.transcribe(speech_wav)
    assert isinstance(result, TranscriptionResult)
    assert result.text == "Testing VoicePilot STT"
    assert result.latency_ms >= 5.0
    assert result.provider == "mock"


@pytest.mark.asyncio
async def test_groq_stt_missing_key_fallback(speech_wav):
    fallback = MockSTTAdapter(default_text="Fallback transcript")
    groq_adapter = GroqSTTAdapter(api_key="", fallback_adapter=fallback)
    result = await groq_adapter.transcribe(speech_wav)
    assert result.text == "Fallback transcript"
    assert result.provider == "mock"


@pytest.mark.asyncio
async def test_groq_stt_missing_key_no_fallback_raises(speech_wav):
    groq_adapter = GroqSTTAdapter(api_key="", fallback_adapter=None)
    with pytest.raises(ProviderAuthError) as exc_info:
        await groq_adapter.transcribe(speech_wav)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_groq_stt_mocked_success(speech_wav):
    groq_adapter = GroqSTTAdapter(api_key="mock-groq-key")

    mock_resp = httpx.Response(
        200,
        json={"text": "Summarize the Green Grid AI project", "language": "en"},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions"),
    )

    with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_resp)):
        result = await groq_adapter.transcribe(speech_wav)
        assert result.text == "Summarize the Green Grid AI project"
        assert result.provider == "groq"


@pytest.mark.asyncio
async def test_groq_stt_rate_limit_and_fallback(speech_wav):
    fallback = MockSTTAdapter(default_text="Rate limit fallback text")
    groq_adapter = GroqSTTAdapter(
        api_key="mock-groq-key",
        max_retries=1,
        fallback_adapter=fallback,
    )

    mock_resp_429 = httpx.Response(
        429,
        text="Rate limit exceeded",
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions"),
    )

    with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_resp_429)):
        result = await groq_adapter.transcribe(speech_wav)
        assert result.text == "Rate limit fallback text"
        assert result.provider == "mock"


@pytest.mark.asyncio
async def test_mock_tts_adapter():
    tts = MockTTSAdapter(simulated_latency_ms=10.0)
    result = await tts.synthesize("Hello from VoicePilot")
    assert result.provider == "mock"
    assert result.format == "wav"
    assert len(result.audio_bytes) > 44
    assert result.latency_ms >= 5.0


@pytest.mark.asyncio
async def test_edge_tts_fallback_on_network_error():
    fallback = MockTTSAdapter()
    edge = EdgeTTSAdapter(timeout_seconds=0.1, fallback_adapter=fallback)
    # Edge TTS will fail or timeout, invoking fallback
    result = await edge.synthesize("Testing fallback")
    assert result.provider == "mock"
    assert len(result.audio_bytes) > 44


def test_adapter_factories():
    settings = Settings(STT_PROVIDER="mock", TTS_PROVIDER="mock")
    stt = get_stt_adapter(settings)
    tts = get_tts_adapter(settings)
    assert isinstance(stt, MockSTTAdapter)
    assert isinstance(tts, MockTTSAdapter)

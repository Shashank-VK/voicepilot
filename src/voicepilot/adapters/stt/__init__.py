"""Speech-to-Text adapters package and factory."""

from voicepilot.adapters.stt.base import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderServiceError,
    ProviderTimeoutError,
    SpeechToTextAdapter,
    STTError,
    TranscriptionResult,
)
from voicepilot.adapters.stt.groq import GroqSTTAdapter
from voicepilot.adapters.stt.local import LocalWhisperSTTAdapter
from voicepilot.adapters.stt.mock import MockSTTAdapter
from voicepilot.config import Settings


def get_stt_adapter(settings: Settings) -> SpeechToTextAdapter:
    """Factory to instantiate the appropriate STT adapter based on configuration."""
    mock_adapter = MockSTTAdapter()

    if settings.STT_PROVIDER == "mock":
        return mock_adapter

    local_adapter = LocalWhisperSTTAdapter(
        model_size=settings.LOCAL_WHISPER_MODEL,
        device=settings.LOCAL_WHISPER_DEVICE,
        compute_type=settings.LOCAL_WHISPER_COMPUTE_TYPE,
        fallback_adapter=mock_adapter,
    )

    if settings.STT_PROVIDER == "local":
        return local_adapter

    if settings.STT_PROVIDER == "groq":
        return GroqSTTAdapter(
            api_key=settings.GROQ_API_KEY,
            timeout_seconds=settings.PROVIDER_TIMEOUT_SECONDS,
            max_retries=settings.STT_MAX_RETRIES,
            fallback_adapter=local_adapter if not settings.OFFLINE_MODE else mock_adapter,
        )

    return mock_adapter


__all__ = [
    "GroqSTTAdapter",
    "LocalWhisperSTTAdapter",
    "MockSTTAdapter",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderServiceError",
    "ProviderTimeoutError",
    "STTError",
    "SpeechToTextAdapter",
    "TranscriptionResult",
    "get_stt_adapter",
]

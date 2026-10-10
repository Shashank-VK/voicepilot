"""Base interface and error types for Speech-to-Text adapters."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class TranscriptionResult(BaseModel):
    """Normalized output from speech transcription."""

    text: str
    latency_ms: float
    provider: str
    language: str = "en"
    confidence: float = 1.0
    segments: list[dict] = Field(default_factory=list)


class STTError(Exception):
    """Base exception for STT provider failures."""

    def __init__(self, message: str, provider: str = "unknown", status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code


class ProviderAuthError(STTError):
    """Authentication failure (e.g. invalid or missing API key)."""


class ProviderRateLimitError(STTError):
    """Rate limit exceeded (HTTP 429)."""


class ProviderTimeoutError(STTError):
    """STT call timed out."""


class ProviderServiceError(STTError):
    """Upstream service error (HTTP 5xx)."""


class SpeechToTextAdapter(ABC):
    """Abstract interface isolating STT providers from agent orchestration."""

    @abstractmethod
    async def transcribe(self, wav_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcode 16kHz mono PCM WAV bytes into text."""

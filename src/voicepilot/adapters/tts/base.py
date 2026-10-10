"""Base interface and types for Text-to-Speech adapters."""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class SynthesisResult(BaseModel):
    """Normalized output from speech synthesis."""

    audio_bytes: bytes
    format: str  # "wav" or "mp3"
    latency_ms: float
    provider: str


class TTSError(Exception):
    """Base exception for TTS failure."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message)
        self.message = message
        self.provider = provider


class TextToSpeechAdapter(ABC):
    """Abstract interface isolating TTS providers from agent orchestration."""

    @abstractmethod
    async def synthesize(self, text: str) -> SynthesisResult:
        """Synthesize text into audio bytes."""

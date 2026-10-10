"""Groq Whisper STT adapter with timeout, retry-once, and provider error mapping."""

import asyncio
import logging
import time

import httpx

from voicepilot.adapters.stt.base import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderServiceError,
    ProviderTimeoutError,
    SpeechToTextAdapter,
    TranscriptionResult,
)

logger = logging.getLogger("voicepilot.stt.groq")


class GroqSTTAdapter(SpeechToTextAdapter):
    """Cloud fast-path STT using Groq Whisper with defensive retry and fallback."""

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        timeout_seconds: float = 8.0,
        max_retries: int = 1,
        fallback_adapter: SpeechToTextAdapter | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.fallback_adapter = fallback_adapter
        self.endpoint = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, wav_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        if not self.api_key:
            if self.fallback_adapter:
                logger.warning("Groq API key not provided; falling back to offline adapter")
                return await self.fallback_adapter.transcribe(wav_bytes, sample_rate)
            raise ProviderAuthError("GROQ_API_KEY is not configured", provider="groq", status_code=401)

        t0 = time.perf_counter()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": self.model, "response_format": "json", "language": "en"}

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False) as client:
                    response = await client.post(
                        self.endpoint,
                        headers=headers,
                        files=files,
                        data=data,
                    )

                if response.status_code == 200:
                    result = response.json()
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    return TranscriptionResult(
                        text=result.get("text", "").strip(),
                        latency_ms=latency_ms,
                        provider="groq",
                        language=result.get("language", "en"),
                    )

                if response.status_code in (401, 403):
                    raise ProviderAuthError(
                        f"Groq STT authentication failed: {response.text}",
                        provider="groq",
                        status_code=response.status_code,
                    )
                elif response.status_code == 429:
                    raise ProviderRateLimitError(
                        "Groq STT rate limit exceeded",
                        provider="groq",
                        status_code=429,
                    )
                elif response.status_code >= 500:
                    if attempt < self.max_retries:
                        logger.warning(
                            "Groq STT returned %s; retrying (attempt %d/%d)...",
                            response.status_code,
                            attempt + 1,
                            self.max_retries,
                        )
                        await asyncio.sleep(0.5)
                        continue
                    raise ProviderServiceError(
                        f"Groq STT service error: {response.status_code}",
                        provider="groq",
                        status_code=response.status_code,
                    )
                else:
                    response.raise_for_status()

            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    logger.warning("Groq STT timed out; retrying once...")
                    await asyncio.sleep(0.5)
                    continue
                last_error = ProviderTimeoutError(
                    f"Groq STT timed out after {self.timeout_seconds}s",
                    provider="groq",
                )
            except (httpx.ConnectError, httpx.NetworkError) as exc:
                if attempt < self.max_retries:
                    logger.warning("Network connection error to Groq STT; retrying once...")
                    await asyncio.sleep(0.5)
                    continue
                last_error = ProviderServiceError(
                    f"Network error contacting Groq STT: {exc}",
                    provider="groq",
                )
            except Exception as exc:
                last_error = exc
                break

        # If cloud call failed and fallback adapter is available, seamlessly fall back
        if self.fallback_adapter:
            logger.warning(
                "Groq STT failed (%s); triggering offline fallback adapter",
                last_error,
            )
            return await self.fallback_adapter.transcribe(wav_bytes, sample_rate)

        if last_error:
            raise last_error
        raise ProviderServiceError("Groq STT transcription failed", provider="groq")

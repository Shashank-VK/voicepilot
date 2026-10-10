"""Edge TTS adapter for high-quality cloud neural speech."""

import asyncio
import logging
import time

import edge_tts

from voicepilot.adapters.tts.base import SynthesisResult, TextToSpeechAdapter, TTSError

logger = logging.getLogger("voicepilot.tts.edge")


class EdgeTTSAdapter(TextToSpeechAdapter):
    """Microsoft Edge Neural TTS adapter with graceful offline fallback."""

    def __init__(
        self,
        voice: str = "en-US-JennyNeural",
        timeout_seconds: float = 8.0,
        fallback_adapter: TextToSpeechAdapter | None = None,
    ):
        self.voice = voice
        self.timeout_seconds = timeout_seconds
        self.fallback_adapter = fallback_adapter

    async def synthesize(self, text: str) -> SynthesisResult:
        if not text or not text.strip():
            text = "I have completed your request."

        t0 = time.perf_counter()
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            audio_data = []

            async def _collect():
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data.append(chunk["data"])

            await asyncio.wait_for(_collect(), timeout=self.timeout_seconds)

            mp3_bytes = b"".join(audio_data)
            if not mp3_bytes:
                raise TTSError("Edge TTS generated empty audio payload", provider="edge-tts")

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return SynthesisResult(
                audio_bytes=mp3_bytes,
                format="mp3",
                latency_ms=latency_ms,
                provider="edge-tts",
            )
        except Exception as exc:
            logger.warning("Edge TTS failed (%s); invoking fallback adapter", exc)
            if self.fallback_adapter:
                return await self.fallback_adapter.synthesize(text)
            raise TTSError(f"Edge TTS synthesis failed: {exc}", provider="edge-tts") from exc

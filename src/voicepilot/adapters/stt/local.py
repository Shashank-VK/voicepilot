"""Local faster-whisper STT adapter for offline execution."""

import asyncio
import io
import logging
import time

from voicepilot.adapters.stt.base import (
    ProviderServiceError,
    SpeechToTextAdapter,
    TranscriptionResult,
)

logger = logging.getLogger("voicepilot.stt.local")


class LocalWhisperSTTAdapter(SpeechToTextAdapter):
    """Local offline STT adapter powered by faster-whisper."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        fallback_adapter: SpeechToTextAdapter | None = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.fallback_adapter = fallback_adapter
        self._model = None
        self._lock = asyncio.Lock()

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info(
                "Loading faster-whisper model '%s' on %s (%s)...",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def _transcribe_sync(self, wav_bytes: bytes) -> tuple[str, str]:
        model = self._get_model()
        audio_stream = io.BytesIO(wav_bytes)
        segments, info = model.transcribe(
            audio_stream,
            beam_size=1,
            language="en",
            condition_on_previous_text=False,
        )
        text_pieces = [seg.text.strip() for seg in segments]
        return " ".join(text_pieces).strip(), info.language

    async def transcribe(self, wav_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        t0 = time.perf_counter()
        try:
            text, language = await asyncio.to_thread(self._transcribe_sync, wav_bytes)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return TranscriptionResult(
                text=text,
                latency_ms=latency_ms,
                provider="local",
                language=language,
            )
        except Exception as exc:
            logger.warning("Local faster-whisper STT error: %s", exc)
            if self.fallback_adapter:
                logger.info("Falling back to secondary adapter from local whisper")
                return await self.fallback_adapter.transcribe(wav_bytes, sample_rate)
            raise ProviderServiceError(
                f"Local faster-whisper STT failed: {exc}",
                provider="local",
            ) from exc

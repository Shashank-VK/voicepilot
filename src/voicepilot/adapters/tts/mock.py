"""Deterministic mock TTS adapter for testing, CI, and offline runs."""

import asyncio
import time

from voicepilot.adapters.tts.base import SynthesisResult, TextToSpeechAdapter
from voicepilot.audio.fixtures import generate_speech_like_wav


class MockTTSAdapter(TextToSpeechAdapter):
    """Deterministic mock TTS returning lightweight synthesized audio."""

    def __init__(self, simulated_latency_ms: float = 40.0):
        self.simulated_latency_ms = simulated_latency_ms

    async def synthesize(self, text: str) -> SynthesisResult:
        t0 = time.perf_counter()
        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000.0)

        # Generate a short 0.5s audio waveform
        wav_bytes = generate_speech_like_wav(duration_s=0.5, sample_rate=16000)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return SynthesisResult(
            audio_bytes=wav_bytes,
            format="wav",
            latency_ms=latency_ms,
            provider="mock",
        )

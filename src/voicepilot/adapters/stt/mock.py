"""Deterministic Mock STT adapter for testing, CI, and reproducible benchmarks."""

import asyncio
import time

from voicepilot.adapters.stt.base import SpeechToTextAdapter, TranscriptionResult


class MockSTTAdapter(SpeechToTextAdapter):
    """Deterministic mock STT adapter returning scripted prompts with controlled latency."""

    def __init__(
        self,
        default_text: str = "Summarize the Green Grid AI project",
        simulated_latency_ms: float = 80.0,
    ):
        self.default_text = default_text
        self.simulated_latency_ms = simulated_latency_ms
        self.canned_responses: list[str] = [
            "Summarize the Green Grid AI project",
            "Calculate a simple savings estimate",
            "Remember my preferred unit for the follow-up as kWh",
        ]
        self._call_count = 0

    def set_next_text(self, text: str) -> None:
        """Inject explicit text for the next call."""
        self.default_text = text

    async def transcribe(self, wav_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        t0 = time.perf_counter()
        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000.0)

        # Select canned response sequentially if default text not overridden
        if self._call_count < len(self.canned_responses) and self.default_text == self.canned_responses[0]:
            text = self.canned_responses[self._call_count]
        else:
            text = self.default_text

        self._call_count += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return TranscriptionResult(
            text=text,
            latency_ms=latency_ms,
            provider="mock",
            language="en",
            confidence=0.99,
        )

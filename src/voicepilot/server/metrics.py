"""Latency instrumentation and p50/p95 summary collector."""

import threading
from typing import Any

import numpy as np


class MetricsCollector:
    """Thread-safe latency metrics collector tracking p50/p90/p95 percentiles."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stages = ["upload", "validate_audio", "stt", "agent", "tts", "total"]
        self._history: dict[str, list[float]] = {s: [] for s in self._stages}
        self._total_turns = 0
        self._failed_turns = 0

    def record_turn(self, timings: dict[str, float], success: bool = True) -> None:
        """Record stage latencies in milliseconds."""
        with self._lock:
            self._total_turns += 1
            if not success:
                self._failed_turns += 1
            for stage, ms in timings.items():
                if stage in self._history and ms is not None:
                    self._history[stage].append(float(ms))

    def get_summary(self) -> dict[str, Any]:
        """Compute percentile statistics for each pipeline stage."""
        with self._lock:
            stats = {
                "total_turns": self._total_turns,
                "failed_turns": self._failed_turns,
                "stages": {},
            }
            for stage, samples in self._history.items():
                if not samples:
                    stats["stages"][stage] = {
                        "count": 0,
                        "mean_ms": 0.0,
                        "p50_ms": 0.0,
                        "p90_ms": 0.0,
                        "p95_ms": 0.0,
                    }
                else:
                    arr = np.array(samples, dtype=np.float64)
                    stats["stages"][stage] = {
                        "count": len(samples),
                        "mean_ms": round(float(np.mean(arr)), 1),
                        "p50_ms": round(float(np.percentile(arr, 50)), 1),
                        "p90_ms": round(float(np.percentile(arr, 90)), 1),
                        "p95_ms": round(float(np.percentile(arr, 95)), 1),
                        "min_ms": round(float(np.min(arr)), 1),
                        "max_ms": round(float(np.max(arr)), 1),
                    }
            return stats

    def reset(self) -> None:
        """Clear recorded latency history."""
        with self._lock:
            self._history = {s: [] for s in self._stages}
            self._total_turns = 0
            self._failed_turns = 0


_global_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    return _global_metrics

"""CLI runner and latency benchmark harness for VoicePilot."""

import argparse
import asyncio

import uvicorn

from voicepilot.adapters.stt.mock import MockSTTAdapter
from voicepilot.adapters.tts.mock import MockTTSAdapter
from voicepilot.agent.graph import AgentTurnState, VoicePilotGraph
from voicepilot.audio.fixtures import (
    generate_speech_like_wav,
)
from voicepilot.config import get_settings
from voicepilot.server.metrics import get_metrics_collector


async def run_benchmark(iterations: int = 10) -> None:
    """Run simulated turns across fixtures to instrument stage latencies."""
    print("=" * 65)
    print(" VoicePilot Benchmark Harness: Cloud vs Local/Mock Latency")
    print("=" * 65)

    settings = get_settings()
    stt = MockSTTAdapter(simulated_latency_ms=75.0)
    tts = MockTTSAdapter(simulated_latency_ms=35.0)
    graph = VoicePilotGraph(stt, tts, settings=settings)

    test_audio = generate_speech_like_wav(duration_s=2.0)
    print(f"[*] Running {iterations} synthetic turns through LangGraph...")

    for i in range(iterations):
        initial_state: AgentTurnState = {
            "turn_id": f"bench_{i}",
            "session_id": "bench_session",
            "audio_bytes": test_audio,
            "timings": {"upload": 45.0},
            "is_cancelled": False,
        }
        res = await graph.run_turn(initial_state)
        get_metrics_collector().record_turn(res.get("timings", {}), success=True)
        print(f"  Turn {i+1:02d}: Total {res.get('timings', {}).get('total', 0):.1f}ms | STT: {res.get('timings', {}).get('stt', 0):.1f}ms | Agent: {res.get('timings', {}).get('agent', 0):.1f}ms | TTS: {res.get('timings', {}).get('tts', 0):.1f}ms")

    summary = get_metrics_collector().get_summary()
    print("\n" + "=" * 65)
    print(" Measured Percentiles (ms):")
    print("=" * 65)
    print(f"{'Stage':<18} | {'Count':<6} | {'Mean':<8} | {'p50':<8} | {'p90':<8} | {'p95':<8}")
    print("-" * 65)
    for stage, s in summary["stages"].items():
        print(f"{stage:<18} | {s['count']:<6} | {s['mean_ms']:<8.1f} | {s['p50_ms']:<8.1f} | {s['p90_ms']:<8.1f} | {s['p95_ms']:<8.1f}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="VoicePilot CLI")
    subparsers = parser.add_subparsers(dest="command")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the VoicePilot API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # bench command
    bench_parser = subparsers.add_parser("bench", help="Run offline latency benchmark")
    bench_parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark turns")

    args = parser.parse_args()

    if args.command == "bench":
        asyncio.run(run_benchmark(iterations=args.iterations))
    elif args.command == "serve" or args.command is None:
        settings = get_settings()
        host = getattr(args, "host", settings.HOST)
        port = getattr(args, "port", settings.PORT)
        reload = getattr(args, "reload", False)
        print(f"Starting VoicePilot on http://{host}:{port} ...")
        uvicorn.run("voicepilot.server.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()

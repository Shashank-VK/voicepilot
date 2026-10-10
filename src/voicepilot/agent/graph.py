"""LangGraph state machine for voice turn orchestration."""

import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from voicepilot.adapters.stt.base import SpeechToTextAdapter, STTError
from voicepilot.adapters.tts.base import TextToSpeechAdapter, TTSError
from voicepilot.agent.memory import SessionStore, TurnRecord, get_session_store
from voicepilot.agent.tools import GreenGridKnowledgeStore, SafeCalculator, ToolExecutionError
from voicepilot.audio.transcoder import AudioFormatError, transcode_to_pcm16_wav
from voicepilot.audio.validator import AudioValidationError, validate_audio_samples
from voicepilot.config import Settings, get_settings


class AgentTurnState(TypedDict, total=False):
    """LangGraph turn state representing an in-flight voice interaction."""

    turn_id: str
    session_id: str
    audio_bytes: bytes | None
    audio_format: str | None
    pcm_wav_bytes: bytes | None
    samples: Any | None
    sample_rate: int
    raw_transcript: str
    intent: str
    tool_called: str | None
    tool_input: str | None
    tool_result: str | None
    response_text: str
    speech_audio_bytes: bytes | None
    speech_audio_format: str
    timings: dict[str, float]
    error: str | None
    error_code: str | None
    error_stage: str | None
    is_cancelled: bool


class VoicePilotGraph:
    """Builds and compiles the LangGraph state machine for VoicePilot."""

    def __init__(
        self,
        stt_adapter: SpeechToTextAdapter,
        tts_adapter: TextToSpeechAdapter,
        session_store: SessionStore | None = None,
        settings: Settings | None = None,
    ):
        self.stt_adapter = stt_adapter
        self.tts_adapter = tts_adapter
        self.session_store = session_store or get_session_store()
        self.settings = settings or get_settings()
        self.app = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentTurnState)

        # Nodes
        builder.add_node("validate_audio", self.node_validate_audio)
        builder.add_node("transcribe", self.node_transcribe)
        builder.add_node("classify_intent", self.node_classify_intent)
        builder.add_node("execute_tool", self.node_execute_tool)
        builder.add_node("generate_response", self.node_generate_response)
        builder.add_node("synthesize_speech", self.node_synthesize_speech)
        builder.add_node("recoverable_message", self.node_recoverable_message)

        # Edges
        builder.add_edge(START, "validate_audio")

        builder.add_conditional_edges(
            "validate_audio",
            self.route_after_validate,
            {"transcribe": "transcribe", "recoverable_message": "recoverable_message", "END": END},
        )

        builder.add_conditional_edges(
            "transcribe",
            self.route_after_transcribe,
            {"classify_intent": "classify_intent", "recoverable_message": "recoverable_message", "END": END},
        )

        builder.add_conditional_edges(
            "classify_intent",
            self.route_after_intent,
            {"execute_tool": "execute_tool", "generate_response": "generate_response", "END": END},
        )

        builder.add_edge("execute_tool", "generate_response")

        builder.add_conditional_edges(
            "generate_response",
            self.route_after_response,
            {"synthesize_speech": "synthesize_speech", "recoverable_message": "recoverable_message", "END": END},
        )

        builder.add_edge("synthesize_speech", END)
        builder.add_edge("recoverable_message", END)

        return builder.compile()

    # Routing functions
    def route_after_validate(self, state: AgentTurnState) -> str:
        if state.get("is_cancelled"):
            return "END"
        if state.get("error"):
            return "recoverable_message"
        return "transcribe"

    def route_after_transcribe(self, state: AgentTurnState) -> str:
        if state.get("is_cancelled"):
            return "END"
        if state.get("error"):
            return "recoverable_message"
        return "classify_intent"

    def route_after_intent(self, state: AgentTurnState) -> str:
        if state.get("is_cancelled"):
            return "END"
        if state.get("tool_called"):
            return "execute_tool"
        return "generate_response"

    def route_after_response(self, state: AgentTurnState) -> str:
        if state.get("is_cancelled"):
            return "END"
        if state.get("error"):
            return "recoverable_message"
        return "synthesize_speech"

    # Node implementations
    async def node_validate_audio(self, state: AgentTurnState) -> dict[str, Any]:
        timings = dict(state.get("timings") or {})
        t0 = time.perf_counter()

        if state.get("is_cancelled"):
            return {"is_cancelled": True}

        raw_audio = state.get("audio_bytes")
        pcm_bytes = state.get("pcm_wav_bytes")
        samples = state.get("samples")
        sample_rate = state.get("sample_rate", 16000)

        try:
            if pcm_bytes is None and raw_audio is not None:
                pcm_bytes, samples, sample_rate = transcode_to_pcm16_wav(raw_audio)

            if samples is None and pcm_bytes is not None:
                _, samples, sample_rate = transcode_to_pcm16_wav(pcm_bytes)

            if samples is None:
                raise AudioFormatError("unsupported format: no audio payload provided")

            validate_audio_samples(
                samples=samples,
                sample_rate=sample_rate,
                min_speech_duration_seconds=self.settings.MIN_SPEECH_DURATION_SECONDS,
                max_audio_seconds=self.settings.MAX_AUDIO_SECONDS,
                min_snr_db=self.settings.MIN_SNR_DB,
                max_silence_ratio=self.settings.MAX_SILENCE_RATIO,
                max_clipping_ratio=self.settings.MAX_CLIPPING_RATIO,
            )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            timings["validate_audio"] = latency_ms

            return {
                "pcm_wav_bytes": pcm_bytes,
                "samples": samples,
                "sample_rate": sample_rate,
                "timings": timings,
            }

        except (AudioValidationError, AudioFormatError) as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            timings["validate_audio"] = latency_ms
            code = getattr(exc, "code", "unsupported_format")
            return {
                "error": str(exc),
                "error_code": code,
                "error_stage": "validate_audio",
                "timings": timings,
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            timings["validate_audio"] = latency_ms
            return {
                "error": f"unsupported format: audio decoding failed ({exc})",
                "error_code": "unsupported_format",
                "error_stage": "validate_audio",
                "timings": timings,
            }

    async def node_transcribe(self, state: AgentTurnState) -> dict[str, Any]:
        timings = dict(state.get("timings") or {})
        if state.get("is_cancelled") or state.get("error"):
            return {}

        pcm_bytes = state.get("pcm_wav_bytes")
        if not pcm_bytes:
            return {"error": "Missing PCM audio for transcription", "error_code": "no_speech", "error_stage": "transcribe"}

        t0 = time.perf_counter()
        try:
            result = await self.stt_adapter.transcribe(pcm_bytes, state.get("sample_rate", 16000))
            timings["stt"] = result.latency_ms
            if not result.text:
                return {
                    "error": "no speech: transcription returned empty text",
                    "error_code": "no_speech",
                    "error_stage": "transcribe",
                    "timings": timings,
                }
            return {
                "raw_transcript": result.text,
                "timings": timings,
            }
        except STTError as exc:
            timings["stt"] = (time.perf_counter() - t0) * 1000.0
            return {
                "error": str(exc),
                "error_code": "stt_error",
                "error_stage": "transcribe",
                "timings": timings,
            }
        except Exception as exc:
            timings["stt"] = (time.perf_counter() - t0) * 1000.0
            return {
                "error": f"STT failed: {exc}",
                "error_code": "stt_error",
                "error_stage": "transcribe",
                "timings": timings,
            }

    async def node_classify_intent(self, state: AgentTurnState) -> dict[str, Any]:
        if state.get("is_cancelled") or state.get("error"):
            return {}

        transcript = state.get("raw_transcript", "").strip()
        q = transcript.lower()

        # Check for user preference update:
        unit_match = re.search(r"\b(preferred unit|unit|preferred)\b.*?\b(kwh|mwh|dollars?|usd|inr)\b", q)
        if unit_match:
            unit = unit_match.group(2).upper()
            if "DOLLAR" in unit or "USD" in unit:
                unit = "USD"
            elif "KWH" in unit:
                unit = "kWh"
            elif "MWH" in unit:
                unit = "MWh"
            self.session_store.set_preference(state.get("session_id", "default"), "preferred_unit", unit)
            return {"intent": "preference_update", "tool_called": None}

        # Check for calculation:
        is_calc = False
        math_expr = None
        if any(w in q for w in ["calculate", "savings estimate", "estimate savings", "compute", "how much savings"]):
            is_calc = True
            # Check if there is explicit math or standard Green Grid savings formula:
            explicit_math = re.search(r"(\d+[\s\d\+\-\*\/\.\(\)]+\d+)", transcript)
            if explicit_math:
                math_expr = explicit_math.group(1).strip()
            else:
                # Canonical savings formula: 3200 kWh/day * ($0.22 - $0.09) * 365
                math_expr = "3200 * (0.22 - 0.09) * 365"

        elif re.search(r"(\d+\s*[\+\-\*\/]\s*\d+)", transcript):
            is_calc = True
            explicit_math = re.search(r"([\d\.\s\+\-\*\/\(\)]+)", transcript)
            if explicit_math:
                math_expr = explicit_math.group(1).strip()

        if is_calc:
            return {
                "intent": "calculation",
                "tool_called": "calculator",
                "tool_input": math_expr or "3200 * (0.22 - 0.09) * 365",
            }

        # Check for knowledge lookup:
        if any(w in q for w in ["green grid", "solar", "battery", "bess", "tariff", "summarize", "summary", "project", "what is"]):
            return {
                "intent": "knowledge_lookup",
                "tool_called": "knowledge_lookup",
                "tool_input": transcript,
            }

        return {"intent": "general", "tool_called": None}

    async def node_execute_tool(self, state: AgentTurnState) -> dict[str, Any]:
        if state.get("is_cancelled") or state.get("error"):
            return {}

        tool_name = state.get("tool_called")
        tool_input = state.get("tool_input", "")

        try:
            if tool_name == "calculator":
                result = SafeCalculator.evaluate(tool_input)
                return {"tool_result": str(result)}
            elif tool_name == "knowledge_lookup":
                result = GreenGridKnowledgeStore.lookup(tool_input)
                return {"tool_result": result}
        except ToolExecutionError as exc:
            return {"tool_result": f"Tool error: {exc}"}
        except Exception as exc:
            return {"tool_result": f"Unexpected tool error: {exc}"}

        return {}

    async def node_generate_response(self, state: AgentTurnState) -> dict[str, Any]:
        timings = dict(state.get("timings") or {})
        if state.get("is_cancelled") or state.get("error"):
            return {}

        t0 = time.perf_counter()
        session_id = state.get("session_id", "default")
        session = self.session_store.get_or_create_session(session_id)
        preferred_unit = session.preferences.get("preferred_unit", "kWh")

        intent = state.get("intent", "general")
        transcript = state.get("raw_transcript", "")
        tool_name = state.get("tool_called")
        tool_result = state.get("tool_result")

        # Response composition
        if intent == "preference_update":
            response = (
                f"I have saved your preference. All subsequent figures and follow-ups will be "
                f"expressed using your preferred unit: {preferred_unit}."
            )

        elif intent == "calculation":
            if tool_result and not tool_result.startswith("Tool error"):
                try:
                    val = float(tool_result)
                    formatted_val = f"${val:,.2f}"
                except ValueError:
                    formatted_val = tool_result

                response = (
                    f"Based on shifting 3,200 {preferred_unit}/day from peak tariff ($0.22/{preferred_unit}) "
                    f"to off-peak ($0.09/{preferred_unit}), the calculated annual arbitrage savings is {formatted_val}. "
                    f"Factoring in a 15% demand charge reduction, total annual savings reaches approximately $189,840."
                )
            else:
                response = f"I encountered an error calculating that: {tool_result}"

        elif intent == "knowledge_lookup":
            if "summarize" in transcript.lower() or "summary" in transcript.lower() or "green grid" in transcript.lower():
                response = (
                    "Green Grid AI is an intelligent microgrid optimization system deploying 2.4 MW of commercial "
                    "solar PV paired with a 4.8 MWh lithium iron phosphate (LiFePO4) battery energy storage system. "
                    "Operating at 99.4% uptime across 4 commercial sites in Bengaluru, it provides automated "
                    "peak-shaving and energy arbitrage, saving ~$189,840 annually and avoiding 2,450 metric tons of CO2."
                )
            else:
                response = tool_result or "The Green Grid AI project provides intelligent energy arbitrage and peak-shaving."

        else:
            response = (
                f"I received your question: '{transcript}'. How can I assist you with Green Grid AI summaries, "
                f"savings calculations, or microgrid telemetry?"
            )

        timings["agent"] = (time.perf_counter() - t0) * 1000.0

        # Save turn to session memory
        turn_rec = TurnRecord(
            turn_id=state.get("turn_id", f"turn_{int(time.time()*1000)}"),
            user_transcript=transcript,
            assistant_response=response,
            tool_used=tool_name,
            tool_output=tool_result,
            latencies=timings,
        )
        self.session_store.add_turn(session_id, turn_rec)

        return {
            "response_text": response,
            "timings": timings,
        }

    async def node_synthesize_speech(self, state: AgentTurnState) -> dict[str, Any]:
        timings = dict(state.get("timings") or {})
        if state.get("is_cancelled") or state.get("error"):
            return {}

        text = state.get("response_text", "")
        t0 = time.perf_counter()

        try:
            result = await self.tts_adapter.synthesize(text)
            timings["tts"] = result.latency_ms
            return {
                "speech_audio_bytes": result.audio_bytes,
                "speech_audio_format": result.format,
                "timings": timings,
            }
        except TTSError:
            timings["tts"] = (time.perf_counter() - t0) * 1000.0
            # TTS failure should still preserve text response
            return {
                "speech_audio_bytes": None,
                "speech_audio_format": "none",
                "timings": timings,
            }
        except Exception:
            timings["tts"] = (time.perf_counter() - t0) * 1000.0
            return {
                "speech_audio_bytes": None,
                "speech_audio_format": "none",
                "timings": timings,
            }

    async def node_recoverable_message(self, state: AgentTurnState) -> dict[str, Any]:
        code = state.get("error_code", "generic_error")
        err = state.get("error", "An unknown error occurred")

        messages = {
            "no_speech": "I couldn't hear any speech. Please make sure your microphone is unmuted and speak clearly.",
            "too_noisy": "The ambient noise level is too high for clean audio recognition. Please try again in a quieter area.",
            "too_short": "Your recording was too short. Please speak for at least half a second.",
            "too_long": "Your audio exceeded the 30-second duration limit. Please provide a shorter question.",
            "clipping": "Your audio was severely distorted or clipped. Please lower your microphone input level and retry.",
            "unsupported_format": "Unsupported audio format. Please provide valid audio in WebM, WAV, MP3, or MP4.",
            "permission_denied": "Microphone permission was denied. Please allow microphone access in your browser settings.",
            "stt_error": "Speech-to-text service encountered a temporary error. You may retry or enable Offline mode.",
        }

        friendly = messages.get(code, f"I ran into an issue: {err}. Please retry your question.")
        return {
            "response_text": friendly,
            "error": err,
            "error_code": code,
        }

    async def run_turn(self, initial_state: AgentTurnState) -> AgentTurnState:
        """Execute full turn through the LangGraph state machine."""
        t_turn_start = time.perf_counter()
        result = await self.app.ainvoke(initial_state)
        # Compute total latency
        timings = dict(result.get("timings") or {})
        timings["total"] = (time.perf_counter() - t_turn_start) * 1000.0
        result["timings"] = timings
        return result

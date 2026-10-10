"""Tests for LangGraph agent, tools, and 3-turn memory window."""

import pytest

from voicepilot.adapters.stt.mock import MockSTTAdapter
from voicepilot.adapters.tts.mock import MockTTSAdapter
from voicepilot.agent.graph import AgentTurnState, VoicePilotGraph
from voicepilot.agent.memory import SessionStore, TurnRecord
from voicepilot.agent.tools import (
    GreenGridKnowledgeStore,
    SafeCalculator,
    ToolExecutionError,
)
from voicepilot.config import Settings


def test_safe_calculator_basic_arithmetic():
    assert SafeCalculator.evaluate("2 + 2") == 4
    assert SafeCalculator.evaluate("10 - 4.5") == 5.5
    # Green Grid annual savings calculation
    expr = "3200 * (0.22 - 0.09) * 365"
    assert round(SafeCalculator.evaluate(expr), 2) == 151840.0


def test_safe_calculator_division_by_zero():
    with pytest.raises(ToolExecutionError) as exc_info:
        SafeCalculator.evaluate("100 / 0")
    assert "division by zero" in str(exc_info.value).lower()


def test_safe_calculator_blocks_arbitrary_code():
    malicious = [
        "__import__('os').system('ls')",
        "eval('2+2')",
        "exec('print(1)')",
        "open('/etc/passwd').read()",
    ]
    for expr in malicious:
        with pytest.raises(ToolExecutionError):
            SafeCalculator.evaluate(expr)


def test_knowledge_lookup_green_grid():
    summary = GreenGridKnowledgeStore.lookup("Tell me about Green Grid AI")
    assert "2.4 MW" in summary
    assert "4.8 MWh" in summary
    assert "Bengaluru" in summary

    tariffs = GreenGridKnowledgeStore.lookup("What are the peak tariffs?")
    assert "$0.22/kWh" in tariffs


def test_session_memory_window_and_preferences():
    store = SessionStore(max_turns=3)
    sid = "test_sess_01"

    store.set_preference(sid, "preferred_unit", "kWh")
    sess = store.get_session(sid)
    assert sess.preferences["preferred_unit"] == "kWh"

    # Add 4 turns; only recent 3 should be in active memory window
    for i in range(4):
        store.add_turn(
            sid,
            TurnRecord(
                turn_id=f"t_{i}",
                user_transcript=f"Query {i}",
                assistant_response=f"Answer {i}",
            ),
        )

    recent = sess.get_recent_turns(3)
    assert len(recent) == 3
    assert recent[0].turn_id == "t_1"
    assert recent[-1].turn_id == "t_3"

    history_str = sess.format_history_prompt(3)
    assert "User Preferences: preferred_unit: kWh" in history_str
    assert "Query 0" not in history_str  # Evicted from active 3-turn window
    assert "Query 1" in history_str
    assert "Query 3" in history_str


@pytest.mark.asyncio
async def test_langgraph_canonical_demo_flow(speech_wav):
    settings = Settings(DEMO_MODE=True, STT_PROVIDER="mock", TTS_PROVIDER="mock")
    stt = MockSTTAdapter()
    tts = MockTTSAdapter()
    store = SessionStore()
    graph = VoicePilotGraph(stt, tts, session_store=store, settings=settings)

    sid = "demo_flow_sess"

    # Turn 1: Summary
    stt.set_next_text("Summarize the Green Grid AI project")
    turn1: AgentTurnState = {
        "turn_id": "turn_1",
        "session_id": sid,
        "audio_bytes": speech_wav,
        "is_cancelled": False,
    }
    res1 = await graph.run_turn(turn1)
    assert "Green Grid AI" in res1["response_text"]
    assert res1["tool_called"] == "knowledge_lookup"
    assert res1.get("speech_audio_bytes") is not None

    # Turn 2: Calculation
    stt.set_next_text("Calculate a simple savings estimate")
    turn2: AgentTurnState = {
        "turn_id": "turn_2",
        "session_id": sid,
        "audio_bytes": speech_wav,
        "is_cancelled": False,
    }
    res2 = await graph.run_turn(turn2)
    assert "$151,840" in res2["response_text"] or "$189,840" in res2["response_text"]
    assert res2["tool_called"] == "calculator"

    # Turn 3: Preference follow-up
    stt.set_next_text("Remember my preferred unit for the follow-up as kWh")
    turn3: AgentTurnState = {
        "turn_id": "turn_3",
        "session_id": sid,
        "audio_bytes": speech_wav,
        "is_cancelled": False,
    }
    res3 = await graph.run_turn(turn3)
    assert "kWh" in res3["response_text"]

    # Verify session store recorded all 3 turns
    sess = store.get_session(sid)
    assert len(sess.turns) == 3
    assert sess.preferences.get("preferred_unit") == "kWh"


@pytest.mark.asyncio
async def test_langgraph_cancellation_state(speech_wav):
    settings = Settings(DEMO_MODE=True, STT_PROVIDER="mock", TTS_PROVIDER="mock")
    stt = MockSTTAdapter()
    tts = MockTTSAdapter()
    graph = VoicePilotGraph(stt, tts, settings=settings)

    turn: AgentTurnState = {
        "turn_id": "cancelled_turn",
        "session_id": "cancel_sess",
        "audio_bytes": speech_wav,
        "is_cancelled": True,
    }
    res = await graph.run_turn(turn)
    # When cancelled, should not produce speech or response mutation
    assert res.get("speech_audio_bytes") is None

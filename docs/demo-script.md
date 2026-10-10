# VoicePilot: Two-Minute Demo & Screen Recording Script

This script guides a 90-to-120-second walkthrough showing the core value proposition:
1. **Happy Path Conversation**: Multi-turn dialogue with tool invocation and preference retention.
2. **Failure Recovery**: Defensive audio validation rejecting bad audio with actionable guidance.
3. **Cancellation & Telemetry**: Instant turn cancellation and stage latency breakdown.

---

## Pre-Flight Check

```bash
# 1. Start VoicePilot in demo mode
make demo

# 2. Open browser at:
# http://localhost:8000
```

---

## Turn 1: Project Knowledge Retrieval (0:00 - 0:30)

- **Action**: Click the microphone button (or the chip button **"1️⃣ Summarize Green Grid"**).
- **Spoken Prompt**: *"Summarize the Green Grid AI project."*
- **What to Observe**:
  - State machine cycles: `idle` → `recording` → `uploading` → `transcribing` → `reasoning` → `speaking` → `idle`.
  - Latency Waterfall updates: STT (~80 ms), Agent (~100 ms), TTS (~40 ms). Total turn under 1.5 seconds.
  - Spoken Answer: VoicePilot summarizes the 2.4 MW solar PV array and 4.8 MWh LiFePO4 battery energy storage system (BESS) in Bengaluru operating at 99.4% uptime.

---

## Turn 2: Safe Tool Invocation (Calculation) (0:30 - 1:00)

- **Action**: Click microphone or **"2️⃣ Calculate Savings"**.
- **Spoken Prompt**: *"Calculate a simple savings estimate."*
- **What to Observe**:
  - The agent classifies intent as `calculation` and routes to the **AST-safe calculator** tool.
  - Computes peak tariff ($0.22/kWh) vs off-peak ($0.09/kWh) arbitrage for 3,200 kWh/day.
  - Evaluates: `3200 * (0.22 - 0.09) * 365 = 151,840.0`
  - Voice response quotes $151,840 annual base arbitrage plus 15% demand reduction totaling ~$189,840.

---

## Turn 3: Context Memory & Unit Follow-Up (1:00 - 1:25)

- **Action**: Click microphone or **"3️⃣ Set Unit: kWh"**.
- **Spoken Prompt**: *"Remember my preferred unit for the follow-up as kWh."*
- **What to Observe**:
  - VoicePilot acknowledges and persists `preferred_unit = "kWh"` inside session memory.
  - Subsequent requests format figures according to this preference.
  - Chat stream retains the prior 3 turns in active memory.

---

## Turn 4: Failure Handling & Audio Rejection (1:25 - 1:45)

- **Action**: Click microphone and stay completely silent for 2 seconds.
- **What to Observe**:
  - Defensive audio validator detects silence (peak amplitude below threshold / VAD zero).
  - STT is skipped entirely (saving provider compute and budget).
  - Actionable error banner appears:
    `⚠️ no speech: audio is silent or unintelligible`
  - Assistant speaks a friendly recovery notice:
    *"I couldn't hear any speech. Please make sure your microphone is unmuted and speak clearly."*

---

## Turn 5: Clean Recovery via Demo Reset (1:45 - 2:00)

- **Action**: Click the **🔄 Reset Demo** button in the header.
- **What to Observe**:
  - Calls `POST /api/demo/reset`.
  - Reseeds synthetic demo fixtures and clears all active session caches in under 100 milliseconds.
  - Telemetry resets to baseline; session is ready for a new clean run.

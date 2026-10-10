/**
 * VoicePilot Client Application
 * Implements:
 * - 8-state explicit client state machine
 * - One AbortController per turn
 * - Waveform visualization & VU meter
 * - Stage latency waterfall & p50/p95 telemetry
 * - Session history rendering & JSON export
 * - Demo reset and permission recovery guidance
 */

class VoicePilotClient {
  constructor() {
    this.VALID_STATES = [
      "idle",
      "recording",
      "uploading",
      "transcribing",
      "reasoning",
      "speaking",
      "cancelled",
      "error",
    ];

    this.state = "idle";
    this.sessionId = null;
    this.currentTurnId = null;
    this.abortController = null;

    // Audio & Recording
    this.mediaStream = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.audioContext = null;
    this.analyser = null;
    this.animFrameId = null;
    this.currentPlaybackAudio = null;

    // WebSocket
    this.socket = null;
    this.isRecording = false;

    this.initDOMElements();
    this.initEventListeners();
    this.initSession();
  }

  initDOMElements() {
    this.dom = {
      sessionIdDisplay: document.getElementById("sessionIdDisplay"),
      resetDemoBtn: document.getElementById("resetDemoBtn"),
      downloadTraceBtn: document.getElementById("downloadTraceBtn"),
      statePill: document.getElementById("statePill"),
      currentStateText: document.getElementById("currentStateText"),
      chatContainer: document.getElementById("chatContainer"),
      waveformCanvas: document.getElementById("waveformCanvas"),
      vuMeterBar: document.getElementById("vuMeterBar"),
      errorBanner: document.getElementById("errorBanner"),
      errorMessageText: document.getElementById("errorMessageText"),
      dismissErrorBtn: document.getElementById("dismissErrorBtn"),
      micBtn: document.getElementById("micBtn"),
      micHint: document.getElementById("micHint"),
      cancelTurnBtn: document.getElementById("cancelTurnBtn"),
      helpPermissionBtn: document.getElementById("helpPermissionBtn"),
      permissionModal: document.getElementById("permissionModal"),
      closeModalBtn: document.getElementById("closeModalBtn"),
      gotItBtn: document.getElementById("gotItBtn"),
      modalBackdrop: document.getElementById("modalBackdrop"),
      turnIdTag: document.getElementById("turnIdTag"),
      valUpload: document.getElementById("valUpload"),
      barUpload: document.getElementById("barUpload"),
      valStt: document.getElementById("valStt"),
      barStt: document.getElementById("barStt"),
      valAgent: document.getElementById("valAgent"),
      barAgent: document.getElementById("barAgent"),
      valTts: document.getElementById("valTts"),
      barTts: document.getElementById("barTts"),
      valTotal: document.getElementById("valTotal"),
      barTotal: document.getElementById("barTotal"),
      refreshStatsBtn: document.getElementById("refreshStatsBtn"),
      statsTableBody: document.getElementById("statsTableBody"),
      turnCountBadge: document.getElementById("turnCountBadge"),
      stateChips: document.querySelectorAll(".state-chips .chip"),
      demoChips: document.querySelectorAll(".demo-helpers .chip-btn"),
    };
  }

  initEventListeners() {
    // Mic click & hold
    this.dom.micBtn.addEventListener("click", () => this.toggleRecording());
    this.dom.cancelTurnBtn.addEventListener("click", () => this.cancelCurrentTurn());

    // Reset & Export
    this.dom.resetDemoBtn.addEventListener("click", () => this.resetDemo());
    this.dom.downloadTraceBtn.addEventListener("click", () => this.exportSessionJson());
    this.dom.refreshStatsBtn.addEventListener("click", () => this.fetchTelemetryStats());

    // Error dismiss
    this.dom.dismissErrorBtn.addEventListener("click", () => {
      this.dom.errorBanner.classList.add("hidden");
    });

    // Permission modal
    this.dom.helpPermissionBtn.addEventListener("click", () => this.togglePermissionModal(true));
    this.dom.closeModalBtn.addEventListener("click", () => this.togglePermissionModal(false));
    this.dom.gotItBtn.addEventListener("click", () => this.togglePermissionModal(false));
    this.dom.modalBackdrop.addEventListener("click", () => this.togglePermissionModal(false));

    // Scripted demo chips
    this.dom.demoChips.forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const prompt = e.currentTarget.getAttribute("data-prompt");
        this.runScriptedTurn(prompt);
      });
    });
  }

  transitionState(newState, reason = "") {
    if (!this.VALID_STATES.includes(newState)) {
      console.error(`Invalid state transition: ${newState}`);
      return;
    }

    this.state = newState;
    this.dom.currentStateText.textContent = newState;
    this.dom.statePill.className = `state-pill state-${newState}`;

    // Update chips in right panel
    this.dom.stateChips.forEach((chip) => {
      chip.classList.toggle("active", chip.getAttribute("data-state") === newState);
    });

    // Enable/disable Cancel button
    const inFlight = ["recording", "uploading", "transcribing", "reasoning", "speaking"].includes(newState);
    this.dom.cancelTurnBtn.disabled = !inFlight;

    // Mic button appearance
    if (newState === "recording") {
      this.dom.micBtn.classList.add("recording");
      this.dom.micHint.textContent = "Recording... Click to Stop";
    } else {
      this.dom.micBtn.classList.remove("recording");
      this.dom.micHint.textContent = inFlight ? `Processing (${newState})...` : "Click to Speak";
    }

    console.log(`[State Transition] -> ${newState} ${reason ? `(${reason})` : ""}`);
  }

  showError(message, code = "error") {
    this.dom.errorMessageText.textContent = message;
    this.dom.errorBanner.classList.remove("hidden");
    this.transitionState("error", message);
  }

  clearError() {
    this.dom.errorBanner.classList.add("hidden");
  }

  togglePermissionModal(show) {
    this.dom.permissionModal.classList.toggle("hidden", !show);
  }

  async initSession() {
    try {
      const res = await fetch("/v1/sessions", { method: "POST" });
      const data = await res.json();
      this.sessionId = data.session_id;
      this.dom.sessionIdDisplay.textContent = this.sessionId;
      this.initWebSocket();
      await this.fetchTelemetryStats();
    } catch (err) {
      console.warn("Failed creating initial session, using default demo session:", err);
      this.sessionId = "demo-session-greengrid";
      this.dom.sessionIdDisplay.textContent = this.sessionId;
      this.loadSessionHistory();
    }
  }

  initWebSocket() {
    if (!this.sessionId) return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/v1/voice/${this.sessionId}`;

    try {
      this.socket = new WebSocket(wsUrl);

      this.socket.onopen = () => {
        console.log("[WebSocket] Connected to VoicePilot");
      };

      this.socket.onmessage = (event) => {
        try {
          const envelope = JSON.parse(event.data);
          this.handleEventEnvelope(envelope);
        } catch (e) {
          console.error("[WebSocket] Parse error:", e);
        }
      };

      this.socket.onclose = () => {
        console.log("[WebSocket] Disconnected. Reconnecting in 3s...");
        setTimeout(() => this.initWebSocket(), 3000);
      };

      this.socket.onerror = (err) => {
        console.warn("[WebSocket] Socket error:", err);
      };
    } catch (e) {
      console.warn("WebSocket not supported or failed to connect, falling back to HTTP turns:", e);
    }
  }

  handleEventEnvelope(envelope) {
    const { type, turn_id, stage, latency_ms, payload } = envelope;

    // Discard stale events if not matching current active turn
    if (this.currentTurnId && turn_id !== this.currentTurnId) {
      console.log(`[Stale Event Ignored] Received ${type} for ${turn_id}, active is ${this.currentTurnId}`);
      return;
    }

    switch (type) {
      case "stage.started":
        if (stage === "validate_audio" || stage === "transcribe") {
          this.transitionState("transcribing");
        } else if (stage === "classify_intent" || stage === "execute_tool" || stage === "generate_response") {
          this.transitionState("reasoning");
        } else if (stage === "synthesize_speech") {
          this.transitionState("speaking");
        }
        break;

      case "stage.completed":
        this.updateLatencyBar(stage, latency_ms);
        if (stage === "transcribe" && payload.text) {
          this.transitionState("reasoning");
        }
        break;

      case "speech.ready":
        this.updateLatencyBar("tts", latency_ms);
        this.transitionState("speaking");
        if (payload.audio_base64) {
          this.playAudioBase64(payload.audio_base64, payload.format);
        }
        break;

      case "turn.completed":
        this.updateWaterfall(payload.timings || {});
        this.appendTurnToChat(
          payload.transcript,
          payload.response,
          payload.tool_used,
          payload.tool_result,
          payload.timings
        );
        this.transitionState("idle", "Turn completed successfully");
        this.fetchTelemetryStats();
        break;

      case "turn.cancelled":
        this.transitionState("cancelled", payload.message);
        break;

      case "error":
        this.showError(payload.message || "Pipeline error", payload.code);
        if (payload.recoverable_text) {
          this.appendTurnToChat("[Error Occurred]", payload.recoverable_text);
        }
        break;
    }
  }

  async toggleRecording() {
    if (this.isRecording) {
      this.stopRecording();
    } else {
      if (this.state !== "idle" && this.state !== "cancelled" && this.state !== "error") {
        console.warn("Cannot start turn while current turn is in flight");
        return;
      }
      await this.startRecording();
    }
  }

  async startRecording() {
    this.clearError();

    // Create fresh AbortController for this turn
    this.abortController = new AbortController();
    this.currentTurnId = `turn_${Date.now()}`;
    this.dom.turnIdTag.textContent = `Turn: ${this.currentTurnId}`;

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch (err) {
      console.error("Microphone access denied:", err);
      this.showError(
        "Microphone access was denied or device is busy. Click 'Mic Help' for resolution.",
        "permission_denied"
      );
      return;
    }

    this.startAudioVisualizer(this.mediaStream);

    // Prefer webm/opus, fallback to mp4 or standard
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/mp4")
      ? "audio/mp4"
      : "";

    this.audioChunks = [];
    try {
      this.mediaRecorder = new MediaRecorder(this.mediaStream, mimeType ? { mimeType } : {});
    } catch (e) {
      this.mediaRecorder = new MediaRecorder(this.mediaStream);
    }

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        this.audioChunks.push(e.data);
      }
    };

    this.mediaRecorder.onstop = () => {
      this.handleRecordingFinished();
    };

    this.mediaRecorder.start(100);
    this.isRecording = true;
    this.transitionState("recording");
  }

  stopRecording() {
    if (!this.isRecording) return;
    this.isRecording = false;
    if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
      this.mediaRecorder.stop();
    }
    this.stopAudioVisualizer();
  }

  cancelCurrentTurn() {
    console.log(`[Cancel Requested] turn_id: ${this.currentTurnId}`);

    // Abort HTTP fetch if in-flight
    if (this.abortController) {
      this.abortController.abort();
    }

    // Stop MediaRecorder if still recording
    if (this.isRecording) {
      this.isRecording = false;
      if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
        this.mediaRecorder.stop();
      }
    }
    this.stopAudioVisualizer();

    // Stop any in-flight audio playback
    if (this.currentPlaybackAudio) {
      this.currentPlaybackAudio.pause();
      this.currentPlaybackAudio = null;
    }

    // Inform server via WebSocket if connected
    if (this.socket && this.socket.readyState === WebSocket.OPEN && this.currentTurnId) {
      this.socket.send(
        JSON.stringify({
          type: "turn.cancel",
          turn_id: this.currentTurnId,
        })
      );
    }

    this.transitionState("cancelled", "User cancelled turn");
  }

  async handleRecordingFinished() {
    if (this.state === "cancelled") return;

    this.transitionState("uploading");
    const audioBlob = new Blob(this.audioChunks, {
      type: this.mediaRecorder.mimeType || "audio/webm",
    });

    const turnId = this.currentTurnId;
    const t0 = performance.now();

    // If WebSocket is open, send via WebSocket
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (this.state === "cancelled" || this.currentTurnId !== turnId) return;
        const base64Data = reader.result.split(",")[1];
        const uploadLatency = performance.now() - t0;
        this.updateLatencyBar("upload", uploadLatency);

        this.socket.send(
          JSON.stringify({
            type: "audio.chunk",
            turn_id: turnId,
            data: base64Data,
          })
        );
        this.socket.send(
          JSON.stringify({
            type: "audio.end",
            turn_id: turnId,
            client_upload_ms: uploadLatency,
          })
        );
      };
      reader.readAsDataURL(audioBlob);
    } else {
      // Fallback: POST /v1/voice/{session_id}/turn
      await this.executeHttpTurn(audioBlob, turnId, t0);
    }
  }

  async executeHttpTurn(audioBlob, turnId, t0) {
    const formData = new FormData();
    formData.append("file", audioBlob, "speech.webm");
    formData.append("turn_id", turnId);
    formData.append("client_upload_ms", (performance.now() - t0).toString());

    try {
      this.transitionState("transcribing");
      const res = await fetch(`/v1/voice/${this.sessionId}/turn`, {
        method: "POST",
        body: formData,
        signal: this.abortController.signal,
      });

      const data = await res.json();
      if (this.currentTurnId !== turnId) return;

      if (!res.ok) {
        this.showError(data.error || "Turn processing failed", data.error_code || "error");
        if (data.assistant_response) {
          this.appendTurnToChat("[Error]", data.assistant_response);
        }
        return;
      }

      this.updateWaterfall(data.timings || {});
      this.appendTurnToChat(
        data.user_transcript,
        data.assistant_response,
        data.tool_used,
        data.tool_result,
        data.timings
      );

      if (data.audio_base64) {
        this.transitionState("speaking");
        this.playAudioBase64(data.audio_base64, data.audio_format);
      } else {
        this.transitionState("idle");
      }

      this.fetchTelemetryStats();
    } catch (err) {
      if (err.name === "AbortError") {
        console.log("[HTTP Turn Aborted by client]");
      } else {
        console.error("HTTP turn error:", err);
        this.showError("Network request failed while sending audio", "network_error");
      }
    }
  }

  async runScriptedTurn(promptText) {
    if (this.state !== "idle" && this.state !== "cancelled" && this.state !== "error") {
      console.warn("Turn in flight; cannot run scripted prompt");
      return;
    }

    this.clearError();
    this.abortController = new AbortController();
    this.currentTurnId = `script_${Date.now()}`;
    this.dom.turnIdTag.textContent = `Turn: ${this.currentTurnId}`;
    this.transitionState("uploading");

    // Generate lightweight synthesized audio fixture in browser or request turn
    try {
      // Create simple synthetic audio buffer (sine harmonic tone 1.5s)
      const sampleRate = 16000;
      const duration = 1.2;
      const numSamples = sampleRate * duration;
      const buffer = new ArrayBuffer(44 + numSamples * 2);
      const view = new DataView(buffer);

      // Write standard 44-byte WAV header
      this.writeWavHeader(view, numSamples, sampleRate);
      for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate;
        const val = 0.3 * Math.sin(2 * Math.PI * 300 * t) + 0.15 * Math.sin(2 * Math.PI * 600 * t);
        view.setInt16(44 + i * 2, val * 32767, true);
      }

      const audioBlob = new Blob([buffer], { type: "audio/wav" });
      const t0 = performance.now();
      await this.executeHttpTurn(audioBlob, this.currentTurnId, t0);
    } catch (err) {
      console.error("Scripted turn error:", err);
      this.showError("Failed to trigger scripted demo turn", "error");
    }
  }

  writeWavHeader(view, numSamples, sampleRate) {
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    writeString(0, "RIFF");
    view.setUint32(4, 36 + numSamples * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, numSamples * 2, true);
  }

  playAudioBase64(base64Data, format = "audio/wav") {
    try {
      const mime = format.startsWith("audio/") ? format : format === "mp3" ? "audio/mp3" : "audio/wav";
      const audioUrl = `data:${mime};base64,${base64Data}`;
      this.currentPlaybackAudio = new Audio(audioUrl);

      this.currentPlaybackAudio.onended = () => {
        this.currentPlaybackAudio = null;
        if (this.state === "speaking") {
          this.transitionState("idle");
        }
      };

      this.currentPlaybackAudio.onerror = (e) => {
        console.warn("Audio playback error:", e);
        this.currentPlaybackAudio = null;
        this.transitionState("idle");
      };

      this.currentPlaybackAudio.play().catch((err) => {
        console.warn("Audio autoplay blocked by browser policy:", err);
        this.transitionState("idle");
      });
    } catch (e) {
      console.warn("Playback initialization error:", e);
      this.transitionState("idle");
    }
  }

  startAudioVisualizer(stream) {
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = this.audioContext.createMediaStreamSource(stream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 64;
      source.connect(this.analyser);

      const canvas = this.dom.waveformCanvas;
      const ctx = canvas.getContext("2d");
      const bufferLength = this.analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const draw = () => {
        this.animFrameId = requestAnimationFrame(draw);
        this.analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = "#0b1120";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        let sum = 0;
        const barWidth = (canvas.width / bufferLength) * 1.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const barHeight = (dataArray[i] / 255.0) * canvas.height;
          sum += dataArray[i];

          ctx.fillStyle = "#38bdf8";
          ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
          x += barWidth;
        }

        const avg = sum / bufferLength;
        const vuPercent = Math.min(100, Math.round((avg / 128.0) * 100));
        this.dom.vuMeterBar.style.width = `${vuPercent}%`;
      };

      draw();
    } catch (err) {
      console.warn("Visualizer failed to initialize:", err);
    }
  }

  stopAudioVisualizer() {
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.audioContext && this.audioContext.state !== "closed") {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
      this.mediaStream = null;
    }
    this.dom.vuMeterBar.style.width = "0%";
    const canvas = this.dom.waveformCanvas;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  updateLatencyBar(stage, latencyMs) {
    const ms = Math.round(latencyMs || 0);
    const maxScaleMs = 4000.0;
    const percent = Math.min(100, Math.max(3, (ms / maxScaleMs) * 100));

    const map = {
      upload: { val: this.dom.valUpload, bar: this.dom.barUpload },
      stt: { val: this.dom.valStt, bar: this.dom.barStt },
      validate_audio: { val: this.dom.valUpload, bar: this.dom.barUpload },
      agent: { val: this.dom.valAgent, bar: this.dom.barAgent },
      execute_tool: { val: this.dom.valAgent, bar: this.dom.barAgent },
      generate_response: { val: this.dom.valAgent, bar: this.dom.barAgent },
      tts: { val: this.dom.valTts, bar: this.dom.barTts },
      synthesize_speech: { val: this.dom.valTts, bar: this.dom.barTts },
      total: { val: this.dom.valTotal, bar: this.dom.barTotal },
    };

    if (map[stage]) {
      map[stage].val.textContent = `${ms} ms`;
      map[stage].bar.style.width = `${percent}%`;
    }
  }

  updateWaterfall(timings) {
    if (timings.upload !== undefined) this.updateLatencyBar("upload", timings.upload);
    if (timings.stt !== undefined) this.updateLatencyBar("stt", timings.stt);
    if (timings.agent !== undefined) this.updateLatencyBar("agent", timings.agent);
    if (timings.tts !== undefined) this.updateLatencyBar("tts", timings.tts);
    if (timings.total !== undefined) this.updateLatencyBar("total", timings.total);
  }

  appendTurnToChat(userText, assistantText, toolName = null, toolResult = null, latencies = null) {
    const card = document.createElement("div");
    card.className = "turn-card";

    let toolHtml = "";
    if (toolName) {
      toolHtml = `<div class="turn-tool-badge">⚙️ Tool (${toolName}): ${toolResult || "Executed"}</div>`;
    }

    let latenciesHtml = "";
    if (latencies && latencies.total) {
      latenciesHtml = `
        <div class="turn-latencies">
          <span>STT: ${Math.round(latencies.stt || 0)}ms</span>
          <span>Agent: ${Math.round(latencies.agent || 0)}ms</span>
          <span>TTS: ${Math.round(latencies.tts || 0)}ms</span>
          <span><strong>Total: ${Math.round(latencies.total)}ms</strong></span>
        </div>
      `;
    }

    card.innerHTML = `
      <div class="turn-user">
        <span class="role-tag">User:</span>
        <span>${escapeHtml(userText)}</span>
      </div>
      ${toolHtml}
      <div class="turn-assistant">${escapeHtml(assistantText)}</div>
      ${latenciesHtml}
    `;

    this.dom.chatContainer.appendChild(card);
    this.dom.chatContainer.scrollTop = this.dom.chatContainer.scrollHeight;
  }

  async loadSessionHistory() {
    if (!this.sessionId) return;
    try {
      const res = await fetch(`/v1/sessions/${this.sessionId}`);
      if (!res.ok) return;
      const data = await res.json();
      if (data.turns && data.turns.length > 0) {
        this.dom.chatContainer.innerHTML = "";
        data.turns.forEach((turn) => {
          this.appendTurnToChat(
            turn.user_transcript,
            turn.assistant_response,
            turn.tool_used,
            turn.tool_output,
            turn.latencies
          );
        });
      }
    } catch (err) {
      console.warn("Failed loading session history:", err);
    }
  }

  async resetDemo() {
    if (!confirm("Reset demo state? This will clear active sessions and reseed the Green Grid demo scenario.")) {
      return;
    }

    try {
      const res = await fetch("/api/demo/reset", {
        method: "POST",
        headers: { "X-Demo-Key": "voicepilot-demo-key" },
      });
      const data = await res.json();
      this.sessionId = data.demo_session_id;
      this.dom.sessionIdDisplay.textContent = this.sessionId;
      this.clearError();
      this.transitionState("idle", "Demo state reseeded");
      await this.loadSessionHistory();
      await this.fetchTelemetryStats();
    } catch (err) {
      this.showError("Failed to reset demo state: " + err.message);
    }
  }

  exportSessionJson() {
    if (!this.sessionId) return;
    window.open(`/v1/sessions/${this.sessionId}/export`, "_blank");
  }

  async fetchTelemetryStats() {
    try {
      const res = await fetch("/v1/metrics");
      if (!res.ok) return;
      const data = await res.json();
      this.dom.turnCountBadge.textContent = `Turns: ${data.total_turns}`;

      const rows = [];
      for (const [stage, s] of Object.entries(data.stages)) {
        rows.push(`
          <tr>
            <td><strong>${stage}</strong></td>
            <td>${s.mean_ms} ms</td>
            <td>${s.p50_ms} ms</td>
            <td>${s.p95_ms} ms</td>
          </tr>
        `);
      }
      this.dom.statsTableBody.innerHTML = rows.join("");
    } catch (err) {
      console.warn("Failed fetching telemetry:", err);
    }
  }
}

function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Bootstrap on DOM loaded
document.addEventListener("DOMContentLoaded", () => {
  window.voicePilotApp = new VoicePilotClient();
});

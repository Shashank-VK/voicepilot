"""Tests for FastAPI HTTP endpoints and WebSocket streaming."""

import io
import json

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")
    assert "ffmpeg" in data["dependencies"]


def test_session_lifecycle_and_export(client: TestClient):
    # 1. Create session
    create_resp = client.post("/v1/sessions")
    assert create_resp.status_code == 201
    sid = create_resp.json()["session_id"]
    assert sid is not None

    # 2. Get session
    get_resp = client.get(f"/v1/sessions/{sid}")
    assert get_resp.status_code == 200
    assert get_resp.json()["session_id"] == sid

    # 3. Export session transcript
    export_resp = client.get(f"/v1/sessions/{sid}/export")
    assert export_resp.status_code == 200
    export_data = export_resp.json()
    assert export_data["session_id"] == sid
    assert "transcript_turns" in export_data


def test_turn_valid_speech(client: TestClient, speech_wav: bytes):
    create_resp = client.post("/v1/sessions")
    sid = create_resp.json()["session_id"]

    files = {"file": ("speech.wav", io.BytesIO(speech_wav), "audio/wav")}
    data = {"turn_id": "test_turn_01", "client_upload_ms": "25.0"}

    resp = client.post(f"/v1/voice/{sid}/turn", files=files, data=data)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["success"] is True
    assert res_data["turn_id"] == "test_turn_01"
    assert "timings" in res_data
    assert res_data["audio_base64"] is not None


def test_turn_webm_audio(client: TestClient, webm_audio: bytes):
    create_resp = client.post("/v1/sessions")
    sid = create_resp.json()["session_id"]

    files = {"file": ("speech.webm", io.BytesIO(webm_audio), "audio/webm")}
    resp = client.post(f"/v1/voice/{sid}/turn", files=files)
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_turn_mp4_audio(client: TestClient, mp4_audio: bytes):
    create_resp = client.post("/v1/sessions")
    sid = create_resp.json()["session_id"]

    files = {"file": ("speech.mp4", io.BytesIO(mp4_audio), "audio/mp4")}
    resp = client.post(f"/v1/voice/{sid}/turn", files=files)
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_turn_silence_rejected_actionable_error(client: TestClient, silence_wav: bytes):
    create_resp = client.post("/v1/sessions")
    sid = create_resp.json()["session_id"]

    files = {"file": ("silence.wav", io.BytesIO(silence_wav), "audio/wav")}
    resp = client.post(f"/v1/voice/{sid}/turn", files=files)
    assert resp.status_code == 400
    res_data = resp.json()
    assert res_data["success"] is False
    assert res_data["error_code"] == "no_speech"
    assert "no speech" in res_data["error"].lower()


def test_turn_noisy_rejected_actionable_error(client: TestClient, noisy_wav: bytes):
    create_resp = client.post("/v1/sessions")
    sid = create_resp.json()["session_id"]

    files = {"file": ("noise.wav", io.BytesIO(noisy_wav), "audio/wav")}
    resp = client.post(f"/v1/voice/{sid}/turn", files=files)
    assert resp.status_code == 400
    res_data = resp.json()
    assert res_data["error_code"] in ("too_noisy", "no_speech")


def test_turn_clipped_rejected_actionable_error(client: TestClient, clipped_wav: bytes):
    create_resp = client.post("/v1/sessions")
    sid = create_resp.json()["session_id"]

    files = {"file": ("clipped.wav", io.BytesIO(clipped_wav), "audio/wav")}
    resp = client.post(f"/v1/voice/{sid}/turn", files=files)
    assert resp.status_code == 400
    res_data = resp.json()
    assert res_data["error_code"] == "clipping"


def test_metrics_endpoint(client: TestClient):
    resp = client.get("/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "stages" in data
    assert "total_turns" in data


def test_websocket_stream_and_turn_completion(client: TestClient, speech_wav: bytes):
    import base64

    create_resp = client.post("/v1/sessions")
    sid = create_resp.json()["session_id"]

    with client.websocket_connect(f"/v1/voice/{sid}") as ws:
        # Send audio in chunks
        chunk_b64 = base64.b64encode(speech_wav).decode("utf-8")
        ws.send_text(json.dumps({"type": "audio.chunk", "turn_id": "ws_turn_01", "data": chunk_b64}))
        ws.send_text(json.dumps({"type": "audio.end", "turn_id": "ws_turn_01", "client_upload_ms": 15.0}))

        # Read envelopes
        envelopes = []
        # Receive events until turn.completed
        for _ in range(10):
            raw = ws.receive_text()
            env = json.loads(raw)
            envelopes.append(env)
            if env.get("type") == "turn.completed":
                break

        types = [e.get("type") for e in envelopes]
        assert "stage.started" in types
        assert "stage.completed" in types
        assert "speech.ready" in types
        assert "turn.completed" in types


def test_websocket_cancellation(client: TestClient):
    create_resp = client.post("/v1/sessions")
    sid = create_resp.json()["session_id"]

    with client.websocket_connect(f"/v1/voice/{sid}") as ws:
        ws.send_text(json.dumps({"type": "turn.cancel", "turn_id": "cancel_01"}))
        raw = ws.receive_text()
        env = json.loads(raw)
        assert env.get("type") == "turn.cancelled"
        assert env.get("turn_id") == "cancel_01"

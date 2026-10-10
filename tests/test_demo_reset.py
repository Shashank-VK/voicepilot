"""Tests for authenticated POST /api/demo/reset endpoint and safety guards."""

from fastapi.testclient import TestClient

from voicepilot.config import Settings
from voicepilot.server.app import create_app, set_app_graph


def test_demo_reset_success(client: TestClient):
    # 1. Create arbitrary session
    res_s = client.post("/v1/sessions")
    custom_sid = res_s.json()["session_id"]

    # 2. Trigger reset
    resp = client.post("/api/demo/reset", headers={"X-Demo-Key": "voicepilot-demo-key"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["demo_session_id"] == "demo-session-greengrid"
    assert data["reseeded_turns"] == 3

    # 3. Old session should be gone
    res_old = client.get(f"/v1/sessions/{custom_sid}")
    assert res_old.status_code == 404

    # 4. Demo session should be available with 3 turns
    res_demo = client.get("/v1/sessions/demo-session-greengrid")
    assert res_demo.status_code == 200
    demo_data = res_demo.json()
    assert len(demo_data["turns"]) == 3
    assert demo_data["preferences"].get("preferred_unit") == "kWh"


def test_demo_reset_idempotent(client: TestClient):
    """Calling demo reset multiple times must always succeed and leave a clean state."""
    for _ in range(3):
        resp = client.post("/api/demo/reset", headers={"X-Demo-Key": "voicepilot-demo-key"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_demo_reset_disabled_outside_demo_mode(test_graph):
    """When DEMO_MODE=False, POST /api/demo/reset must return 403 Forbidden."""
    prod_settings = Settings(
        DEMO_MODE=False,
        STT_PROVIDER="mock",
        TTS_PROVIDER="mock",
    )
    prod_app = create_app(prod_settings)
    set_app_graph(test_graph)

    with TestClient(prod_app) as prod_client:
        resp = prod_client.post("/api/demo/reset")
        assert resp.status_code == 403
        assert "disabled outside demo mode" in resp.json()["detail"].lower()


def test_demo_reset_invalid_key_fails(client: TestClient):
    """Invalid demo key should return 401 Unauthorized."""
    resp = client.post("/api/demo/reset", headers={"X-Demo-Key": "wrong-key"})
    assert resp.status_code == 401

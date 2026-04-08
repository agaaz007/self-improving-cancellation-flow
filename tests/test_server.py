"""Tests for the online bandit serving API."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

os.environ["BANDIT_STATE_DIR"] = ""  # overridden per test


def _make_state_dir(tmp_path: Path) -> Path:
    state_dir = tmp_path / "policy"
    state_dir.mkdir()
    os.environ["BANDIT_STATE_DIR"] = str(state_dir)
    return state_dir


def _reset_server():
    import cta_autoresearch.server as srv
    srv._runtime = None
    srv._last_mtime = 0.0
    srv._last_reload = 0.0
    srv._STATE_DIR = Path(os.environ["BANDIT_STATE_DIR"])


@pytest.fixture(autouse=True)
def _setup(tmp_path):
    _make_state_dir(tmp_path)
    _reset_server()
    yield
    _reset_server()


def _client():
    from fastapi.testclient import TestClient
    from cta_autoresearch.server import app
    return TestClient(app)


SAMPLE_CONTEXT = {
    "session_id": "sess_test_001",
    "user_id_hash": "user_abc",
    "plan_tier": "starter",
    "tenure_days": 90,
    "engagement_7d": 0.6,
    "engagement_30d": 0.4,
    "prior_cancel_attempts_30d": 0,
    "discount_exposures_30d": 0,
    "transcript_extraction": {
        "primary_reason": "price",
        "intent_strength": 0.7,
        "save_openness": 0.5,
    },
}


def test_decide_returns_decision():
    client = _client()
    resp = client.post("/decide", json=SAMPLE_CONTEXT)
    assert resp.status_code == 200
    data = resp.json()
    assert "decision" in data
    assert "action" in data
    decision = data["decision"]
    assert "decision_id" in decision
    assert "action_id" in decision
    assert "propensity" in decision


def test_decide_invalid_body():
    client = _client()
    resp = client.post("/decide", json={"bad": "data"})
    assert resp.status_code == 422


def test_outcome_records():
    client = _client()
    # First make a decision
    resp = client.post("/decide", json=SAMPLE_CONTEXT)
    decision = resp.json()["decision"]

    # Record outcome
    outcome_body = {
        "decision_id": decision["decision_id"],
        "session_id": SAMPLE_CONTEXT["session_id"],
        "saved_flag": True,
        "cancel_completed_flag": False,
        "support_escalation_flag": False,
        "complaint_flag": False,
    }
    resp = client.post("/outcome", json=outcome_body)
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"


def test_health():
    client = _client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "decisions" in data
    assert "last_reload" in data


def test_hot_reload(tmp_path):
    client = _client()

    # Make a decision to init runtime
    client.post("/decide", json=SAMPLE_CONTEXT)

    # Get current health
    health1 = client.get("/health").json()

    # Write new state with different exploration_rate
    state_dir = Path(os.environ["BANDIT_STATE_DIR"])
    state_path = state_dir / "policy_state.json"
    state = json.loads(state_path.read_text())
    state["config"]["exploration_rate"] = 0.99
    # Touch with future mtime
    state_path.write_text(json.dumps(state))
    os.utime(state_path, (time.time() + 1, time.time() + 1))

    # Next decide should reload
    client.post("/decide", json=SAMPLE_CONTEXT)
    health2 = client.get("/health").json()
    assert health2["exploration_rate"] == 0.99


def test_reload_endpoint():
    client = _client()
    resp = client.post("/reload")
    assert resp.status_code == 200
    assert resp.json()["status"] == "reloaded"


def test_dashboard_api_arms():
    client = _client()
    # Make a decision to populate state
    client.post("/decide", json=SAMPLE_CONTEXT)
    resp = client.get("/api/arms")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) > 0
    first = next(iter(data.values()))
    assert "mean" in first
    assert "ci_low" in first
    assert "impressions" in first


def test_dashboard_api_replay():
    client = _client()
    # Decision + outcome to have replay data
    resp = client.post("/decide", json=SAMPLE_CONTEXT)
    decision = resp.json()["decision"]
    client.post("/outcome", json={
        "decision_id": decision["decision_id"],
        "session_id": SAMPLE_CONTEXT["session_id"],
        "saved_flag": True,
        "cancel_completed_flag": False,
        "support_escalation_flag": False,
        "complaint_flag": False,
    })
    resp = client.get("/api/replay")
    assert resp.status_code == 200
    data = resp.json()
    assert "save_rate" in data
    assert "by_action" in data
    assert "by_reason" in data


def test_dashboard_api_strategies():
    client = _client()
    resp = client.get("/api/strategies")
    assert resp.status_code == 200


def test_dashboard_api_generation():
    client = _client()
    resp = client.get("/api/generation")
    assert resp.status_code == 200
    data = resp.json()
    assert "config" in data

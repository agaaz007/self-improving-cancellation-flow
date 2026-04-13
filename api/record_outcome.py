"""Vercel serverless: record retention outcome and update bandit posteriors."""
from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from cta_autoresearch import redis_state


def _reward(outcome: dict) -> float:
    r = 1.0 if outcome.get("saved_flag") else 0.0
    if outcome.get("support_escalation_flag"):
        r -= 0.3
    if outcome.get("complaint_flag"):
        r -= 0.2
    return r


def _update_posteriors(state: dict, action_id: str, reason: str, plan_tier: str, saved: bool, reward: float) -> None:
    # Global arm
    g = state.setdefault("arms_global", {}).setdefault(action_id, {"alpha": 1, "beta": 1, "impressions": 0, "outcomes": 0})
    s = 1.0 if saved else 0.0
    g["alpha"] = float(g["alpha"]) + s + max(reward, 0)
    g["beta"] = float(g["beta"]) + (1 - s) + max(-reward, 0)
    g["impressions"] = float(g["impressions"]) + 1
    g["outcomes"] = float(g["outcomes"]) + s

    # Context arm
    ctx_key = f"{reason}|{plan_tier}|{action_id}"
    c = state.setdefault("arms_context", {}).setdefault(ctx_key, {"alpha": 1, "beta": 1, "impressions": 0, "outcomes": 0})
    c["alpha"] = float(c["alpha"]) + s + max(reward, 0)
    c["beta"] = float(c["beta"]) + (1 - s) + max(-reward, 0)
    c["impressions"] = float(c["impressions"]) + 1
    c["outcomes"] = float(c["outcomes"]) + s

    # Metrics
    m = state.setdefault("metrics", {"decisions": 0, "outcomes": 0, "fallbacks": 0, "blocked_actions_total": 0})
    m["outcomes"] = int(m.get("outcomes", 0)) + 1


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        if not redis_state.available():
            return self._respond(503, {"error": "Redis not configured"})

        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except (json.JSONDecodeError, ValueError):
            return self._respond(400, {"error": "Invalid JSON"})

        decision_id = str(body.get("decision_id") or "").strip()
        if not decision_id:
            return self._respond(400, {"error": "decision_id required"})

        client_id = str(body.get("client_id") or "zeo_auto")

        decision = redis_state.get_decision(client_id, decision_id)
        if not decision:
            return self._respond(404, {"error": f"Unknown decision_id: {decision_id}"})

        reward = _reward(body)
        action_id = decision.get("action_id", "")
        reason = decision.get("primary_reason", "other")
        plan_tier = decision.get("plan_tier", "unknown")

        state = redis_state.get_policy(client_id)
        if not state:
            return self._respond(500, {"error": "No policy state in Redis"})

        _update_posteriors(state, action_id, reason, plan_tier, bool(body.get("saved_flag")), reward)
        state["updated_at"] = time.time()
        redis_state.set_policy(client_id, state)

        outcome_record = {
            "decision_id": decision_id,
            "action_id": action_id,
            "primary_reason": reason,
            "plan_tier": plan_tier,
            "saved_flag": bool(body.get("saved_flag")),
            "cancel_completed_flag": bool(body.get("cancel_completed_flag")),
            "support_escalation_flag": bool(body.get("support_escalation_flag")),
            "complaint_flag": bool(body.get("complaint_flag")),
            "reward": reward,
            "outcome_timestamp": time.time(),
        }
        redis_state.save_outcome(client_id, outcome_record)

        self._respond(200, {"status": "recorded", "reward": reward, "action_id": action_id})

    def do_GET(self):
        self._respond(200, {
            "endpoint": "POST /api/record_outcome",
            "required": {"decision_id": "str", "client_id": "str (zeo_auto|jungle_ai)"},
            "optional": {"saved_flag": "bool", "cancel_completed_flag": "bool", "support_escalation_flag": "bool", "complaint_flag": "bool"},
        })

    def _respond(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

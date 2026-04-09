"""Vercel serverless: dashboard data API — serves all views from Redis state."""
from __future__ import annotations

import json
import math
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from cta_autoresearch import redis_state


def _beta_ci(alpha, beta, z=1.96):
    a, b = float(alpha), float(beta)
    mean = a / (a + b) if (a + b) > 0 else 0.5
    var = (a * b) / ((a + b) ** 2 * (a + b + 1)) if (a + b + 1) > 0 else 0.01
    hw = z * math.sqrt(var)
    return mean, max(0, mean - hw), min(1, mean + hw)


def _view_health(state):
    m = state.get("metrics", {})
    cfg = state.get("config", {})
    return {
        "status": "ok",
        "policy_version": state.get("policy_version", "?"),
        "last_reload": state.get("updated_at", 0),
        "decisions": m.get("decisions", 0),
        "outcomes": m.get("outcomes", 0),
        "fallbacks": m.get("fallbacks", 0),
        "blocked_actions_total": m.get("blocked_actions_total", 0),
        "exploration_rate": cfg.get("exploration_rate", 0),
        "holdout_rate": cfg.get("holdout_rate", 0),
        "actions": len(state.get("arms_global", {})),
        "latency": {},
    }


def _view_arms(state):
    out = {}
    for action_id, arm in state.get("arms_global", {}).items():
        mean, lo, hi = _beta_ci(arm.get("alpha", 1), arm.get("beta", 1))
        imp = arm.get("impressions", 0)
        outcomes = arm.get("outcomes", 0)
        out[action_id] = {
            "label": action_id.replace("_", " ").title(),
            "mean": round(mean, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "impressions": int(imp),
            "win_rate": round(outcomes / imp, 4) if imp > 0 else 0,
        }
    return out


def _view_context_arms(state):
    return state.get("arms_context", {})


def _view_replay(state, outcomes):
    by_action, by_reason = {}, {}
    for o in outcomes:
        for group, key in [(by_action, o.get("action_id", "")), (by_reason, o.get("primary_reason", ""))]:
            if key not in group:
                group[key] = {"rows": 0, "saved": 0, "total_reward": 0}
            group[key]["rows"] += 1
            if o.get("saved_flag"):
                group[key]["saved"] += 1
            group[key]["total_reward"] += o.get("reward", 0)
    for group in (by_action, by_reason):
        for v in group.values():
            v["save_rate"] = round(v["saved"] / v["rows"], 4) if v["rows"] else 0
            v["average_reward"] = round(v["total_reward"] / v["rows"], 4) if v["rows"] else 0
    return {"by_action": by_action, "by_reason": by_reason}


def _view_strategies(state):
    pool = state.get("strategy_pool", {})
    out = {}
    sa = state.get("strategy_arms", {})
    for action_id, strats in pool.items():
        items = []
        for sid, s in sorted(strats.items(), key=lambda x: -float(x[1].get("mean_score", 0))):
            arm_key = f"{action_id}:{sid}"
            arm = sa.get(arm_key, {})
            arm_mean, _, _ = _beta_ci(arm.get("alpha", 1), arm.get("beta", 1)) if arm else (0, 0, 0)
            items.append({
                "id": sid,
                "mean_score": round(float(s.get("mean_score", 0)), 4),
                "arm_mean": round(arm_mean, 4),
                "generation": s.get("generation", 0),
                "dims": s.get("dims", {}),
            })
        out[action_id] = items
    return out


def _view_generation(state):
    return {
        "generation_meta": state.get("generation_meta", {}),
        "config": state.get("config", {}),
        "updated_at": state.get("updated_at", 0),
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            return self._handle_get()
        except Exception as e:
            return self._respond(500, {"error": str(e), "type": type(e).__name__})

    def _handle_get(self):
        qs = parse_qs(urlparse(self.path).query)
        client_id = qs.get("client", ["zeo_auto"])[0]
        view = qs.get("view", ["health"])[0]

        if not redis_state.available():
            return self._respond(503, {"error": "Redis not configured"})

        state = redis_state.get_policy(client_id)
        if not state:
            return self._respond(404, {"error": f"No policy for {client_id}"})

        if view == "health":
            data = _view_health(state)
        elif view == "arms":
            data = _view_arms(state)
        elif view == "context-arms":
            data = _view_context_arms(state)
        elif view == "replay":
            outcomes = redis_state.recent_outcomes(client_id, 500)
            data = _view_replay(state, outcomes)
        elif view == "strategies":
            data = _view_strategies(state)
        elif view == "generation":
            data = _view_generation(state)
        elif view == "recent-decisions":
            data = redis_state.recent_decisions(client_id, 30)
        elif view == "recent-outcomes":
            data = redis_state.recent_outcomes(client_id, 30)
        else:
            return self._respond(400, {"error": f"Unknown view: {view}"})

        self._respond(200, data)

    def _respond(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

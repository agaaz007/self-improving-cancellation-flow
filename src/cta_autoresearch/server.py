"""Online bandit serving API.

Wraps CancelPolicyRuntime in a FastAPI app with hot-reload.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cta_autoresearch.cancel_policy import (
    CancelContextV1,
    CancelOutcomeV1,
    CancelPolicyRuntime,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="CTA Bandit", version="0.1.0")

_STATE_DIR = Path(os.environ.get("BANDIT_STATE_DIR", "serving_data/policy"))
_runtime: CancelPolicyRuntime | None = None
_last_mtime: float = 0.0
_last_reload: float = 0.0


def _get_runtime() -> CancelPolicyRuntime:
    global _runtime, _last_mtime, _last_reload
    state_path = _STATE_DIR / "policy_state.json"

    if _runtime is None:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _runtime = CancelPolicyRuntime(str(_STATE_DIR))
        _last_mtime = state_path.stat().st_mtime if state_path.exists() else 0.0
        _last_reload = time.time()
        logger.info("Runtime loaded from %s", _STATE_DIR)
        return _runtime

    # Hot-reload: check mtime
    try:
        current_mtime = state_path.stat().st_mtime
    except FileNotFoundError:
        return _runtime

    if current_mtime > _last_mtime:
        _runtime.state = _runtime._load_state()
        config = _runtime.state.get("config") or {}
        if "exploration_rate" in config:
            _runtime.exploration_rate = float(config["exploration_rate"])
        if "holdout_rate" in config:
            _runtime.holdout_rate = float(config["holdout_rate"])
        if "discount_cap_30d" in config:
            _runtime.discount_cap_30d = int(config["discount_cap_30d"])
        _last_mtime = current_mtime
        _last_reload = time.time()
        logger.info("Hot-reloaded policy state (mtime %.0f)", current_mtime)

    return _runtime


@app.post("/decide")
async def decide(request: Request) -> JSONResponse:
    body = await request.json()
    try:
        context = CancelContextV1.from_dict(body)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    runtime = _get_runtime()
    decision = runtime.decide(context)

    action_def = runtime.actions.get(decision.action_id)
    return JSONResponse({
        "decision": decision.to_dict(),
        "action": action_def.to_dict() if action_def else None,
    })


@app.post("/outcome")
async def outcome(request: Request) -> JSONResponse:
    body = await request.json()
    try:
        cancel_outcome = CancelOutcomeV1.from_dict(body)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    runtime = _get_runtime()
    result = runtime.record_outcome(cancel_outcome)
    return JSONResponse(result)


@app.get("/health")
async def health() -> JSONResponse:
    runtime = _get_runtime()
    payload = runtime.health()
    payload["last_reload"] = _last_reload
    payload["state_dir"] = str(_STATE_DIR)
    return JSONResponse(payload)


@app.post("/reload")
async def reload() -> JSONResponse:
    global _runtime
    _runtime = None
    runtime = _get_runtime()
    return JSONResponse({"status": "reloaded", "policy_version": runtime.state.get("policy_version")})


# ---------------------------------------------------------------------------
# Dashboard data endpoints
# ---------------------------------------------------------------------------

@app.get("/api/arms")
async def api_arms() -> JSONResponse:
    """Global arm posteriors with derived stats."""
    runtime = _get_runtime()
    arms = runtime.state.get("arms_global", {})
    result = {}
    for action_id, arm in arms.items():
        a, b = float(arm["alpha"]), float(arm["beta"])
        n = float(arm.get("impressions", 0))
        outcomes = float(arm.get("outcomes", 0))
        mean = a / (a + b) if (a + b) > 0 else 0.5
        # 95% credible interval approximation
        std = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))) if (a + b) > 1 else 0.25
        label = runtime.actions[action_id].label if action_id in runtime.actions else action_id
        result[action_id] = {
            "label": label,
            "alpha": round(a, 2),
            "beta": round(b, 2),
            "mean": round(mean, 4),
            "ci_low": round(max(0, mean - 1.96 * std), 4),
            "ci_high": round(min(1, mean + 1.96 * std), 4),
            "impressions": int(n),
            "outcomes": int(outcomes),
            "win_rate": round(outcomes / n, 4) if n > 0 else 0.0,
        }
    return JSONResponse(result)


@app.get("/api/context-arms")
async def api_context_arms() -> JSONResponse:
    """Context arm posteriors grouped by reason|plan."""
    runtime = _get_runtime()
    ctx_arms = runtime.state.get("arms_context", {})
    result = {}
    for key, arm in ctx_arms.items():
        a, b = float(arm["alpha"]), float(arm["beta"])
        n = float(arm.get("impressions", 0))
        mean = a / (a + b) if (a + b) > 0 else 0.5
        result[key] = {
            "alpha": round(a, 2),
            "beta": round(b, 2),
            "mean": round(mean, 4),
            "impressions": int(n),
            "outcomes": int(arm.get("outcomes", 0)),
        }
    return JSONResponse(result)


@app.get("/api/replay")
async def api_replay() -> JSONResponse:
    """Replay metrics from recorded decisions+outcomes."""
    runtime = _get_runtime()
    return JSONResponse(runtime.replay())


@app.get("/api/strategies")
async def api_strategies() -> JSONResponse:
    """Strategy pool with scores per action."""
    runtime = _get_runtime()
    pool = runtime.state.get("strategy_pool", {})
    strategy_arms = runtime.state.get("strategy_arms", {})
    result = {}
    for action_id, strategies in pool.items():
        action_strats = []
        for sid, sdata in strategies.items():
            arm_key = f"{action_id}:{sid}"
            arm = strategy_arms.get(arm_key, {})
            a = float(arm.get("alpha", 1))
            b = float(arm.get("beta", 1))
            action_strats.append({
                "id": sid,
                "dims": sdata.get("dims", {}),
                "mean_score": round(float(sdata.get("mean_score", 0)), 4),
                "score_std": round(float(sdata.get("score_std", 0)), 4),
                "n_evals": int(sdata.get("n_evals", 0)),
                "generation": int(sdata.get("generation", 0)),
                "arm_mean": round(a / (a + b), 4) if (a + b) > 0 else 0.5,
            })
        action_strats.sort(key=lambda s: s["mean_score"], reverse=True)
        result[action_id] = action_strats
    return JSONResponse(result)


@app.get("/api/generation")
async def api_generation() -> JSONResponse:
    """Optimization generation metadata."""
    runtime = _get_runtime()
    meta = runtime.state.get("generation_meta", {})
    config = runtime.state.get("config", {})
    return JSONResponse({
        "generation_meta": meta,
        "config": config,
        "updated_at": runtime.state.get("updated_at"),
        "created_at": runtime.state.get("created_at"),
    })


@app.get("/api/recent-decisions")
async def api_recent_decisions() -> JSONResponse:
    """Last 50 decisions from journal."""
    runtime = _get_runtime()
    decisions = list(runtime._decision_index.values())
    decisions.sort(key=lambda d: float(d.get("timestamp", 0)), reverse=True)
    return JSONResponse(decisions[:50])


@app.get("/api/recent-outcomes")
async def api_recent_outcomes() -> JSONResponse:
    """Last 50 outcomes from journal."""
    runtime = _get_runtime()
    outcomes = list(runtime._outcome_by_decision.values())
    outcomes.sort(key=lambda o: float(o.get("outcome_timestamp", 0)), reverse=True)
    return JSONResponse(outcomes[:50])


@app.get("/dashboard")
async def dashboard() -> HTMLResponse:
    """Serve the monitoring dashboard."""
    dashboard_path = Path(__file__).parent.parent.parent / "dashboard" / "monitor.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return HTMLResponse(dashboard_path.read_text())


def main() -> None:
    import uvicorn

    port = int(os.environ.get("BANDIT_PORT", "8080"))
    log_level = os.environ.get("BANDIT_LOG_LEVEL", "info").lower()
    uvicorn.run(
        "cta_autoresearch.server:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()

"""Thin Upstash Redis REST wrapper — zero dependencies (stdlib only)."""
from __future__ import annotations

import json
import os
import urllib.request

def _env():
    return os.environ.get("UPSTASH_REDIS_REST_URL", "").strip(), os.environ.get("UPSTASH_REDIS_REST_TOKEN", "").strip()


def _cmd(*args: str) -> object:
    """Execute a single Redis command via Upstash REST API."""
    url, token = _env()
    if not url or not token:
        return None
    data = json.dumps(list(args)).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read()).get("result")


def available() -> bool:
    url, token = _env()
    return bool(url and token)


# -- Policy state --

def get_policy(client_id: str) -> dict | None:
    raw = _cmd("GET", f"policy:{client_id}")
    return json.loads(raw) if raw else None


def set_policy(client_id: str, state: dict) -> None:
    _cmd("SET", f"policy:{client_id}", json.dumps(state, separators=(",", ":")))


# -- Decision index --

def save_decision(client_id: str, decision: dict) -> None:
    blob = json.dumps(decision, separators=(",", ":"))
    _cmd("HSET", f"didx:{client_id}", decision["decision_id"], blob)
    _cmd("LPUSH", f"dlog:{client_id}", blob)
    _cmd("LTRIM", f"dlog:{client_id}", "0", "199")


def get_decision(client_id: str, decision_id: str) -> dict | None:
    raw = _cmd("HGET", f"didx:{client_id}", decision_id)
    return json.loads(raw) if raw else None


def recent_decisions(client_id: str, n: int = 30) -> list[dict]:
    raw = _cmd("LRANGE", f"dlog:{client_id}", "0", str(n - 1))
    return [json.loads(r) for r in (raw or [])]


# -- Outcome log --

def save_outcome(client_id: str, outcome: dict) -> None:
    blob = json.dumps(outcome, separators=(",", ":"))
    _cmd("LPUSH", f"olog:{client_id}", blob)
    _cmd("LTRIM", f"olog:{client_id}", "0", "499")


def recent_outcomes(client_id: str, n: int = 30) -> list[dict]:
    raw = _cmd("LRANGE", f"olog:{client_id}", "0", str(n - 1))
    return [json.loads(r) for r in (raw or [])]

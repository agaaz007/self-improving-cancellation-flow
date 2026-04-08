"""Vercel serverless function: present_offers.

Receives transcript + user_id from 11 Labs agent, runs the trained
Thompson-sampling bandit to pick the best retention action, then
optionally personalizes the UI text via LLM using the optimizer's
best strategy dimensions as creative guardrails.

Flow: bandit picks ACTION → strategy_pool gives best DIMS →
      LLM writes personalized text within those constraints →
      falls back to fixed text if no API key or LLM fails.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

logger = logging.getLogger(__name__)

# Add src/ to import path so cta_autoresearch is importable
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from cta_autoresearch.cancel_policy import (
    CancelContextV1,
    CancelPolicyRuntime,
    TranscriptExtractor,
    TranscriptExtractionV1,
)
from cta_autoresearch.clients import load_client

# ---------------------------------------------------------------------------
# Fixed UI data per action (Jungle AI default)
# ---------------------------------------------------------------------------
_UI_JUNGLE = {
    "control_empathic_exit":  {"header": "We understand", "body": "We can make this easier. Tell us what is not working.", "cta": "Tell us more", "offer_kind": "none"},
    "pause_plan_relief":      {"header": "Take a break instead?", "body": "Pause now and resume when your schedule stabilizes.", "cta": "Pause my plan", "offer_kind": "pause"},
    "downgrade_lite_switch":  {"header": "A lighter plan might work", "body": "Switch to a lighter plan and keep your progress.", "cta": "Switch to Lite", "offer_kind": "downgrade"},
    "targeted_discount_20":   {"header": "We'd like to offer you a deal", "body": "Stay on track with a limited 20% discount.", "cta": "Claim 20% off", "offer_kind": "discount"},
    "targeted_discount_40":   {"header": "We'd like to offer you a deal", "body": "High-savings offer available now if you continue.", "cta": "Claim 40% off", "offer_kind": "discount"},
    "concierge_recovery":     {"header": "Let us help", "body": "Get a guided reset from our learning support team.", "cta": "Talk to support", "offer_kind": "support"},
    "exam_sprint_focus":      {"header": "Finish what you started", "body": "Use an exam sprint path to finish your current goal.", "cta": "Start sprint", "offer_kind": "extension"},
    "feature_value_recap":    {"header": "You're missing out", "body": "Quickly recover value from features you have not used yet.", "cta": "Show me what I'm missing", "offer_kind": "credit"},
    "billing_clarity_reset":  {"header": "Let's sort this out", "body": "See billing details and simpler options before deciding.", "cta": "View billing options", "offer_kind": "billing"},
}

_UI_ZEO = {
    "control_graceful_exit":       {"header": "We understand", "body": "We understand. Let us make the transition easy.", "cta": "Continue canceling", "offer_kind": "none"},
    "pause_plan":                  {"header": "Take a break instead?", "body": "Pause your plan and resume when your routes pick up again.", "cta": "Pause my plan", "offer_kind": "pause"},
    "downgrade_basic":             {"header": "A lighter plan might work", "body": "Switch to a lighter plan and keep your route history.", "cta": "Switch to Basic", "offer_kind": "downgrade"},
    "discount_20":                 {"header": "We'd like to offer you a deal", "body": "Stay on track with a limited 20% discount on your next cycle.", "cta": "Claim 20% off", "offer_kind": "discount"},
    "discount_40":                 {"header": "We'd like to offer you a deal", "body": "High-savings offer available now if you continue your plan.", "cta": "Claim 40% off", "offer_kind": "discount"},
    "route_optimization_demo":     {"header": "Let us show you more", "body": "Let us show you how to get more value from route optimization.", "cta": "Show me how", "offer_kind": "support"},
    "fleet_rightsize":             {"header": "Let's adjust your plan", "body": "Right-size your plan to match your current fleet needs.", "cta": "Adjust my plan", "offer_kind": "credit"},
}

_UI_BY_CLIENT = {
    "jungle_ai": _UI_JUNGLE,
    "zeo_auto": _UI_ZEO,
}

# ---------------------------------------------------------------------------
# Singleton state — survives across warm invocations
# ---------------------------------------------------------------------------
_runtime: CancelPolicyRuntime | None = None
_extractor: TranscriptExtractor | None = None
_ui_data: dict | None = None
_client_id: str | None = None


def _init():
    global _runtime, _extractor, _ui_data, _client_id

    if _runtime is not None:
        return

    _client_id = os.environ.get("CLIENT_ID", "zeo_auto")
    client = load_client(_client_id)

    # Copy bundled policy_state.json to /tmp where the runtime can write
    tmp_dir = Path("/tmp/bandit_policy")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    bundled = Path(__file__).parent / "policy_state.json"
    dst = tmp_dir / "policy_state.json"
    if bundled.exists():
        shutil.copy2(str(bundled), str(dst))

    _runtime = CancelPolicyRuntime(
        root=str(tmp_dir),
        actions=client.ACTIONS,
        control_action_id=client.CONTROL_ACTION_ID,
        reason_denylist=client.REASON_DENYLIST,
        primary_reasons=tuple(client.PRIMARY_REASONS),
        plan_tiers=list(client.PLAN_TIERS),
    )
    _extractor = TranscriptExtractor()
    _ui_data = _UI_BY_CLIENT.get(_client_id, _UI_JUNGLE)


def _build_context(user_id: str, extraction: TranscriptExtractionV1, metadata: dict) -> CancelContextV1:
    return CancelContextV1(
        session_id=metadata.get("session_id") or f"session_{user_id}_{int(time.time())}",
        user_id_hash=user_id,
        timestamp=time.time(),
        plan_tier=str(metadata.get("plan_tier") or "unknown").strip().lower(),
        tenure_days=int(metadata.get("tenure_days") or 0),
        engagement_7d=float(metadata.get("engagement_7d") or 0.0),
        engagement_30d=float(metadata.get("engagement_30d") or 0.0),
        prior_cancel_attempts_30d=int(metadata.get("prior_cancel_attempts_30d") or 0),
        discount_exposures_30d=int(metadata.get("discount_exposures_30d") or 0),
        transcript_extraction=extraction,
    )


def _action_to_ui(action_id: str) -> dict:
    """Fixed fallback UI — always works, no LLM needed."""
    entry = _ui_data.get(action_id, {})
    return {
        "header_title": entry.get("header", "Before you go..."),
        "body_text": entry.get("body", ""),
        "cta_button_text": entry.get("cta", "Learn more"),
        "cta_button_action": action_id,
        "offer_kind": entry.get("offer_kind", "none"),
        "secondary_action_text": "No thanks, continue canceling",
        "secondary_action": "dismiss",
    }


def _best_strategy_dims(action_id: str) -> dict | None:
    """Get the top-scoring strategy dimensions for this action from the optimizer."""
    pool = _runtime.state.get("strategy_pool", {})
    action_strats = pool.get(action_id, {})
    if not action_strats:
        return None
    # Pick the strategy with the highest mean_score
    best = max(action_strats.values(), key=lambda s: float(s.get("mean_score", 0)))
    return best.get("dims", {})


def _personalize_ui(action_id: str, transcript: str, extraction: TranscriptExtractionV1, strategy_dims: dict) -> dict | None:
    """Use LLM to personalize UI text, constrained by the bandit's action + optimizer's strategy dims.

    Returns personalized UI dict, or None if LLM unavailable/fails (caller falls back to fixed).
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    fallback = _ui_data.get(action_id, {})
    offer_kind = fallback.get("offer_kind", "none")

    prompt = f"""You are writing retention screen copy for a user who wants to cancel.

The system has already decided the best ACTION for this user: {action_id} ({fallback.get('header', '')})
Offer type: {offer_kind}

The optimizer found these are the best creative dimensions for this action:
- message_angle: {strategy_dims.get('message_angle', 'contextual')}
- proof_style: {strategy_dims.get('proof_style', 'none')}
- personalization: {strategy_dims.get('personalization', 'contextual')}
- contextual_grounding: {strategy_dims.get('contextual_grounding', 'generic')}

User's cancel reason: {extraction.primary_reason}
User's frustration level: {extraction.frustration_level:.1f}/1.0
User's save openness: {extraction.save_openness:.1f}/1.0

Relevant transcript excerpt (last few turns):
{transcript[-500:] if isinstance(transcript, str) else json.dumps(transcript[-3:] if isinstance(transcript, list) else transcript)}

Write JSON with exactly these keys:
- header_title: 3-8 words, empathetic, matches the message_angle
- body_text: 1-2 sentences, personalized to their situation, uses the contextual_grounding
- cta_button_text: 2-5 words, action-oriented, matches the offer type

Rules:
- Do NOT change the offer type. If the action is a pause, the CTA must be about pausing.
- If frustration is high (>0.6), be more empathetic and less salesy.
- Keep it short. This is a UI card, not an email.
- Return ONLY valid JSON, no markdown."""

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("PERSONALIZE_MODEL", "gpt-4o-mini"),
            input=prompt,
            max_output_tokens=200,
        )
        raw = response.output_text or ""
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        result = json.loads(match.group(0))

        # Validate required keys exist and are strings
        for key in ("header_title", "body_text", "cta_button_text"):
            if not isinstance(result.get(key), str) or not result[key].strip():
                return None

        return {
            "header_title": result["header_title"].strip(),
            "body_text": result["body_text"].strip(),
            "cta_button_text": result["cta_button_text"].strip(),
            "cta_button_action": action_id,
            "offer_kind": offer_kind,
            "secondary_action_text": "No thanks, continue canceling",
            "secondary_action": "dismiss",
        }
    except Exception as exc:
        logger.warning("LLM personalization failed, using fixed text: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Vercel handler
# ---------------------------------------------------------------------------
class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
        except (json.JSONDecodeError, ValueError):
            self._respond(400, {"error": "Invalid JSON"})
            return

        user_id = str(body.get("user_id") or "anonymous")
        transcript = body.get("transcript", "")
        metadata = body.get("metadata") or {}

        _init()

        personalize = str(body.get("personalize", "true")).lower() in ("true", "1", "yes")

        # 1. Extract transcript features
        extraction = _extractor.extract(transcript, metadata=metadata)

        # 2. Build context and run bandit
        context = _build_context(user_id, extraction, metadata)
        decision = _runtime.decide(context)

        # 3. Try LLM personalization, fall back to fixed UI
        personalized = False
        strategy_dims = None
        ui = None

        if personalize:
            strategy_dims = _best_strategy_dims(decision.action_id)
            if strategy_dims:
                ui = _personalize_ui(decision.action_id, transcript, extraction, strategy_dims)
                if ui:
                    personalized = True

        if ui is None:
            ui = _action_to_ui(decision.action_id)

        self._respond(200, {
            "decision_id": decision.decision_id,
            "action_id": decision.action_id,
            "ui": ui,
            "meta": {
                "policy_version": decision.policy_version,
                "exploration_flag": decision.exploration_flag,
                "primary_reason": extraction.primary_reason,
                "client_id": _client_id,
                "personalized": personalized,
                "strategy_dims": strategy_dims,
            },
        })

    def do_GET(self):
        self._respond(200, {"status": "ok", "endpoint": "POST /api/present_offers"})

    def _respond(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

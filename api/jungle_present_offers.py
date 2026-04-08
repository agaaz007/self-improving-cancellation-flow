"""Vercel serverless function: present_offers (Jungle AI / edtech).

Receives transcript + user_id from 11 Labs agent, runs reason-based routing
or Thompson-sampling bandit to pick the best retention action, then
optionally personalizes the UI text via LLM.

Flow: extract reason → route to ACTION → LLM personalizes text →
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
# Fixed UI data per action (Jungle AI — edtech)
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

# ---------------------------------------------------------------------------
# Reason-based flow routing — Jungle AI (edtech)
#   price → escalating discounts (20% if open, 40% if frustrated)
#   graduating → empathic exit (life event, unsaveable)
#   break → pause (natural fit — they'll be back)
#   quality_bug → concierge recovery + empathy
#   feature_gap → feature value recap (show what they're missing)
#   competition → targeted discount + value recap
#   billing_confusion → billing clarity
#   other → pause (safe default)
# ---------------------------------------------------------------------------
_REASON_TO_ACTION_JUNGLE = {
    "price":             "targeted_discount_40",    # price-sensitive → strong discount
    "graduating":        "control_empathic_exit",   # life event — let them go gracefully
    "break":             "pause_plan_relief",       # seasonal break → pause is perfect
    "quality_bug":       "concierge_recovery",      # bugs → empathize + guided support
    "feature_gap":       "feature_value_recap",     # missing features → show what they haven't tried
    "competition":       "targeted_discount_40",    # switching to competitor → discount to retain
    "billing_confusion": "billing_clarity_reset",   # billing issue → clarify
    "other":             "pause_plan_relief",       # default → soft pause offer
}

# ---------------------------------------------------------------------------
# Singleton state — survives across warm invocations
# ---------------------------------------------------------------------------
_runtime: CancelPolicyRuntime | None = None
_extractor: TranscriptExtractor | None = None
_client_id: str = "jungle_ai"


def _init():
    global _runtime, _extractor

    if _runtime is not None:
        return

    client = load_client(_client_id)

    # Copy bundled policy_state.json to /tmp where the runtime can write
    tmp_dir = Path("/tmp/bandit_policy_jungle")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    bundled = Path(__file__).parent / "jungle_policy_state.json"
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
    entry = _UI_JUNGLE.get(action_id, {})
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
    best = max(action_strats.values(), key=lambda s: float(s.get("mean_score", 0)))
    return best.get("dims", {})


def _personalize_ui(action_id: str, transcript: str, extraction: TranscriptExtractionV1, strategy_dims: dict) -> dict | None:
    """Use LLM to personalize UI text, constrained by the action + optimizer's strategy dims."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    fallback = _UI_JUNGLE.get(action_id, {})
    offer_kind = fallback.get("offer_kind", "none")

    prompt = f"""You are writing retention screen copy for a student who wants to cancel their learning app subscription.

The system has already decided the best ACTION for this user: {action_id} ({fallback.get('header', '')})
Offer type: {offer_kind}

Creative direction:
- message_angle: {strategy_dims.get('message_angle', 'empathetic and direct')}
- proof_style: {strategy_dims.get('proof_style', 'acknowledge their specific situation')}
- personalization: {strategy_dims.get('personalization', 'contextual — reference their situation')}
- contextual_grounding: {strategy_dims.get('contextual_grounding', 'their stated reason for leaving')}

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
- This is an edtech/learning app — reference their learning goals where appropriate.
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
        use_bandit = str(body.get("use_bandit", "false")).lower() in ("true", "1", "yes")

        # 1. Extract transcript features (empty transcript = silent churn)
        transcript_text = transcript if isinstance(transcript, str) else json.dumps(transcript)
        if not transcript_text.strip():
            extraction = TranscriptExtractionV1(
                primary_reason="other",
                frustration_level=0.0,
                save_openness=0.3,
                summary="No transcript — silent cancel",
            )
        else:
            extraction = _extractor.extract(transcript, metadata=metadata)

        # 2. Pick the action — reason-based routing or bandit
        if use_bandit:
            context = _build_context(user_id, extraction, metadata)
            decision = _runtime.decide(context)
            action_id = decision.action_id
            decision_id = decision.decision_id
            exploration_flag = decision.exploration_flag
            policy_version = decision.policy_version
            routing_mode = "bandit"
        else:
            reason = extraction.primary_reason
            action_id = _REASON_TO_ACTION_JUNGLE.get(reason, "pause_plan_relief")

            # Escalation logic for price
            if reason == "price" and extraction.save_openness > 0.5:
                action_id = "targeted_discount_20"  # open to staying → start lower
            elif reason == "price" and extraction.frustration_level > 0.6:
                action_id = "targeted_discount_40"  # frustrated + price = stronger offer

            # Competition: if they're open, show value; if not, discount
            if reason == "competition" and extraction.save_openness > 0.4:
                action_id = "feature_value_recap"

            # Bug/quality: always empathize first
            if reason == "quality_bug":
                action_id = "concierge_recovery"

            from uuid import uuid4
            decision_id = f"dec_{uuid4().hex[:12]}"
            exploration_flag = False
            policy_version = "reason_routing_v1"
            routing_mode = "reason"

        # 3. Try LLM personalization, fall back to fixed UI
        personalized = False
        strategy_dims = None
        ui = None

        if personalize:
            strategy_dims = _best_strategy_dims(action_id) if use_bandit else None
            ui = _personalize_ui(action_id, transcript, extraction, strategy_dims or {})
            if ui:
                personalized = True

        if ui is None:
            ui = _action_to_ui(action_id)

        self._respond(200, {
            "decision_id": decision_id,
            "action_id": action_id,
            "ui": ui,
            "meta": {
                "policy_version": policy_version,
                "exploration_flag": exploration_flag,
                "primary_reason": extraction.primary_reason,
                "frustration_level": round(extraction.frustration_level, 2),
                "save_openness": round(extraction.save_openness, 2),
                "client_id": _client_id,
                "personalized": personalized,
                "routing_mode": routing_mode,
                "strategy_dims": strategy_dims,
            },
        })

    def do_GET(self):
        self._respond(200, {"status": "ok", "endpoint": "POST /api/jungle_present_offers"})

    def _respond(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

"""Jungle AI (edtech) client configuration.

Re-exports existing constants so the pipeline stays unchanged when
CLIENT_ID=jungle_ai (the default).
"""
from __future__ import annotations

from cta_autoresearch.cancel_policy import (
    DEFAULT_ACTIONS,
    DEFAULT_REASON_DENYLIST,
    PRIMARY_REASONS as _ORIG_REASONS,
)
from cta_autoresearch.strategy_policy import (
    BASE_OFFERS,
    CONTEXTUAL_GROUNDINGS,
    CREATIVE_TREATMENTS,
    CTAS,
    FRICTION_REDUCERS,
    MESSAGE_ANGLES,
    PERSONALIZATION_LEVELS,
    PROOF_STYLES,
)
from cta_autoresearch.user_model import (
    ARCHETYPES as _ORIG_ARCHETYPES,
    ACTION_TO_CANDIDATE as _ORIG_ACTION_TO_CANDIDATE,
    MUTABLE_DIMENSIONS as _ORIG_MUTABLE_DIMENSIONS,
    enriched_row_to_persona,
)

PRIMARY_REASONS = _ORIG_REASONS
ACTIONS = DEFAULT_ACTIONS
REASON_DENYLIST = DEFAULT_REASON_DENYLIST
CONTROL_ACTION_ID = "control_empathic_exit"
PLAN_TIERS = ["free", "starter", "super_learner"]
LLM_DOMAIN_CONTEXT = "a learning app for students preparing for exams"

MUTABLE_DIMENSIONS = dict(_ORIG_MUTABLE_DIMENSIONS)
ARCHETYPES = dict(_ORIG_ARCHETYPES)
ACTION_TO_CANDIDATE = dict(_ORIG_ACTION_TO_CANDIDATE)

DIMENSION_CATALOGS = {
    "message_angles": dict(MESSAGE_ANGLES),
    "proof_styles": dict(PROOF_STYLES),
    "base_offers": dict(BASE_OFFERS),
    "ctas": dict(CTAS),
    "personalization_levels": dict(PERSONALIZATION_LEVELS),
    "contextual_groundings": dict(CONTEXTUAL_GROUNDINGS),
    "creative_treatments": dict(CREATIVE_TREATMENTS),
    "friction_reducers": dict(FRICTION_REDUCERS),
}


def row_to_persona(row, index=0):
    """Map an eval cohort row to a Persona. Delegates to existing function."""
    return enriched_row_to_persona(row, index)


def reason_from_raw(raw_reason: str, raw_note: str = "") -> str:
    """Map raw cancel reason text to a normalized reason."""
    text = f"{raw_reason} {raw_note}".lower()
    keyword_map = {
        "price": ("expensive", "cost", "price", "afford", "budget", "too much"),
        "graduating": ("graduat", "finished school", "done with school"),
        "break": ("summer break", "winter break", "on break", "vacation"),
        "quality_bug": ("bug", "crash", "slow", "error", "broken"),
        "feature_gap": ("missing", "feature", "flashcard", "anki", "audio"),
        "competition": ("chatgpt", "gemini", "notebooklm", "competitor"),
        "billing_confusion": ("billing", "charge", "charged", "quota", "refund"),
    }
    for reason, keywords in keyword_map.items():
        if any(kw in text for kw in keywords):
            return reason
    return "other"

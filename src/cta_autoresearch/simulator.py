"""LLM-as-judge scorer for cancel flow strategies.

The heuristic formula factory is gone. Scoring is now done by an LLM that
evaluates whether a strategy matches a user's situation. Scores are cached
aggressively so the overnight optimization loop stays fast.

Architecture:
    1. Call configure_scorer() once at startup with an OpenAI client
    2. score_candidate_details(persona, candidate) dispatches to LLM + cache
    3. When no client is configured (tests, CI), a minimal fallback runs

Cache key: (segment, offer, message_angle, cta) — the dimensions that most
affect whether a strategy saves a user. This groups many persona×candidate
pairs into the same bucket, keeping LLM calls manageable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cta_autoresearch.features import clamp
from cta_autoresearch.models import Persona, StrategyCandidate
from cta_autoresearch.strategy_policy import OFFERS


# ── Module-level scorer state ────────────────────────────────────────

_llm_client: Any = None
_llm_model: str = "gpt-4o-mini"
_score_cache: dict[str, dict[str, float]] = {}
_cache_path: Path | None = None
_domain_context: str = "a learning app"


def configure_scorer(
    *,
    client: Any = None,
    model: str = "gpt-4o-mini",
    cache_path: str | Path | None = None,
) -> None:
    """Configure the LLM scorer. Call once at startup.

    Args:
        client: OpenAI client instance (chat.completions compatible)
        model: Model name for scoring calls
        cache_path: Path to persist score cache across runs
    """
    global _llm_client, _llm_model, _score_cache, _cache_path
    _llm_client = client
    _llm_model = model
    _score_cache = {}  # Always reset cache on reconfigure
    _cache_path = Path(cache_path) if cache_path else None
    if _cache_path and _cache_path.exists():
        try:
            loaded = json.loads(_cache_path.read_text())
            if isinstance(loaded, dict):
                _score_cache = loaded
                print(f"[scorer] Loaded {len(_score_cache)} cached scores from {_cache_path}")
        except (json.JSONDecodeError, OSError):
            pass


def persist_cache() -> None:
    """Write score cache to disk. Call after optimization loop."""
    if _cache_path and _score_cache:
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        _cache_path.write_text(json.dumps(_score_cache, indent=2))


def configure_domain(context: str) -> None:
    """Set the domain context string for LLM prompts."""
    global _domain_context
    _domain_context = context


def reset_scorer() -> None:
    """Reset scorer to fallback mode and clear cache. Use in tests."""
    global _llm_client, _score_cache
    _llm_client = None
    _score_cache = {}


def _cache_key(persona: Persona, candidate: StrategyCandidate) -> str:
    """Cache key captures the dimensions that matter for scoring."""
    return f"{persona.features.segment}|{candidate.offer}|{candidate.message_angle}|{candidate.cta}|{candidate.friction_reducer}"


# ── LLM judge ────────────────────────────────────────────────────────


def _persona_summary(persona: Persona) -> str:
    f = persona.features
    p = persona.profile
    top_features = sorted(
        [(k, getattr(f, k)) for k in (
            "habit_strength", "price_sensitivity", "urgency", "support_need",
            "feature_awareness_gap", "trust_sensitivity", "friction_sensitivity",
            "deadline_pressure", "fatigue_risk", "discount_affinity",
        )],
        key=lambda x: x[1], reverse=True,
    )[:5]
    features_str = ", ".join(f"{k.replace('_', ' ')}={v:.2f}" for k, v in top_features)
    return (
        f"Segment: {f.segment}. Plan: {p.plan}, {p.billing_period}. "
        f"Tenure: {p.lifetime_days}d. Status: {p.status}. "
        f"Dormancy: {p.dormancy_days}d. Context: {p.study_context}. "
        f"Top features: {features_str}."
    )


def _candidate_summary(candidate: StrategyCandidate) -> str:
    return (
        f"Action: {candidate.offer.replace('_', ' ')}. "
        f"Angle: {candidate.message_angle.replace('_', ' ')}. "
        f"Proof: {candidate.proof_style.replace('_', ' ')}. "
        f"CTA: {candidate.cta.replace('_', ' ')}. "
        f"Personalization: {candidate.personalization}. "
        f"Grounding: {candidate.contextual_grounding.replace('_', ' ')}. "
        f"Treatment: {candidate.creative_treatment.replace('_', ' ')}. "
        f"Friction reducer: {candidate.friction_reducer.replace('_', ' ')}."
    )


_LLM_PROMPT = (
    "You are scoring a cancellation save strategy for a specific user of {domain}.\n\n"
    "User: {persona}\n\n"
    "Strategy: {candidate}\n\n"
    "Score these three dimensions from 0.0 to 1.0:\n"
    "- retention: probability this intervention saves the user from cancelling. "
    "Consider fit between the user's reason for leaving and what the strategy offers.\n"
    "- revenue: revenue preservation (1.0 = full price retained, 0.0 = total loss). "
    "Discounts reduce this. Pause/downgrade partially reduce it. No-offer = 1.0.\n"
    "- trust: trust preservation (1.0 = respectful, 0.0 = manipulative). "
    "Aggressive urgency on non-urgent users damages trust. Over-personalization feels creepy.\n\n"
    "Score based on fit, not just generosity. A discount for a graduating student wastes money. "
    "A pause for a price-sensitive user misses the point.\n\n"
    'Return JSON only: {{"retention": 0.X, "revenue": 0.X, "trust": 0.X}}'
)


def _llm_score(persona: Persona, candidate: StrategyCandidate) -> dict[str, float]:
    """Score via LLM. Assumes _llm_client is set.

    Uses the OpenAI Responses API (client.responses.create) which is what
    the rest of the codebase uses. Falls back gracefully on API errors so
    a single bad call doesn't crash the overnight run.
    """
    prompt = _LLM_PROMPT.format(
        domain=_domain_context,
        persona=_persona_summary(persona),
        candidate=_candidate_summary(candidate),
    )
    try:
        response = _llm_client.responses.create(
            model=_llm_model,
            input=prompt,
            max_output_tokens=100,
        )
        raw = str(response.output_text or "")
    except Exception:
        # API error — return neutral scores rather than crashing
        return _build_score_dict(0.4, 0.5, 0.7)

    match = re.search(r"\{[^}]+\}", raw)
    if match:
        try:
            parsed = json.loads(match.group(0))
            retention = clamp(float(parsed.get("retention", 0.4)))
            revenue = clamp(float(parsed.get("revenue", 0.5)))
            trust = clamp(float(parsed.get("trust", 0.7)))
        except (json.JSONDecodeError, TypeError, ValueError):
            retention, revenue, trust = 0.4, 0.5, 0.7
    else:
        retention, revenue, trust = 0.4, 0.5, 0.7

    return _build_score_dict(retention, revenue, trust)


# ── Fallback scorer (no API key) ─────────────────────────────────────


def _fallback_score(persona: Persona, candidate: StrategyCandidate) -> dict[str, float]:
    """Minimal fallback when no LLM client is configured (tests, CI).

    Not a replacement for the LLM judge — just enough to keep the pipeline
    runnable and produce sane relative rankings. Active interventions beat
    control, and feature-relevant offers beat generic ones.
    """
    f = persona.features
    offer_meta = OFFERS.get(candidate.offer, {"kind": "none", "generosity": 0.0})
    kind = offer_meta.get("kind", "none")

    # Base retention by offer kind
    if kind == "none":
        retention = 0.20
    elif kind == "discount":
        retention = 0.40 + 0.15 * f.price_sensitivity
    elif kind in ("pause", "downgrade"):
        retention = 0.38 + 0.12 * f.friction_sensitivity
    elif kind == "support":
        retention = 0.35 + 0.15 * f.support_need
    elif kind == "credit":
        retention = 0.35 + 0.15 * f.feature_awareness_gap
    elif kind == "extension":
        retention = 0.35 + 0.15 * f.deadline_pressure
    elif kind == "billing":
        retention = 0.33 + 0.12 * f.price_sensitivity
    else:
        retention = 0.30

    # Angle fit bonus
    angle_boosts = {
        "cost_value_reframe": f.price_sensitivity * 0.08,
        "flexibility_relief": f.friction_sensitivity * 0.08,
        "feature_unlock": f.feature_awareness_gap * 0.08,
        "goal_deadline": f.deadline_pressure * 0.08,
        "mistake_recovery": f.support_need * 0.08,
        "progress_reflection": f.habit_strength * 0.06,
        "momentum_protection": f.habit_strength * 0.06,
        "fresh_start_reset": f.fatigue_risk * 0.06,
        "outcome_proof": f.proof_need * 0.06,
        "habit_identity": f.habit_strength * 0.05,
        "empathetic_exit": f.trust_sensitivity * 0.04,
    }
    retention += angle_boosts.get(candidate.message_angle, 0.0)

    # Proof style bonus
    proof_boosts = {
        "quantified_outcome": f.proof_need * 0.04,
        "personal_usage_signal": f.habit_strength * 0.04,
        "similar_user_story": f.rescue_readiness * 0.03,
        "expert_validation": f.trust_sensitivity * 0.03,
    }
    retention += proof_boosts.get(candidate.proof_style, 0.0)

    # Grounding bonus
    grounding_boosts = {
        "deadline_countdown": f.deadline_pressure * 0.04,
        "deadline_pressure": f.deadline_pressure * 0.04,
        "unused_value": f.feature_awareness_gap * 0.04,
        "pricing_context": f.price_sensitivity * 0.04,
        "progress_snapshot": f.habit_strength * 0.03,
        "habit_streak": f.habit_strength * 0.03,
        "support_signal": f.support_need * 0.03,
    }
    retention += grounding_boosts.get(candidate.contextual_grounding, 0.0)

    # Treatment bonus
    treatment_boosts = {
        "progress_thermometer": f.habit_strength * 0.03,
        "feature_collage": f.feature_awareness_gap * 0.03,
        "study_timeline": f.deadline_pressure * 0.03,
        "coach_note": f.support_need * 0.03,
        "options_table": f.friction_sensitivity * 0.02,
    }
    retention += treatment_boosts.get(candidate.creative_treatment, 0.0)

    # Friction reducer bonus
    friction_boosts = {
        "single_tap_pause": f.friction_sensitivity * 0.04,
        "billing_date_shift": f.price_sensitivity * 0.04,
        "prefilled_downgrade": f.price_sensitivity * 0.03,
        "concierge_setup": f.support_need * 0.03,
        "smart_resume_date": f.fatigue_risk * 0.03,
    }
    retention += friction_boosts.get(candidate.friction_reducer, 0.0)

    # Dormancy penalty for no-action
    if kind == "none" and persona.profile.dormancy_days > 30:
        retention -= 0.05

    # Revenue: discounts cost money
    generosity = float(offer_meta.get("generosity", 0.0))
    revenue = clamp(1.0 - generosity)

    # Trust: high personalization + high trust sensitivity = bad
    trust = 0.85
    if candidate.personalization in ("behavioral", "highly_specific") and f.trust_sensitivity > 0.65:
        trust -= 0.10

    return _build_score_dict(clamp(retention), revenue, trust)


# ── Shared helpers ───────────────────────────────────────────────────


def _build_score_dict(retention: float, revenue: float, trust: float) -> dict[str, float]:
    """Build the standard score dict from the three core dimensions."""
    composite = clamp(0.60 * retention + 0.40 * revenue)
    return {
        "score": composite,
        "retention": retention,
        "revenue": revenue,
        "trust": trust,
        "base_retention": retention,
        "angle_fit": 0.5,
        "proof_fit": 0.5,
        "offer_fit": 0.5,
        "cta_fit": 0.5,
        "personalization_fit": 0.5,
        "grounding_fit": 0.5,
        "treatment_fit": 0.5,
        "friction_fit": 0.5,
        "trust_penalty": clamp(1.0 - trust),
    }


# ── Public API ───────────────────────────────────────────────────────


def score_candidate_details(persona: Persona, candidate: StrategyCandidate) -> dict[str, float]:
    """Score a (persona, candidate) pair. Same signature as the old heuristic.

    Dispatches to LLM judge (with cache) when configured, fallback otherwise.
    Cache is only used for LLM calls — fallback is instant and stateless.
    """
    if _llm_client is not None:
        key = _cache_key(persona, candidate)
        if key in _score_cache:
            return _score_cache[key]
        result = _llm_score(persona, candidate)
        _score_cache[key] = result
        return result

    return _fallback_score(persona, candidate)


def score_candidate(persona: Persona, candidate: StrategyCandidate) -> tuple[float, float]:
    details = score_candidate_details(persona, candidate)
    return details["score"], details["trust"]


def prewarm_cache(
    personas: list[Persona],
    candidates: list[StrategyCandidate],
) -> int:
    """Pre-score representative (persona, candidate) pairs to warm the cache.

    Call after configure_scorer() and before the optimization loop.
    Returns the number of new LLM calls made.
    """
    calls = 0
    for persona in personas:
        for candidate in candidates:
            key = _cache_key(persona, candidate)
            if key not in _score_cache:
                score_candidate_details(persona, candidate)
                calls += 1
    if calls:
        persist_cache()
        print(f"[scorer] Pre-warmed cache with {calls} new scores ({len(_score_cache)} total cached)")
    return calls

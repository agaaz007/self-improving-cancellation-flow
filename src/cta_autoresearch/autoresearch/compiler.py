from __future__ import annotations

from cta_autoresearch.autoresearch.schemas import ExperimentSpec, FlowResearchSpec, ResearchFinding
from cta_autoresearch.models import StrategyCandidate
from cta_autoresearch.research_settings import ResearchSettings
from cta_autoresearch.strategy_policy import (
    candidate_key,
    offer_catalog,
    valid_candidate,
)


_CANDIDATE_FIELDS = (
    "message_angle", "proof_style", "offer", "cta",
    "personalization", "contextual_grounding", "creative_treatment", "friction_reducer",
)

# Weights for dimension matching — offer and message_angle matter most
_DIM_WEIGHTS = {
    "message_angle": 3.0, "offer": 3.0, "cta": 2.0,
    "proof_style": 1.0, "personalization": 1.0, "contextual_grounding": 1.0,
    "creative_treatment": 1.0, "friction_reducer": 1.0,
}


def _dim_distance(a: StrategyCandidate, b: StrategyCandidate) -> float:
    """Weighted dimension distance: 0 = identical, higher = more different."""
    dist = 0.0
    for field in _CANDIDATE_FIELDS:
        if getattr(a, field) != getattr(b, field):
            dist += _DIM_WEIGHTS.get(field, 1.0)
    return dist


def compile_flow_spec(
    spec: FlowResearchSpec,
    *,
    candidate_universe: list[StrategyCandidate],
    settings: ResearchSettings,
    fallback_candidate: StrategyCandidate | None = None,
) -> tuple[StrategyCandidate, list[str]]:
    """Compile a FlowResearchSpec into the nearest valid StrategyCandidate.

    Strategy: prefer the fallback (LLM's explicit choice) if it's valid and in
    the universe. Otherwise find the nearest candidate by weighted dimension
    distance from the fallback. Keyword regex is gone — the LLM's structured
    output is a better signal than regex on hypothesis text.
    """
    notes: list[str] = []
    universe_keys = {candidate_key(c) for c in candidate_universe}

    # 1. Try the fallback candidate directly (the LLM's explicit choice)
    if fallback_candidate is not None:
        if valid_candidate(fallback_candidate, settings) and candidate_key(fallback_candidate) in universe_keys:
            notes.append("LLM candidate matched a valid strategy in the universe directly.")
            return fallback_candidate, notes

    # 2. Find nearest candidate in universe by weighted dimension distance
    if fallback_candidate is not None and candidate_universe:
        best = min(candidate_universe, key=lambda c: _dim_distance(c, fallback_candidate))
        dist = _dim_distance(best, fallback_candidate)
        if dist == 0:
            notes.append("Exact match found in universe.")
        else:
            notes.append(f"Nearest match in universe (distance={dist:.1f}).")
        return best, notes

    # 3. No fallback — return first candidate in universe
    if candidate_universe:
        notes.append("No fallback candidate; using first candidate in universe.")
        return candidate_universe[0], notes

    # 4. Nothing at all — construct default
    notes.append("Empty universe; constructed default candidate.")
    return StrategyCandidate(
        message_angle="empathetic_exit", proof_style="none", offer="none",
        cta="tell_us_why", personalization="generic",
    ), notes


def build_experiment_spec(spec: FlowResearchSpec, candidate: StrategyCandidate) -> ExperimentSpec:
    treatment = spec.copy_blocks[0] if spec.copy_blocks else spec.rescue_objective
    return ExperimentSpec(
        target_segment=spec.target_segment,
        hypothesis=spec.user_state_hypothesis,
        control_description="Current cancellation save flow for the same eligible user segment.",
        treatment_description=f"{treatment} via {candidate.cta.replace('_', ' ')} with {candidate.offer.replace('_', ' ')}.",
        primary_metric="retention/save rate",
        secondary_metrics=("economic impact", "post-save engagement quality"),
        guardrails=("complaint rate", "support burden", "trust sentiment"),
        rollout_suggestion="Start with a holdout-backed experiment capped to a narrow eligible segment.",
        rollback_trigger="Rollback if trust complaints or support burden rise before save rate improves.",
    )


def flow_spec_to_payload(spec: FlowResearchSpec) -> dict[str, object]:
    return spec.to_dict()


def research_trace_payload(
    *,
    spec: FlowResearchSpec,
    findings: list[ResearchFinding],
    compile_notes: list[str],
    evaluation_summary: str = "",
) -> dict[str, object]:
    payload = {
        "target_segment": spec.target_segment,
        "agent_role": spec.agent_role,
        "user_state_hypothesis": spec.user_state_hypothesis,
        "cancellation_moment_hypothesis": spec.cancellation_moment_hypothesis,
        "rescue_objective": spec.rescue_objective,
        "findings": [finding.to_dict() for finding in findings],
        "compile_notes": list(compile_notes),
    }
    if evaluation_summary:
        payload["evaluation_summary"] = evaluation_summary
    return payload

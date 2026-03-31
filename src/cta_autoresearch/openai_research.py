from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from cta_autoresearch.features import clamp
from cta_autoresearch.models import Persona, StrategyCandidate
from cta_autoresearch.research_settings import ResearchSettings
from cta_autoresearch.strategy_policy import candidate_key, candidate_label, offer_catalog


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    match = re.search(r"\[\s*{.*}\s*]", raw, flags=re.DOTALL)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _target_summary(persona: Persona, target_id: str) -> dict[str, Any]:
    return {
        "id": target_id,
        "segment": persona.features.segment,
        "study_context": persona.profile.study_context,
        "recent_behavior": persona.profile.recent_behavior,
        "plan": persona.profile.plan,
        "status": persona.profile.status,
        "features": {
            key: round(value, 4)
            for key, value in vars(persona.features).items()
            if key != "segment"
        },
    }


def _candidate_summary(candidate: StrategyCandidate, settings: ResearchSettings) -> dict[str, Any]:
    offers = offer_catalog(settings)
    return {
        "candidate_id": candidate_key(candidate),
        "label": candidate_label(candidate, offers=offers),
        "message_angle": candidate.message_angle,
        "proof_style": candidate.proof_style,
        "offer": candidate.offer,
        "cta": candidate.cta,
        "personalization": candidate.personalization,
        "contextual_grounding": candidate.contextual_grounding,
        "creative_treatment": candidate.creative_treatment,
        "friction_reducer": candidate.friction_reducer,
    }


def _batched(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _normalize_detail(payload: dict[str, Any]) -> dict[str, float]:
    normalized = {}
    for key in (
        "score",
        "retention",
        "revenue",
        "trust",
        "angle_fit",
        "proof_fit",
        "offer_fit",
        "cta_fit",
        "personalization_fit",
        "grounding_fit",
        "treatment_fit",
        "friction_fit",
        "trust_penalty",
    ):
        try:
            value = float(payload.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if key == "trust_penalty":
            normalized[key] = clamp(value, 0.0, 1.0)
        else:
            normalized[key] = clamp(value)
    return normalized


def _api_score_targets(
    *,
    target_payloads: list[dict[str, Any]],
    candidates: list[StrategyCandidate],
    settings: ResearchSettings,
    progress_callback=None,
    progress_range: tuple[float, float] = (0.62, 0.9),
    stage_label: str = "validation",
) -> tuple[dict[str, dict[str, dict[str, float]]] | None, list[str]]:
    warnings: list[str] = []
    if OpenAI is None:
        warnings.append("OpenAI SDK is not installed, so api_only mode fell back to local scoring.")
        return None, warnings
    if not settings.has_api_key:
        warnings.append("OPENAI_API_KEY is not set, so api_only mode fell back to local scoring.")
        return None, warnings

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    score_map: dict[str, dict[str, dict[str, float]]] = {}
    target_count = max(len(target_payloads), 1)
    effective_batch_size = max(1, min(settings.api_batch_size, max(1, 18 // target_count)))
    batches = _batched(candidates, effective_batch_size)
    total_batches = max(len(batches), 1)
    for index, batch in enumerate(batches, start=1):
        if progress_callback:
            start, end = progress_range
            progress = start + (end - start) * ((index - 1) / total_batches)
            progress_callback(progress, stage_label, f"Submitting OpenAI scoring batch {index} of {total_batches}.")
        prompt = "\n".join(
            [
                "You are scoring churn-reduction strategies for a cancellation flow.",
                "Return JSON only as an array.",
                "For each candidate, return: candidate_id and targets.",
                "targets must be an object keyed by target id.",
                "Each target object must contain numeric values between 0 and 1 for:",
                "score, retention, revenue, trust, angle_fit, proof_fit, offer_fit, cta_fit, personalization_fit, grounding_fit, treatment_fit, friction_fit, trust_penalty.",
                "Use the target summaries and candidate specs below.",
                "Target summaries:",
                json.dumps(target_payloads, indent=2),
                "Candidates:",
                json.dumps([_candidate_summary(candidate, settings) for candidate in batch], indent=2),
            ]
        )
        response = client.responses.create(
            model=settings.model_name,
            input=prompt,
            reasoning={"effort": settings.openai_reasoning_effort},
            max_output_tokens=min(12000, max(4000, 420 * len(batch) * target_count)),
        )
        payload = _parse_json_array(response.output_text)
        for item in payload:
            candidate_id = str(item.get("candidate_id", ""))
            if not candidate_id:
                continue
            target_scores = item.get("targets", {})
            if not isinstance(target_scores, dict):
                continue
            score_map[candidate_id] = {
                str(target_id): _normalize_detail(detail if isinstance(detail, dict) else {})
                for target_id, detail in target_scores.items()
            }
        if progress_callback:
            start, end = progress_range
            progress = start + (end - start) * (index / total_batches)
            progress_callback(progress, stage_label, f"Completed OpenAI scoring batch {index} of {total_batches}.")

    missing = [
        candidate_key(candidate)
        for candidate in candidates
        if candidate_key(candidate) not in score_map
    ]
    if missing:
        warnings.append(
            f"OpenAI evaluation returned incomplete scores for {len(missing)} candidates; local scoring was used for the missing items."
        )
    return score_map, warnings


def evaluate_candidates_via_api(
    *,
    cohort_persona: Persona,
    segment_personas: dict[str, Persona],
    candidates: list[StrategyCandidate],
    settings: ResearchSettings,
    progress_callback=None,
) -> tuple[dict[str, dict[str, dict[str, float]]] | None, list[str]]:
    target_payloads = [_target_summary(cohort_persona, "cohort")]
    target_payloads.extend(
        _target_summary(persona, f"segment::{segment}")
        for segment, persona in sorted(segment_personas.items())
    )
    return _api_score_targets(
        target_payloads=target_payloads,
        candidates=candidates,
        settings=settings,
        progress_callback=progress_callback,
        progress_range=(0.62, 0.78),
        stage_label="cohort-judge",
    )


def evaluate_persona_shortlist_via_api(
    *,
    personas: list[Persona],
    candidates: list[StrategyCandidate],
    settings: ResearchSettings,
    progress_callback=None,
) -> tuple[dict[str, dict[str, dict[str, float]]] | None, list[str]]:
    target_payloads = [
        _target_summary(persona, f"persona::{persona.name}")
        for persona in personas
    ]
    return _api_score_targets(
        target_payloads=target_payloads,
        candidates=candidates,
        settings=settings,
        progress_callback=progress_callback,
        progress_range=(0.8, 0.92),
        stage_label="persona-judge",
    )

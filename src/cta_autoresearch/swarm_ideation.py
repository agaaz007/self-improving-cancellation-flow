from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from statistics import mean

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from cta_autoresearch.autoresearch import (
    FlowResearchSpec,
    ResearchFinding,
    build_experiment_spec,
    compile_flow_spec,
    flow_spec_to_payload,
    research_trace_payload,
)
from cta_autoresearch.models import FeatureVector, IdeaProposal, Persona, StrategyCandidate
from cta_autoresearch.research_settings import ResearchSettings
from cta_autoresearch.simulator import score_candidate_details
from cta_autoresearch.strategy_policy import (
    CONTEXTUAL_GROUNDINGS,
    CREATIVE_TREATMENTS,
    CTAS,
    FRICTION_REDUCERS,
    MESSAGE_ANGLES,
    PERSONALIZATION_LEVELS,
    PROOF_STYLES,
    candidate_key,
    candidate_label,
    offer_catalog,
    render_message,
)


ROLE_PRIORS = {
    "behavioral_scientist": {
        "segment_weights": ("urgency", "friction_sensitivity", "loss_aversion"),
        "angle_bonus": {"goal_deadline": 0.05, "momentum_protection": 0.05, "mistake_recovery": 0.04},
        "offer_bonus": {"exam_sprint": 0.04, "pause_plan": 0.03},
    },
    "pricing_strategist": {
        "segment_weights": ("price_sensitivity", "discount_affinity", "friction_sensitivity"),
        "angle_bonus": {"cost_value_reframe": 0.04, "flexibility_relief": 0.04},
        "offer_bonus": {"flexible_billing": 0.05, "downgrade_lite": 0.05, "pause_plan": 0.04},
    },
    "retention_copywriter": {
        "segment_weights": ("habit_strength", "loss_aversion", "proof_need"),
        "angle_bonus": {"progress_reflection": 0.05, "habit_identity": 0.05, "feature_unlock": 0.03},
        "offer_bonus": {"none": 0.03, "bonus_credits": 0.02},
    },
    "student_success_lead": {
        "segment_weights": ("support_need", "trust_sensitivity", "urgency"),
        "angle_bonus": {"mistake_recovery": 0.05, "empathetic_exit": 0.04, "fresh_start_reset": 0.04},
        "offer_bonus": {"concierge_support": 0.05, "pause_plan": 0.03},
    },
    "product_marketer": {
        "segment_weights": ("feature_awareness_gap", "activation_score", "proof_need"),
        "angle_bonus": {"feature_unlock": 0.06, "outcome_proof": 0.04},
        "offer_bonus": {"bonus_credits": 0.04, "exam_sprint": 0.03},
    },
    "trust_guardian": {
        "segment_weights": ("trust_sensitivity", "support_need", "friction_sensitivity"),
        "angle_bonus": {"empathetic_exit": 0.05, "flexibility_relief": 0.04},
        "offer_bonus": {"pause_plan": 0.04, "none": 0.03, "concierge_support": 0.03},
    },
    "lifecycle_pm": {
        "segment_weights": ("activation_score", "urgency", "price_sensitivity"),
        "angle_bonus": {"progress_reflection": 0.04, "goal_deadline": 0.03, "flexibility_relief": 0.03},
        "offer_bonus": {"pause_plan": 0.03, "flexible_billing": 0.03, "downgrade_lite": 0.03},
    },
    "winback_operator": {
        "segment_weights": ("dormancy_days", "support_need", "feature_awareness_gap"),
        "angle_bonus": {"fresh_start_reset": 0.05, "feature_unlock": 0.04, "empathetic_exit": 0.03},
        "offer_bonus": {"pause_plan": 0.04, "bonus_credits": 0.03, "concierge_support": 0.03},
    },
}

ROLE_LABEL_TO_ID = {
    "Retention Psychologist": "behavioral_scientist",
    "Offer Economist": "pricing_strategist",
    "Lifecycle Strategist": "lifecycle_pm",
    "Product Storyteller": "product_marketer",
    "Support Concierge": "student_success_lead",
    "Deadline Operator": "behavioral_scientist",
    "Trust Guardian": "trust_guardian",
    "Visual CTA Designer": "retention_copywriter",
    "Win-Back Researcher": "winback_operator",
}


def _feature_fields() -> list[str]:
    return [field for field in FeatureVector.__dataclass_fields__ if field != "segment"]


def _aggregate_persona(personas: list[Persona], label: str) -> Persona:
    base = personas[0]
    averaged = {
        key: mean(getattr(persona.features, key) for persona in personas)
        for key in _feature_fields()
    }
    aggregate_features = FeatureVector(segment=label, **averaged)
    return Persona(name=label, profile=base.profile, features=aggregate_features)


def _segment_strength(persona: Persona, role_id: str) -> float:
    priors = ROLE_PRIORS[role_id]
    total = 0.0
    for field in priors["segment_weights"]:
        if field == "dormancy_days":
            total += min(persona.profile.dormancy_days / 60.0, 1.0)
        else:
            total += getattr(persona.features, field)
    return total / len(priors["segment_weights"])


def _role_bonus(candidate: StrategyCandidate, role_id: str) -> float:
    priors = ROLE_PRIORS[role_id]
    bonus = priors["angle_bonus"].get(candidate.message_angle, 0.0)
    bonus += priors["offer_bonus"].get(candidate.offer, 0.0)
    if role_id == "trust_guardian" and candidate.personalization == "highly_specific":
        bonus -= 0.08
    if role_id == "pricing_strategist" and candidate.offer.startswith("discount_"):
        generosity = float(candidate.offer.split("_")[1]) / 100.0
        bonus += 0.03 if generosity <= 0.25 else -0.04
    if role_id == "retention_copywriter" and candidate.proof_style in {"personal_usage_signal", "similar_user_story"}:
        bonus += 0.02
    return bonus


def _idea_thesis(role_label: str, segment: str, candidate: StrategyCandidate, settings: ResearchSettings) -> str:
    offers = offer_catalog(settings)
    return (
        f"{role_label} wants to win back {segment.replace('_', ' ')} users with "
        f"{MESSAGE_ANGLES[candidate.message_angle]['label'].lower()}, "
        f"{offers[candidate.offer]['label'].lower()}, and "
        f"{CTAS[candidate.cta]['label'].lower()}."
    )


def _idea_rationale(role_id: str, candidate: StrategyCandidate) -> str:
    focus = ", ".join(ROLE_PRIORS[role_id]["segment_weights"])
    return (
        f"Selected because this role over-weights {focus.replace('_', ' ')}, "
        f"and the candidate leans into {MESSAGE_ANGLES[candidate.message_angle]['label']} "
        f"without defaulting to an unnecessarily expensive save."
    )


def _findings_for_spec(spec: FlowResearchSpec) -> list[ResearchFinding]:
    findings = [
        ResearchFinding(
            title="Primary hypothesis",
            detail=spec.user_state_hypothesis,
            lens="retention",
            severity="high",
        ),
        ResearchFinding(
            title="Cancellation moment",
            detail=spec.cancellation_moment_hypothesis,
            lens="behavior",
            severity="medium",
        ),
    ]
    findings.extend(
        ResearchFinding(title="Trust risk", detail=item, lens="trust", severity="medium")
        for item in spec.trust_risks
    )
    findings.extend(
        ResearchFinding(title="Economic risk", detail=item, lens="economics", severity="medium")
        for item in spec.economic_risks
    )
    return findings


def _proposal_from_spec(
    *,
    proposal_id: str,
    role_label: str,
    spec: FlowResearchSpec,
    candidate: StrategyCandidate,
    compile_notes: list[str],
    settings: ResearchSettings,
    sample_persona: Persona,
    label: str,
    thesis: str,
    rationale: str,
) -> IdeaProposal:
    findings = _findings_for_spec(spec)
    experiment = build_experiment_spec(spec, candidate)
    return IdeaProposal(
        id=proposal_id,
        agent_role=role_label,
        label=label,
        thesis=thesis,
        rationale=rationale,
        target_segment=spec.target_segment,
        confidence=spec.confidence,
        candidate=candidate,
        sample_message=render_message(sample_persona, candidate, settings),
        flow_spec=flow_spec_to_payload(spec),
        research_trace=research_trace_payload(
            spec=spec,
            findings=findings,
            compile_notes=compile_notes,
        ),
        experiment_spec=experiment.to_dict(),
    )


def _heuristic_proposals(
    personas: list[Persona],
    candidate_universe: list[StrategyCandidate],
    settings: ResearchSettings,
) -> list[IdeaProposal]:
    """Generate proposals from heuristic scoring — no fabricated research specs.

    Heuristic proposals are honest about what they are: ranked candidates from
    the simulator, not hypothesis-driven research. Only LLM-generated proposals
    carry research_trace/flow_spec/experiment_spec metadata.
    """
    by_segment: dict[str, list[Persona]] = defaultdict(list)
    for persona in personas:
        by_segment[persona.features.segment].append(persona)

    segment_aggregates = {
        segment: _aggregate_persona(group, segment)
        for segment, group in by_segment.items()
    }

    selected: list[IdeaProposal] = []
    used_candidates: set[str] = set()
    offers = offer_catalog(settings)

    for role_label in settings.available_roles():
        role_id = ROLE_LABEL_TO_ID[role_label]
        segment_persona = max(
            segment_aggregates.values(),
            key=lambda persona: _segment_strength(persona, role_id),
        )

        ranked = sorted(
            candidate_universe,
            key=lambda candidate: (
                score_candidate_details(segment_persona, candidate)["score"]
                + _role_bonus(candidate, role_id)
            ),
            reverse=True,
        )

        proposals_for_role = 0
        for candidate in ranked:
            key = candidate_key(candidate)
            if key in used_candidates:
                continue
            confidence = round(
                min(
                    score_candidate_details(segment_persona, candidate)["trust"]
                    + _role_bonus(candidate, role_id),
                    0.98,
                ),
                4,
            )
            proposal_id = f"{role_id}::{key}"
            selected.append(IdeaProposal(
                id=proposal_id,
                agent_role=role_label,
                label=candidate_label(candidate, offers=offers),
                thesis=_idea_thesis(role_label, segment_persona.features.segment, candidate, settings),
                rationale=_idea_rationale(role_id, candidate),
                target_segment=segment_persona.features.segment,
                confidence=confidence,
                candidate=candidate,
                sample_message=render_message(segment_persona, candidate, settings),
                research_trace=None,
                flow_spec=None,
                experiment_spec=None,
            ))
            used_candidates.add(key)
            proposals_for_role += 1
            if proposals_for_role >= settings.idea_proposals_per_agent:
                break

    return selected[: settings.generated_idea_limit]


def _catalog_prompt(settings: ResearchSettings) -> str:
    offers = offer_catalog(settings)
    return json.dumps(
        {
            "message_angles": sorted(MESSAGE_ANGLES.keys()),
            "proof_styles": sorted(PROOF_STYLES.keys()),
            "offers": sorted(offers.keys()),
            "ctas": sorted(CTAS.keys()),
            "personalization": sorted(PERSONALIZATION_LEVELS.keys()),
            "contextual_groundings": sorted(CONTEXTUAL_GROUNDINGS.keys()),
            "creative_treatments": sorted(CREATIVE_TREATMENTS.keys()),
            "friction_reducers": sorted(FRICTION_REDUCERS.keys()),
        },
        indent=2,
    )


def _persona_prompt(personas: list[Persona], settings: ResearchSettings) -> str:
    by_segment: dict[str, list[Persona]] = defaultdict(list)
    for persona in personas:
        by_segment[persona.features.segment].append(persona)

    blocks = []
    for segment, group in sorted(by_segment.items(), key=lambda item: len(item[1]), reverse=True)[: settings.segment_focus_limit]:
        exemplar = group[0]
        blocks.append(
            {
                "segment": segment,
                "count": len(group),
                "study_context": exemplar.profile.study_context,
                "recent_behavior": exemplar.profile.recent_behavior,
                "features": {
                    key: round(mean(getattr(persona.features, key) for persona in group), 4)
                    for key in _feature_fields()
                },
            }
        )
    return json.dumps(blocks, indent=2)


def _parse_response_payload(raw: str) -> list[dict]:
    match = re.search(r"\[\s*{.*}\s*]", raw, flags=re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def _openai_proposals(
    personas: list[Persona],
    candidate_universe: list[StrategyCandidate],
    settings: ResearchSettings,
    progress_callback=None,
) -> tuple[list[IdeaProposal], str | None]:
    if OpenAI is None:
        return [], "OpenAI SDK is not installed."
    if not settings.has_api_key:
        return [], "OPENAI_API_KEY is not set."

    allowed = {candidate_key(candidate): candidate for candidate in candidate_universe}
    offers = offer_catalog(settings)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    proposals: list[IdeaProposal] = []

    roles = settings.available_roles()
    total_roles = max(len(roles), 1)
    for index, role_label in enumerate(roles, start=1):
        role_id = ROLE_LABEL_TO_ID[role_label]
        if progress_callback:
            progress = 0.48 + 0.14 * ((index - 1) / total_roles)
            progress_callback(progress, "ideation", f"{role_label} is drafting hypothesis-first cancellation rescues.")
        prompt = "\n".join(
            [
                "You are one specialist agent in a churn-reduction research swarm.",
                f"Specialty: {role_label}.",
                f"Propose exactly {settings.idea_proposals_per_agent} cancellation-save flow ideas.",
                "Think from first principles about why the user is cancelling and what rescue flow would most likely retain them.",
                "You may reason in open-ended terms first, but you must still supply the closest matching structured candidate fields.",
                "Output JSON only as an array of objects with keys:",
                (
                    "label, thesis, rationale, target_segment, confidence, user_state_hypothesis, "
                    "cancellation_moment_hypothesis, rescue_objective, step_sequence, copy_blocks, offer_logic, "
                    "cta_logic, branch_logic, trust_risks, economic_risks, evaluation_notes, falsifiable_assumption, "
                    "message_angle, proof_style, offer, cta, personalization, contextual_grounding, creative_treatment, friction_reducer."
                ),
                "Allowed catalog:",
                _catalog_prompt(settings),
                "Persona segments:",
                _persona_prompt(personas, settings),
            ]
        )
        response = client.responses.create(
            model=settings.model_name,
            input=prompt,
            reasoning={"effort": settings.openai_reasoning_effort},
            max_output_tokens=max(1400, 650 * settings.idea_proposals_per_agent),
        )
        for item in _parse_response_payload(response.output_text):
            segment = str(item.get("target_segment", "unknown"))
            fallback_persona = next((persona for persona in personas if persona.features.segment == segment), personas[0])
            explicit_candidate = StrategyCandidate(
                message_angle=str(item.get("message_angle", "progress_reflection")),
                proof_style=str(item.get("proof_style", "similar_user_story")),
                offer=str(item.get("offer", "none")),
                cta=str(item.get("cta", "stay_on_current_plan")),
                personalization=str(item.get("personalization", "contextual")),
                contextual_grounding=str(item.get("contextual_grounding", "generic")),
                creative_treatment=str(item.get("creative_treatment", "plain_note")),
                friction_reducer=str(item.get("friction_reducer", "none")),
            )
            spec = FlowResearchSpec(
                id=f"{role_id}::{index}::{len(proposals)}",
                agent_role=role_label,
                target_segment=segment,
                user_state_hypothesis=str(item.get("user_state_hypothesis") or f"{segment.replace('_', ' ')} users need a more relevant rescue path."),
                cancellation_moment_hypothesis=str(item.get("cancellation_moment_hypothesis") or "The user is leaving because the current cancellation screen does not address the real reason for churn."),
                rescue_objective=str(item.get("rescue_objective") or str(item.get("thesis") or "")),
                step_sequence=tuple(str(part) for part in item.get("step_sequence", []) if str(part).strip()) or (
                    "Acknowledge the user intent.",
                    "Offer the most relevant rescue path.",
                    "Make the next action feel reversible and safe.",
                ),
                copy_blocks=tuple(str(part) for part in item.get("copy_blocks", []) if str(part).strip()) or (
                    str(item.get("label") or "Save concept"),
                    str(item.get("rationale") or "Model-generated rescue flow."),
                ),
                offer_logic=str(item.get("offer_logic") or str(item.get("offer") or "No offer.")),
                cta_logic=str(item.get("cta_logic") or str(item.get("cta") or "Provide the clearest next step.")),
                branch_logic=str(item.get("branch_logic") or "If the primary rescue fails, reveal a lower-friction fallback."),
                trust_risks=tuple(str(part) for part in item.get("trust_risks", []) if str(part).strip()),
                economic_risks=tuple(str(part) for part in item.get("economic_risks", []) if str(part).strip()),
                evaluation_notes=tuple(str(part) for part in item.get("evaluation_notes", []) if str(part).strip()),
                falsifiable_assumption=str(item.get("falsifiable_assumption") or ""),
                confidence=float(item.get("confidence", 0.72)),
            )
            candidate, compile_notes = compile_flow_spec(
                spec,
                candidate_universe=list(allowed.values()),
                settings=settings,
                fallback_candidate=explicit_candidate,
            )
            key = candidate_key(candidate)
            sample_persona = fallback_persona
            proposals.append(
                _proposal_from_spec(
                    proposal_id=f"{role_id}::{key}",
                    role_label=role_label,
                    spec=spec,
                    candidate=candidate,
                    compile_notes=compile_notes,
                    settings=settings,
                    sample_persona=sample_persona,
                    label=str(item.get("label") or candidate_label(candidate, offers=offers)),
                    thesis=str(item.get("thesis") or _idea_thesis(role_label, segment, candidate, settings)),
                    rationale=str(item.get("rationale") or "Generated from an open-ended cancellation hypothesis."),
                )
            )
        if progress_callback:
            progress = 0.48 + 0.14 * (index / total_roles)
            progress_callback(progress, "ideation", f"{role_label} finished proposing and compiling ideas.")

    unique: dict[str, IdeaProposal] = {}
    for proposal in proposals:
        unique[proposal.id] = proposal
    return list(unique.values())[: settings.generated_idea_limit], None


def generate_ideas(
    personas: list[Persona],
    candidate_universe: list[StrategyCandidate],
    settings: ResearchSettings,
    progress_callback=None,
) -> tuple[list[IdeaProposal], list[str]]:
    warnings: list[str] = []
    heuristic = _heuristic_proposals(personas, candidate_universe, settings)
    if progress_callback:
        progress_callback(0.44, "ideation", f"Heuristic ideation prepared {len(heuristic)} proposals.")
    if not settings.openai_for_research:
        return heuristic, warnings

    proposals, warning = _openai_proposals(personas, candidate_universe, settings, progress_callback=progress_callback)
    if warning:
        warnings.append(warning)
        return heuristic, warnings
    if not proposals:
        warnings.append("Model ideation returned no valid structured strategies; using heuristic agent ideas.")
        return heuristic, warnings

    merged: dict[str, IdeaProposal] = {proposal.id: proposal for proposal in heuristic}
    for proposal in proposals:
        merged[proposal.id] = proposal
    return list(merged.values())[: settings.generated_idea_limit], warnings

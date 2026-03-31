from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean

from cta_autoresearch.models import IdeaProposal, Persona, StrategyCandidate, StrategyScore
from cta_autoresearch.research_config import PERSONA_RICHNESS, ResearchConfig
from cta_autoresearch.strategy_policy import (
    CONTEXTUAL_GROUNDINGS,
    CREATIVE_TREATMENTS,
    FRICTION_REDUCERS,
    OFFERS,
    candidate_key,
    candidate_label,
    render_message,
    valid_candidate,
)


AGENT_LIBRARY = [
    {
        "role": "Retention Psychologist",
        "thesis": "Use identity and momentum before the learner rationalizes churn.",
        "angles": ["progress_reflection", "habit_identity", "momentum_protection"],
        "groundings": ["progress_snapshot", "recovery_moment", "study_goal"],
        "treatments": ["progress_thermometer", "coach_note", "plain_note"],
        "reducers": ["plan_comparison", "smart_resume_date", "none"],
    },
    {
        "role": "Offer Economist",
        "thesis": "Spend margin only when price pain is real and rescue odds justify it.",
        "angles": ["cost_value_reframe", "flexibility_relief", "outcome_proof"],
        "groundings": ["unused_value", "study_goal", "deadline_countdown"],
        "treatments": ["plain_note", "feature_collage", "before_after_frame"],
        "reducers": ["billing_date_shift", "prefilled_downgrade", "plan_comparison"],
    },
    {
        "role": "Lifecycle Strategist",
        "thesis": "Route people to the least-destructive next action instead of pushing a single stay choice.",
        "angles": ["flexibility_relief", "empathetic_exit", "fresh_start_reset"],
        "groundings": ["recovery_moment", "unused_value", "study_goal"],
        "treatments": ["coach_note", "plain_note", "before_after_frame"],
        "reducers": ["single_tap_pause", "prefilled_downgrade", "plan_comparison"],
    },
    {
        "role": "Product Storyteller",
        "thesis": "Show value already created so the save attempt feels earned, not desperate.",
        "angles": ["feature_unlock", "outcome_proof", "progress_reflection"],
        "groundings": ["unused_value", "progress_snapshot", "study_goal"],
        "treatments": ["feature_collage", "progress_thermometer", "social_proof_card"],
        "reducers": ["plan_comparison", "smart_resume_date", "none"],
    },
    {
        "role": "Support Concierge",
        "thesis": "Users who are stuck often need friction removal more than a deeper discount.",
        "angles": ["mistake_recovery", "empathetic_exit", "fresh_start_reset"],
        "groundings": ["recovery_moment", "study_goal", "unused_value"],
        "treatments": ["coach_note", "plain_note", "feature_collage"],
        "reducers": ["concierge_setup", "smart_resume_date", "plan_comparison"],
    },
    {
        "role": "Deadline Operator",
        "thesis": "When urgency is real, frame the next few days as the point of maximum leverage.",
        "angles": ["goal_deadline", "momentum_protection", "progress_reflection"],
        "groundings": ["deadline_countdown", "study_goal", "progress_snapshot"],
        "treatments": ["before_after_frame", "progress_thermometer", "coach_note"],
        "reducers": ["single_tap_pause", "smart_resume_date", "none"],
    },
    {
        "role": "Trust Guardian",
        "thesis": "Reduce creepiness while preserving useful specificity and choice architecture.",
        "angles": ["empathetic_exit", "flexibility_relief", "feature_unlock"],
        "groundings": ["generic", "unused_value", "study_goal"],
        "treatments": ["plain_note", "social_proof_card", "before_after_frame"],
        "reducers": ["plan_comparison", "prefilled_downgrade", "none"],
    },
    {
        "role": "Visual CTA Designer",
        "thesis": "Change the treatment, not just the words, so the intervention feels materially different.",
        "angles": ["feature_unlock", "progress_reflection", "cost_value_reframe"],
        "groundings": ["progress_snapshot", "unused_value", "study_goal"],
        "treatments": ["feature_collage", "progress_thermometer", "before_after_frame"],
        "reducers": ["plan_comparison", "billing_date_shift", "smart_resume_date"],
    },
    {
        "role": "Win-Back Researcher",
        "thesis": "Dormant and tired users need a comeback path, not a harder sell.",
        "angles": ["fresh_start_reset", "mistake_recovery", "empathetic_exit"],
        "groundings": ["recovery_moment", "unused_value", "study_goal"],
        "treatments": ["coach_note", "social_proof_card", "plain_note"],
        "reducers": ["smart_resume_date", "single_tap_pause", "concierge_setup"],
    },
]


def _average_persona(personas: list[Persona], key: str) -> float:
    return mean(getattr(persona.features, key) for persona in personas)


def select_persona_representatives(personas: list[Persona], config: ResearchConfig) -> list[Persona]:
    representative_count = PERSONA_RICHNESS[config.persona_richness]["representative_count"]
    by_segment: dict[str, list[Persona]] = defaultdict(list)
    for persona in personas:
        by_segment[persona.features.segment].append(persona)

    representatives: list[Persona] = []
    for segment_group in by_segment.values():
        ranked = sorted(
            segment_group,
            key=lambda persona: (
                persona.features.rescue_readiness,
                persona.features.deadline_pressure,
                persona.features.fatigue_risk,
                persona.features.feature_awareness_gap,
            ),
            reverse=True,
        )
        representatives.append(ranked[0])

    extras = sorted(
        personas,
        key=lambda persona: (
            persona.features.rescue_readiness,
            persona.features.habit_fragility,
            persona.features.price_sensitivity,
        ),
        reverse=True,
    )
    for persona in extras:
        if persona.name not in {item.name for item in representatives}:
            representatives.append(persona)
        if len(representatives) >= representative_count:
            break
    return representatives[:representative_count]


def _preferred_offer(persona: Persona, anchor: StrategyScore | None, rng: random.Random) -> str:
    features = persona.features
    if features.price_sensitivity > 0.82:
        return f"discount_{rng.choice([20, 30, 40, 60] if features.habit_strength > 0.55 else [40, 60, 80, 100])}"
    if features.habit_fragility > 0.68:
        return "pause_plan"
    if features.deadline_pressure > 0.76:
        return "exam_sprint"
    if features.support_need > 0.70:
        return "concierge_support"
    if features.feature_awareness_gap > 0.62:
        return "bonus_credits"
    if anchor:
        return anchor.candidate.offer
    return "flexible_billing"


def _preferred_cta(offer: str, persona: Persona) -> str:
    kind = OFFERS[offer]["kind"]
    if kind == "pause":
        return "pause_instead"
    if kind == "downgrade":
        return "switch_to_lite"
    if kind == "support":
        return "talk_to_learning_support"
    if persona.features.deadline_pressure > 0.74:
        return "finish_current_goal"
    if kind in {"discount", "credit", "billing"}:
        return "claim_offer"
    if persona.features.rescue_readiness > 0.72:
        return "see_plan_options"
    return "stay_on_current_plan"


def _preferred_personalization(persona: Persona) -> str:
    features = persona.features
    if features.trust_sensitivity > 0.72:
        return "contextual"
    if features.value_realization > 0.66 and features.habit_strength > 0.58:
        return "behavioral"
    if features.deadline_pressure > 0.82 and features.trust_sensitivity < 0.45:
        return "highly_specific"
    return "contextual"


def _proposal_from_agent(agent: dict[str, object], persona: Persona, anchor: StrategyScore | None, rng: random.Random) -> StrategyCandidate:
    features = persona.features
    message_angle = rng.choice(agent["angles"])
    proof_style = "personal_usage_signal" if features.value_realization > 0.65 else "similar_user_story"
    if features.proof_need > 0.70:
        proof_style = "expert_validation" if features.support_need > 0.62 else "quantified_outcome"
    if anchor and rng.random() < 0.4:
        message_angle = anchor.candidate.message_angle
        proof_style = anchor.candidate.proof_style

    candidate = StrategyCandidate(
        message_angle=message_angle,
        proof_style=proof_style,
        offer=_preferred_offer(persona, anchor, rng),
        cta="stay_on_current_plan",
        personalization=_preferred_personalization(persona),
        contextual_grounding=rng.choice(agent["groundings"]),
        creative_treatment=rng.choice(agent["treatments"]),
        friction_reducer=rng.choice(agent["reducers"]),
    )
    candidate = StrategyCandidate(
        **{**candidate.__dict__, "cta": _preferred_cta(candidate.offer, persona)}
    )

    if valid_candidate(candidate):
        return candidate

    fallback = StrategyCandidate(
        message_angle="empathetic_exit",
        proof_style="similar_user_story",
        offer="flexible_billing",
        cta="see_plan_options",
        personalization="contextual",
        contextual_grounding="study_goal",
        creative_treatment="plain_note",
        friction_reducer="none",
    )
    return fallback


def generate_idea_proposals(
    personas: list[Persona],
    anchors: list[StrategyScore],
    config: ResearchConfig,
) -> tuple[list[IdeaProposal], dict[str, object]]:
    rng = random.Random(config.seed)
    representatives = select_persona_representatives(personas, config)
    selected_agents = AGENT_LIBRARY[: config.ideation_agents]
    proposals: list[IdeaProposal] = []
    seen_ids: set[str] = set()

    for round_index in range(config.ideation_rounds):
        for agent_index, agent in enumerate(selected_agents):
            for persona_index, persona in enumerate(representatives):
                anchor = anchors[(round_index + agent_index + persona_index) % len(anchors)] if anchors else None
                candidate = _proposal_from_agent(agent, persona, anchor, rng)
                proposal_id = candidate_key(candidate)
                if proposal_id in seen_ids:
                    continue
                seen_ids.add(proposal_id)
                confidence = clamp(
                    0.34 * persona.features.rescue_readiness
                    + 0.22 * persona.features.value_realization
                    + 0.18 * persona.features.deadline_pressure
                    + 0.26 * (1.0 - persona.features.trust_sensitivity)
                )
                proposals.append(
                    IdeaProposal(
                        id=proposal_id,
                        agent_role=str(agent["role"]),
                        label=candidate_label(candidate),
                        thesis=str(agent["thesis"]),
                        rationale=(
                            f"{agent['role']} targeted {persona.features.segment} with "
                            f"{candidate.contextual_grounding.replace('_', ' ')} grounding and a "
                            f"{candidate.creative_treatment.replace('_', ' ')} treatment."
                        ),
                        target_segment=persona.features.segment,
                        confidence=round(confidence, 4),
                        candidate=candidate,
                        sample_message=render_message(persona, candidate),
                    )
                )
                if len(proposals) >= config.validation_budget:
                    break
            if len(proposals) >= config.validation_budget:
                break
        if len(proposals) >= config.validation_budget:
            break

    meta = {
        "agents_used": [agent["role"] for agent in selected_agents],
        "representatives_used": [persona.name for persona in representatives],
        "ideation_rounds": config.ideation_rounds,
        "validation_budget": config.validation_budget,
        "average_representative_rescue_readiness": round(_average_persona(representatives, "rescue_readiness"), 4),
    }
    return proposals, meta


def generate_ideas(
    personas: list[Persona],
    candidate_universe: list[StrategyCandidate],
    settings: object | None = None,
) -> tuple[list[IdeaProposal], list[str]]:
    config = ResearchConfig.from_overrides(
        population=len(personas),
        seed=getattr(settings, "seed", 7) if settings is not None else 7,
        top_n=getattr(settings, "top_n", 10) if settings is not None else 10,
        depth_mode="extreme" if getattr(settings, "depth", 2) >= 4 else "deep" if getattr(settings, "depth", 2) >= 3 else "standard" if getattr(settings, "depth", 2) >= 2 else "quick",
        persona_richness="extreme" if getattr(settings, "persona_richness", 2) >= 3 else "rich" if getattr(settings, "persona_richness", 2) >= 2 else "standard",
        validation_budget=getattr(settings, "effective_validation_budget", len(candidate_universe)) if settings is not None else len(candidate_universe),
        ideation_rounds=getattr(settings, "ideation_rounds", 1) if settings is not None else 1,
        model_name=getattr(settings, "model", None) if settings is not None else None,
        model_provider="openai" if getattr(settings, "use_llm", False) else "heuristic",
    )
    anchors = [
        StrategyScore(
            candidate=candidate,
            average_score=0.0,
            baseline_lift=0.0,
            retention_score=0.0,
            revenue_score=0.0,
            trust_safety_score=0.0,
            component_scores={},
        )
        for candidate in candidate_universe[: max(12, min(len(candidate_universe), config.anchor_count))]
    ]
    proposals, meta = generate_idea_proposals(personas, anchors, config)
    warnings: list[str] = []
    if config.model_provider != "heuristic" and not config.provider_status()["available"]:
        warnings.append(config.provider_status()["warning"])
    if config.validation_budget < 100:
        warnings.append("Validation budget is low; deeper idea exploration may uncover additional winners.")
    warnings.append(f"{len(meta['agents_used'])} idea agents contributed to this run.")
    return proposals, warnings


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))

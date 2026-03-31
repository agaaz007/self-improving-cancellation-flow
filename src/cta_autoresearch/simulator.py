from __future__ import annotations

from cta_autoresearch.features import clamp
from cta_autoresearch.models import FeatureVector, Persona, StrategyCandidate
from cta_autoresearch.strategy_policy import (
    CONTEXTUAL_GROUNDINGS,
    CREATIVE_TREATMENTS,
    FRICTION_REDUCERS,
    OFFERS,
    PERSONALIZATION_LEVELS,
)


def _angle_fit(features: FeatureVector, angle: str) -> float:
    formulas = {
        "progress_reflection": 0.46 * features.habit_strength + 0.18 * features.activation_score + 0.22 * features.loss_aversion + 0.14 * features.value_realization,
        "outcome_proof": 0.40 * features.proof_need + 0.24 * features.price_sensitivity + 0.20 * features.feature_awareness_gap + 0.16 * features.deadline_pressure,
        "mistake_recovery": 0.34 * features.support_need + 0.22 * features.urgency + 0.20 * features.activation_score + 0.24 * features.rescue_readiness,
        "habit_identity": 0.46 * features.habit_strength + 0.18 * features.loss_aversion + 0.14 * features.activation_score + 0.22 * (1.0 - features.habit_fragility),
        "empathetic_exit": 0.34 * features.trust_sensitivity + 0.24 * features.support_need + 0.22 * features.friction_sensitivity + 0.20 * features.fatigue_risk,
        "feature_unlock": 0.44 * features.feature_awareness_gap + 0.20 * features.proof_need + 0.16 * features.activation_score + 0.20 * (1.0 - features.value_realization),
        "momentum_protection": 0.36 * features.habit_strength + 0.24 * features.urgency + 0.18 * features.loss_aversion + 0.22 * features.habit_fragility,
        "cost_value_reframe": 0.34 * features.price_sensitivity + 0.22 * features.proof_need + 0.20 * features.feature_awareness_gap + 0.24 * (1.0 - features.value_realization),
        "goal_deadline": 0.42 * features.deadline_pressure + 0.22 * features.loss_aversion + 0.18 * features.habit_strength + 0.18 * features.urgency,
        "flexibility_relief": 0.36 * features.friction_sensitivity + 0.28 * features.trust_sensitivity + 0.20 * features.price_sensitivity + 0.16 * features.fatigue_risk,
        "fresh_start_reset": 0.28 * features.support_need + 0.24 * features.feature_awareness_gap + 0.22 * (1.0 - features.activation_score) + 0.26 * features.fatigue_risk,
    }
    return clamp(formulas[angle])


def _proof_fit(features: FeatureVector, proof_style: str) -> float:
    formulas = {
        "none": 0.30 * features.trust_sensitivity + 0.40 * features.habit_strength + 0.30 * features.value_realization,
        "quantified_outcome": 0.40 * features.proof_need + 0.20 * features.price_sensitivity + 0.20 * features.urgency + 0.20 * features.deadline_pressure,
        "peer_testimonial": 0.32 * features.proof_need + 0.26 * features.trust_sensitivity + 0.20 * features.feature_awareness_gap + 0.22 * features.fatigue_risk,
        "similar_user_story": 0.28 * features.proof_need + 0.24 * features.feature_awareness_gap + 0.20 * features.urgency + 0.28 * features.rescue_readiness,
        "expert_validation": 0.28 * features.proof_need + 0.24 * features.support_need + 0.28 * features.trust_sensitivity + 0.20 * features.deadline_pressure,
        "personal_usage_signal": 0.30 * features.habit_strength + 0.18 * features.activation_score + 0.20 * features.loss_aversion + 0.32 * features.value_realization,
    }
    return clamp(formulas[proof_style])


def _offer_fit(features: FeatureVector, offer: str) -> float:
    offer_meta = OFFERS[offer]
    if offer_meta["kind"] == "discount":
        generosity = offer_meta["generosity"]
        discount_weight = 0.32 + 0.30 * generosity
        urgency_weight = max(0.08, 0.22 - 0.10 * generosity)
        return clamp(
            discount_weight * features.discount_affinity
            + 0.26 * features.price_sensitivity
            + urgency_weight * features.urgency
            + 0.20 * features.fatigue_risk
        )

    formulas = {
        "none": 0.30 * features.habit_strength + 0.24 * features.trust_sensitivity + 0.20 * features.urgency + 0.26 * features.value_realization,
        "pause_plan": 0.30 * features.friction_sensitivity + 0.20 * features.trust_sensitivity + 0.16 * features.habit_strength + 0.34 * features.habit_fragility,
        "downgrade_lite": 0.28 * features.price_sensitivity + 0.20 * features.habit_strength + 0.22 * features.friction_sensitivity + 0.30 * features.rescue_readiness,
        "exam_sprint": 0.38 * features.deadline_pressure + 0.18 * features.activation_score + 0.18 * features.loss_aversion + 0.26 * features.urgency,
        "bonus_credits": 0.34 * features.feature_awareness_gap + 0.16 * features.proof_need + 0.14 * features.activation_score + 0.36 * features.rescue_readiness,
        "flexible_billing": 0.28 * features.price_sensitivity + 0.24 * features.friction_sensitivity + 0.16 * features.trust_sensitivity + 0.32 * features.fatigue_risk,
        "concierge_support": 0.34 * features.support_need + 0.16 * features.trust_sensitivity + 0.16 * features.feature_awareness_gap + 0.34 * features.rescue_readiness,
        "study_plan_reset": 0.28 * features.support_need + 0.18 * features.feature_awareness_gap + 0.18 * features.fatigue_risk + 0.36 * features.rescue_readiness,
        "priority_review_pack": 0.30 * features.deadline_pressure + 0.18 * features.urgency + 0.14 * features.proof_need + 0.38 * features.rescue_readiness,
        "deadline_extension_plus": 0.36 * features.deadline_pressure + 0.22 * features.urgency + 0.14 * features.loss_aversion + 0.28 * features.rescue_readiness,
        "office_hours_access": 0.32 * features.support_need + 0.18 * features.trust_sensitivity + 0.16 * features.proof_need + 0.34 * features.rescue_readiness,
    }
    return clamp(formulas[offer])


def _cta_fit(features: FeatureVector, cta: str) -> float:
    formulas = {
        "stay_on_current_plan": 0.34 * features.habit_strength + 0.18 * features.loss_aversion + 0.14 * features.activation_score + 0.34 * features.value_realization,
        "claim_offer": 0.30 * features.discount_affinity + 0.18 * features.price_sensitivity + 0.14 * features.proof_need + 0.38 * features.rescue_readiness,
        "pause_instead": 0.28 * features.friction_sensitivity + 0.18 * features.trust_sensitivity + 0.14 * features.habit_strength + 0.40 * features.habit_fragility,
        "switch_to_lite": 0.30 * features.price_sensitivity + 0.20 * features.friction_sensitivity + 0.12 * features.habit_strength + 0.38 * features.rescue_readiness,
        "finish_current_goal": 0.34 * features.deadline_pressure + 0.18 * features.loss_aversion + 0.10 * features.activation_score + 0.38 * features.urgency,
        "talk_to_learning_support": 0.28 * features.support_need + 0.18 * features.trust_sensitivity + 0.12 * features.proof_need + 0.42 * features.rescue_readiness,
        "tell_us_why": 0.22 * features.support_need + 0.18 * features.trust_sensitivity + 0.18 * features.feature_awareness_gap + 0.42 * features.fatigue_risk,
        "see_plan_options": 0.18 * features.friction_sensitivity + 0.18 * features.price_sensitivity + 0.16 * features.trust_sensitivity + 0.48 * features.rescue_readiness,
        "remind_me_later": 0.24 * features.friction_sensitivity + 0.18 * features.trust_sensitivity + 0.16 * (1.0 - features.urgency) + 0.42 * features.fatigue_risk,
    }
    return clamp(formulas[cta])


def _personalization_fit(features: FeatureVector, personalization: str) -> float:
    intensity = PERSONALIZATION_LEVELS[personalization]["intensity"]
    sweet_spot = 0.16 + 0.26 * features.habit_strength + 0.20 * features.urgency + 0.18 * features.value_realization - 0.28 * features.trust_sensitivity
    return clamp(1.0 - abs(intensity - clamp(sweet_spot)))


def _grounding_fit(features: FeatureVector, grounding: str) -> float:
    formulas = {
        "generic": 0.42 * features.trust_sensitivity + 0.28 * features.fatigue_risk + 0.30 * features.value_realization,
        "study_goal": 0.38 * features.urgency + 0.36 * features.deadline_pressure + 0.26 * features.value_realization,
        "recent_progress": 0.36 * features.habit_strength + 0.28 * features.activation_score + 0.36 * features.value_realization,
        "deadline_pressure": 0.52 * features.deadline_pressure + 0.24 * features.urgency + 0.24 * features.loss_aversion,
        "unused_value": 0.40 * features.feature_awareness_gap + 0.30 * (1.0 - features.value_realization) + 0.30 * features.proof_need,
        "comeback_window": 0.30 * features.habit_fragility + 0.22 * features.fatigue_risk + 0.20 * features.support_need + 0.28 * features.rescue_readiness,
        "pricing_context": 0.36 * features.price_sensitivity + 0.26 * features.discount_affinity + 0.18 * features.friction_sensitivity + 0.20 * features.fatigue_risk,
        "support_signal": 0.38 * features.support_need + 0.18 * features.trust_sensitivity + 0.16 * features.feature_awareness_gap + 0.28 * features.rescue_readiness,
        "progress_snapshot": 0.36 * features.habit_strength + 0.28 * features.activation_score + 0.36 * features.value_realization,
        "deadline_countdown": 0.52 * features.deadline_pressure + 0.24 * features.urgency + 0.24 * features.loss_aversion,
        "recovery_moment": 0.30 * features.habit_fragility + 0.22 * features.fatigue_risk + 0.20 * features.support_need + 0.28 * features.rescue_readiness,
        "habit_streak": 0.36 * features.habit_strength + 0.28 * features.activation_score + 0.36 * features.value_realization,
    }
    return clamp(formulas[grounding])


def _treatment_fit(features: FeatureVector, treatment: str) -> float:
    formulas = {
        "plain_note": 0.42 * features.trust_sensitivity + 0.30 * features.fatigue_risk + 0.28 * features.value_realization,
        "progress_snapshot": 0.30 * features.habit_strength + 0.20 * features.activation_score + 0.20 * features.loss_aversion + 0.30 * features.value_realization,
        "feature_visual": 0.30 * features.feature_awareness_gap + 0.20 * features.proof_need + 0.14 * features.activation_score + 0.36 * features.rescue_readiness,
        "study_timeline": 0.34 * features.deadline_pressure + 0.20 * features.urgency + 0.16 * features.loss_aversion + 0.30 * features.rescue_readiness,
        "peer_story_card": 0.30 * features.proof_need + 0.22 * features.trust_sensitivity + 0.14 * features.feature_awareness_gap + 0.34 * features.rescue_readiness,
        "options_table": 0.26 * features.friction_sensitivity + 0.22 * features.price_sensitivity + 0.16 * features.trust_sensitivity + 0.36 * features.rescue_readiness,
        "coach_plan": 0.22 * features.support_need + 0.20 * features.urgency + 0.18 * features.activation_score + 0.40 * features.rescue_readiness,
        "feature_collage": 0.30 * features.feature_awareness_gap + 0.20 * features.proof_need + 0.14 * features.activation_score + 0.36 * features.rescue_readiness,
        "progress_thermometer": 0.30 * features.habit_strength + 0.20 * features.activation_score + 0.20 * features.loss_aversion + 0.30 * features.value_realization,
        "comeback_plan": 0.22 * features.support_need + 0.20 * features.urgency + 0.18 * features.activation_score + 0.40 * features.rescue_readiness,
        "social_proof_card": 0.30 * features.proof_need + 0.22 * features.trust_sensitivity + 0.14 * features.feature_awareness_gap + 0.34 * features.rescue_readiness,
        "coach_note": 0.22 * features.support_need + 0.20 * features.urgency + 0.18 * features.activation_score + 0.40 * features.rescue_readiness,
        "before_after_frame": 0.26 * features.friction_sensitivity + 0.22 * features.price_sensitivity + 0.16 * features.trust_sensitivity + 0.36 * features.rescue_readiness,
    }
    return clamp(formulas[treatment])


def _friction_reducer_fit(features: FeatureVector, reducer: str) -> float:
    formulas = {
        "none": 0.38 * features.value_realization + 0.32 * features.habit_strength + 0.30 * (1.0 - features.friction_sensitivity),
        "one_tap_pause": 0.26 * features.friction_sensitivity + 0.16 * features.trust_sensitivity + 0.20 * features.habit_fragility + 0.38 * features.rescue_readiness,
        "one_tap_downgrade": 0.24 * features.price_sensitivity + 0.18 * features.friction_sensitivity + 0.18 * features.habit_strength + 0.40 * features.rescue_readiness,
        "billing_shift": 0.32 * features.price_sensitivity + 0.20 * features.fatigue_risk + 0.10 * features.trust_sensitivity + 0.38 * features.rescue_readiness,
        "keep_history": 0.20 * features.habit_strength + 0.18 * features.loss_aversion + 0.18 * features.trust_sensitivity + 0.44 * features.habit_fragility,
        "guided_reset": 0.28 * features.support_need + 0.16 * features.feature_awareness_gap + 0.18 * features.fatigue_risk + 0.38 * features.rescue_readiness,
        "human_concierge": 0.30 * features.support_need + 0.18 * features.trust_sensitivity + 0.16 * features.feature_awareness_gap + 0.36 * features.rescue_readiness,
        "single_tap_pause": 0.26 * features.friction_sensitivity + 0.16 * features.trust_sensitivity + 0.20 * features.habit_fragility + 0.38 * features.rescue_readiness,
        "prefilled_downgrade": 0.24 * features.price_sensitivity + 0.18 * features.friction_sensitivity + 0.18 * features.habit_strength + 0.40 * features.rescue_readiness,
        "billing_date_shift": 0.32 * features.price_sensitivity + 0.20 * features.fatigue_risk + 0.10 * features.trust_sensitivity + 0.38 * features.rescue_readiness,
        "concierge_setup": 0.30 * features.support_need + 0.18 * features.trust_sensitivity + 0.16 * features.feature_awareness_gap + 0.36 * features.rescue_readiness,
        "smart_resume_date": 0.28 * features.support_need + 0.16 * features.feature_awareness_gap + 0.18 * features.fatigue_risk + 0.38 * features.rescue_readiness,
        "plan_comparison": 0.20 * features.habit_strength + 0.18 * features.loss_aversion + 0.18 * features.trust_sensitivity + 0.44 * features.habit_fragility,
    }
    return clamp(formulas[reducer])


def _trust_penalty(features: FeatureVector, candidate: StrategyCandidate) -> float:
    intensity = PERSONALIZATION_LEVELS[candidate.personalization]["intensity"]
    offer_meta = OFFERS[candidate.offer]

    penalty = 0.0
    if intensity > 0.70 and features.trust_sensitivity > 0.68:
        penalty += 0.08
    if candidate.message_angle == "goal_deadline" and offer_meta["generosity"] >= 0.40 and features.trust_sensitivity > 0.55:
        penalty += 0.04
    if candidate.proof_style == "personal_usage_signal" and intensity > 0.70 and features.trust_sensitivity > 0.55:
        penalty += 0.05
    if candidate.message_angle == "empathetic_exit" and candidate.offer.startswith("discount_"):
        penalty += 0.02
    if candidate.contextual_grounding in {"recent_progress", "support_signal"} and intensity > 0.70 and features.trust_sensitivity > 0.60:
        penalty += 0.03
    if candidate.creative_treatment in {"progress_snapshot", "feature_visual"} and features.trust_sensitivity > 0.72:
        penalty += 0.02
    if candidate.friction_reducer == "human_concierge" and features.trust_sensitivity > 0.70:
        penalty += 0.03
    return penalty


def _economic_score(candidate: StrategyCandidate, features: FeatureVector) -> float:
    generosity = OFFERS[candidate.offer]["generosity"]
    base = 1.0 - generosity

    if candidate.offer == "pause_plan":
        base += 0.08
    elif candidate.offer == "downgrade_lite":
        base += 0.06
    elif candidate.offer == "flexible_billing":
        base += 0.05
    elif candidate.offer == "concierge_support":
        base -= 0.02
    elif candidate.offer == "bonus_credits":
        base += 0.03
    elif candidate.offer == "study_plan_reset":
        base += 0.02
    elif candidate.offer == "priority_review_pack":
        base += 0.01
    elif candidate.offer == "deadline_extension_plus":
        base -= 0.01
    elif candidate.offer == "office_hours_access":
        base -= 0.01

    if candidate.friction_reducer in {"one_tap_pause", "one_tap_downgrade", "billing_shift"}:
        base += 0.03
    if candidate.creative_treatment in {"progress_snapshot", "feature_visual", "coach_plan"}:
        base -= 0.01

    if generosity >= 0.40 and features.habit_strength > 0.70:
        base -= 0.10
    if generosity >= 0.70:
        base -= 0.08
    if generosity == 1.0:
        base -= 0.15

    return clamp(base)


def score_candidate_details(persona: Persona, candidate: StrategyCandidate) -> dict[str, float]:
    features = persona.features
    base_retention = (
        0.10
        + 0.20 * features.habit_strength
        + 0.10 * features.activation_score
        + 0.08 * features.urgency
        + 0.08 * features.value_realization
        - 0.06 * features.feature_awareness_gap
        - 0.04 * features.friction_sensitivity
        - 0.05 * features.fatigue_risk
    )
    if persona.profile.plan.lower() == "free":
        base_retention -= 0.04
    if persona.profile.status != "active":
        base_retention -= 0.03
    if persona.profile.dormancy_days > 30:
        base_retention -= 0.05

    angle_fit = _angle_fit(features, candidate.message_angle)
    proof_fit = _proof_fit(features, candidate.proof_style)
    offer_fit = _offer_fit(features, candidate.offer)
    cta_fit = _cta_fit(features, candidate.cta)
    personalization_fit = _personalization_fit(features, candidate.personalization)
    grounding_fit = _grounding_fit(features, candidate.contextual_grounding)
    treatment_fit = _treatment_fit(features, candidate.creative_treatment)
    friction_fit = _friction_reducer_fit(features, candidate.friction_reducer)

    trust_penalty = _trust_penalty(features, candidate)
    trust_safety = clamp(1.0 - trust_penalty)
    revenue_score = _economic_score(candidate, features)

    retention_score = clamp(
        base_retention
        + 0.16 * angle_fit
        + 0.10 * proof_fit
        + 0.12 * offer_fit
        + 0.10 * cta_fit
        + 0.06 * personalization_fit
        + 0.10 * grounding_fit
        + 0.10 * treatment_fit
        + 0.12 * friction_fit
        - trust_penalty
    )

    composite_score = clamp(
        0.58 * retention_score + 0.24 * revenue_score + 0.18 * trust_safety
    )

    return {
        "score": composite_score,
        "retention": retention_score,
        "revenue": revenue_score,
        "trust": trust_safety,
        "base_retention": clamp(base_retention),
        "angle_fit": angle_fit,
        "proof_fit": proof_fit,
        "offer_fit": offer_fit,
        "cta_fit": cta_fit,
        "personalization_fit": personalization_fit,
        "grounding_fit": grounding_fit,
        "treatment_fit": treatment_fit,
        "friction_fit": friction_fit,
        "trust_penalty": trust_penalty,
    }


def score_candidate(persona: Persona, candidate: StrategyCandidate) -> tuple[float, float]:
    details = score_candidate_details(persona, candidate)
    return details["score"], details["trust"]

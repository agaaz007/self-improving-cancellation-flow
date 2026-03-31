from __future__ import annotations

from cta_autoresearch.models import FeatureVector, UserProfile


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def norm(value: float, ceiling: float) -> float:
    return clamp(value / ceiling if ceiling else 0.0)


def infer_segment(profile: UserProfile, habit_strength: float, price_sensitivity: float, urgency: float) -> str:
    if profile.plan.lower() == "free":
        return "dormant_free_explorer" if profile.total_sessions < 60 else "engaged_free_user"
    if habit_strength > 0.75 and profile.retry_after_mistake:
        return "committed_power_learner"
    if urgency > 0.70 and habit_strength < 0.60:
        return "exam_cram_risk"
    if price_sensitivity > 0.72:
        return "price_sensitive_subscriber"
    return "steady_subscriber"


def derive_features(profile: UserProfile) -> FeatureVector:
    usage_ratio = 1.0
    if profile.monthly_generations_total:
        usage_ratio = 1.0 - (
            profile.monthly_generations_remaining / profile.monthly_generations_total
        )

    high_stakes_context = any(
        token in profile.study_context.lower()
        for token in ("exam", "test", "quiz", "final", "course", "science")
    )
    overload_signal = any(
        token in profile.recent_behavior.lower()
        for token in ("late-night", "heavy", "cram", "rush", "stalled", "overwhelmed")
    )
    high_accuracy = "high" in profile.accuracy_signal
    low_accuracy = "low" in profile.accuracy_signal

    habit_strength = clamp(
        0.34 * norm(profile.total_sessions, 600)
        + 0.28 * norm(profile.time_in_app_hours, 150)
        + 0.18 * norm(profile.lifetime_days, 900)
        + 0.20 * norm(profile.multi_device_count, 16)
    )
    activation_score = clamp(
        0.35 * norm(profile.card_sets_generated, 60)
        + 0.20 * usage_ratio
        + 0.20 * (1.0 if profile.retry_after_mistake else 0.0)
        + 0.10 * (1.0 if profile.source_context_usage else 0.0)
        + 0.15 * (1.0 if profile.status == "active" else 0.2)
    )
    price_sensitivity = clamp(
        0.35 * (1.0 if profile.plan.lower() == "free" else 0.55)
        + 0.25 * (1.0 - usage_ratio)
        + 0.20 * norm(profile.dormancy_days, 45)
        + 0.20 * (1.0 if profile.billing_period == "monthly" else 0.1)
    )
    urgency = clamp(
        0.35 * (1.0 if high_stakes_context else 0.2)
        + 0.25 * (1.0 if profile.retry_after_mistake else 0.1)
        + 0.20 * (1.0 if high_accuracy else 0.3)
        + 0.20 * (1.0 if profile.source_context_usage else 0.1)
    )
    proof_need = clamp(
        0.40 * (1.0 - activation_score)
        + 0.20 * (1.0 if profile.plan.lower() == "free" else 0.2)
        + 0.20 * norm(profile.dormancy_days, 60)
        + 0.20 * (1.0 if low_accuracy else 0.2)
    )
    discount_affinity = clamp(
        0.55 * price_sensitivity
        + 0.25 * (1.0 if profile.status != "active" else 0.35)
        + 0.20 * (1.0 if profile.billing_period == "monthly" else 0.15)
    )
    support_need = clamp(
        0.35 * norm(profile.dormancy_days, 60)
        + 0.30 * (1.0 - activation_score)
        + 0.20 * (1.0 if profile.retry_after_mistake else 0.25)
        + 0.15 * (1.0 if "wrong" in profile.recent_behavior.lower() else 0.2)
    )
    feature_awareness_gap = clamp(
        0.50 * (1.0 - usage_ratio)
        + 0.25 * (1.0 - activation_score)
        + 0.25 * (1.0 if "signup" in profile.recent_behavior.lower() else 0.1)
    )
    trust_sensitivity = clamp(
        0.35 * (1.0 if profile.plan.lower() == "free" else 0.2)
        + 0.25 * norm(profile.dormancy_days, 60)
        + 0.20 * (1.0 if "high school" in profile.user_type.lower() else 0.3)
        + 0.20 * (1.0 if profile.status != "active" else 0.3)
    )
    friction_sensitivity = clamp(
        0.40 * norm(profile.dormancy_days, 45)
        + 0.30 * (1.0 if profile.status != "active" else 0.2)
        + 0.30 * (1.0 if profile.plan.lower() == "free" else 0.4)
    )
    loss_aversion = clamp(0.60 * habit_strength + 0.40 * urgency)
    deadline_pressure = clamp(
        0.48 * (1.0 if high_stakes_context else 0.2)
        + 0.22 * urgency
        + 0.18 * (1.0 if profile.dormancy_days <= 5 else 0.35)
        + 0.12 * (1.0 if "final" in profile.study_context.lower() else 0.2)
    )
    value_realization = clamp(
        0.34 * usage_ratio
        + 0.26 * activation_score
        + 0.22 * habit_strength
        + 0.18 * (1.0 if profile.source_context_usage else 0.25)
    )
    fatigue_risk = clamp(
        0.32 * norm(profile.dormancy_days, 45)
        + 0.24 * (1.0 - activation_score)
        + 0.22 * (1.0 if overload_signal else 0.25)
        + 0.22 * (1.0 if profile.total_sessions > 100 else 0.35)
    )
    rescue_readiness = clamp(
        0.30 * habit_strength
        + 0.22 * urgency
        + 0.24 * support_need
        + 0.24 * (1.0 - trust_sensitivity)
    )
    habit_fragility = clamp(
        0.38 * friction_sensitivity
        + 0.24 * fatigue_risk
        + 0.20 * urgency
        + 0.18 * (1.0 - activation_score)
    )
    segment = infer_segment(profile, habit_strength, price_sensitivity, urgency)

    return FeatureVector(
        segment=segment,
        habit_strength=habit_strength,
        activation_score=activation_score,
        price_sensitivity=price_sensitivity,
        urgency=urgency,
        proof_need=proof_need,
        discount_affinity=discount_affinity,
        support_need=support_need,
        feature_awareness_gap=feature_awareness_gap,
        trust_sensitivity=trust_sensitivity,
        friction_sensitivity=friction_sensitivity,
        loss_aversion=loss_aversion,
        deadline_pressure=deadline_pressure,
        value_realization=value_realization,
        fatigue_risk=fatigue_risk,
        rescue_readiness=rescue_readiness,
        habit_fragility=habit_fragility,
    )

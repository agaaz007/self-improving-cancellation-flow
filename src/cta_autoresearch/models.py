from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UserProfile:
    name: str
    cohort: str
    plan: str
    status: str
    billing_period: str
    user_type: str
    lifetime_days: int
    total_sessions: int
    total_events: int
    time_in_app_hours: float
    card_sets_generated: int
    monthly_generations_remaining: int
    monthly_generations_total: int
    chat_messages_remaining: int
    answer_feedback_remaining: int
    multi_device_count: int
    acquisition_source: str
    recent_behavior: str
    study_context: str
    retry_after_mistake: bool
    source_context_usage: bool
    accuracy_signal: str
    dormancy_days: int

    @classmethod
    def from_dict(cls, payload: dict) -> "UserProfile":
        return cls(**payload)


@dataclass(frozen=True)
class FeatureVector:
    segment: str
    habit_strength: float
    activation_score: float
    price_sensitivity: float
    urgency: float
    proof_need: float
    discount_affinity: float
    support_need: float
    feature_awareness_gap: float
    trust_sensitivity: float
    friction_sensitivity: float
    loss_aversion: float
    deadline_pressure: float
    value_realization: float
    fatigue_risk: float
    rescue_readiness: float
    habit_fragility: float


@dataclass(frozen=True)
class PersonaInsights:
    behavioral_trace: dict[str, str]
    risk_factors: tuple[str, ...]
    retention_motivators: tuple[str, ...]
    likely_objections: tuple[str, ...]
    recommended_hooks: tuple[str, ...]
    narrative: str


@dataclass(frozen=True)
class Persona:
    name: str
    profile: UserProfile
    features: FeatureVector
    insights: PersonaInsights | None = None


@dataclass(frozen=True)
class StrategyCandidate:
    message_angle: str
    proof_style: str
    offer: str
    cta: str
    personalization: str
    contextual_grounding: str = "generic"
    creative_treatment: str = "plain_note"
    friction_reducer: str = "none"


@dataclass(frozen=True)
class StrategyScore:
    candidate: StrategyCandidate
    average_score: float
    baseline_lift: float
    retention_score: float
    revenue_score: float
    trust_safety_score: float
    component_scores: dict[str, float]


@dataclass(frozen=True)
class IdeaProposal:
    id: str
    agent_role: str
    label: str
    thesis: str
    rationale: str
    target_segment: str
    confidence: float
    candidate: StrategyCandidate
    sample_message: str

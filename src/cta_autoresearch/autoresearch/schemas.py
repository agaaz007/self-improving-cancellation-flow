from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ResearchFinding:
    title: str
    detail: str
    lens: str
    severity: str = "medium"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentSpec:
    target_segment: str
    hypothesis: str
    control_description: str
    treatment_description: str
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    guardrails: tuple[str, ...]
    rollout_suggestion: str
    rollback_trigger: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["secondary_metrics"] = list(self.secondary_metrics)
        payload["guardrails"] = list(self.guardrails)
        return payload


@dataclass(frozen=True)
class FlowResearchSpec:
    id: str
    agent_role: str
    target_segment: str
    user_state_hypothesis: str
    cancellation_moment_hypothesis: str
    rescue_objective: str
    step_sequence: tuple[str, ...]
    copy_blocks: tuple[str, ...]
    offer_logic: str
    cta_logic: str
    branch_logic: str
    trust_risks: tuple[str, ...] = field(default_factory=tuple)
    economic_risks: tuple[str, ...] = field(default_factory=tuple)
    evaluation_notes: tuple[str, ...] = field(default_factory=tuple)
    falsifiable_assumption: str = ""
    confidence: float = 0.7

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["step_sequence"] = list(self.step_sequence)
        payload["copy_blocks"] = list(self.copy_blocks)
        payload["trust_risks"] = list(self.trust_risks)
        payload["economic_risks"] = list(self.economic_risks)
        payload["evaluation_notes"] = list(self.evaluation_notes)
        return payload

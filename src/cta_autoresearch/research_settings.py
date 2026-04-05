from __future__ import annotations

import os
from dataclasses import asdict, dataclass


HEURISTIC_MODEL = "heuristic-simulator"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
REASONING_EFFORT_OPTIONS = {
    "low": {"label": "Low"},
    "medium": {"label": "Medium"},
    "high": {"label": "High"},
}
DEPTH_OPTIONS = {
    "quick": {"validation_budget": 80, "discount_step": 10, "grounding_limit": 1, "treatment_limit": 1, "friction_limit": 1},
    "standard": {"validation_budget": 240, "discount_step": 10, "grounding_limit": 1, "treatment_limit": 1, "friction_limit": 1},
    "deep": {"validation_budget": 1400, "discount_step": 5, "grounding_limit": 2, "treatment_limit": 1, "friction_limit": 1},
    "extreme": {"validation_budget": 2400, "discount_step": 5, "grounding_limit": 7, "treatment_limit": 7, "friction_limit": 6},
}
PERSONA_RICHNESS_OPTIONS = {
    "standard": {"population_boost": 0},
    "rich": {"population_boost": 20},
    "extreme": {"population_boost": 40},
}
AGENT_ROLES = [
    "Retention Psychologist",
    "Offer Economist",
    "Lifecycle Strategist",
    "Product Storyteller",
    "Support Concierge",
    "Deadline Operator",
    "Trust Guardian",
    "Visual CTA Designer",
    "Win-Back Researcher",
]


def _safe_int(value: object, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


@dataclass(frozen=True)
class ResearchSettings:
    population: int = 120
    top_n: int = 25
    strategy_depth: str = "standard"
    persona_richness: str = "rich"
    ideation_agents: int = 5
    validation_budget: int = DEPTH_OPTIONS["standard"]["validation_budget"]
    model_name: str = DEFAULT_OPENAI_MODEL
    seed: int = 7
    discount_step: int = DEPTH_OPTIONS["standard"]["discount_step"]
    discount_floor: int = 0
    discount_ceiling: int = 100
    grounding_limit: int = DEPTH_OPTIONS["standard"]["grounding_limit"]
    treatment_limit: int = DEPTH_OPTIONS["standard"]["treatment_limit"]
    friction_limit: int = DEPTH_OPTIONS["standard"]["friction_limit"]
    idea_proposals_per_agent: int = 2
    persona_shortlist_multiplier: int = 3
    segment_focus_limit: int = 5
    archetype_template_count: int = 0
    persona_blend_every: int = 0
    openai_reasoning_effort: str = "medium"
    api_batch_size: int = 4

    @property
    def uses_openai(self) -> bool:
        return self.model_name != HEURISTIC_MODEL

    @property
    def has_api_key(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    @property
    def openai_for_research(self) -> bool:
        """Whether OpenAI is configured and available for ideation."""
        return self.uses_openai and self.has_api_key

    @property
    def effective_validation_budget(self) -> int:
        return max(self.validation_budget, DEPTH_OPTIONS[self.strategy_depth]["validation_budget"])

    @property
    def workbench_limit(self) -> int:
        return min(self.effective_validation_budget, 1500)

    @property
    def generated_idea_limit(self) -> int:
        return max(self.ideation_agents * self.idea_proposals_per_agent, self.ideation_agents)

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def strategy_budget(self) -> int:
        return self.effective_validation_budget

    @property
    def depth(self) -> int:
        return {"quick": 1, "standard": 2, "deep": 3, "extreme": 4}[self.strategy_depth]

    def available_roles(self) -> list[str]:
        return AGENT_ROLES[: self.ideation_agents]

    def discount_levels(self) -> list[int]:
        step = self.discount_step
        levels = list(range(step, 101, step))
        if 100 not in levels:
            levels.append(100)
        return sorted(set(levels))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["uses_openai"] = self.uses_openai
        payload["effective_validation_budget"] = self.effective_validation_budget
        payload["validation_budget"] = self.effective_validation_budget
        payload["workbench_limit"] = self.workbench_limit
        payload["generated_idea_limit"] = self.generated_idea_limit
        payload["persona_shortlist_size"] = max(self.top_n * self.persona_shortlist_multiplier, 12)
        payload["discount_levels"] = self.discount_levels()
        payload["openai_for_research"] = self.openai_for_research
        payload["roles"] = self.available_roles()
        return payload

    def as_dict(self) -> dict[str, object]:
        return self.to_dict()


def build_settings(overrides: dict[str, object] | None = None) -> ResearchSettings:
    overrides = overrides or {}
    strategy_depth = str(overrides.get("strategy_depth") or "standard")
    if strategy_depth not in DEPTH_OPTIONS:
        strategy_depth = "standard"
    persona_richness = str(overrides.get("persona_richness") or "rich")
    if persona_richness not in PERSONA_RICHNESS_OPTIONS:
        persona_richness = "rich"
    reasoning_effort = str(overrides.get("openai_reasoning_effort") or "medium")
    if reasoning_effort not in REASONING_EFFORT_OPTIONS:
        reasoning_effort = "medium"
    validation_budget = _safe_int(
        overrides.get("validation_budget"),
        DEPTH_OPTIONS[strategy_depth]["validation_budget"],
        minimum=1,
    )
    depth_defaults = DEPTH_OPTIONS[strategy_depth]
    discount_floor = _safe_int(overrides.get("discount_floor"), 0, minimum=0, maximum=100)
    discount_ceiling = _safe_int(overrides.get("discount_ceiling"), 100, minimum=0, maximum=100)
    if discount_floor > discount_ceiling:
        discount_floor, discount_ceiling = discount_ceiling, discount_floor
    return ResearchSettings(
        population=_safe_int(overrides.get("population"), 120, minimum=1),
        top_n=_safe_int(overrides.get("top_n"), 25, minimum=1),
        strategy_depth=strategy_depth,
        persona_richness=persona_richness,
        ideation_agents=_safe_int(overrides.get("ideation_agents"), 5, minimum=1, maximum=len(AGENT_ROLES)),
        validation_budget=validation_budget,
        model_name=str(overrides["model_name"] if "model_name" in overrides else DEFAULT_OPENAI_MODEL),
        seed=_safe_int(overrides.get("seed"), 7, minimum=0),
        discount_step=_safe_int(overrides.get("discount_step"), depth_defaults["discount_step"], minimum=1),
        discount_floor=discount_floor,
        discount_ceiling=discount_ceiling,
        grounding_limit=_safe_int(overrides.get("grounding_limit"), depth_defaults["grounding_limit"], minimum=1, maximum=7),
        treatment_limit=_safe_int(overrides.get("treatment_limit"), depth_defaults["treatment_limit"], minimum=1, maximum=7),
        friction_limit=_safe_int(overrides.get("friction_limit"), depth_defaults["friction_limit"], minimum=1, maximum=6),
        idea_proposals_per_agent=_safe_int(overrides.get("idea_proposals_per_agent"), 2, minimum=1, maximum=8),
        persona_shortlist_multiplier=_safe_int(overrides.get("persona_shortlist_multiplier"), 3, minimum=1, maximum=12),
        segment_focus_limit=_safe_int(overrides.get("segment_focus_limit"), 5, minimum=1, maximum=12),
        archetype_template_count=_safe_int(overrides.get("archetype_template_count"), 0, minimum=0, maximum=12),
        persona_blend_every=_safe_int(overrides.get("persona_blend_every"), 0, minimum=0, maximum=20),
        openai_reasoning_effort=reasoning_effort,
        api_batch_size=_safe_int(overrides.get("api_batch_size"), 4, minimum=1),
    )


def build_settings_catalog() -> dict[str, object]:
    return {
        "defaults": build_settings().to_dict(),
        "depth_options": {
            key: {"label": key.title(), **value}
            for key, value in DEPTH_OPTIONS.items()
        },
        "persona_richness_options": {
            key: {"label": key.title(), **value}
            for key, value in PERSONA_RICHNESS_OPTIONS.items()
        },
        "model_options": {
            "gpt-5.4-mini": {"label": "GPT-5.4 Mini", "model_name": "gpt-5.4-mini"},
            "gpt-5.4": {"label": "GPT-5.4", "model_name": "gpt-5.4"},
        },
        "reasoning_effort_options": {
            key: {"label": value["label"]}
            for key, value in REASONING_EFFORT_OPTIONS.items()
        },
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "advanced_control_ranges": {
            "discount_step": {"min": 5, "max": 25, "step": 5},
            "discount_floor": {"min": 0, "max": 100, "step": 5},
            "discount_ceiling": {"min": 0, "max": 100, "step": 5},
            "grounding_limit": {"min": 1, "max": 7, "step": 1},
            "treatment_limit": {"min": 1, "max": 7, "step": 1},
            "friction_limit": {"min": 1, "max": 6, "step": 1},
            "idea_proposals_per_agent": {"min": 1, "max": 8, "step": 1},
            "persona_shortlist_multiplier": {"min": 1, "max": 12, "step": 1},
            "segment_focus_limit": {"min": 1, "max": 12, "step": 1},
            "archetype_template_count": {"min": 0, "max": 12, "step": 1},
            "persona_blend_every": {"min": 0, "max": 20, "step": 1},
            "api_batch_size": {"min": 1, "max": 12, "step": 1},
        },
    }

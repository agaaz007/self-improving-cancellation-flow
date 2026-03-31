from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from cta_autoresearch.models import Persona, UserProfile
from cta_autoresearch.personas import generate_personas


DEFAULT_MODEL = "heuristic-simulator"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"


def _normalize_positive_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


@dataclass(frozen=True)
class ResearchSettings:
    base_population: int = 120
    depth: int = 1
    strategy_budget: int = 1500
    persona_richness: int = 1
    ideation_rounds: int = 1
    model: str = DEFAULT_MODEL
    use_llm: bool = False
    api_key_env: str = DEFAULT_API_KEY_ENV
    seed: int = 7

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_population", _normalize_positive_int(self.base_population, 120))
        object.__setattr__(self, "depth", _normalize_positive_int(self.depth, 1))
        object.__setattr__(self, "strategy_budget", max(0, int(self.strategy_budget if self.strategy_budget is not None else 1500)))
        object.__setattr__(self, "persona_richness", _normalize_positive_int(self.persona_richness, 1))
        object.__setattr__(self, "ideation_rounds", _normalize_positive_int(self.ideation_rounds, 1))
        object.__setattr__(self, "model", (self.model or DEFAULT_MODEL).strip() or DEFAULT_MODEL)
        object.__setattr__(self, "api_key_env", (self.api_key_env or DEFAULT_API_KEY_ENV).strip() or DEFAULT_API_KEY_ENV)
        object.__setattr__(self, "seed", int(self.seed))

    @classmethod
    def from_namespace(cls, namespace: Any) -> "ResearchSettings":
        return cls(
            base_population=getattr(namespace, "population", getattr(namespace, "base_population", 120)),
            depth=getattr(namespace, "depth", 1),
            strategy_budget=getattr(namespace, "strategy_budget", 0),
            persona_richness=getattr(namespace, "persona_richness", 1),
            ideation_rounds=getattr(namespace, "ideation_rounds", 1),
            model=getattr(namespace, "model", DEFAULT_MODEL),
            use_llm=bool(getattr(namespace, "use_llm", False)),
            api_key_env=getattr(namespace, "api_key_env", DEFAULT_API_KEY_ENV),
            seed=getattr(namespace, "seed", 7),
        )

    @property
    def research_intensity(self) -> int:
        return self.depth * self.persona_richness * self.ideation_rounds

    @property
    def strategy_depth(self) -> str:
        return "exhaustive" if self.strategy_budget <= 0 and self.depth >= 4 else "budgeted"

    @property
    def effective_population_per_round(self) -> int:
        richness_boost = (self.persona_richness - 1) * max(10, self.base_population // 4)
        depth_boost = (self.depth - 1) * max(8, self.base_population // 6)
        return max(1, self.base_population + richness_boost + depth_boost)

    @property
    def effective_total_personas(self) -> int:
        return self.effective_population_per_round * self.ideation_rounds

    @property
    def effective_workbench_limit(self) -> int:
        if self.strategy_budget > 0:
            return self.strategy_budget
        return max(250, self.depth * 500 * self.ideation_rounds)

    @property
    def effective_validation_budget(self) -> int:
        return self.effective_workbench_limit

    @property
    def workbench_limit(self) -> int:
        return self.effective_workbench_limit

    @property
    def api_key_present(self) -> bool:
        return bool(os.getenv(self.api_key_env, "").strip())

    @property
    def active_model(self) -> str:
        if self.use_llm and self.api_key_present:
            return self.model
        return DEFAULT_MODEL

    @property
    def model_name(self) -> str:
        return self.active_model

    @property
    def top_n(self) -> int:
        return 25

    def as_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["strategy_depth"] = self.strategy_depth
        payload["validation_budget"] = self.effective_validation_budget
        payload["model_name"] = self.model_name
        payload["population"] = self.base_population
        return payload

    @property
    def llm_runtime_state(self) -> str:
        if not self.use_llm:
            return "disabled"
        if self.api_key_present:
            return "requested-but-not-wired"
        return "requested-key-missing"

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_population": self.base_population,
            "depth": self.depth,
            "strategy_budget": self.strategy_budget,
            "persona_richness": self.persona_richness,
            "ideation_rounds": self.ideation_rounds,
            "model": self.model,
            "active_model": self.active_model,
            "use_llm": self.use_llm,
            "api_key_env": self.api_key_env,
            "api_key_present": self.api_key_present,
            "llm_runtime_state": self.llm_runtime_state,
            "research_intensity": self.research_intensity,
            "effective_population_per_round": self.effective_population_per_round,
            "effective_total_personas": self.effective_total_personas,
            "effective_workbench_limit": self.effective_workbench_limit,
        }


def build_persona_cohort(seed_profiles: list[UserProfile], settings: ResearchSettings) -> list[Persona]:
    personas: list[Persona] = []
    richness_lookup = {1: "standard", 2: "rich"}
    richness = richness_lookup.get(settings.persona_richness, "extreme")
    for round_index in range(settings.ideation_rounds):
        round_seed = settings.seed + (round_index * 997)
        round_personas = generate_personas(
            seed_profiles=seed_profiles,
            population=settings.effective_population_per_round,
            seed=round_seed,
            richness=richness,
        )
        for index, persona in enumerate(round_personas, start=1):
            if round_index == 0:
                personas.append(persona)
                continue

            suffix = f" [R{round_index + 1}:{index}]"
            renamed_profile = replace(
                persona.profile,
                name=f"{persona.profile.name}{suffix}",
                cohort=f"{persona.profile.cohort}_r{round_index + 1}",
            )
            personas.append(replace(persona, name=renamed_profile.name, profile=renamed_profile))
    return personas


def backend_metadata(settings: ResearchSettings) -> dict[str, Any]:
    return {
        "strategy_engine": "multi-agent-heuristic-lab",
        "active_model": settings.active_model,
        "configured_model": settings.model,
        "use_llm": settings.use_llm,
        "llm_runtime_state": settings.llm_runtime_state,
        "api_key_env": settings.api_key_env,
        "api_key_present": settings.api_key_present,
        "research_intensity": settings.research_intensity,
    }


def annotate_dashboard_payload(payload: dict[str, Any], settings: ResearchSettings) -> dict[str, Any]:
    annotated = dict(payload)
    meta = dict(annotated.get("meta", {}))
    meta.update(
        {
            "base_population": settings.base_population,
            "depth": settings.depth,
            "strategy_budget": settings.strategy_budget,
            "persona_richness": settings.persona_richness,
            "ideation_rounds": settings.ideation_rounds,
            "configured_model": settings.model,
            "active_model": settings.active_model,
            "use_llm": settings.use_llm,
            "llm_runtime_state": settings.llm_runtime_state,
            "api_key_env": settings.api_key_env,
            "api_key_present": settings.api_key_present,
            "research_intensity": settings.research_intensity,
            "effective_population_per_round": settings.effective_population_per_round,
            "effective_total_personas": settings.effective_total_personas,
            "effective_workbench_limit": settings.effective_workbench_limit,
        }
    )
    annotated["meta"] = meta
    annotated["research_settings"] = settings.to_dict()
    annotated["backend"] = backend_metadata(settings)

    workbench_limit = settings.effective_workbench_limit
    if "all_candidates" in annotated:
        annotated["all_candidates"] = annotated["all_candidates"][:workbench_limit]
        annotated["meta"]["workbench_candidate_count"] = len(annotated["all_candidates"])

    return annotated

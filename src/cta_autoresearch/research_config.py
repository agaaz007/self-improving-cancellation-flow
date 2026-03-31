from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace


DEPTH_PRESETS = {
    "quick": {
        "label": "Quick",
        "description": "Fast pass across the strongest structural ideas.",
        "ideation_agents": 3,
        "ideation_rounds": 1,
        "validation_budget": 72,
        "anchor_count": 10,
        "workbench_limit": 120,
    },
    "standard": {
        "label": "Standard",
        "description": "Balanced strategy generation and validation depth.",
        "ideation_agents": 5,
        "ideation_rounds": 2,
        "validation_budget": 180,
        "anchor_count": 18,
        "workbench_limit": 180,
    },
    "deep": {
        "label": "Deep",
        "description": "Broader ideation and more strategy validation per run.",
        "ideation_agents": 7,
        "ideation_rounds": 3,
        "validation_budget": 420,
        "anchor_count": 28,
        "workbench_limit": 260,
    },
    "extreme": {
        "label": "Extreme",
        "description": "Maximum persona detail and the largest local search budget.",
        "ideation_agents": 9,
        "ideation_rounds": 5,
        "validation_budget": 960,
        "anchor_count": 40,
        "workbench_limit": 360,
    },
}

PERSONA_RICHNESS = {
    "standard": {
        "label": "Standard",
        "description": "Compact persona summaries with core risk and value signals.",
        "representative_count": 6,
    },
    "rich": {
        "label": "Rich",
        "description": "Adds objections, hooks, and richer behavioral traces.",
        "representative_count": 10,
    },
    "extreme": {
        "label": "Extreme",
        "description": "Maximizes persona detail and broadens the ideation slice.",
        "representative_count": 14,
    },
}

MODEL_PROVIDERS = {
    "heuristic": {
        "label": "Heuristic research engine",
        "default_model": "heuristic-simulator-v2",
        "api_key_env": "",
    },
    "openai": {
        "label": "OpenAI hybrid ideation",
        "default_model": "gpt-5-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
}


@dataclass(frozen=True)
class ResearchConfig:
    population: int = 120
    seed: int = 7
    top_n: int = 10
    depth_mode: str = "deep"
    persona_richness: str = "rich"
    ideation_agents: int = DEPTH_PRESETS["deep"]["ideation_agents"]
    ideation_rounds: int = DEPTH_PRESETS["deep"]["ideation_rounds"]
    validation_budget: int = DEPTH_PRESETS["deep"]["validation_budget"]
    anchor_count: int = DEPTH_PRESETS["deep"]["anchor_count"]
    workbench_limit: int = DEPTH_PRESETS["deep"]["workbench_limit"]
    model_provider: str = "heuristic"
    model_name: str = MODEL_PROVIDERS["heuristic"]["default_model"]

    @classmethod
    def from_overrides(cls, **overrides: object) -> "ResearchConfig":
        depth_mode = str(overrides.get("depth_mode") or cls.depth_mode)
        persona_richness = str(overrides.get("persona_richness") or cls.persona_richness)
        depth_defaults = DEPTH_PRESETS.get(depth_mode, DEPTH_PRESETS[cls.depth_mode])
        provider = str(overrides.get("model_provider") or cls.model_provider)
        provider_meta = MODEL_PROVIDERS.get(provider, MODEL_PROVIDERS[cls.model_provider])
        config = cls(
            population=int(overrides.get("population", cls.population)),
            seed=int(overrides.get("seed", cls.seed)),
            top_n=int(overrides.get("top_n", cls.top_n)),
            depth_mode=depth_mode if depth_mode in DEPTH_PRESETS else cls.depth_mode,
            persona_richness=persona_richness if persona_richness in PERSONA_RICHNESS else cls.persona_richness,
            ideation_agents=int(overrides.get("ideation_agents", depth_defaults["ideation_agents"])),
            ideation_rounds=int(overrides.get("ideation_rounds", depth_defaults["ideation_rounds"])),
            validation_budget=int(overrides.get("validation_budget", depth_defaults["validation_budget"])),
            anchor_count=int(overrides.get("anchor_count", depth_defaults["anchor_count"])),
            workbench_limit=int(overrides.get("workbench_limit", depth_defaults["workbench_limit"])),
            model_provider=provider if provider in MODEL_PROVIDERS else cls.model_provider,
            model_name=str(overrides.get("model_name") or provider_meta["default_model"]),
        )
        return config.clamped()

    def clamped(self) -> "ResearchConfig":
        return replace(
            self,
            population=max(12, min(self.population, 500)),
            top_n=max(3, min(self.top_n, 50)),
            ideation_agents=max(1, min(self.ideation_agents, 9)),
            ideation_rounds=max(1, min(self.ideation_rounds, 12)),
            validation_budget=max(24, min(self.validation_budget, 3000)),
            anchor_count=max(6, min(self.anchor_count, 120)),
            workbench_limit=max(50, min(self.workbench_limit, 600)),
        )

    def provider_status(self) -> dict[str, object]:
        provider_meta = MODEL_PROVIDERS[self.model_provider]
        api_key_env = provider_meta["api_key_env"]
        api_key_present = bool(api_key_env and os.environ.get(api_key_env))
        available = self.model_provider == "heuristic" or api_key_present
        requested_model = self.model_name or provider_meta["default_model"]
        active_provider = self.model_provider if available else "heuristic"
        active_model = requested_model if available else MODEL_PROVIDERS["heuristic"]["default_model"]
        warning = ""
        if self.model_provider != "heuristic" and not available:
            warning = f"{api_key_env} is not set, so the run falls back to the heuristic engine."
        return {
            "requested_provider": self.model_provider,
            "requested_model": requested_model,
            "active_provider": active_provider,
            "active_model": active_model,
            "available": available,
            "api_key_env": api_key_env,
            "api_key_present": api_key_present,
            "warning": warning,
        }

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["provider_status"] = self.provider_status()
        return payload


def control_payload(config: ResearchConfig) -> dict[str, object]:
    return {
        "selected": config.to_dict(),
        "depth_modes": [
            {"value": key, **value}
            for key, value in DEPTH_PRESETS.items()
        ],
        "persona_richness": [
            {"value": key, **value}
            for key, value in PERSONA_RICHNESS.items()
        ],
        "model_providers": [
            {
                "value": key,
                "label": value["label"],
                "default_model": value["default_model"],
                "requires_api_key": bool(value["api_key_env"]),
                "api_key_env": value["api_key_env"],
            }
            for key, value in MODEL_PROVIDERS.items()
        ],
    }

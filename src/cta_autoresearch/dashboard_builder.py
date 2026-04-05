from __future__ import annotations

import json
from pathlib import Path

from cta_autoresearch.config import ResearchSettings as RuntimeSettings, annotate_dashboard_payload, build_persona_cohort
from cta_autoresearch.optimizer import build_dashboard_payload
from cta_autoresearch.research_settings import build_settings, build_settings_catalog
from cta_autoresearch.sample_data import load_seed_profiles


def _runtime_settings(ui_settings) -> RuntimeSettings:
    depth_map = {"quick": 1, "standard": 2, "deep": 3, "extreme": 4}
    richness_map = {"standard": 1, "rich": 2, "extreme": 3}
    return RuntimeSettings(
        base_population=ui_settings.population + (20 if ui_settings.persona_richness == "rich" else 40 if ui_settings.persona_richness == "extreme" else 0),
        depth=depth_map[ui_settings.strategy_depth],
        strategy_budget=ui_settings.validation_budget,
        persona_richness=richness_map[ui_settings.persona_richness],
        ideation_rounds=max(1, depth_map[ui_settings.strategy_depth] - 1),
        model=ui_settings.model_name,
        use_llm=ui_settings.uses_openai,
        seed=ui_settings.seed,
    )


def build_dashboard_dataset(**overrides: object) -> dict:
    if "settings" in overrides and overrides["settings"] is not None:
        runtime = overrides["settings"]
        ui_settings = build_settings(
            {
                "population": runtime.base_population,
                "strategy_depth": "extreme" if runtime.depth >= 4 else "deep" if runtime.depth >= 3 else "standard" if runtime.depth >= 2 else "quick",
                "persona_richness": "extreme" if runtime.persona_richness >= 3 else "rich" if runtime.persona_richness >= 2 else "standard",
                "ideation_agents": min(5 + max(runtime.depth - 2, 0), 9),
                "validation_budget": runtime.strategy_budget or runtime.effective_workbench_limit,
                "model_name": runtime.model,
                "seed": runtime.seed,
            }
        )
    else:
        ui_settings = build_settings(overrides)
        runtime = _runtime_settings(ui_settings)
    seed_profiles = load_seed_profiles()
    personas = build_persona_cohort(seed_profiles=seed_profiles, settings=runtime)
    payload = annotate_dashboard_payload(build_dashboard_payload(personas, settings=runtime), runtime)
    catalog = build_settings_catalog()
    payload["controls"] = {
        **catalog,
        "selected": {
            **ui_settings.to_dict(),
            "depth_mode": ui_settings.strategy_depth,
            "model_provider": "openai",
            "top_n": 25,
        },
        "depth_modes": [{"value": key, "label": value["label"]} for key, value in catalog["depth_options"].items()],
        "persona_richness": [{"value": key, "label": value["label"]} for key, value in catalog["persona_richness_options"].items()],
        "model_providers": [
            {"value": "openai", "label": "OpenAI hybrid ideation", "default_model": ui_settings.model_name, "requires_api_key": True},
        ],
    }
    payload["meta"]["research_settings"] = ui_settings.to_dict()
    return payload


def write_dashboard_data(output_dir: str | Path, **overrides: object) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = build_dashboard_dataset(**overrides)
    output = destination / "data.json"
    output.write_text(json.dumps(payload, indent=2))
    return output

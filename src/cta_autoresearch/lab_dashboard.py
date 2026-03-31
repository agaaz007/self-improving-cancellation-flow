from __future__ import annotations

import json
from pathlib import Path

from cta_autoresearch.lab_optimizer import build_dashboard_payload
from cta_autoresearch.personas import generate_personas
from cta_autoresearch.research_settings import build_settings, build_settings_catalog
from cta_autoresearch.sample_data import load_seed_profiles


def build_dashboard_dataset(*, progress_callback=None, **overrides: object) -> dict:
    settings = build_settings(overrides)
    if progress_callback:
        progress_callback(0.04, "settings", "Loaded research settings and control catalog.")
    seed_profiles = load_seed_profiles()
    if progress_callback:
        progress_callback(0.12, "personas", "Loaded seed profiles for persona synthesis.")
    personas = generate_personas(
        seed_profiles=seed_profiles,
        population=settings.population,
        seed=settings.seed,
        richness=settings.persona_richness,
        archetype_template_count=settings.archetype_template_count or None,
        blend_every=settings.persona_blend_every or None,
    )
    if progress_callback:
        progress_callback(0.28, "personas", f"Built {len(personas)} personas for this run.")
    payload = build_dashboard_payload(personas, settings=settings, progress_callback=progress_callback)
    payload["controls"] = build_settings_catalog()
    if progress_callback:
        progress_callback(1.0, "complete", "Prepared dashboard payload and control metadata.")
    return payload


def write_dashboard_data(output_dir: str | Path, *, progress_callback=None, **overrides: object) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    payload = build_dashboard_dataset(progress_callback=progress_callback, **overrides)
    output = destination / "data.json"
    output.write_text(json.dumps(payload, indent=2))
    return output

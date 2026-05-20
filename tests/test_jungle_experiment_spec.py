from __future__ import annotations

from cta_autoresearch.gbrain_memory import seed_memories
from cta_autoresearch.jungle_experiment_spec import SPEC_VERSION, build_jungle_experiment_spec


def test_build_jungle_experiment_spec_is_ui_generator_ready():
    spec = build_jungle_experiment_spec({
        "client_id": "jungle_ai",
        "research_run_ids": ["run_test_001"],
        "baseline_metrics": {"paywall_accept_rate": 0.18},
    }, memory_items=seed_memories("jungle_ai"), created_at="2026-05-20T00:00:00Z")

    assert spec["spec_version"] == SPEC_VERSION
    assert spec["client_id"] == "jungle_ai"
    assert spec["status"] == "ready_for_ui_generation"
    assert spec["validation"]["is_ready"] is True
    assert spec["source"]["gbrain_memory_ids"]
    assert spec["experiment"]["variant"] == "progress_recap_value_proof"
    assert spec["strategy"]["primary_strategy"] == "progress_recap_plus_value_proof"
    assert "react_component" in spec["generation_instructions"]["return_format"]["accepted"]
    assert spec["ui_generator_payload"]["task"] == "generate_paywall_variants"
    assert "progress_recap" in spec["ui_generator_payload"]["must_use_sections"]


def test_spec_contains_real_world_paywall_constraints():
    spec = build_jungle_experiment_spec(memory_items=seed_memories("jungle_ai"), created_at="2026-05-20T00:00:00Z")
    sections = {section["id"]: section for section in spec["ui_requirements"]["sections"]}

    assert {"progress_recap", "value_proof", "plan_offer", "cta"} <= set(sections)
    assert sections["progress_recap"]["data_bindings"]
    assert sections["plan_offer"]["copy_slots"]["price_note"] == "Use the real price from the plan catalog."
    assert "fake countdown" in spec["ui_requirements"]["copy_constraints"]["must_not_include"]
    assert "claims about progress that are not backed by data" in spec["ui_requirements"]["copy_constraints"]["must_not_include"]
    assert spec["data_requirements"]["hard_rule"].startswith("If a data field is missing")
    assert "factuality_gate" in {judge["id"] for judge in spec["judge_plan"]["post_generation_judges"]}


def test_spec_uses_supplied_context_without_inventing_metrics():
    spec = build_jungle_experiment_spec({
        "surface": "checkout_paywall",
        "target_segment": "activated_exam_prep_users",
        "control_id": "jungle_control_v3",
        "generate_variants": 2,
        "product_context": {"plan_name": "Jungle Premium", "market": "India"},
        "constraints": {"rollout": "Preview only until tracking is verified."},
    }, memory_items=seed_memories("jungle_ai"), created_at="2026-05-20T00:00:00Z")

    assert spec["context"]["surface"] == "checkout_paywall"
    assert spec["context"]["target_segment"] == "activated_exam_prep_users"
    assert spec["context"]["control_id"] == "jungle_control_v3"
    assert spec["context"]["product_context"]["market"] == "India"
    assert spec["generation_instructions"]["generate_variants"] == 2
    assert spec["implementation_notes"]["rollout"] == "Preview only until tracking is verified."


def test_spec_tolerates_invalid_variant_count():
    spec = build_jungle_experiment_spec(
        {"generate_variants": "not-a-number"},
        memory_items=seed_memories("jungle_ai"),
        created_at="2026-05-20T00:00:00Z",
    )

    assert spec["generation_instructions"]["generate_variants"] == 3

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from cta_autoresearch.gbrain_memory import normalize_memory_list, seed_memories


SPEC_VERSION = "jungle_experiment_ui_spec_v1"
DEFAULT_CLIENT_ID = "jungle_ai"
DEFAULT_SURFACE = "paywall"
DEFAULT_TARGET_SEGMENT = "high_intent_learners"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _as_int(value: Any, default: int, *, lower: int = 1, upper: int = 10) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(lower, min(number, upper))


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "jungle_experiment"


def _memory_refs(items: list[dict[str, Any]], category: str, limit: int = 3) -> list[dict[str, Any]]:
    refs = []
    for item in items:
        if item.get("category") != category or item.get("status") == "archived":
            continue
        refs.append({
            "memory_id": item.get("id"),
            "title": item.get("title"),
            "lesson": item.get("lesson"),
            "recommendation": item.get("recommendation"),
            "module_id": item.get("module_id"),
            "confidence": item.get("confidence"),
            "impact": item.get("impact"),
        })
    refs.sort(key=lambda ref: (float(ref.get("impact") or 0), float(ref.get("confidence") or 0)), reverse=True)
    return refs[:limit]


def _lessons(refs: list[dict[str, Any]], fallback: list[str]) -> list[str]:
    lessons = [str(ref.get("lesson") or ref.get("recommendation") or "").strip() for ref in refs]
    lessons = [lesson for lesson in lessons if lesson]
    return lessons or fallback


def _memory_ids(*groups: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for group in groups:
        for item in group:
            memory_id = str(item.get("memory_id") or "").strip()
            if memory_id and memory_id not in ids:
                ids.append(memory_id)
    return ids


def _validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    required_top_level = [
        "spec_version",
        "client_id",
        "experiment_id",
        "experiment",
        "strategy",
        "ui_requirements",
        "generation_instructions",
    ]
    missing = [key for key in required_top_level if key not in spec]
    section_ids = {
        str(section.get("id"))
        for section in _as_dict(spec.get("ui_requirements")).get("sections", [])
        if isinstance(section, dict)
    }
    required_sections = {"progress_recap", "value_proof", "plan_offer", "cta"}
    missing_sections = sorted(required_sections - section_ids)
    warnings = []
    if not _as_dict(spec.get("strategy")).get("blocked_strategies"):
        warnings.append("No blocked strategies were supplied to the UI generator.")
    if not _as_dict(spec.get("data_requirements")).get("required"):
        warnings.append("No required data bindings were supplied; generated copy may become generic.")
    return {
        "is_ready": not missing and not missing_sections,
        "missing_top_level_fields": missing,
        "missing_required_sections": missing_sections,
        "warnings": warnings,
    }


def build_jungle_experiment_spec(
    payload: dict[str, Any] | None = None,
    *,
    memory_items: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the next Jungle experiment spec for UI/UX generation."""
    body = payload or {}
    client_id = str(body.get("client_id") or DEFAULT_CLIENT_ID)
    surface = str(body.get("surface") or DEFAULT_SURFACE)
    target_segment = str(body.get("target_segment") or DEFAULT_TARGET_SEGMENT)
    control_id = str(body.get("control_id") or "current_jungle_paywall")
    research_run_ids = _as_list(body.get("research_run_ids"))
    baseline_metrics = _as_dict(body.get("baseline_metrics"))
    product_context = _as_dict(body.get("product_context"))
    constraints = _as_dict(body.get("constraints"))
    generate_variants = _as_int(body.get("generate_variants"), 3, lower=1, upper=6)

    memories = normalize_memory_list(memory_items or seed_memories(client_id), client_id=client_id)
    winning = _memory_refs(memories, "winning_lesson")
    promoted = _memory_refs(memories, "promoted_strategy")
    failures = _memory_refs(memories, "repeated_failure")
    contradicted = _memory_refs(memories, "contradicted_memory", limit=2)
    stale = _memory_refs(memories, "stale_assumption", limit=2)
    blocked = _memory_refs(memories, "blocked_strategy")

    experiment_name = "Progress recap + value proof paywall"
    experiment_id = str(body.get("experiment_id") or f"jungle_{_slug(experiment_name)}_v1")
    generated_at = created_at or str(body.get("created_at") or _now_iso())

    promoted_lessons = _lessons(
        promoted + winning,
        [
            "Show concrete learner progress before asking for payment.",
            "Explain the next useful outcome before any urgency or offer framing.",
        ],
    )
    failure_patterns = _lessons(
        failures + contradicted + stale,
        [
            "Discount-first framing can hurt trust and revenue quality.",
            "Generic social proof is weaker than user-specific value proof.",
            "Urgency should only appear when the product has a factual deadline.",
        ],
    )
    blocked_strategies = _lessons(
        blocked,
        [
            "Do not invent scarcity, countdowns, limited seats, or unsupported deadlines.",
            "Do not lead with a large discount before showing value.",
        ],
    )

    spec: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "client_id": client_id,
        "experiment_id": experiment_id,
        "created_at": generated_at,
        "status": "ready_for_ui_generation",
        "source": {
            "recommendation": "progress_recap_plus_value_proof",
            "research_run_ids": research_run_ids,
            "gbrain_memory_ids": _memory_ids(winning, promoted, failures, contradicted, stale, blocked),
            "selected_by": "deterministic_jungle_spec_builder",
            "memory_categories_used": [
                "winning_lesson",
                "repeated_failure",
                "contradicted_memory",
                "stale_assumption",
                "promoted_strategy",
                "blocked_strategy",
            ],
        },
        "context": {
            "surface": surface,
            "target_segment": target_segment,
            "control_id": control_id,
            "product_context": product_context,
            "baseline_metrics": baseline_metrics,
            "assumptions": [
                "The user has at least one meaningful progress or activation signal.",
                "The paywall can access enough context to avoid generic progress copy.",
                "The variant is judged before shipment and instrumented before rollout.",
            ],
        },
        "experiment": {
            "name": experiment_name,
            "hypothesis": (
                "Showing a personalized progress recap followed by concrete value proof will improve "
                "Jungle paywall acceptance for high-intent learners without using discount-first or fake urgency framing."
            ),
            "surface": surface,
            "target_segment": target_segment,
            "control": control_id,
            "variant": "progress_recap_value_proof",
            "primary_metric": "paywall_accept_rate",
            "secondary_metrics": ["expected_revenue_per_exposure", "trial_start_rate", "plan_select_rate"],
            "guardrail_metrics": ["complaint_rate", "support_escalation_rate", "refund_intent_rate"],
            "minimum_sample_guidance": "Run until each arm has enough exposures for a directional read; do not call a winner from early synthetic-only scores.",
        },
        "modules": [
            {
                "id": "gbrain_memory_recall",
                "role": "strategy_input",
                "uses": ["promoted strategies", "blocked strategies", "repeated failure lessons"],
                "output": "memory-grounded design constraints",
            },
            {
                "id": "progress_recap_module",
                "role": "hero_content",
                "uses": ["completed lessons", "streak", "saved vocabulary", "recent learning goal"],
                "output": "specific learner progress proof",
            },
            {
                "id": "value_proof_module",
                "role": "conversion_support",
                "uses": ["next learning outcome", "plan benefits", "feature access"],
                "output": "why paid plan helps the next step",
            },
            {
                "id": "trust_guardrail_module",
                "role": "constraint_checker",
                "uses": ["blocked strategies", "factuality checks", "copy constraints"],
                "output": "safe-to-generate UI brief",
            },
            {
                "id": "judge_panel_module",
                "role": "post_generation_evaluator",
                "uses": ["conversion", "trust", "revenue", "factuality", "novelty"],
                "output": "accept/reject verdict with clustered reasons",
            },
        ],
        "strategy": {
            "primary_strategy": "progress_recap_plus_value_proof",
            "rationale": (
                "This is safer than discounting, sharper than generic social proof, and directly uses the strongest "
                "current memory: show concrete value before asking for payment."
            ),
            "promoted_lessons": promoted_lessons,
            "winning_memory_refs": winning,
            "promoted_memory_refs": promoted,
            "blocked_strategies": blocked_strategies,
            "blocked_memory_refs": blocked,
            "failure_patterns_to_avoid": failure_patterns,
            "failure_memory_refs": failures + contradicted + stale,
        },
        "data_requirements": {
            "required": [
                {
                    "field": "user_progress_summary",
                    "type": "string",
                    "purpose": "One factual sentence about what the learner has completed or unlocked.",
                    "fallback": "Reference the learner's current goal instead of inventing progress.",
                },
                {
                    "field": "next_learning_outcome",
                    "type": "string",
                    "purpose": "The next concrete benefit unlocked by the paid plan.",
                    "fallback": "Use a generic but factual plan benefit from the product catalog.",
                },
                {
                    "field": "recommended_plan",
                    "type": "object",
                    "purpose": "Plan name, price, billing period, and included features.",
                    "fallback": "Show the default Jungle premium plan.",
                },
            ],
            "optional": [
                "streak_count",
                "completed_lessons_count",
                "saved_words_count",
                "exam_or_goal_name",
                "trial_eligibility",
            ],
            "hard_rule": "If a data field is missing, the UI generator must use the fallback and must not fabricate user history.",
        },
        "ui_requirements": {
            "output_type": "mobile_first_subscription_paywall",
            "tone": "specific, calm, encouraging, not pushy",
            "layout_intent": "progress proof first, value explanation second, plan clarity third, single CTA last",
            "sections": [
                {
                    "id": "progress_recap",
                    "role": "hero",
                    "priority": 1,
                    "required": True,
                    "content_goal": "Show the learner what they have already done and why continuing now is rational.",
                    "data_bindings": ["user_progress_summary", "streak_count", "completed_lessons_count"],
                    "copy_slots": {
                        "headline": "You have already built real momentum",
                        "supporting_line": "Use actual progress data here; never invent completions.",
                    },
                },
                {
                    "id": "value_proof",
                    "role": "supporting_evidence",
                    "priority": 2,
                    "required": True,
                    "content_goal": "Connect the paid plan to the user's next learning outcome.",
                    "data_bindings": ["next_learning_outcome", "exam_or_goal_name"],
                    "copy_slots": {
                        "headline": "Keep the next step unlocked",
                        "supporting_line": "Explain the specific benefit, not a vague premium promise.",
                    },
                },
                {
                    "id": "plan_offer",
                    "role": "conversion",
                    "priority": 3,
                    "required": True,
                    "content_goal": "Show plan, price, billing period, and included benefits without hiding cost.",
                    "data_bindings": ["recommended_plan"],
                    "copy_slots": {
                        "plan_label": "Jungle Premium",
                        "price_note": "Use the real price from the plan catalog.",
                    },
                },
                {
                    "id": "cta",
                    "role": "action",
                    "priority": 4,
                    "required": True,
                    "content_goal": "Use one primary CTA and one low-pressure secondary action.",
                    "data_bindings": ["trial_eligibility"],
                    "copy_slots": {
                        "primary": "Continue with Premium",
                        "secondary": "Not now",
                    },
                },
            ],
            "copy_constraints": {
                "must_include": [
                    "one factual progress reference or fallback goal reference",
                    "one clear next benefit",
                    "one visible price or trial condition",
                    "one primary CTA",
                ],
                "must_not_include": [
                    "fake countdown",
                    "unsupported scarcity",
                    "large discount as the first message",
                    "claims about progress that are not backed by data",
                    "more than one primary CTA",
                ],
            },
            "visual_constraints": {
                "style": "clean mobile subscription paywall",
                "density": "medium",
                "hierarchy": ["progress recap", "value proof", "plan offer", "CTA"],
                "avoid": [
                    "dark manipulative urgency UI",
                    "cluttered nested cards",
                    "multiple competing plans on the first screen",
                    "testimonial-heavy generic social proof",
                ],
            },
            "accessibility_constraints": [
                "price and billing period must be readable without hover",
                "CTA text must be descriptive",
                "secondary action must be visible",
                "do not rely on color alone for plan recommendation",
            ],
        },
        "generation_instructions": {
            "target_tools": ["claude_design", "v0", "internal_variant_generator"],
            "generate_variants": generate_variants,
            "variant_dimensions": ["headline framing", "progress visualization", "benefit hierarchy", "CTA wording"],
            "return_format": {
                "preferred": "react_component",
                "accepted": ["html_css", "react_component", "figma_like_json"],
                "must_return": [
                    "renderable_ui",
                    "variant_id",
                    "copy_blocks",
                    "data_bindings_used",
                    "constraints_satisfied",
                    "constraints_violated",
                ],
            },
            "design_prompt": (
                "Generate a mobile-first Jungle paywall variant. Lead with factual learner progress, then show the next "
                "paid-plan value, then the plan offer, then one primary CTA. Do not use fake scarcity, discount-first copy, "
                "or unsupported user claims."
            ),
        },
        "ui_generator_payload": {
            "task": "generate_paywall_variants",
            "brief": (
                "Create mobile-first Jungle paywall variants for a progress recap plus value proof experiment. "
                "The design must feel useful and factual, not urgent or discount-led."
            ),
            "input_spec_version": SPEC_VERSION,
            "must_use_sections": ["progress_recap", "value_proof", "plan_offer", "cta"],
            "must_preserve_constraints": [
                "use factual progress data or fallback copy",
                "show the real plan price or trial condition",
                "use one primary CTA",
                "do not use fake scarcity or countdown urgency",
                "do not lead with discount copy",
            ],
            "deliverables": [
                "3 variant concepts unless generate_variants says otherwise",
                "renderable UI for each variant",
                "copy blocks for each section",
                "data bindings used by each section",
                "list of constraints satisfied and violated",
            ],
            "acceptance_bar": "A generated variant is usable only if it can pass factuality, trust, conversion, revenue, and novelty judges.",
        },
        "judge_plan": {
            "pre_generation_checks": [
                "All required UI sections exist.",
                "Blocked strategies are present as hard constraints.",
                "Required data bindings include fallbacks.",
            ],
            "post_generation_judges": [
                {"id": "conversion_judge", "threshold": 0.70, "focus": "likelihood of paid acceptance"},
                {"id": "trust_judge", "threshold": 0.78, "focus": "non-manipulative and factual"},
                {"id": "revenue_judge", "threshold": 0.68, "focus": "does not over-discount or hide price"},
                {"id": "novelty_judge", "threshold": 0.62, "focus": "meaningfully different from generic social proof"},
                {"id": "factuality_gate", "threshold": 1.0, "focus": "no invented progress, scarcity, or price claims"},
            ],
            "rejection_reasons_to_cluster": [
                "missing factual progress data",
                "generic value proof",
                "discount-first framing",
                "fake urgency",
                "unclear price",
                "CTA clutter",
            ],
        },
        "event_tracking": {
            "exposure_event": "tranzmit_experiment_exposed",
            "primary_outcome_event": "jungle_paywall_accepted",
            "guardrail_events": ["jungle_paywall_dismissed", "jungle_support_escalated", "jungle_refund_intent"],
            "required_properties": [
                "experiment_id",
                "variant_id",
                "client_id",
                "surface",
                "target_segment",
                "gbrain_memory_ids",
            ],
        },
        "implementation_notes": {
            "rollout": constraints.get("rollout", "Start as a preview or low-traffic experiment until judge and tracking checks pass."),
            "statsig": constraints.get("statsig", "Use one experiment key with control and progress_recap_value_proof variant."),
            "storage": "Store this JSON spec, generated UI variants, judge verdicts, and final outcome under the same research run.",
        },
    }
    spec["validation"] = _validate_spec(spec)
    return spec

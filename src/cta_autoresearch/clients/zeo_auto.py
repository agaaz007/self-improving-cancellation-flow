"""Zeo Auto (route planning SaaS) client configuration.

Zeo Auto is a route planning and delivery optimization app for logistics
teams and drivers. This module provides all domain-specific configuration
needed to run the overnight autoresearch pipeline.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from cta_autoresearch.cancel_policy import ActionDefinition
from cta_autoresearch.features import clamp
from cta_autoresearch.models import FeatureVector, Persona, StrategyCandidate, UserProfile
from cta_autoresearch.user_model import Archetype

# ---------------------------------------------------------------------------
# Cancel reasons
# ---------------------------------------------------------------------------

PRIMARY_REASONS = (
    "price",
    "low_usage",
    "route_quality",
    "no_need",
    "job_change",
    "webhook",
    "user_initiated",
    "other",
)

# ---------------------------------------------------------------------------
# Plan tiers (from payment_interval in Zeo CSV)
# ---------------------------------------------------------------------------

PLAN_TIERS = ["weekly", "monthly", "quarterly", "annual"]

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

ACTIONS: dict[str, ActionDefinition] = {
    "control_graceful_exit": ActionDefinition(
        id="control_graceful_exit",
        label="Control - Graceful Exit",
        offer_kind="none",
        message="We understand. Let us make the transition easy.",
    ),
    "pause_plan": ActionDefinition(
        id="pause_plan",
        label="Pause Plan",
        offer_kind="pause",
        message="Pause your plan and resume when your routes pick up again.",
    ),
    "downgrade_basic": ActionDefinition(
        id="downgrade_basic",
        label="Downgrade to Basic",
        offer_kind="downgrade",
        message="Switch to a lighter plan and keep your route history.",
    ),
    "discount_20": ActionDefinition(
        id="discount_20",
        label="20% Discount",
        offer_kind="discount",
        is_discount=True,
        message="Stay on track with a limited 20% discount on your next cycle.",
    ),
    "discount_40": ActionDefinition(
        id="discount_40",
        label="40% Discount",
        offer_kind="discount",
        is_discount=True,
        aggressive_urgency=True,
        message="High-savings offer available now if you continue your plan.",
    ),
    "route_optimization_demo": ActionDefinition(
        id="route_optimization_demo",
        label="Route Optimization Demo",
        offer_kind="support",
        message="Let us show you how to get more value from route optimization.",
    ),
    "fleet_rightsize": ActionDefinition(
        id="fleet_rightsize",
        label="Fleet Right-sizing",
        offer_kind="credit",
        message="We can adjust your plan to match your actual fleet size.",
    ),
}

CONTROL_ACTION_ID = "control_graceful_exit"

REASON_DENYLIST: dict[str, set[str]] = {
    "webhook": {"discount_40"},
    "job_change": {"route_optimization_demo", "discount_40"},
    "user_initiated": {"discount_40"},
}

LLM_DOMAIN_CONTEXT = "a route planning and delivery optimization app for logistics teams and drivers"

# ---------------------------------------------------------------------------
# Strategy dimension catalogs — broad, LLM-friendly
# ---------------------------------------------------------------------------

_MESSAGE_ANGLES = {
    "efficiency_gain": {
        "label": "Efficiency Gain",
        "specificity": 0.72,
        "description": "Highlight time saved, fuel savings, and route optimization results.",
    },
    "cost_control": {
        "label": "Cost Control",
        "specificity": 0.60,
        "description": "Frame continuity in terms of cost-per-delivery and fleet expense reduction.",
    },
    "team_coordination": {
        "label": "Team Coordination",
        "specificity": 0.68,
        "description": "Emphasize workflow continuity for the team and drivers who depend on routes.",
    },
    "growth_potential": {
        "label": "Growth Potential",
        "specificity": 0.55,
        "description": "Show how the platform supports scaling to more routes and drivers.",
    },
    "reliability": {
        "label": "Reliability",
        "specificity": 0.50,
        "description": "Focus on consistent, dependable route quality and uptime.",
    },
    "workflow_continuity": {
        "label": "Workflow Continuity",
        "specificity": 0.45,
        "description": "Frame cancellation as a disruption to established delivery workflows.",
    },
    "competitive_edge": {
        "label": "Competitive Edge",
        "specificity": 0.42,
        "description": "Position route optimization as a competitive advantage over manual planning.",
    },
    "seasonal_timing": {
        "label": "Seasonal Timing",
        "specificity": 0.38,
        "description": "Connect the decision to seasonal delivery volume patterns.",
    },
    "empathetic_exit": {
        "label": "Empathetic Exit",
        "specificity": 0.25,
        "description": "Meet the user with empathy and reduce resistance before asking for a choice.",
    },
    "flexibility_relief": {
        "label": "Flexibility Relief",
        "specificity": 0.28,
        "description": "Show softer paths than all-or-nothing cancellation.",
    },
}

_PROOF_STYLES = {
    "none": {
        "label": "No explicit proof",
        "description": "Rely on the message angle without added evidence.",
    },
    "similar_user_story": {
        "label": "Similar user story",
        "description": "Show a logistics team with a similar fleet size succeeding.",
    },
    "metric_snapshot": {
        "label": "Metric snapshot",
        "description": "Use concrete route optimization metrics and savings data.",
    },
    "roi_calculation": {
        "label": "ROI calculation",
        "description": "Show cost-per-route or time-per-delivery savings.",
    },
    "team_impact": {
        "label": "Team impact",
        "description": "Reference the team members and drivers who depend on this workflow.",
    },
    "before_after": {
        "label": "Before / after comparison",
        "description": "Compare manual planning vs. optimized route performance.",
    },
}

_BASE_OFFERS = {
    "none": {"label": "No offer", "kind": "none", "generosity": 0.00},
    "discount_10": {"label": "10% discount", "kind": "discount", "generosity": 0.10},
    "discount_20": {"label": "20% discount", "kind": "discount", "generosity": 0.20},
    "discount_30": {"label": "30% discount", "kind": "discount", "generosity": 0.30},
    "discount_40": {"label": "40% discount", "kind": "discount", "generosity": 0.40},
    "discount_50": {"label": "50% discount", "kind": "discount", "generosity": 0.50},
    "pause_plan": {"label": "Pause plan", "kind": "pause", "generosity": 0.18},
    "downgrade_basic": {"label": "Downgrade to basic", "kind": "downgrade", "generosity": 0.28},
    "fleet_credit": {"label": "Fleet credit", "kind": "credit", "generosity": 0.16},
    "route_demo": {"label": "Route optimization demo", "kind": "support", "generosity": 0.08},
    "flexible_billing": {"label": "Flexible billing", "kind": "billing", "generosity": 0.12},
    "peak_season_ext": {"label": "Peak season extension", "kind": "extension", "generosity": 0.22},
}

_CTAS = {
    "keep_plan": {
        "label": "Keep my plan",
        "allowed_offer_kinds": {"none", "discount", "extension"},
    },
    "claim_offer": {
        "label": "Claim this offer",
        "allowed_offer_kinds": {"discount", "credit", "extension", "billing"},
    },
    "pause_instead": {
        "label": "Pause my plan",
        "allowed_offer_kinds": {"pause"},
    },
    "switch_to_basic": {
        "label": "Switch to basic plan",
        "allowed_offer_kinds": {"downgrade"},
    },
    "see_options": {
        "label": "See my options",
        "allowed_offer_kinds": {"pause", "downgrade", "billing", "discount", "credit", "support"},
    },
    "talk_to_support": {
        "label": "Talk to support",
        "allowed_offer_kinds": {"support", "none"},
    },
    "schedule_demo": {
        "label": "Schedule a demo",
        "allowed_offer_kinds": {"support", "none"},
    },
    "tell_us_why": {
        "label": "Tell us why you're leaving",
        "allowed_offer_kinds": {"none", "support"},
    },
}

_PERSONALIZATION_LEVELS = {
    "generic": {
        "label": "Generic",
        "intensity": 0.15,
        "description": "Broad category-level copy with little user-specific detail.",
    },
    "contextual": {
        "label": "Contextual",
        "intensity": 0.38,
        "description": "Use the user's fleet size, route volume, or business context.",
    },
    "behavioral": {
        "label": "Behavioral",
        "intensity": 0.62,
        "description": "Reference meaningful usage patterns or optimization results.",
    },
    "highly_specific": {
        "label": "Highly specific",
        "intensity": 0.88,
        "description": "Use deeply personalized phrasing based on specific fleet and route data.",
    },
}

_CONTEXTUAL_GROUNDINGS = {
    "generic": {"label": "Generic grounding", "specificity": 0.10},
    "route_efficiency": {"label": "Route efficiency", "specificity": 0.60},
    "fleet_utilization": {"label": "Fleet utilization", "specificity": 0.55},
    "delivery_reliability": {"label": "Delivery reliability", "specificity": 0.50},
    "cost_per_mile": {"label": "Cost per mile", "specificity": 0.58},
    "seasonal_demand": {"label": "Seasonal demand", "specificity": 0.45},
    "team_workflow": {"label": "Team workflow", "specificity": 0.52},
    "pricing_context": {"label": "Pricing context", "specificity": 0.36},
}

_CREATIVE_TREATMENTS = {
    "plain_note": {"label": "Plain note", "boldness": 0.10},
    "efficiency_report": {"label": "Efficiency report", "boldness": 0.54},
    "cost_analysis": {"label": "Cost analysis", "boldness": 0.52},
    "usage_summary": {"label": "Usage summary", "boldness": 0.46},
    "team_dashboard": {"label": "Team dashboard", "boldness": 0.48},
    "optimization_replay": {"label": "Optimization replay", "boldness": 0.44},
    "before_after_frame": {"label": "Before / after frame", "boldness": 0.42},
    "options_table": {"label": "Options table", "boldness": 0.40},
}

_FRICTION_REDUCERS = {
    "none": {"label": "No friction reducer", "assist": 0.00},
    "one_click_pause": {"label": "One-click pause", "assist": 0.54},
    "instant_downgrade": {"label": "Instant downgrade", "assist": 0.52},
    "callback_scheduling": {"label": "Callback scheduling", "assist": 0.42},
    "trial_extension": {"label": "Trial extension", "assist": 0.48},
    "dedicated_onboarding": {"label": "Dedicated onboarding", "assist": 0.44},
    "billing_shift": {"label": "Billing date shift", "assist": 0.44},
}

DIMENSION_CATALOGS = {
    "message_angles": _MESSAGE_ANGLES,
    "proof_styles": _PROOF_STYLES,
    "base_offers": _BASE_OFFERS,
    "ctas": _CTAS,
    "personalization_levels": _PERSONALIZATION_LEVELS,
    "contextual_groundings": _CONTEXTUAL_GROUNDINGS,
    "creative_treatments": _CREATIVE_TREATMENTS,
    "friction_reducers": _FRICTION_REDUCERS,
}

MUTABLE_DIMENSIONS: dict[str, list[str]] = {
    "message_angle": list(_MESSAGE_ANGLES.keys()),
    "proof_style": list(_PROOF_STYLES.keys()),
    "personalization": list(_PERSONALIZATION_LEVELS.keys()),
    "contextual_grounding": list(_CONTEXTUAL_GROUNDINGS.keys()),
    "creative_treatment": list(_CREATIVE_TREATMENTS.keys()),
    "friction_reducer": list(_FRICTION_REDUCERS.keys()),
}

# ---------------------------------------------------------------------------
# Action → default StrategyCandidate mapping
# ---------------------------------------------------------------------------

ACTION_TO_CANDIDATE: dict[str, StrategyCandidate] = {
    "pause_plan": StrategyCandidate(
        message_angle="flexibility_relief",
        proof_style="similar_user_story",
        offer="pause_plan",
        cta="pause_instead",
        personalization="contextual",
        contextual_grounding="seasonal_demand",
        creative_treatment="plain_note",
        friction_reducer="one_click_pause",
    ),
    "downgrade_basic": StrategyCandidate(
        message_angle="cost_control",
        proof_style="roi_calculation",
        offer="downgrade_basic",
        cta="switch_to_basic",
        personalization="contextual",
        contextual_grounding="pricing_context",
        creative_treatment="options_table",
        friction_reducer="instant_downgrade",
    ),
    "discount_20": StrategyCandidate(
        message_angle="cost_control",
        proof_style="metric_snapshot",
        offer="discount_20",
        cta="claim_offer",
        personalization="contextual",
        contextual_grounding="pricing_context",
        creative_treatment="plain_note",
        friction_reducer="none",
    ),
    "discount_40": StrategyCandidate(
        message_angle="cost_control",
        proof_style="roi_calculation",
        offer="discount_40",
        cta="claim_offer",
        personalization="contextual",
        contextual_grounding="cost_per_mile",
        creative_treatment="cost_analysis",
        friction_reducer="none",
    ),
    "route_optimization_demo": StrategyCandidate(
        message_angle="efficiency_gain",
        proof_style="before_after",
        offer="route_demo",
        cta="schedule_demo",
        personalization="behavioral",
        contextual_grounding="route_efficiency",
        creative_treatment="optimization_replay",
        friction_reducer="callback_scheduling",
    ),
    "fleet_rightsize": StrategyCandidate(
        message_angle="cost_control",
        proof_style="metric_snapshot",
        offer="fleet_credit",
        cta="talk_to_support",
        personalization="behavioral",
        contextual_grounding="fleet_utilization",
        creative_treatment="usage_summary",
        friction_reducer="dedicated_onboarding",
    ),
    "control_graceful_exit": StrategyCandidate(
        message_angle="empathetic_exit",
        proof_style="none",
        offer="none",
        cta="tell_us_why",
        personalization="generic",
        contextual_grounding="generic",
        creative_treatment="plain_note",
        friction_reducer="none",
    ),
}

# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------

ARCHETYPES: dict[str, Archetype] = {
    "high_volume_fleet": Archetype(
        id="high_volume_fleet",
        label="High Volume Fleet",
        root_cause=(
            "Large fleet operator with many routes. Highly dependent on the product. "
            "Cancellation is likely driven by cost at scale or a specific reliability issue."
        ),
        save_potential=0.55,
        recommended_actions=["discount_20", "fleet_rightsize", "route_optimization_demo"],
        anti_actions=["control_graceful_exit"],
        bandit_priors={
            "discount_20": {"alpha_boost": 3.0, "beta_boost": 0.0},
            "fleet_rightsize": {"alpha_boost": 4.0, "beta_boost": 0.0},
            "route_optimization_demo": {"alpha_boost": 2.0, "beta_boost": 0.0},
            "control_graceful_exit": {"alpha_boost": 0.0, "beta_boost": 4.0},
        },
    ),
    "cost_optimizer": Archetype(
        id="cost_optimizer",
        label="Cost Optimizer",
        root_cause=(
            "Price-sensitive user looking for savings. May respond well to discounts "
            "or a lower-tier plan that still meets core needs."
        ),
        save_potential=0.60,
        recommended_actions=["discount_20", "discount_40", "downgrade_basic"],
        anti_actions=["control_graceful_exit", "route_optimization_demo"],
        bandit_priors={
            "discount_20": {"alpha_boost": 4.0, "beta_boost": 0.0},
            "discount_40": {"alpha_boost": 3.0, "beta_boost": 0.0},
            "downgrade_basic": {"alpha_boost": 3.0, "beta_boost": 0.0},
            "control_graceful_exit": {"alpha_boost": 0.0, "beta_boost": 5.0},
        },
    ),
    "light_user": Archetype(
        id="light_user",
        label="Light User",
        root_cause=(
            "Low sessions, few routes created. User drifted away without realizing "
            "full value. May respond to a demo or pause rather than a discount."
        ),
        save_potential=0.40,
        recommended_actions=["route_optimization_demo", "pause_plan", "downgrade_basic"],
        anti_actions=["discount_40"],
        bandit_priors={
            "route_optimization_demo": {"alpha_boost": 4.0, "beta_boost": 0.0},
            "pause_plan": {"alpha_boost": 3.0, "beta_boost": 0.0},
            "downgrade_basic": {"alpha_boost": 2.0, "beta_boost": 0.0},
            "discount_40": {"alpha_boost": 0.0, "beta_boost": 3.0},
        },
    ),
    "quality_frustrated": Archetype(
        id="quality_frustrated",
        label="Quality Frustrated",
        root_cause=(
            "Route quality complaints. The product isn't meeting expectations for "
            "route accuracy or optimization. High frustration, but saveable if "
            "the issue is addressed directly."
        ),
        save_potential=0.35,
        recommended_actions=["route_optimization_demo", "fleet_rightsize", "pause_plan"],
        anti_actions=["discount_20", "discount_40", "control_graceful_exit"],
        bandit_priors={
            "route_optimization_demo": {"alpha_boost": 5.0, "beta_boost": 0.0},
            "fleet_rightsize": {"alpha_boost": 3.0, "beta_boost": 0.0},
            "pause_plan": {"alpha_boost": 2.0, "beta_boost": 0.0},
            "discount_20": {"alpha_boost": 0.0, "beta_boost": 3.0},
            "control_graceful_exit": {"alpha_boost": 0.0, "beta_boost": 4.0},
        },
    ),
    "seasonal_user": Archetype(
        id="seasonal_user",
        label="Seasonal User",
        root_cause=(
            "Usage spikes in certain periods. May not need the product year-round. "
            "Pause is the ideal intervention — they'll come back next season."
        ),
        save_potential=0.65,
        recommended_actions=["pause_plan", "downgrade_basic", "discount_20"],
        anti_actions=["control_graceful_exit"],
        bandit_priors={
            "pause_plan": {"alpha_boost": 5.0, "beta_boost": 0.0},
            "downgrade_basic": {"alpha_boost": 3.0, "beta_boost": 0.0},
            "discount_20": {"alpha_boost": 2.0, "beta_boost": 0.0},
            "control_graceful_exit": {"alpha_boost": 0.0, "beta_boost": 5.0},
        },
    ),
    "burned_bridge": Archetype(
        id="burned_bridge",
        label="Burned Bridge",
        root_cause=(
            "Very high frustration, very low save openness. Something went badly wrong "
            "and the user has checked out. Any aggressive save attempt will backfire."
        ),
        save_potential=0.05,
        recommended_actions=["control_graceful_exit"],
        anti_actions=["discount_20", "discount_40", "route_optimization_demo"],
        bandit_priors={
            "control_graceful_exit": {"alpha_boost": 4.0, "beta_boost": 0.0},
            "discount_20": {"alpha_boost": 0.0, "beta_boost": 5.0},
            "discount_40": {"alpha_boost": 0.0, "beta_boost": 6.0},
        },
    ),
}

# ---------------------------------------------------------------------------
# Raw reason mapping (from Zeo CSV cancel_reason/cancel_note)
# ---------------------------------------------------------------------------


def reason_from_raw(raw_reason: str, raw_note: str = "") -> str:
    """Map Zeo's raw cancel reason text to a normalized reason."""
    text = f"{raw_reason} {raw_note}".lower().strip()

    if not text or text == "nan":
        return "other"

    if any(kw in text for kw in ("expensive", "cost", "price", "afford", "too much")):
        return "price"
    if any(kw in text for kw in ("routes were not proper", "route quality", "not accurate")):
        return "route_quality"
    if any(kw in text for kw in ("don't use", "don't need", "not using", "low usage")):
        return "low_usage"
    if any(kw in text for kw in ("don't need route", "we don't need", "no need", "not listed")):
        return "no_need"
    if any(kw in text for kw in ("changed job", "job change", "left company")):
        return "job_change"
    if any(kw in text for kw in ("webhook", "ios webhook")):
        return "webhook"
    if any(kw in text for kw in ("user canceled", "user cancelled", "canceled the subscription")):
        return "user_initiated"
    return "other"


# ---------------------------------------------------------------------------
# Row → Persona adapter (Zeo CSV → autoresearch Persona)
# ---------------------------------------------------------------------------


def _float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def row_to_persona(row: dict[str, Any], index: int = 0) -> Persona:
    """Convert a Zeo CSV row into a Persona for simulator scoring."""
    # --- Plan tier from payment_interval ---
    interval = str(row.get("payment_interval", "month")).lower().strip()
    plan_map = {"week": "weekly", "month": "monthly", "quarter": "quarterly", "year": "annual"}
    plan = plan_map.get(interval, "monthly")

    # --- Tenure from first_used ---
    tenure_days = 90
    first_used = str(row.get("first_used", ""))
    if first_used and first_used != "nan":
        try:
            first_dt = datetime.fromisoformat(first_used.replace("Z", "+00:00"))
            tenure_days = max(1, (datetime.now(timezone.utc) - first_dt).days)
        except (ValueError, TypeError):
            pass

    # --- Engagement from num_sessions + last_used recency ---
    num_sessions = _int(row.get("num_sessions"), 0)
    last_used = str(row.get("last_used", ""))
    recency_days = 30
    if last_used and last_used != "nan":
        try:
            last_dt = datetime.fromisoformat(last_used.replace("Z", "+00:00"))
            recency_days = max(0, (datetime.now(timezone.utc) - last_dt).days)
        except (ValueError, TypeError):
            pass

    engagement_7d = clamp(1.0 - (recency_days / 30.0)) if recency_days < 30 else 0.05
    engagement_30d = clamp(min(num_sessions / 100.0, 1.0) * (1.0 - recency_days / 90.0))

    # --- Domain-specific features ---
    fleet_size = _int(row.get("fleet_size"), 1)
    routes_created = _int(row.get("total_routes_created"), 0)
    routes_optimized = _int(row.get("total_routes_optimized"), 0)
    stops_planned = _int(row.get("total_stops_planned"), 0)
    revenue = _float(row.get("revenue", 0))

    # --- Derive frustration / save_openness from context ---
    raw_reason = str(row.get("cancel_reason", ""))
    raw_note = str(row.get("cancel_note", ""))
    primary_reason = reason_from_raw(raw_reason, raw_note)

    # Heuristic: route_quality → higher frustration, price → higher save_openness
    frustration = 0.3
    save_openness = 0.4
    trust_risk = 0.2
    if primary_reason == "route_quality":
        frustration = 0.7
        trust_risk = 0.4
    elif primary_reason == "price":
        save_openness = 0.6
    elif primary_reason == "low_usage":
        save_openness = 0.5
    elif primary_reason in ("webhook", "user_initiated"):
        frustration = 0.2
        save_openness = 0.3
    elif primary_reason == "job_change":
        save_openness = 0.1

    # --- Build UserProfile (map Zeo fields into the generic structure) ---
    dormancy_days = min(recency_days, 60)
    total_sessions_val = max(5, num_sessions)
    time_in_app = max(0.5, total_sessions_val * 0.12)
    status = "active" if recency_days < 14 else "inactive"

    # study_context → domain_context (route planning context)
    context_parts = []
    if fleet_size > 5:
        context_parts.append(f"fleet of {fleet_size}")
    if routes_created > 50:
        context_parts.append(f"{routes_created} routes created")
    if stops_planned > 100:
        context_parts.append(f"{stops_planned} stops planned")
    domain_context = ", ".join(context_parts) if context_parts else "route planning"

    profile = UserProfile(
        name=f"zeo_user_{index}",
        cohort="zeo_cancel_cohort",
        plan=plan,
        status=status,
        billing_period=interval if interval in ("week", "month") else "monthly",
        user_type="logistics",
        lifetime_days=tenure_days,
        total_sessions=total_sessions_val,
        total_events=total_sessions_val * 6,
        time_in_app_hours=round(time_in_app, 1),
        card_sets_generated=max(1, routes_created),
        monthly_generations_remaining=max(0, 50 - routes_optimized % 50),
        monthly_generations_total=50,
        chat_messages_remaining=10,
        answer_feedback_remaining=5,
        multi_device_count=min(fleet_size, 10),
        acquisition_source="organic",
        recent_behavior=f"routes:{routes_created} optimized:{routes_optimized}",
        study_context=domain_context,
        retry_after_mistake=save_openness > 0.4,
        source_context_usage=routes_optimized > 10,
        accuracy_signal="high" if routes_optimized > routes_created * 0.5 else "medium",
        dormancy_days=dormancy_days,
    )

    # Derive features using the generic feature engine
    from cta_autoresearch.features import derive_features
    fv = derive_features(profile)

    return Persona(name=profile.name, profile=profile, features=fv)


# ---------------------------------------------------------------------------
# Agent roles for Phase A (strategy evolution swarm)
# ---------------------------------------------------------------------------

AGENT_ROLES = [
    {
        "id": "logistics_roi_analyst",
        "label": "Logistics ROI Analyst",
        "thesis": "Fleet operators track cost-per-stop and fuel savings above all else. Show the math.",
        "focus": "arm priors — boost discount and fleet-rightsizing actions for cost-driven churners",
        "preferred_angles": ["cost_control", "efficiency_gain"],
        "preferred_groundings": ["cost_per_mile", "fleet_utilization", "route_efficiency"],
        "preferred_treatments": ["cost_analysis", "usage_summary"],
        "preferred_reducers": ["none", "instant_discount"],
        "preferred_proof_styles": ["roi_calculation", "metric_snapshot"],
        "preferred_personalization": ["behavioral", "contextual"],
    },
    {
        "id": "operations_empathist",
        "label": "Operations Empathist",
        "thesis": "Dispatchers and owner-operators are time-starved. Friction kills saves. Remove every barrier.",
        "focus": "friction_reducer and creative_treatment — minimize clicks, maximize one-tap actions",
        "preferred_angles": ["team_coordination", "flexibility_relief"],
        "preferred_groundings": ["time_saved", "dispatcher_workload"],
        "preferred_treatments": ["plain_note", "before_after_frame"],
        "preferred_reducers": ["pause_and_return", "callback_scheduling", "dedicated_onboarding"],
        "preferred_proof_styles": ["before_after", "quantified_outcome"],
        "preferred_personalization": ["contextual", "behavioral"],
    },
    {
        "id": "growth_retention_strategist",
        "label": "Growth Retention Strategist",
        "thesis": "The best save for a seasonal fleet is a pause, not a cancel. Match intervention to business cycle.",
        "focus": "context arms — strengthen reason+plan combos for seasonal and job-change churners",
        "preferred_angles": ["seasonal_flexibility", "fleet_dependency"],
        "preferred_groundings": ["seasonal_usage", "fleet_utilization"],
        "preferred_treatments": ["plain_note", "usage_summary"],
        "preferred_reducers": ["pause_and_return", "dedicated_onboarding"],
        "preferred_proof_styles": ["similar_user_story", "metric_snapshot"],
        "preferred_personalization": ["contextual", "generic"],
    },
    {
        "id": "product_evangelist",
        "label": "Product Evangelist",
        "thesis": "Route quality complaints are often feature-awareness gaps. A live demo closes more than a discount.",
        "focus": "route_optimization_demo arm — surface it for route_quality and low_usage churners",
        "preferred_angles": ["efficiency_gain", "competitive_edge"],
        "preferred_groundings": ["route_efficiency", "driver_experience"],
        "preferred_treatments": ["optimization_replay", "before_after_frame"],
        "preferred_reducers": ["callback_scheduling", "none"],
        "preferred_proof_styles": ["before_after", "roi_calculation"],
        "preferred_personalization": ["behavioral", "contextual"],
    },
    {
        "id": "trust_guardian",
        "label": "Trust Guardian",
        "thesis": "Aggressive saves on job-change or personal churn erode long-term brand health. Let go gracefully.",
        "focus": "reason routing — block high-pressure actions for job_change and user_initiated reasons",
        "preferred_angles": ["empathetic_exit", "flexibility_relief"],
        "preferred_groundings": ["generic", "dispatcher_workload"],
        "preferred_treatments": ["plain_note"],
        "preferred_reducers": ["none"],
        "preferred_proof_styles": ["none", "similar_user_story"],
        "preferred_personalization": ["generic", "contextual"],
    },
    {
        "id": "experiment_operator",
        "label": "Experiment Operator",
        "thesis": "Statistical power requires the right exploration/holdout balance for fleet-size segments.",
        "focus": "exploration rate and holdout rate tuning based on segment volume",
    },
]

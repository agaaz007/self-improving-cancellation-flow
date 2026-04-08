"""User modeling from enriched cancel data.

Derives root-cause archetypes from session signals, maps each archetype
to the interventions most likely to save that user, and produces
context-aware priors for the cancel policy bandit.

The 3 raw cancel reasons (feature_gap, quality_bug, other) collapse real
user intent into buckets too coarse to act on.  This module re-derives
the TRUE root cause from the enriched features:

  Raw reason        + Enriched signals              => Root cause archetype
  ─────────────────────────────────────────────────────────────────────────
  feature_gap       + flashcard_flip request         => MISSING_CORE_FEATURE
  feature_gap       + medical student + image_support=> POWER_USER_GAP
  feature_gap       + free plan + upgrade clicks     => FREE_PLAN_CEILING
  quality_bug       + pdf_upload_failure             => BROKEN_WORKFLOW
  quality_bug       + crash / slow_loading           => RELIABILITY_EROSION
  other             + teacher tag                    => WRONG_FIT
  other             + low frustration + high openness=> SOFT_CHURN (saveable!)
  *                 + high frustration + low openness=> BURNED_BRIDGE

Each archetype carries:
  - A root cause explanation (what's actually wrong)
  - Save potential score (how likely an intervention works)
  - Recommended actions ranked by expected effectiveness
  - Context arm prior boosts for the bandit
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ── Root cause archetypes ─────────────────────────────────────────────


@dataclass(frozen=True)
class Archetype:
    """A root-cause archetype with intervention recommendations."""

    id: str
    label: str
    root_cause: str
    save_potential: float  # 0.0 (unsaveable) to 1.0 (very saveable)
    recommended_actions: list[str]  # ranked best-to-worst
    anti_actions: list[str]  # actions that will backfire
    bandit_priors: dict[str, dict[str, float]]  # action_id -> {alpha_boost, beta_boost}


ARCHETYPES: dict[str, Archetype] = {
    "missing_core_feature": Archetype(
        id="missing_core_feature",
        label="Missing Core Feature",
        root_cause=(
            "User needs a specific feature (flashcard flip, audio mode, anki export) "
            "that doesn't exist yet. They like the product but can't use it for their workflow."
        ),
        save_potential=0.35,
        recommended_actions=[
            "feature_value_recap",      # show them what IS there + roadmap
            "pause_plan_relief",        # pause until the feature ships
            "exam_sprint_focus",        # reframe around what works now
        ],
        anti_actions=[
            "targeted_discount_20",     # discount doesn't fix missing features
            "targeted_discount_40",
            "control_empathic_exit",
        ],
        bandit_priors={
            "feature_value_recap":   {"alpha_boost": 3.0, "beta_boost": 0.0},
            "pause_plan_relief":     {"alpha_boost": 2.0, "beta_boost": 0.0},
            "exam_sprint_focus":     {"alpha_boost": 1.5, "beta_boost": 0.0},
            "targeted_discount_20":  {"alpha_boost": 0.0, "beta_boost": 2.0},
            "targeted_discount_40":  {"alpha_boost": 0.0, "beta_boost": 2.5},
            "control_empathic_exit": {"alpha_boost": 0.0, "beta_boost": 5.0},
        },
    ),
    "power_user_gap": Archetype(
        id="power_user_gap",
        label="Power User Gap",
        root_cause=(
            "Advanced user (medical/nursing student, grad student) who needs domain-specific "
            "features: image support for anatomy, better question quality, specialized content. "
            "They're invested but hitting ceiling of what the product offers their field."
        ),
        save_potential=0.45,
        recommended_actions=[
            "concierge_recovery",       # personal outreach, understand their specific need
            "feature_value_recap",      # show advanced features they may not know about
            "pause_plan_relief",        # hold them while building for their vertical
        ],
        anti_actions=[
            "control_empathic_exit",
            "billing_clarity_reset",
        ],
        bandit_priors={
            "concierge_recovery":    {"alpha_boost": 4.0, "beta_boost": 0.0},
            "feature_value_recap":   {"alpha_boost": 2.5, "beta_boost": 0.0},
            "pause_plan_relief":     {"alpha_boost": 2.0, "beta_boost": 0.0},
            "control_empathic_exit": {"alpha_boost": 0.0, "beta_boost": 5.0},
        },
    ),
    "free_plan_ceiling": Archetype(
        id="free_plan_ceiling",
        label="Free Plan Ceiling",
        root_cause=(
            "Free user hitting limits (generation quota, chat messages). They clicked "
            "upgrade buttons but didn't convert — price or commitment is the barrier, "
            "not product quality."
        ),
        save_potential=0.55,
        recommended_actions=[
            "targeted_discount_20",     # lower the price barrier
            "downgrade_lite_switch",    # offer a middle tier
            "exam_sprint_focus",        # short-term access for their exam
        ],
        anti_actions=[
            "control_empathic_exit",
            "feature_value_recap",      # they already know the value, price is the issue
        ],
        bandit_priors={
            "targeted_discount_20":  {"alpha_boost": 4.0, "beta_boost": 0.0},
            "targeted_discount_40":  {"alpha_boost": 3.0, "beta_boost": 0.0},
            "downgrade_lite_switch": {"alpha_boost": 3.0, "beta_boost": 0.0},
            "exam_sprint_focus":     {"alpha_boost": 2.0, "beta_boost": 0.0},
            "control_empathic_exit": {"alpha_boost": 0.0, "beta_boost": 5.0},
        },
    ),
    "broken_workflow": Archetype(
        id="broken_workflow",
        label="Broken Workflow",
        root_cause=(
            "Specific feature is broken for them: PDF upload fails, card generation errors. "
            "They WANT to use the product but literally can't complete their task. "
            "High frustration but also high save potential if the bug is fixed."
        ),
        save_potential=0.50,
        recommended_actions=[
            "concierge_recovery",       # human support to workaround/fix their issue
            "pause_plan_relief",        # pause while we fix it
            "billing_clarity_reset",    # acknowledge the issue + refund/credit
        ],
        anti_actions=[
            "feature_value_recap",      # tone-deaf when something is broken
            "targeted_discount_20",     # discount on a broken product is insulting
            "control_empathic_exit",
        ],
        bandit_priors={
            "concierge_recovery":    {"alpha_boost": 5.0, "beta_boost": 0.0},
            "pause_plan_relief":     {"alpha_boost": 3.0, "beta_boost": 0.0},
            "billing_clarity_reset": {"alpha_boost": 2.5, "beta_boost": 0.0},
            "targeted_discount_20":  {"alpha_boost": 0.0, "beta_boost": 3.0},
            "feature_value_recap":   {"alpha_boost": 0.0, "beta_boost": 4.0},
            "control_empathic_exit": {"alpha_boost": 0.0, "beta_boost": 5.0},
        },
    ),
    "reliability_erosion": Archetype(
        id="reliability_erosion",
        label="Reliability Erosion",
        root_cause=(
            "Repeated crashes, slow loading, error messages have eroded trust over time. "
            "No single catastrophic bug — just death by a thousand cuts. "
            "Frustration is very high, save openness is low."
        ),
        save_potential=0.20,
        recommended_actions=[
            "concierge_recovery",       # only human touch can rebuild trust
            "pause_plan_relief",        # stop bleeding while we stabilize
        ],
        anti_actions=[
            "targeted_discount_20",
            "targeted_discount_40",
            "feature_value_recap",
            "exam_sprint_focus",
        ],
        bandit_priors={
            "concierge_recovery":    {"alpha_boost": 3.0, "beta_boost": 0.0},
            "pause_plan_relief":     {"alpha_boost": 2.0, "beta_boost": 0.0},
            "targeted_discount_20":  {"alpha_boost": 0.0, "beta_boost": 3.0},
            "targeted_discount_40":  {"alpha_boost": 0.0, "beta_boost": 3.5},
            "feature_value_recap":   {"alpha_boost": 0.0, "beta_boost": 4.0},
            "control_empathic_exit": {"alpha_boost": 0.0, "beta_boost": 3.0},
        },
    ),
    "wrong_fit": Archetype(
        id="wrong_fit",
        label="Wrong Product Fit",
        root_cause=(
            "User is a teacher, professional, or non-student who signed up but the "
            "product isn't designed for their use case. No bug, no missing feature — "
            "the product just isn't for them."
        ),
        save_potential=0.10,
        recommended_actions=[
            "control_empathic_exit",    # let them go gracefully
            "billing_clarity_reset",    # ensure clean billing
        ],
        anti_actions=[
            "targeted_discount_20",     # discount on wrong product is waste
            "targeted_discount_40",
            "concierge_recovery",       # nothing to recover — it's a mismatch
        ],
        bandit_priors={
            "control_empathic_exit": {"alpha_boost": 2.0, "beta_boost": 0.0},
            "billing_clarity_reset": {"alpha_boost": 1.0, "beta_boost": 0.0},
            "targeted_discount_20":  {"alpha_boost": 0.0, "beta_boost": 3.0},
            "concierge_recovery":    {"alpha_boost": 0.0, "beta_boost": 2.0},
        },
    ),
    "soft_churn": Archetype(
        id="soft_churn",
        label="Soft Churn (Saveable)",
        root_cause=(
            "Low frustration, moderate save openness, no strong complaint. "
            "Often seasonal (end of semester), financial, or just drifted away. "
            "These users don't hate the product — they just need a reason to stay."
        ),
        save_potential=0.65,
        recommended_actions=[
            "pause_plan_relief",        # they'll come back next semester
            "targeted_discount_20",     # small nudge is enough
            "exam_sprint_focus",        # reactivate around their next exam
            "downgrade_lite_switch",    # lower commitment option
        ],
        anti_actions=[
            "control_empathic_exit",    # don't just let the saveable ones go!
        ],
        bandit_priors={
            "pause_plan_relief":     {"alpha_boost": 5.0, "beta_boost": 0.0},
            "targeted_discount_20":  {"alpha_boost": 4.0, "beta_boost": 0.0},
            "exam_sprint_focus":     {"alpha_boost": 3.0, "beta_boost": 0.0},
            "downgrade_lite_switch": {"alpha_boost": 3.0, "beta_boost": 0.0},
            "targeted_discount_40":  {"alpha_boost": 2.0, "beta_boost": 0.0},
            "control_empathic_exit": {"alpha_boost": 0.0, "beta_boost": 5.0},
        },
    ),
    "burned_bridge": Archetype(
        id="burned_bridge",
        label="Burned Bridge",
        root_cause=(
            "Very high frustration, very low save openness. Something went badly wrong "
            "and the user has emotionally checked out. Any aggressive save attempt will "
            "generate a complaint or support escalation."
        ),
        save_potential=0.05,
        recommended_actions=[
            "control_empathic_exit",    # graceful exit preserves brand
            "billing_clarity_reset",    # make sure billing is clean
        ],
        anti_actions=[
            "targeted_discount_20",
            "targeted_discount_40",
            "concierge_recovery",
            "feature_value_recap",
            "exam_sprint_focus",
        ],
        bandit_priors={
            "control_empathic_exit": {"alpha_boost": 4.0, "beta_boost": 0.0},
            "billing_clarity_reset": {"alpha_boost": 1.5, "beta_boost": 0.0},
            "targeted_discount_20":  {"alpha_boost": 0.0, "beta_boost": 5.0},
            "targeted_discount_40":  {"alpha_boost": 0.0, "beta_boost": 6.0},
            "concierge_recovery":    {"alpha_boost": 0.0, "beta_boost": 3.0},
        },
    ),
}


# ── Root cause classifier ─────────────────────────────────────────────


@dataclass
class UserDiagnosis:
    """Result of classifying a user into a root-cause archetype."""

    archetype_id: str
    archetype: Archetype
    confidence: float
    signals: list[str]  # evidence that led to this classification
    save_potential: float
    recommended_actions: list[str]
    root_cause_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "archetype_id": self.archetype_id,
            "archetype_label": self.archetype.label,
            "confidence": round(self.confidence, 3),
            "signals": self.signals,
            "save_potential": round(self.save_potential, 3),
            "recommended_actions": self.recommended_actions,
            "root_cause_summary": self.root_cause_summary,
        }


def classify_user(row: dict[str, Any]) -> UserDiagnosis:
    """Classify a user into a root-cause archetype from enriched data.

    Input is a row from the enriched dataset (either JSON or CSV format).
    Uses a decision-tree style classifier over the enriched features.
    """
    # Normalize inputs from either JSON or CSV format
    reason = str(row.get("primary_reason", "other")).lower().strip()
    plan = str(row.get("plan_tier", "unknown")).lower().strip()
    student_type = str(row.get("student_type", "")).lower().strip()

    features = row.get("features", {})
    if not features:
        # CSV format — features are top-level
        features = row

    save_openness = _float(features.get("save_openness", 0.3))
    frustration = _float(features.get("frustration_level", 0.5))
    trust_risk = _float(features.get("trust_risk", 0.3))
    churn_risk = _float(features.get("churn_risk_score", 0.5))

    # Parse list fields (handle both JSON strings and actual lists)
    feature_requests = _parse_list(features.get("feature_requests",
                                   features.get("feature_requests_json", [])))
    bug_signals = _parse_list(features.get("bug_signals",
                              features.get("bug_signals_json", [])))
    tags = _parse_list(features.get("tags", features.get("tags_json", [])))

    tags_lower = {t.lower() for t in tags}
    signals: list[str] = []

    # ── Decision tree ─────────────────────────────────────────────

    # 1. Burned bridge: high frustration + low openness = let them go
    if frustration >= 0.7 and save_openness < 0.25:
        signals.append(f"frustration={frustration:.2f} (very high)")
        signals.append(f"save_openness={save_openness:.2f} (very low)")
        return _diagnosis("burned_bridge", 0.85, signals)

    # 2. Broken workflow: specific bugs blocking their task
    if bug_signals and reason in ("quality_bug", "feature_gap"):
        for bug in bug_signals:
            signals.append(f"bug_signal: {bug}")
        if frustration >= 0.6:
            signals.append(f"frustration={frustration:.2f}")
            return _diagnosis("broken_workflow", 0.90, signals)
        return _diagnosis("broken_workflow", 0.75, signals)

    # 3. Reliability erosion: quality_bug + no specific bug + high frustration
    if reason == "quality_bug" and frustration >= 0.6:
        signals.append(f"quality_bug + high frustration ({frustration:.2f})")
        signals.append("no specific bug signal — general reliability issues")
        crash_tags = tags_lower & {"error messages", "error", "crash", "frustration", "technical issues"}
        for t in crash_tags:
            signals.append(f"tag: {t}")
        return _diagnosis("reliability_erosion", 0.80, signals)

    # 4. Free plan ceiling: free user who explored upgrades
    if plan == "free" and reason == "feature_gap":
        upgrade_signals = tags_lower & {"upgrade exploration", "free plan", "free plan limitations", "upgrade interactions"}
        if upgrade_signals:
            for s in upgrade_signals:
                signals.append(f"tag: {s}")
            signals.append("free user hitting plan limits")
            return _diagnosis("free_plan_ceiling", 0.85, signals)

    # 5. Power user gap: medical/nursing/grad student needing domain features
    if reason == "feature_gap" and student_type in ("medical", "nursing", "other graduate"):
        signals.append(f"student_type: {student_type}")
        for req in feature_requests:
            signals.append(f"feature_request: {req}")
        if feature_requests:
            return _diagnosis("power_user_gap", 0.85, signals)
        return _diagnosis("power_user_gap", 0.70, signals)

    # 6. Missing core feature: feature_gap + specific requests
    if reason == "feature_gap" and feature_requests:
        for req in feature_requests:
            signals.append(f"feature_request: {req}")
        return _diagnosis("missing_core_feature", 0.80, signals)

    # 7. Wrong fit: teacher, professional, or non-student user
    if "teacher" in tags_lower or student_type == "teacher":
        signals.append("user is a teacher — product designed for students")
        return _diagnosis("wrong_fit", 0.80, signals)

    if reason == "other" and frustration < 0.3 and save_openness < 0.2:
        signals.append("other reason + very low engagement signals")
        return _diagnosis("wrong_fit", 0.60, signals)

    # 8. Soft churn: low frustration, moderate openness, no strong complaint
    if frustration < 0.5 and save_openness >= 0.3:
        signals.append(f"low frustration ({frustration:.2f})")
        signals.append(f"moderate save_openness ({save_openness:.2f})")
        signals.append("no strong complaint — likely seasonal or financial")
        return _diagnosis("soft_churn", 0.75, signals)

    # 9. Remaining feature_gap without specific signals
    if reason == "feature_gap":
        signals.append("feature_gap without specific feature request or bug")
        return _diagnosis("missing_core_feature", 0.50, signals)

    # 10. Remaining quality_bug
    if reason == "quality_bug":
        signals.append("quality_bug with moderate signals")
        return _diagnosis("reliability_erosion", 0.55, signals)

    # 11. Default: soft churn (most saveable default)
    signals.append(f"unclassified: reason={reason}, frustration={frustration:.2f}, openness={save_openness:.2f}")
    if save_openness >= 0.3:
        return _diagnosis("soft_churn", 0.45, signals)
    return _diagnosis("wrong_fit", 0.40, signals)


def _diagnosis(archetype_id: str, confidence: float, signals: list[str]) -> UserDiagnosis:
    arch = ARCHETYPES.get(archetype_id)
    if arch is None:
        # Archetype not in current client catalog — pick closest or first available
        arch = next(iter(ARCHETYPES.values()))
        archetype_id = next(iter(ARCHETYPES))
    return UserDiagnosis(
        archetype_id=archetype_id,
        archetype=arch,
        confidence=confidence,
        signals=signals,
        save_potential=arch.save_potential,
        recommended_actions=arch.recommended_actions,
        root_cause_summary=arch.root_cause,
    )


def _float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _parse_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        v = v.strip()
        if v.startswith("["):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return []
        return [v] if v else []
    return []


# ── Batch classification + bandit prior generation ────────────────────


@dataclass
class CohortAnalysis:
    """Analysis of a cohort of users classified into archetypes."""

    total_users: int
    archetype_counts: dict[str, int]
    archetype_pcts: dict[str, float]
    saveable_count: int
    saveable_pct: float
    avg_save_potential: float
    context_arm_priors: dict[str, dict[str, float]]  # context_key -> {alpha, beta}
    diagnoses: list[UserDiagnosis]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_users": self.total_users,
            "archetype_counts": self.archetype_counts,
            "archetype_pcts": {k: round(v, 3) for k, v in self.archetype_pcts.items()},
            "saveable_count": self.saveable_count,
            "saveable_pct": round(self.saveable_pct, 3),
            "avg_save_potential": round(self.avg_save_potential, 3),
            "context_arm_priors": {
                k: {kk: round(vv, 2) for kk, vv in v.items()}
                for k, v in self.context_arm_priors.items()
            },
        }


def analyze_cohort(rows: list[dict[str, Any]]) -> CohortAnalysis:
    """Classify all users and produce aggregate stats + bandit priors."""
    diagnoses = [classify_user(r) for r in rows]

    # Counts
    counts: dict[str, int] = Counter(d.archetype_id for d in diagnoses)
    total = len(diagnoses)
    pcts = {k: v / total for k, v in counts.items()} if total > 0 else {}

    # Saveability
    saveable = [d for d in diagnoses if d.save_potential >= 0.3]
    avg_potential = sum(d.save_potential for d in diagnoses) / total if total > 0 else 0.0

    # Generate context arm priors from archetype distributions
    # For each reason|plan combo, accumulate the archetype's bandit priors
    # weighted by how many users fall into that archetype
    context_priors: dict[str, dict[str, float]] = defaultdict(lambda: {"alpha": 1.0, "beta": 1.0})

    for row, diagnosis in zip(rows, diagnoses):
        reason = str(row.get("primary_reason", "other")).lower().strip()
        plan = str(row.get("plan_tier", "unknown")).lower().strip()
        arch = diagnosis.archetype

        for action_id, boosts in arch.bandit_priors.items():
            key = f"{reason}|{plan}|{action_id}"
            alpha_b = boosts.get("alpha_boost", 0.0) * diagnosis.confidence
            beta_b = boosts.get("beta_boost", 0.0) * diagnosis.confidence
            context_priors[key]["alpha"] = context_priors[key].get("alpha", 1.0) + alpha_b
            context_priors[key]["beta"] = context_priors[key].get("beta", 1.0) + beta_b

    return CohortAnalysis(
        total_users=total,
        archetype_counts=dict(counts),
        archetype_pcts=dict(pcts),
        saveable_count=len(saveable),
        saveable_pct=len(saveable) / total if total > 0 else 0.0,
        avg_save_potential=avg_potential,
        context_arm_priors=dict(context_priors),
        diagnoses=diagnoses,
    )


def apply_cohort_priors(
    runtime: Any,
    analysis: CohortAnalysis,
) -> dict[str, Any]:
    """Apply the cohort-derived context arm priors to a CancelPolicyRuntime.

    This seeds the bandit with informed priors so it knows from day one:
    - broken_workflow users need concierge, not discounts
    - free_plan_ceiling users respond to discounts
    - burned_bridge users should be let go gracefully
    """
    applied = 0
    for context_key, priors in analysis.context_arm_priors.items():
        arms_context = runtime.state.setdefault("arms_context", {})
        if context_key not in arms_context:
            arms_context[context_key] = {
                "alpha": priors["alpha"],
                "beta": priors["beta"],
                "impressions": 0,
                "outcomes": 0,
            }
        else:
            arms_context[context_key]["alpha"] = float(arms_context[context_key]["alpha"]) + priors["alpha"] - 1.0
            arms_context[context_key]["beta"] = float(arms_context[context_key]["beta"]) + priors["beta"] - 1.0
        applied += 1

    runtime._persist_state()

    return {
        "context_arms_seeded": applied,
        "archetypes_found": len(analysis.archetype_counts),
        "saveable_pct": round(analysis.saveable_pct, 3),
        "avg_save_potential": round(analysis.avg_save_potential, 3),
    }


# ── Simulator harness bridge ──────────────────────────────────────────

from cta_autoresearch.models import Persona, StrategyCandidate, UserProfile
from cta_autoresearch.features import derive_features, clamp
from cta_autoresearch.strategy_policy import (
    CONTEXTUAL_GROUNDINGS,
    CREATIVE_TREATMENTS,
    FRICTION_REDUCERS,
    MESSAGE_ANGLES,
    PERSONALIZATION_LEVELS,
    PROOF_STYLES,
    valid_candidate,
)


# The 6 presentation dimensions the optimizer can mutate per action.
# offer + cta are FIXED per action (business logic — WHAT we offer).
# These 6 control HOW the offer is presented.
MUTABLE_DIMENSIONS: dict[str, list[str]] = {
    "message_angle": list(MESSAGE_ANGLES.keys()),
    "proof_style": list(PROOF_STYLES.keys()),
    "personalization": list(PERSONALIZATION_LEVELS.keys()),
    "contextual_grounding": list(CONTEXTUAL_GROUNDINGS.keys()),
    "creative_treatment": list(CREATIVE_TREATMENTS.keys()),
    "friction_reducer": list(FRICTION_REDUCERS.keys()),
}


def configure_dimensions(dims: dict[str, list[str]]) -> None:
    """Override MUTABLE_DIMENSIONS with client-specific dimension keys."""
    MUTABLE_DIMENSIONS.clear()
    MUTABLE_DIMENSIONS.update(dims)


def configure_archetypes(archs: dict[str, Archetype]) -> None:
    """Override ARCHETYPES with client-specific archetypes."""
    ARCHETYPES.clear()
    ARCHETYPES.update(archs)


def configure_action_candidates(candidates: dict[str, StrategyCandidate]) -> None:
    """Override ACTION_TO_CANDIDATE with client-specific action→strategy mapping."""
    ACTION_TO_CANDIDATE.clear()
    ACTION_TO_CANDIDATE.update(candidates)


ACTION_TO_CANDIDATE: dict[str, StrategyCandidate] = {
    "pause_plan_relief": StrategyCandidate(
        message_angle="flexibility_relief",
        proof_style="personal_usage_signal",
        offer="pause_plan",
        cta="pause_instead",
        personalization="contextual",
        contextual_grounding="comeback_window",
        creative_treatment="plain_note",
        friction_reducer="single_tap_pause",
    ),
    "downgrade_lite_switch": StrategyCandidate(
        message_angle="cost_value_reframe",
        proof_style="similar_user_story",
        offer="downgrade_lite",
        cta="switch_to_lite",
        personalization="contextual",
        contextual_grounding="pricing_context",
        creative_treatment="options_table",
        friction_reducer="prefilled_downgrade",
    ),
    "targeted_discount_20": StrategyCandidate(
        message_angle="cost_value_reframe",
        proof_style="quantified_outcome",
        offer="discount_20",
        cta="claim_offer",
        personalization="contextual",
        contextual_grounding="pricing_context",
        creative_treatment="plain_note",
        friction_reducer="none",
    ),
    "targeted_discount_40": StrategyCandidate(
        message_angle="cost_value_reframe",
        proof_style="quantified_outcome",
        offer="discount_40",
        cta="claim_offer",
        personalization="contextual",
        contextual_grounding="pricing_context",
        creative_treatment="plain_note",
        friction_reducer="none",
    ),
    "concierge_recovery": StrategyCandidate(
        message_angle="mistake_recovery",
        proof_style="personal_usage_signal",
        offer="concierge_support",
        cta="talk_to_learning_support",
        personalization="behavioral",
        contextual_grounding="support_signal",
        creative_treatment="coach_plan",
        friction_reducer="human_concierge",
    ),
    "exam_sprint_focus": StrategyCandidate(
        message_angle="goal_deadline",
        proof_style="quantified_outcome",
        offer="exam_sprint",
        cta="finish_current_goal",
        personalization="contextual",
        contextual_grounding="deadline_pressure",
        creative_treatment="study_timeline",
        friction_reducer="none",
    ),
    "feature_value_recap": StrategyCandidate(
        message_angle="feature_unlock",
        proof_style="personal_usage_signal",
        offer="study_plan_reset",
        cta="talk_to_learning_support",
        personalization="behavioral",
        contextual_grounding="unused_value",
        creative_treatment="feature_visual",
        friction_reducer="guided_reset",
    ),
    "billing_clarity_reset": StrategyCandidate(
        message_angle="flexibility_relief",
        proof_style="none",
        offer="flexible_billing",
        cta="see_plan_options",
        personalization="contextual",
        contextual_grounding="pricing_context",
        creative_treatment="options_table",
        friction_reducer="billing_shift",
    ),
    "control_empathic_exit": StrategyCandidate(
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


def action_to_candidate(action_id: str) -> StrategyCandidate:
    """Map a cancel policy action_id to a StrategyCandidate for simulator scoring."""
    if action_id in ACTION_TO_CANDIDATE:
        return ACTION_TO_CANDIDATE[action_id]
    return next(iter(ACTION_TO_CANDIDATE.values()))


def default_candidate_strategies() -> dict[str, dict[str, str]]:
    """Return the default presentation strategy for each action as mutable dicts.

    Only includes the 6 mutable dimensions. offer + cta are fixed per action.
    """
    strategies: dict[str, dict[str, str]] = {}
    for action_id, cand in ACTION_TO_CANDIDATE.items():
        strategies[action_id] = {
            "message_angle": cand.message_angle,
            "proof_style": cand.proof_style,
            "personalization": cand.personalization,
            "contextual_grounding": cand.contextual_grounding,
            "creative_treatment": cand.creative_treatment,
            "friction_reducer": cand.friction_reducer,
        }
    return strategies


def build_candidate_with_overrides(
    action_id: str,
    strategies: dict[str, dict[str, str]] | None = None,
) -> StrategyCandidate:
    """Build a StrategyCandidate from defaults + mutable strategy overrides.

    The offer and CTA are always fixed per action (business logic).
    The 6 presentation dimensions come from the strategies dict, falling
    back to ACTION_TO_CANDIDATE defaults.
    """
    # Fall back to the first action in the dict (client-specific control action)
    default = ACTION_TO_CANDIDATE.get(action_id) or next(iter(ACTION_TO_CANDIDATE.values()))

    if not strategies or action_id not in strategies:
        return default

    s = strategies[action_id]
    return StrategyCandidate(
        message_angle=s.get("message_angle", default.message_angle),
        proof_style=s.get("proof_style", default.proof_style),
        offer=default.offer,       # FIXED — what we offer
        cta=default.cta,           # FIXED — what we offer
        personalization=s.get("personalization", default.personalization),
        contextual_grounding=s.get("contextual_grounding", default.contextual_grounding),
        creative_treatment=s.get("creative_treatment", default.creative_treatment),
        friction_reducer=s.get("friction_reducer", default.friction_reducer),
    )


def enriched_row_to_persona(row: dict[str, Any], index: int = 0) -> Persona:
    """Convert an enriched cancel row into a Persona for simulator scoring.

    Maps the enriched features (frustration, trust_risk, save_openness) into
    UserProfile fields that derive_features() will pick up, producing a
    FeatureVector that reflects the user's actual state.
    """
    features = row.get("features", {})
    if not features:
        features = row

    frustration = _float(features.get("frustration_level", 0.5))
    trust_risk = _float(features.get("trust_risk", 0.3))
    save_openness = _float(features.get("save_openness", 0.3))
    churn_risk = _float(features.get("churn_risk_score", 0.5))

    plan_tier = str(row.get("plan_tier", "super_learner")).lower().strip()
    plan_map = {"free": "free", "starter": "starter", "super_learner": "super_learner"}
    plan = plan_map.get(plan_tier, "super_learner")

    student_type = str(row.get("student_type", "")).strip()
    user_type = student_type if student_type else "College"

    tags = _parse_list(features.get("tags", []))
    bug_signals = _parse_list(features.get("bug_signals", []))
    feature_requests = _parse_list(features.get("feature_requests", []))

    # Map enriched signals to UserProfile fields that drive feature derivation:
    # - frustration → dormancy_days (frustrated users are dormant)
    # - trust_risk → affects trust_sensitivity via plan + dormancy
    # - save_openness → affects engagement signals
    # - churn_risk → affects status and sessions
    dormancy_days = int(frustration * 40 + trust_risk * 15)
    total_sessions = max(5, int((1.0 - churn_risk) * 300 + save_openness * 100))
    time_in_app_hours = max(0.5, total_sessions * 0.15)
    card_sets_generated = max(1, int(total_sessions * 0.3))

    # High save_openness = still engaged, low = checked out
    status = "active" if save_openness > 0.35 else "inactive"
    retry_after_mistake = save_openness > 0.4
    source_context_usage = save_openness > 0.5

    # Build behavior text from signals
    behavior_parts = []
    for bug in bug_signals:
        behavior_parts.append(f"bug:{bug}")
    for req in feature_requests:
        behavior_parts.append(f"request:{req}")
    if frustration > 0.7:
        behavior_parts.append("frustrated")
    if frustration > 0.5:
        behavior_parts.append("stalled")
    recent_behavior = " ".join(behavior_parts) if behavior_parts else "normal usage"

    # Build study context from tags
    study_parts = [t for t in tags if t.lower() not in {"free plan", "student user"}]
    study_context = " ".join(study_parts[:3]) if study_parts else "general study"

    # Estimate lifetime from timestamp if available
    lifetime_days = 90
    timestamp = row.get("timestamp", "")
    if timestamp and isinstance(timestamp, str) and "2025" in timestamp:
        lifetime_days = 120

    # Usage ratio — higher save_openness means they've used more
    gen_total = 50
    gen_remaining = max(0, int(gen_total * (1.0 - save_openness * 0.7)))

    accuracy = "high" if save_openness > 0.5 and frustration < 0.4 else "medium"

    profile = UserProfile(
        name=f"user_{index}",
        cohort="cancel_cohort",
        plan=plan,
        status=status,
        billing_period="monthly",
        user_type=user_type,
        lifetime_days=lifetime_days,
        total_sessions=total_sessions,
        total_events=total_sessions * 8,
        time_in_app_hours=round(time_in_app_hours, 1),
        card_sets_generated=card_sets_generated,
        monthly_generations_remaining=gen_remaining,
        monthly_generations_total=gen_total,
        chat_messages_remaining=max(0, int(20 * (1.0 - frustration))),
        answer_feedback_remaining=max(0, int(10 * save_openness)),
        multi_device_count=2 if save_openness > 0.4 else 1,
        acquisition_source="organic",
        recent_behavior=recent_behavior,
        study_context=study_context,
        retry_after_mistake=retry_after_mistake,
        source_context_usage=source_context_usage,
        accuracy_signal=accuracy,
        dormancy_days=dormancy_days,
    )

    fv = derive_features(profile)
    return Persona(name=profile.name, profile=profile, features=fv)


@dataclass
class SimulatorEvalResult:
    """Result of evaluating a policy against the simulator harness."""

    total_users: int
    expected_retention_score: float
    expected_revenue_score: float
    trust_safety_score: float
    composite_score: float
    alignment_score: float  # fraction where action matches archetype recommendation
    per_archetype: dict[str, dict[str, float]]

    @property
    def save_rate(self) -> float:
        return self.expected_retention_score

    @property
    def average_reward(self) -> float:
        return self.composite_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_users": self.total_users,
            "expected_retention_score": round(self.expected_retention_score, 4),
            "expected_revenue_score": round(self.expected_revenue_score, 4),
            "trust_safety_score": round(self.trust_safety_score, 4),
            "composite_score": round(self.composite_score, 4),
            "alignment_score": round(self.alignment_score, 4),
            "per_archetype": {
                k: {kk: round(vv, 4) for kk, vv in v.items()}
                for k, v in self.per_archetype.items()
            },
        }


def simulator_eval(
    rows: list[dict[str, Any]],
    simulate_action: Callable[[dict[str, Any]], Any],
    personas: list[Persona] | None = None,
    candidate_resolver: Callable[..., StrategyCandidate] | None = None,
) -> SimulatorEvalResult:
    """Evaluate a policy using the LLM judge as the fixed Karpathy harness.

    For each user row:
      1. Convert to Persona (or use cached)
      2. simulate_action(row) → action_id (str) or (action_id, strategy_id) tuple
      3. candidate_resolver(action_id[, strategy_id]) → StrategyCandidate
      4. score_candidate_details(persona, candidate) → scores (LLM or fallback)
      5. Check alignment: is action in archetype's recommended list?
    """
    from cta_autoresearch.simulator import score_candidate_details

    scorer = score_candidate_details

    if personas is None:
        personas = [enriched_row_to_persona(r, i) for i, r in enumerate(rows)]

    resolve = candidate_resolver or action_to_candidate

    retention_scores = []
    revenue_scores = []
    trust_scores = []
    composite_scores = []
    alignment_hits = 0
    archetype_accum: dict[str, dict[str, list[float]]] = {}

    for row, persona in zip(rows, personas):
        # Get the action the bandit would choose
        action_result = simulate_action(row)

        # Handle both (action_id, strategy_id) tuples and plain action_id strings
        if isinstance(action_result, tuple):
            action_id = action_result[0]
            strategy_id = action_result[1] if len(action_result) > 1 else "s0"
        else:
            action_id = action_result
            strategy_id = "s0"

        # Map to StrategyCandidate (uses mutable dimensions if resolver provided)
        try:
            candidate = resolve(action_id, strategy_id)
        except TypeError:
            # Backward compat: resolver only accepts action_id
            candidate = resolve(action_id)

        # Score with the harness (heuristic or calibrated)
        scores = scorer(persona, candidate)

        retention_scores.append(scores["retention"])
        revenue_scores.append(scores["revenue"])
        trust_scores.append(scores["trust"])
        composite_scores.append(scores["score"])

        # Check alignment with archetype recommendations
        diagnosis = classify_user(row)
        if action_id in diagnosis.recommended_actions:
            alignment_hits += 1

        # Accumulate per-archetype
        arch_id = diagnosis.archetype_id
        if arch_id not in archetype_accum:
            archetype_accum[arch_id] = {
                "retention": [], "revenue": [], "trust": [],
                "composite": [], "alignment": [],
            }
        archetype_accum[arch_id]["retention"].append(scores["retention"])
        archetype_accum[arch_id]["revenue"].append(scores["revenue"])
        archetype_accum[arch_id]["trust"].append(scores["trust"])
        archetype_accum[arch_id]["composite"].append(scores["score"])
        archetype_accum[arch_id]["alignment"].append(
            1.0 if action_id in diagnosis.recommended_actions else 0.0
        )

    n = len(rows) or 1
    per_archetype = {}
    for arch_id, accum in archetype_accum.items():
        cnt = len(accum["composite"]) or 1
        per_archetype[arch_id] = {
            "count": float(cnt),
            "retention": sum(accum["retention"]) / cnt,
            "revenue": sum(accum["revenue"]) / cnt,
            "trust": sum(accum["trust"]) / cnt,
            "composite": sum(accum["composite"]) / cnt,
            "alignment": sum(accum["alignment"]) / cnt,
        }

    return SimulatorEvalResult(
        total_users=len(rows),
        expected_retention_score=sum(retention_scores) / n,
        expected_revenue_score=sum(revenue_scores) / n,
        trust_safety_score=sum(trust_scores) / n,
        composite_score=sum(composite_scores) / n,
        alignment_score=alignment_hits / n,
        per_archetype=per_archetype,
    )


# ── CLI for standalone analysis ───────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze cancel cohort and derive user archetypes")
    parser.add_argument("data_file", help="Path to enriched JSON (with 'rows' key) or CSV")
    parser.add_argument("--apply-to", type=str, default="", help="Policy data dir to seed priors into")
    parser.add_argument("--output", type=str, default="", help="Output JSON path for analysis")
    args = parser.parse_args()

    # Load data
    path = Path(args.data_file)
    if path.suffix == ".csv":
        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))
    else:
        payload = json.loads(path.read_text())
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload

    # Analyze
    analysis = analyze_cohort(rows)

    # Print summary
    print(f"Total users: {analysis.total_users}")
    print(f"Saveable: {analysis.saveable_count} ({analysis.saveable_pct:.0%})")
    print(f"Avg save potential: {analysis.avg_save_potential:.3f}")
    print()
    print("Archetype breakdown:")
    for arch_id, count in sorted(analysis.archetype_counts.items(), key=lambda x: -x[1]):
        arch = ARCHETYPES[arch_id]
        pct = analysis.archetype_pcts[arch_id]
        print(f"  {arch.label:25s}  {count:3d}  ({pct:.0%})  save_potential={arch.save_potential:.2f}")
        print(f"    root cause: {arch.root_cause[:90]}...")
        print(f"    best actions: {', '.join(arch.recommended_actions[:3])}")
    print()

    # Optionally apply to runtime
    if args.apply_to:
        from cta_autoresearch.cancel_policy import CancelPolicyRuntime
        runtime = CancelPolicyRuntime(Path(args.apply_to))
        result = apply_cohort_priors(runtime, analysis)
        print(f"Applied to policy: {result}")

    # Optionally save
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(analysis.to_dict(), indent=2))
        print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

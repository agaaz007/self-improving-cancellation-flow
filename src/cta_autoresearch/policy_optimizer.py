"""Self-improving policy optimizer — two-phase Karpathy autoresearch architecture.

Phase A (Strategy Evolution): LLM swarm proposes full 6D presentation strategies,
  simulator scores them, top K per action survive, bottom pruned.
  Triggered adaptively when bandit tuning stagnates.

Phase B (Bandit Tuning): Every iteration — mutate arm posteriors, exploration rate,
  discount cap, reason routing. Hierarchical Thompson Sampling: Level 1 picks action,
  Level 2 picks best strategy within action (ε-greedy on simulator scores offline,
  Thompson Sampling from real outcomes online).

Key invariant:
  Simulator scores → ranking/pruning only (strategy_pool[].mean_score)
  Bandit posteriors → real outcomes only (strategy_arms[].alpha/beta)
  These NEVER mix.

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │  Phase A: Strategy Evolution (on stagnation)             │
    │                                                          │
    │  LLM swarm ──→ propose 6D strategies (with perf feedback)│
    │       ↑         │ dedup filter                           │
    │       │    Simulator ──→ score × users → mean, std       │
    │       │         │ prune to top K by LCB                  │
    │       └──── top/worst performers fed back                │
    ├──────────────────────────────────────────────────────────┤
    │  Phase B: Bandit Tuning (every iteration)                │
    │                                                          │
    │  Level 1 Thompson → action                               │
    │  Level 2 ε-greedy → strategy (by simulator score)        │
    │  Mutate arm posteriors / config → eval → keep/discard    │
    └──────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import copy
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from cta_autoresearch.cancel_policy import (
    CancelContextV1,
    CancelOutcomeV1,
    CancelPolicyRuntime,
    DEFAULT_ACTIONS,
    PRIMARY_REASONS,
    TranscriptExtractor,
    _clamp,
    _normalize_reason,
)
from cta_autoresearch.user_model import (
    MUTABLE_DIMENSIONS,
    SimulatorEvalResult,
    build_candidate_with_overrides,
    classify_user,
    default_candidate_strategies,
    enriched_row_to_persona,
    simulator_eval,
)
from cta_autoresearch.strategy_policy import valid_candidate


# ── Strategy pool constants ───────────────────────────────────────────

MAX_STRATEGIES_PER_ACTION = 5
STAGNATION_THRESHOLD = 5
LCB_LAMBDA = 1.0
STRATEGY_EXPLORATION_RATE = 0.15
DEDUP_MIN_DIFF = 2
STRATEGIES_PER_GENERATION = 3


# ── Specialist agent roles (ported from Melbourne swarm) ──────────────

# Maps mutable dimension name -> role preference key
_DIM_TO_PREF_KEY: dict[str, str] = {
    "message_angle": "preferred_angles",
    "proof_style": "preferred_proof_styles",
    "personalization": "preferred_personalization",
    "contextual_grounding": "preferred_groundings",
    "creative_treatment": "preferred_treatments",
    "friction_reducer": "preferred_reducers",
}


AGENT_ROLES = [
    {
        "id": "retention_psychologist",
        "label": "Retention Psychologist",
        "thesis": "Use identity and momentum before the learner rationalizes churn.",
        "focus": "arm priors — boost actions that leverage sunk cost and habit loops",
        "preferred_angles": ["progress_reflection", "momentum_protection"],
        "preferred_groundings": ["progress_snapshot", "habit_streak"],
        "preferred_treatments": ["progress_thermometer", "coach_note", "plain_note"],
        "preferred_reducers": ["smart_resume_date", "none"],
        "preferred_proof_styles": ["personal_usage_signal", "quantified_outcome"],
        "preferred_personalization": ["behavioral", "contextual"],
    },
    {
        "id": "offer_economist",
        "label": "Offer Economist",
        "thesis": "Right-size the discount: too low gets ignored, too high destroys LTV.",
        "focus": "discount cap and discount arm priors — find the efficient frontier",
        "preferred_angles": ["cost_value_reframe", "flexibility_relief"],
        "preferred_groundings": ["unused_value", "pricing_context"],
        "preferred_treatments": ["plain_note", "before_after_frame"],
        "preferred_reducers": ["billing_date_shift", "prefilled_downgrade"],
        "preferred_proof_styles": ["quantified_outcome", "similar_user_story"],
        "preferred_personalization": ["contextual", "generic"],
    },
    {
        "id": "lifecycle_strategist",
        "label": "Lifecycle Strategist",
        "thesis": "Match the intervention to the user's lifecycle stage.",
        "focus": "context arms — strengthen reason+plan combos that match lifecycle signals",
        "preferred_angles": ["flexibility_relief", "empathetic_exit", "fresh_start_reset"],
        "preferred_groundings": ["recovery_moment", "unused_value"],
        "preferred_treatments": ["coach_note", "plain_note", "before_after_frame"],
        "preferred_reducers": ["single_tap_pause", "prefilled_downgrade"],
        "preferred_proof_styles": ["similar_user_story", "none"],
        "preferred_personalization": ["contextual", "behavioral"],
    },
    {
        "id": "support_concierge",
        "label": "Support Concierge",
        "thesis": "A warm handoff saves more users than any automated offer.",
        "focus": "concierge_recovery arm priors and exploration rate",
        "preferred_angles": ["mistake_recovery", "empathetic_exit", "fresh_start_reset"],
        "preferred_groundings": ["recovery_moment", "support_signal"],
        "preferred_treatments": ["coach_note", "plain_note"],
        "preferred_reducers": ["human_concierge", "smart_resume_date", "guided_reset"],
        "preferred_proof_styles": ["personal_usage_signal", "none"],
        "preferred_personalization": ["behavioral", "contextual"],
    },
    {
        "id": "trust_guardian",
        "label": "Trust Guardian",
        "thesis": "Aggressive saves erode trust. Protect long-term brand health.",
        "focus": "reason routing denylists — block actions that feel manipulative for certain reasons",
        "preferred_angles": ["empathetic_exit", "flexibility_relief", "feature_unlock"],
        "preferred_groundings": ["generic", "unused_value"],
        "preferred_treatments": ["plain_note", "before_after_frame"],
        "preferred_reducers": ["none", "prefilled_downgrade"],
        "preferred_proof_styles": ["none", "similar_user_story"],
        "preferred_personalization": ["generic", "contextual"],
    },
    {
        "id": "experiment_operator",
        "label": "Experiment Operator",
        "thesis": "Statistical power requires the right exploration/holdout balance.",
        "focus": "exploration rate and holdout rate tuning based on data volume",
    },
    {
        "id": "product_storyteller",
        "label": "Product Storyteller",
        "thesis": "Show value already created so the save attempt feels earned, not desperate.",
        "focus": "feature awareness and value realization through product-led presentation",
        "preferred_angles": ["feature_unlock", "outcome_proof", "progress_reflection"],
        "preferred_groundings": ["unused_value", "progress_snapshot"],
        "preferred_treatments": ["feature_collage", "progress_thermometer"],
        "preferred_reducers": ["smart_resume_date", "none"],
        "preferred_proof_styles": ["quantified_outcome", "personal_usage_signal"],
        "preferred_personalization": ["behavioral", "contextual"],
    },
    {
        "id": "deadline_operator",
        "label": "Deadline Operator",
        "thesis": "When urgency is real, frame the next few days as the point of maximum leverage.",
        "focus": "deadline-aware presentation for exam/goal-driven users",
        "preferred_angles": ["goal_deadline", "momentum_protection", "progress_reflection"],
        "preferred_groundings": ["deadline_countdown", "deadline_pressure", "progress_snapshot"],
        "preferred_treatments": ["before_after_frame", "progress_thermometer", "study_timeline"],
        "preferred_reducers": ["single_tap_pause", "smart_resume_date", "none"],
        "preferred_proof_styles": ["quantified_outcome", "personal_usage_signal"],
        "preferred_personalization": ["contextual", "behavioral"],
    },
    {
        "id": "winback_researcher",
        "label": "Win-Back Researcher",
        "thesis": "Dormant and tired users need a comeback path, not a harder sell.",
        "focus": "re-engagement strategies for dormant or fatigued users",
        "preferred_angles": ["fresh_start_reset", "mistake_recovery", "empathetic_exit"],
        "preferred_groundings": ["recovery_moment", "comeback_window", "unused_value"],
        "preferred_treatments": ["coach_note", "plain_note"],
        "preferred_reducers": ["smart_resume_date", "single_tap_pause", "human_concierge"],
        "preferred_proof_styles": ["similar_user_story", "none"],
        "preferred_personalization": ["contextual", "generic"],
    },
]


def _build_agent_prompt(
    role: dict[str, str],
    state_summary: dict[str, Any],
    history_summary: str,
    available_actions: list[str],
) -> str:
    """Build the LLM prompt for one specialist agent iteration (Phase B: bandit tuning)."""
    return f"""You are a specialist agent in a self-improving cancel-policy optimization loop.

Role: {role['label']}
Thesis: {role['thesis']}
Focus area: {role['focus']}

You are looking at the current policy state and past optimization history.
Propose ONE specific mutation to improve save_rate without harming trust.

NOTE: Strategy presentation (HOW actions are displayed) is managed separately by
the strategy evolution system. Focus on bandit parameters: which actions to favor,
exploration balance, and routing rules.

## Current Policy State
{json.dumps(state_summary, indent=2)}

## Past Optimization History (most recent first)
{history_summary}

## Available Actions
{json.dumps(available_actions)}

## Mutation Types You Can Propose
1. arm_priors: Adjust alpha/beta for a specific action. Specify action_id, alpha_delta, beta_delta.
2. context_arms: Adjust alpha/beta for a reason|plan|action combo. Specify context_key, alpha_delta, beta_delta.
3. exploration_rate: Change exploration rate. Specify new_rate (0.02 to 0.35).
4. discount_cap: Change discount_cap_30d. Specify new_cap (0 to 5).
5. reason_routing: Propose a deny rule. Specify reason and action_id to block.

Respond with JSON only. No markdown, no explanation outside the JSON.
{{
  "mutation_type": "<one of the 5 types above>",
  "parameters": {{ ... type-specific parameters ... }},
  "rationale": "<one sentence explaining why this should improve save_rate>"
}}"""


def _parse_llm_mutation(
    response_text: str,
    state: dict[str, Any],
    rng: random.Random,
    intensity: float,
) -> tuple[str, dict[str, Any], str]:
    """Parse LLM JSON response into a mutation type, mutated state, and description."""
    # Extract JSON from response (handle markdown fences)
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    proposal = json.loads(text)
    mutation_type = proposal["mutation_type"]
    params = proposal.get("parameters", {})
    rationale = proposal.get("rationale", "")

    state = copy.deepcopy(state)

    if mutation_type == "arm_priors":
        action_id = params["action_id"]
        arms = state.get("arms_global", {})
        if action_id in arms:
            arms[action_id]["alpha"] = float(arms[action_id]["alpha"]) + float(params.get("alpha_delta", 0)) * intensity
            arms[action_id]["beta"] = float(arms[action_id]["beta"]) + float(params.get("beta_delta", 0)) * intensity
            state["arms_global"] = arms
        desc = f"[LLM] {action_id} alpha+={params.get('alpha_delta', 0)}, beta+={params.get('beta_delta', 0)} — {rationale}"

    elif mutation_type == "context_arms":
        context_key = params["context_key"]
        arms = state.get("arms_context", {})
        if context_key in arms:
            arms[context_key]["alpha"] = float(arms[context_key]["alpha"]) + float(params.get("alpha_delta", 0)) * intensity
            arms[context_key]["beta"] = float(arms[context_key]["beta"]) + float(params.get("beta_delta", 0)) * intensity
        else:
            arms[context_key] = {
                "alpha": 1.0 + float(params.get("alpha_delta", 0)) * intensity,
                "beta": 1.0 + float(params.get("beta_delta", 0)) * intensity,
                "impressions": 0,
                "outcomes": 0,
            }
        state["arms_context"] = arms
        desc = f"[LLM] context {context_key} alpha+={params.get('alpha_delta', 0)}, beta+={params.get('beta_delta', 0)} — {rationale}"

    elif mutation_type == "exploration_rate":
        new_rate = _clamp(float(params["new_rate"]), low=0.02, high=0.35)
        config = state.get("config", {})
        old = float(config.get("exploration_rate", 0.15))
        config["exploration_rate"] = round(new_rate, 4)
        state["config"] = config
        desc = f"[LLM] exploration_rate {old:.3f} -> {new_rate:.3f} — {rationale}"

    elif mutation_type == "discount_cap":
        new_cap = max(0, min(5, int(params["new_cap"])))
        config = state.get("config", {})
        old = int(config.get("discount_cap_30d", 1))
        config["discount_cap_30d"] = new_cap
        state["config"] = config
        desc = f"[LLM] discount_cap {old} -> {new_cap} — {rationale}"

    elif mutation_type == "reason_routing":
        denylist = state.get("_denylist_proposals", [])
        denylist.append({
            "reason": params["reason"],
            "action_id": params["action_id"],
            "win_rate": 0.0,
            "impressions": 0,
        })
        state["_denylist_proposals"] = denylist
        desc = f"[LLM] deny {params['action_id']} for {params['reason']} — {rationale}"

    else:
        raise ValueError(f"Unknown mutation_type: {mutation_type}")

    return mutation_type, state, desc


def _summarize_state_for_llm(state: dict[str, Any]) -> dict[str, Any]:
    """Build a compact summary of policy state for the LLM prompt."""
    summary: dict[str, Any] = {}

    # Global arm stats
    arms = state.get("arms_global", {})
    arm_table = {}
    for action_id, arm in arms.items():
        imps = float(arm.get("impressions", 0))
        outs = float(arm.get("outcomes", 0))
        arm_table[action_id] = {
            "alpha": round(float(arm.get("alpha", 1)), 2),
            "beta": round(float(arm.get("beta", 1)), 2),
            "impressions": int(imps),
            "win_rate": round(outs / imps, 3) if imps > 0 else None,
        }
    summary["arms_global"] = arm_table

    # Top context arms (limit to 15 most-sampled)
    ctx_arms = state.get("arms_context", {})
    ctx_list = []
    for key, arm in ctx_arms.items():
        imps = float(arm.get("impressions", 0))
        outs = float(arm.get("outcomes", 0))
        ctx_list.append((key, imps, outs))
    ctx_list.sort(key=lambda x: x[1], reverse=True)
    summary["top_context_arms"] = {
        k: {"impressions": int(i), "win_rate": round(o / i, 3) if i > 0 else None}
        for k, i, o in ctx_list[:15]
    }

    # Config
    config = state.get("config", {})
    summary["config"] = {
        "exploration_rate": config.get("exploration_rate", 0.15),
        "holdout_rate": config.get("holdout_rate", 0.10),
        "discount_cap_30d": config.get("discount_cap_30d", 1),
    }

    # Strategy pool — show best strategy per action with score
    pool = state.get("strategy_pool", {})
    if pool:
        strat_summary = {}
        for action_id, strategies in pool.items():
            if not strategies:
                continue
            best_sid, best = max(strategies.items(), key=lambda x: x[1].get("mean_score", 0.0))
            strat_summary[action_id] = {
                "best_strategy": best_sid,
                "mean_score": round(best.get("mean_score", 0.0), 4),
                "n_strategies": len(strategies),
            }
        summary["strategy_pool"] = strat_summary

    # Metrics
    metrics = state.get("metrics", {})
    summary["metrics"] = {
        "decisions": int(metrics.get("decisions", 0)),
        "outcomes": int(metrics.get("outcomes", 0)),
    }

    return summary


def _summarize_history_for_llm(history: list, max_rows: int = 10) -> str:
    """Build a compact text summary of recent optimization results."""
    if not history:
        return "No previous iterations."

    recent = history[-max_rows:]
    lines = []
    for r in reversed(recent):
        status_icon = "+" if r.status == "keep" else "-" if r.status == "discard" else "!"
        lines.append(
            f"  [{status_icon}] run {r.run_id}: {r.mutation_type} | "
            f"save_lift={r.save_lift:+.4f} reward_lift={r.reward_lift:+.4f} "
            f"regression={'pass' if r.regression_pass else 'FAIL'} | {r.mutation_description[:80]}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class RunResult:
    """Immutable record of a single optimization run."""

    run_id: int
    mutation_type: str
    mutation_description: str
    save_rate: float
    average_reward: float
    baseline_save_rate: float
    baseline_average_reward: float
    save_lift: float
    reward_lift: float
    regression_pass: bool
    status: str  # "keep" | "discard" | "crash"
    duration_s: float
    policy_snapshot: dict[str, Any]
    alignment_score: float = 0.0
    trust_score: float = 0.0

    def to_tsv_row(self) -> str:
        return "\t".join([
            str(self.run_id),
            f"{self.save_rate:.4f}",
            f"{self.average_reward:.4f}",
            f"{self.save_lift:.4f}",
            f"{self.reward_lift:.4f}",
            f"{self.alignment_score:.4f}",
            f"{self.trust_score:.4f}",
            "pass" if self.regression_pass else "fail",
            self.status,
            self.mutation_type,
            self.mutation_description,
        ])


RESULTS_HEADER = "run_id\tsave_rate\taverage_reward\tsave_lift\treward_lift\talignment\ttrust\tregression\tstatus\tmutation_type\tdescription"


# ── Mutation strategies ────────────────────────────────────────────────


def _mutate_arm_priors(
    state: dict[str, Any],
    rng: random.Random,
    intensity: float = 1.0,
) -> tuple[dict[str, Any], str]:
    """Nudge Thompson Sampling arm priors based on observed performance."""
    state = copy.deepcopy(state)
    arms_global = state.get("arms_global", {})
    if not arms_global:
        return state, "no arms to mutate"

    # Find the best and worst performing arms by win rate
    arm_stats = []
    for action_id, arm in arms_global.items():
        impressions = float(arm.get("impressions", 0))
        outcomes = float(arm.get("outcomes", 0))
        if impressions > 0:
            win_rate = outcomes / impressions
        else:
            win_rate = 0.5
        arm_stats.append((action_id, win_rate, impressions))

    arm_stats.sort(key=lambda x: x[1], reverse=True)
    if len(arm_stats) < 2:
        return state, "not enough arms"

    best_id = arm_stats[0][0]
    worst_id = arm_stats[-1][0]

    # Boost the best arm's alpha, penalize the worst arm's beta
    boost = rng.uniform(0.5, 2.0) * intensity
    arms_global[best_id]["alpha"] = float(arms_global[best_id]["alpha"]) + boost
    arms_global[worst_id]["beta"] = float(arms_global[worst_id]["beta"]) + boost * 0.6

    state["arms_global"] = arms_global
    description = (
        f"boost {best_id} alpha +{boost:.2f}, "
        f"penalize {worst_id} beta +{boost * 0.6:.2f}"
    )
    return state, description


def _mutate_context_arms(
    state: dict[str, Any],
    rng: random.Random,
    intensity: float = 1.0,
) -> tuple[dict[str, Any], str]:
    """Strengthen context-specific arm priors where a reason+action combo shows signal."""
    state = copy.deepcopy(state)
    arms_context = state.get("arms_context", {})
    if not arms_context:
        return state, "no context arms to mutate"

    # Find context arms with enough data and strong signal
    candidates = []
    for context_key, arm in arms_context.items():
        impressions = float(arm.get("impressions", 0))
        outcomes = float(arm.get("outcomes", 0))
        if impressions >= 3:
            win_rate = outcomes / impressions
            candidates.append((context_key, win_rate, impressions))

    if not candidates:
        return state, "no context arms with sufficient data"

    candidates.sort(key=lambda x: x[1], reverse=True)
    # Boost the top performer and dampen the bottom
    top_key = candidates[0][0]
    boost = rng.uniform(0.3, 1.5) * intensity
    arms_context[top_key]["alpha"] = float(arms_context[top_key]["alpha"]) + boost

    description = f"boost context arm {top_key} alpha +{boost:.2f}"

    if len(candidates) >= 2:
        bottom_key = candidates[-1][0]
        arms_context[bottom_key]["beta"] = float(arms_context[bottom_key]["beta"]) + boost * 0.5
        description += f", penalize {bottom_key} beta +{boost * 0.5:.2f}"

    state["arms_context"] = arms_context
    return state, description


def _mutate_exploration_rate(
    state: dict[str, Any],
    rng: random.Random,
    intensity: float = 1.0,
) -> tuple[dict[str, Any], str]:
    """Adjust exploration rate. Early: explore more. Later: exploit more."""
    state = copy.deepcopy(state)
    config = state.get("config", {})
    current = float(config.get("exploration_rate", 0.15))

    # Read total decisions to decide direction
    metrics = state.get("metrics", {})
    total_decisions = int(metrics.get("decisions", 0))

    if total_decisions < 100:
        # Still early, maybe explore more
        delta = rng.uniform(0.02, 0.08) * intensity
    else:
        # Enough data, lean toward exploiting
        delta = rng.uniform(-0.06, 0.02) * intensity

    new_rate = _clamp(current + delta, low=0.02, high=0.35)
    config["exploration_rate"] = round(new_rate, 4)
    state["config"] = config

    direction = "increase" if new_rate > current else "decrease"
    return state, f"{direction} exploration_rate {current:.3f} -> {new_rate:.3f}"


def _mutate_discount_cap(
    state: dict[str, Any],
    rng: random.Random,
    intensity: float = 1.0,
) -> tuple[dict[str, Any], str]:
    """Tighten or loosen the discount exposure cap."""
    state = copy.deepcopy(state)
    config = state.get("config", {})
    current = int(config.get("discount_cap_30d", 1))

    delta = rng.choice([-1, 0, 1])
    if intensity > 1.5:
        delta = rng.choice([-1, -1, 0, 1, 2])
    new_cap = max(0, min(5, current + delta))
    config["discount_cap_30d"] = new_cap
    state["config"] = config

    return state, f"discount_cap_30d {current} -> {new_cap}"


def _mutate_reason_routing(
    state: dict[str, Any],
    rng: random.Random,
    intensity: float = 1.0,
) -> tuple[dict[str, Any], str]:
    """Propose adding or removing a reason->action deny rule based on outcome data."""
    state = copy.deepcopy(state)
    arms_context = state.get("arms_context", {})

    # Parse context keys to find bad reason+action combos
    bad_combos = []
    for context_key, arm in arms_context.items():
        parts = context_key.split("|")
        if len(parts) != 3:
            continue
        reason, plan, action_id = parts
        impressions = float(arm.get("impressions", 0))
        outcomes = float(arm.get("outcomes", 0))
        if impressions >= 5:
            win_rate = outcomes / impressions
            if win_rate < 0.25:
                bad_combos.append((reason, action_id, win_rate, impressions))

    if not bad_combos:
        return state, "no bad reason+action combos found"

    bad_combos.sort(key=lambda x: x[2])
    reason, action_id, win_rate, impressions = bad_combos[0]

    # Store the proposed denylist change in state metadata
    denylist_updates = state.get("_denylist_proposals", [])
    denylist_updates.append({
        "reason": reason,
        "action_id": action_id,
        "win_rate": round(win_rate, 4),
        "impressions": int(impressions),
    })
    state["_denylist_proposals"] = denylist_updates

    return state, f"propose deny {action_id} for reason={reason} (win_rate={win_rate:.2f}, n={int(impressions)})"


MUTATION_STRATEGIES = [
    ("arm_priors", _mutate_arm_priors),
    ("context_arms", _mutate_context_arms),
    ("exploration_rate", _mutate_exploration_rate),
    ("discount_cap", _mutate_discount_cap),
    ("reason_routing", _mutate_reason_routing),
]


# ── Strategy pool helpers ─────────────────────────────────────────────


def _default_strategy_pool() -> dict[str, dict[str, dict[str, Any]]]:
    """Seed the strategy pool with one default strategy per action."""
    strategies = default_candidate_strategies()
    pool: dict[str, dict[str, dict[str, Any]]] = {}
    for action_id, dims in strategies.items():
        pool[action_id] = {
            "s0": {
                "dims": dict(dims),
                "mean_score": 0.0,
                "score_std": 0.0,
                "n_evals": 0,
                "generation": 0,
                "source": "default",
            }
        }
    return pool


def _default_strategy_arms(pool: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Create flat prior (1.0, 1.0) for each action:strategy in the pool."""
    arms: dict[str, dict[str, float]] = {}
    for action_id, strategies in pool.items():
        for sid in strategies:
            arms[f"{action_id}:{sid}"] = {"alpha": 1.0, "beta": 1.0}
    return arms


def _lcb(strat: dict[str, Any], lam: float = 1.0) -> float:
    """Lower confidence bound: mean - λ·(std/√n). Used for pruning."""
    mean = float(strat.get("mean_score", 0.0))
    std = float(strat.get("score_std", 0.0))
    n = max(1, int(strat.get("n_evals", 1)))
    return mean - lam * (std / math.sqrt(n))


# ── Synthetic traffic generator ────────────────────────────────────────


def generate_synthetic_traffic(
    runtime: CancelPolicyRuntime,
    count: int = 200,
    seed: int = 42,
    save_probability: float = 0.45,
) -> None:
    """Generate synthetic decide+outcome pairs to bootstrap the eval harness.

    This gives the optimizer something to work with when there's no real
    production data yet. The synthetic traffic is intentionally noisy so the
    optimizer has room to find improvements.
    """
    rng = random.Random(seed)
    reasons = list(PRIMARY_REASONS)
    plans = ["starter", "super_learner", "free"]

    for i in range(count):
        reason = rng.choice(reasons)
        plan = rng.choice(plans)
        engagement = round(rng.uniform(0.1, 0.9), 2)

        context = CancelContextV1.from_dict({
            "session_id": f"syn_{seed}_{i}",
            "user_id_hash": f"user_{seed}_{i}",
            "timestamp": time.time() - rng.randint(0, 86400 * 30),
            "plan_tier": plan,
            "tenure_days": rng.randint(7, 600),
            "engagement_7d": engagement,
            "engagement_30d": round(engagement * rng.uniform(0.8, 1.2), 2),
            "prior_cancel_attempts_30d": rng.choice([0, 0, 0, 1, 2]),
            "discount_exposures_30d": rng.choice([0, 0, 0, 1]),
            "transcript_extraction": {
                "primary_reason": reason,
                "secondary_reasons": [],
                "intent_strength": round(rng.uniform(0.3, 0.95), 2),
                "save_openness": round(rng.uniform(0.1, 0.8), 2),
                "frustration_level": round(rng.uniform(0.1, 0.7), 2),
                "trust_risk": round(rng.uniform(0.1, 0.5), 2),
                "billing_confusion_flag": reason == "billing_confusion",
                "competitor_mentions": [],
                "feature_requests": [],
                "bug_signals": [],
                "summary": f"synthetic {reason}",
                "confidence": 0.7,
                "extractor_version": "synthetic-v1",
            },
        })

        decision = runtime.decide(context)

        # Simulate outcomes with noise. Better actions save more often.
        action = runtime.actions.get(decision.action_id)
        action_boost = 0.0
        if action:
            if action.offer_kind == "pause":
                action_boost = 0.12
            elif action.offer_kind == "downgrade":
                action_boost = 0.08
            elif action.is_discount:
                action_boost = 0.15
            elif action.offer_kind == "support":
                action_boost = 0.06
            elif action.offer_kind == "extension":
                action_boost = 0.10

        # Holdout always gets control, so lower base save rate
        base_p = save_probability + action_boost
        if decision.holdout_flag:
            base_p = save_probability * 0.7

        saved = rng.random() < base_p
        support_escalation = rng.random() < 0.05
        complaint = rng.random() < 0.02

        runtime.record_outcome(CancelOutcomeV1.from_dict({
            "decision_id": decision.decision_id,
            "session_id": context.session_id,
            "saved_flag": saved,
            "cancel_completed_flag": not saved,
            "support_escalation_flag": support_escalation,
            "complaint_flag": complaint,
        }))


# ── The optimizer loop ─────────────────────────────────────────────────


class PolicyOptimizer:
    """Two-phase autoresearch optimizer for the cancel policy.

    Phase A (Strategy Evolution): LLM swarm proposes full 6D presentation
    strategies, simulator scores them, top K survive. Triggered on stagnation.

    Phase B (Bandit Tuning): Every iteration — mutate arm posteriors, exploration
    rate, discount cap, reason routing. Hierarchical Thompson Sampling picks
    action (Level 1) then strategy within action (Level 2, ε-greedy on sim scores).

    Key invariant: simulator scores for ranking only, bandit posteriors for real
    outcomes only — never mixed.
    """

    def __init__(
        self,
        runtime: CancelPolicyRuntime,
        output_dir: Path,
        *,
        seed: int = 7,
        intensity: float = 1.0,
        min_samples_for_eval: int = 40,
        mode: str = "random",
        openai_model: str = "gpt-4o-mini",
        eval_cohort_path: str | Path = "",
    ) -> None:
        self.runtime = runtime
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.seed = seed
        self.intensity = intensity
        self.min_samples_for_eval = min_samples_for_eval
        self.mode = mode
        self.openai_model = openai_model
        self.results_path = self.output_dir / "results.tsv"
        self.history: list[RunResult] = []
        self._best_save_rate = 0.0
        self._best_reward = 0.0
        self._openai_client: Any = None
        self._role_index = 0
        self._eval_cohort: list[dict] | None = None
        self._eval_personas: list[Any] | None = None
        if eval_cohort_path:
            self._load_eval_cohort(Path(eval_cohort_path))
        # Initialize strategy pool (migrate from old candidate_strategies if needed)
        self._ensure_strategy_pool()

    def _ensure_strategy_pool(self) -> None:
        """Initialize or migrate strategy pool in runtime state."""
        state = self.runtime.state
        if "strategy_pool" not in state:
            # Migrate from old candidate_strategies or build fresh
            old = state.get("candidate_strategies")
            if old:
                pool: dict[str, dict[str, dict[str, Any]]] = {}
                for action_id, dims in old.items():
                    pool[action_id] = {
                        "s0": {
                            "dims": dict(dims),
                            "mean_score": 0.0,
                            "score_std": 0.0,
                            "n_evals": 0,
                            "generation": 0,
                            "source": "default",
                        }
                    }
                state["strategy_pool"] = pool
            else:
                state["strategy_pool"] = _default_strategy_pool()
        if "strategy_arms" not in state:
            state["strategy_arms"] = _default_strategy_arms(state["strategy_pool"])
        if "generation_meta" not in state:
            state["generation_meta"] = {
                "last_generation": 0,
                "next_strategy_id": 1,
                "stagnation_count": 0,
                "stagnation_threshold": STAGNATION_THRESHOLD,
                "total_generations": 0,
            }

    def _load_eval_cohort(self, path: Path) -> None:
        """Load eval cohort rows and pre-compute personas."""
        payload = json.loads(path.read_text())
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        self._eval_cohort = rows
        self._eval_personas = [
            enriched_row_to_persona(r, i) for i, r in enumerate(rows)
        ]

    # ── Hierarchical action + strategy selection ──────────────────────

    def _simulate_action(self, row: dict[str, Any]) -> tuple[str, str]:
        """Hierarchical Thompson Sampling: Level 1 picks action, Level 2 picks strategy.

        Level 1: Pure Thompson Sampling exploitation on action posteriors.
        Level 2: ε-greedy on simulator scores (offline mode).

        Returns (action_id, strategy_id) tuple.
        """
        reason = str(row.get("primary_reason", "other")).lower().strip()
        plan = str(row.get("plan_tier", "unknown")).lower().strip()
        user_hash = str(row.get("user_id_hash", ""))

        # Use deterministic seed per user for reproducible draws
        user_seed = hash(user_hash) if user_hash else id(row)
        rng = random.Random(user_seed ^ self.seed)

        # Level 1: Thompson Sample for action
        best_action = "control_empathic_exit"
        best_draw = -1.0

        for action_id in self.runtime.actions:
            context_key = f"{reason}|{plan}|{action_id}"
            arms_global = self.runtime.state.get("arms_global", {})
            arms_context = self.runtime.state.get("arms_context", {})

            g = arms_global.get(action_id, {})
            alpha_g = float(g.get("alpha", 1.0))
            beta_g = float(g.get("beta", 1.0))

            c = arms_context.get(context_key, {})
            alpha_c = float(c.get("alpha", 1.0))
            beta_c = float(c.get("beta", 1.0))

            alpha = max(0.01, alpha_g + alpha_c - 1.0)
            beta = max(0.01, beta_g + beta_c - 1.0)

            draw = rng.betavariate(alpha, beta)
            if draw > best_draw:
                best_draw = draw
                best_action = action_id

        # Level 2: Pick strategy within action (ε-greedy on simulator scores)
        pool = self.runtime.state.get("strategy_pool", {})
        strategies = pool.get(best_action, {})
        if not strategies:
            return best_action, "s0"

        strategy_ids = list(strategies.keys())
        if len(strategy_ids) == 1:
            return best_action, strategy_ids[0]

        eps = float(self.runtime.state.get("config", {}).get(
            "strategy_exploration_rate", STRATEGY_EXPLORATION_RATE,
        ))
        if rng.random() < eps:
            # Explore: random strategy
            return best_action, rng.choice(strategy_ids)
        else:
            # Exploit: best by simulator score
            best_sid = max(strategy_ids, key=lambda s: strategies[s].get("mean_score", 0.0))
            return best_action, best_sid

    def _resolve_candidate(self, action_id: str, strategy_id: str = "s0") -> Any:
        """Build a StrategyCandidate from strategy pool dims."""
        pool = self.runtime.state.get("strategy_pool", {})
        strategies = pool.get(action_id, {})
        strat = strategies.get(strategy_id)
        if strat and "dims" in strat:
            return build_candidate_with_overrides(action_id, {action_id: strat["dims"]})
        return build_candidate_with_overrides(action_id)

    # ── Evaluation ────────────────────────────────────────────────────

    def _evaluate_strategy(self, action_id: str, dims: dict[str, str]) -> dict[str, float]:
        """Score one strategy against the eval cohort. Returns mean, std, n."""
        from cta_autoresearch.simulator import score_candidate_details

        if not self._eval_personas:
            return {"mean": 0.0, "std": 0.0, "n": 0}

        candidate = build_candidate_with_overrides(action_id, {action_id: dims})
        scores = []
        for persona in self._eval_personas:
            details = score_candidate_details(persona, candidate)
            scores.append(details["score"])

        n = len(scores)
        mean = statistics.mean(scores) if scores else 0.0
        std = statistics.stdev(scores) if n > 1 else 0.0
        return {"mean": mean, "std": std, "n": n}

    def _baseline_metrics(self) -> dict[str, float]:
        """Get current policy performance using hierarchical eval."""
        if self._eval_cohort is not None:
            result = simulator_eval(
                self._eval_cohort,
                self._simulate_action,
                personas=self._eval_personas,
                candidate_resolver=self._resolve_candidate,
            )
            return {
                "save_rate": result.expected_retention_score,
                "average_reward": result.composite_score,
                "rows": result.total_users,
                "alignment_score": result.alignment_score,
                "trust_score": result.trust_safety_score,
            }

        replay = self.runtime.replay()
        return {
            "save_rate": float(replay.get("save_rate", 0.0)),
            "average_reward": float(replay.get("average_reward", 0.0)),
            "rows": int(replay.get("rows", 0)),
            "alignment_score": 0.0,
            "trust_score": 0.0,
        }

    # ── Phase A: Strategy Evolution ───────────────────────────────────

    def _is_duplicate(self, action_id: str, new_dims: dict[str, str]) -> bool:
        """Reject if fewer than DEDUP_MIN_DIFF dimensions differ from any existing."""
        pool = self.runtime.state.get("strategy_pool", {})
        for sid, existing in pool.get(action_id, {}).items():
            diffs = sum(
                1 for d in MUTABLE_DIMENSIONS
                if new_dims.get(d) != existing.get("dims", {}).get(d)
            )
            if diffs < DEDUP_MIN_DIFF:
                return True
        return False

    def _propose_strategies_heuristic(
        self, action_id: str, count: int,
    ) -> list[dict[str, str]]:
        """Generate strategy proposals using role-guided random exploration."""
        proposals: list[dict[str, str]] = []
        for _ in range(count * 3):  # over-generate, filter dupes
            if len(proposals) >= count:
                break
            role = self.rng.choice(AGENT_ROLES)
            dims: dict[str, str] = {}
            for dim_name, catalog in MUTABLE_DIMENSIONS.items():
                pref_key = _DIM_TO_PREF_KEY.get(dim_name, "")
                preferred = role.get(pref_key, [])
                # 70% from role preferences, 30% from full catalog
                if preferred and self.rng.random() < 0.7:
                    dims[dim_name] = self.rng.choice(preferred)
                else:
                    dims[dim_name] = self.rng.choice(catalog)
            if not self._is_duplicate(action_id, dims):
                proposals.append(dims)
        return proposals

    def _generation_round(self) -> list[str]:
        """Phase A: generate new strategies, evaluate all, prune to top K."""
        if not self._eval_personas:
            return ["generation skipped: no eval cohort"]

        logs: list[str] = []
        pool = self.runtime.state.get("strategy_pool", {})
        strategy_arms = self.runtime.state.get("strategy_arms", {})
        gen_meta = self.runtime.state.get("generation_meta", {})
        next_id = int(gen_meta.get("next_strategy_id", 1))

        # 1. Generate new strategies for each action
        for action_id in list(pool.keys()):
            new_strats = self._propose_strategies_heuristic(action_id, STRATEGIES_PER_GENERATION)
            for dims in new_strats:
                sid = f"s{next_id}"
                next_id += 1
                pool[action_id][sid] = {
                    "dims": dims,
                    "mean_score": 0.0,
                    "score_std": 0.0,
                    "n_evals": 0,
                    "generation": gen_meta.get("total_generations", 0) + 1,
                    "source": "heuristic",
                }
                strategy_arms[f"{action_id}:{sid}"] = {"alpha": 1.0, "beta": 1.0}
                logs.append(f"generated {action_id}:{sid}")

        # 2. Evaluate ALL strategies
        for action_id, strategies in pool.items():
            for sid, strat in strategies.items():
                result = self._evaluate_strategy(action_id, strat["dims"])
                strat["mean_score"] = result["mean"]
                strat["score_std"] = result["std"]
                strat["n_evals"] = result["n"]

        # 3. Prune to top K per action using LCB
        lcb_lambda = float(self.runtime.state.get("config", {}).get("lcb_lambda", LCB_LAMBDA))
        for action_id in list(pool.keys()):
            strategies = pool[action_id]
            if len(strategies) <= MAX_STRATEGIES_PER_ACTION:
                continue

            # Rank by lower confidence bound
            ranked = sorted(
                strategies.items(),
                key=lambda x: _lcb(x[1], lcb_lambda),
                reverse=True,
            )
            kept = dict(ranked[:MAX_STRATEGIES_PER_ACTION])
            pruned_ids = [sid for sid, _ in ranked[MAX_STRATEGIES_PER_ACTION:]]

            pool[action_id] = kept
            for sid in pruned_ids:
                strategy_arms.pop(f"{action_id}:{sid}", None)
            if pruned_ids:
                logs.append(f"pruned {action_id}: {pruned_ids}")

        # 4. Update metadata
        gen_meta["next_strategy_id"] = next_id
        gen_meta["total_generations"] = gen_meta.get("total_generations", 0) + 1
        gen_meta["stagnation_count"] = 0

        self.runtime.state["strategy_pool"] = pool
        self.runtime.state["strategy_arms"] = strategy_arms
        self.runtime.state["generation_meta"] = gen_meta
        self.runtime._persist_state()

        logs.append(f"generation {gen_meta['total_generations']} complete")
        return logs

    # ── State management ──────────────────────────────────────────────

    def _snapshot_state(self) -> dict[str, Any]:
        """Deep copy the current policy state."""
        return copy.deepcopy(self.runtime.state)

    def _restore_state(self, snapshot: dict[str, Any]) -> None:
        """Rollback policy state to a previous snapshot."""
        self.runtime.state = copy.deepcopy(snapshot)
        self.runtime._persist_state()

    def _apply_state(self, new_state: dict[str, Any]) -> None:
        """Apply a mutated state to the runtime (Phase B bandit tuning only)."""
        self.runtime.state["arms_global"] = copy.deepcopy(new_state.get("arms_global", {}))
        self.runtime.state["arms_context"] = copy.deepcopy(new_state.get("arms_context", {}))
        if "config" in new_state:
            config = new_state["config"]
            self.runtime.exploration_rate = float(config.get("exploration_rate", self.runtime.exploration_rate))
            self.runtime.holdout_rate = float(config.get("holdout_rate", self.runtime.holdout_rate))
            self.runtime.discount_cap_30d = int(config.get("discount_cap_30d", self.runtime.discount_cap_30d))
            self.runtime.state["config"] = copy.deepcopy(config)
        # strategy_pool, strategy_arms, generation_meta are Phase A's responsibility
        # and are NOT mutated by Phase B bandit tuning
        self.runtime._persist_state()

    def _log_result(self, result: RunResult) -> None:
        """Append to results.tsv in the autoresearch format."""
        if not self.results_path.exists():
            self.results_path.write_text(RESULTS_HEADER + "\n")
        with self.results_path.open("a") as f:
            f.write(result.to_tsv_row() + "\n")

    def _get_openai_client(self) -> Any:
        """Lazy-init OpenAI client."""
        if self._openai_client is None:
            if OpenAI is None:
                raise RuntimeError("openai package not installed — run: pip install openai")
            self._openai_client = OpenAI()
        return self._openai_client

    def _propose_random(
        self, snapshot: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str]:
        """Random mutation selection (original mode)."""
        strategy_name, strategy_fn = self.rng.choice(MUTATION_STRATEGIES)
        mutated_state, description = strategy_fn(snapshot, self.rng, self.intensity)
        return strategy_name, mutated_state, description

    def _propose_agent(
        self, snapshot: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str]:
        """LLM-driven mutation proposal from a rotating specialist role."""
        # Rotate through specialist roles
        role = AGENT_ROLES[self._role_index % len(AGENT_ROLES)]
        self._role_index += 1

        # Build context for the LLM
        state_summary = _summarize_state_for_llm(snapshot)
        history_summary = _summarize_history_for_llm(self.history)
        available_actions = list(self.runtime.actions.keys())

        prompt = _build_agent_prompt(role, state_summary, history_summary, available_actions)

        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=500,
        )
        response_text = response.choices[0].message.content or ""

        mutation_type, mutated_state, description = _parse_llm_mutation(
            response_text, snapshot, self.rng, self.intensity,
        )
        description = f"[{role['label']}] {description}"
        return mutation_type, mutated_state, description

    def run_one(self, run_id: int) -> RunResult:
        """Execute a single optimize-evaluate-decide iteration.

        Phase B (bandit tuning) runs every iteration. Phase A (strategy
        evolution) triggers adaptively when stagnation is detected.
        """
        start = time.time()

        # 1. Measure baseline
        baseline = self._baseline_metrics()
        if baseline["rows"] < self.min_samples_for_eval:
            return RunResult(
                run_id=run_id,
                mutation_type="skip",
                mutation_description=f"not enough data ({baseline['rows']} < {self.min_samples_for_eval})",
                save_rate=baseline["save_rate"],
                average_reward=baseline["average_reward"],
                baseline_save_rate=baseline["save_rate"],
                baseline_average_reward=baseline["average_reward"],
                save_lift=0.0,
                reward_lift=0.0,
                regression_pass=True,
                status="discard",
                duration_s=time.time() - start,
                policy_snapshot={},
            )

        # 2. Snapshot current state
        snapshot = self._snapshot_state()

        # 3. Propose mutation (Phase B: bandit tuning only)
        try:
            if self.mode == "agent":
                strategy_name, mutated_state, description = self._propose_agent(snapshot)
            else:
                strategy_name, mutated_state, description = self._propose_random(snapshot)
        except Exception as exc:
            if self.mode == "agent":
                # Fall back to random mode and warn once
                if not getattr(self, "_agent_fallback_warned", False):
                    print(f"[warn] agent proposal failed: {exc}")
                    print("[warn] falling back to random mode for remaining iterations")
                    self._agent_fallback_warned = True
                strategy_name, mutated_state, description = self._propose_random(snapshot)
            else:
                return RunResult(
                    run_id=run_id,
                    mutation_type="random_error",
                    mutation_description=f"proposal failed: {exc}",
                    save_rate=baseline["save_rate"],
                    average_reward=baseline["average_reward"],
                    baseline_save_rate=baseline["save_rate"],
                    baseline_average_reward=baseline["average_reward"],
                    save_lift=0.0,
                    reward_lift=0.0,
                    regression_pass=True,
                    status="crash",
                    duration_s=time.time() - start,
                    policy_snapshot={},
                )

        # 4. Apply mutation
        self._apply_state(mutated_state)

        # 5. Evaluate via harness
        post_metrics = self._baseline_metrics()
        save_lift = post_metrics["save_rate"] - baseline["save_rate"]
        reward_lift = post_metrics["average_reward"] - baseline["average_reward"]

        # 6. Regression check
        if self._eval_cohort is not None:
            trust_ok = post_metrics.get("trust_score", 1.0) >= baseline.get("trust_score", 0.0) - 0.02
            reward_ok = reward_lift >= -0.01
            regression_pass = trust_ok and reward_ok
        else:
            regression = self.runtime.regression_check(
                min_treatment_samples=max(20, self.min_samples_for_eval // 3),
                min_holdout_samples=max(5, self.min_samples_for_eval // 10),
                min_save_lift=-0.05,
                max_support_delta=0.05,
                max_complaint_delta=0.03,
            )
            regression_pass = bool(regression.get("pass", False))

        # 7. Keep or discard
        improved = save_lift >= 0.0 and reward_lift >= -0.01
        keep = improved and regression_pass

        if keep:
            status = "keep"
            if post_metrics["save_rate"] > self._best_save_rate:
                self._best_save_rate = post_metrics["save_rate"]
                self._best_reward = post_metrics["average_reward"]
        else:
            self._restore_state(snapshot)
            status = "discard"

        result = RunResult(
            run_id=run_id,
            mutation_type=strategy_name,
            mutation_description=description,
            save_rate=post_metrics["save_rate"],
            average_reward=post_metrics["average_reward"],
            baseline_save_rate=baseline["save_rate"],
            baseline_average_reward=baseline["average_reward"],
            save_lift=save_lift,
            reward_lift=reward_lift,
            regression_pass=regression_pass,
            status=status,
            duration_s=time.time() - start,
            policy_snapshot=mutated_state if keep else {},
            alignment_score=post_metrics.get("alignment_score", 0.0),
            trust_score=post_metrics.get("trust_score", 0.0),
        )

        self._log_result(result)
        self.history.append(result)

        # 8. Adaptive generation trigger (Phase A)
        gen_meta = self.runtime.state.get("generation_meta", {})
        if save_lift <= 0 and self._eval_cohort is not None:
            gen_meta["stagnation_count"] = gen_meta.get("stagnation_count", 0) + 1
        else:
            gen_meta["stagnation_count"] = 0
        self.runtime.state["generation_meta"] = gen_meta

        threshold = int(gen_meta.get("stagnation_threshold", STAGNATION_THRESHOLD))
        if gen_meta.get("stagnation_count", 0) >= threshold and self._eval_cohort is not None:
            gen_logs = self._generation_round()
            for log in gen_logs:
                self._log_generation(run_id, log)
            gen_meta["last_generation"] = run_id

        return result

    def _log_generation(self, run_id: int, message: str) -> None:
        """Log a generation round event to results.tsv."""
        gen_result = RunResult(
            run_id=run_id,
            mutation_type="generation_round",
            mutation_description=message,
            save_rate=0.0,
            average_reward=0.0,
            baseline_save_rate=0.0,
            baseline_average_reward=0.0,
            save_lift=0.0,
            reward_lift=0.0,
            regression_pass=True,
            status="keep",
            duration_s=0.0,
            policy_snapshot={},
        )
        self._log_result(gen_result)

    def optimize(
        self,
        iterations: int = 20,
        *,
        bootstrap_traffic: int = 0,
        bootstrap_seed: int = 42,
        bootstrap_save_probability: float = 0.45,
    ) -> list[RunResult]:
        """Run the full optimization loop.

        If bootstrap_traffic > 0, generates synthetic traffic first so the
        optimizer has data to work with even without production outcomes.
        """
        if bootstrap_traffic > 0:
            generate_synthetic_traffic(
                self.runtime,
                count=bootstrap_traffic,
                seed=bootstrap_seed,
                save_probability=bootstrap_save_probability,
            )

        # Measure starting baseline
        baseline = self._baseline_metrics()
        self._best_save_rate = baseline["save_rate"]
        self._best_reward = baseline["average_reward"]

        results: list[RunResult] = []
        for i in range(iterations):
            result = self.run_one(run_id=i + 1)
            results.append(result)

        return results

    def summary(self) -> dict[str, Any]:
        """Produce a summary report of the optimization run."""
        if not self.history:
            return {"status": "no_runs", "iterations": 0}

        kept = [r for r in self.history if r.status == "keep"]
        discarded = [r for r in self.history if r.status == "discard"]
        crashed = [r for r in self.history if r.status == "crash"]

        first = self.history[0]
        last = self.history[-1]
        total_lift = last.save_rate - first.baseline_save_rate

        # Collect all mutation types that appeared
        all_types = sorted({r.mutation_type for r in self.history if r.mutation_type != "skip"})

        return {
            "status": "complete",
            "mode": self.mode,
            "iterations": len(self.history),
            "kept": len(kept),
            "discarded": len(discarded),
            "crashed": len(crashed),
            "starting_save_rate": round(first.baseline_save_rate, 4),
            "ending_save_rate": round(last.save_rate, 4),
            "best_save_rate": round(self._best_save_rate, 4),
            "total_save_lift": round(total_lift, 4),
            "starting_reward": round(first.baseline_average_reward, 4),
            "ending_reward": round(last.average_reward, 4),
            "best_reward": round(self._best_reward, 4),
            "mutation_breakdown": {
                name: {
                    "attempted": sum(1 for r in self.history if r.mutation_type == name),
                    "kept": sum(1 for r in kept if r.mutation_type == name),
                }
                for name in all_types
            },
            "alignment_score": round(last.alignment_score, 4) if last.alignment_score else None,
            "trust_score": round(last.trust_score, 4) if last.trust_score else None,
            "eval_mode": "simulator" if self._eval_cohort is not None else "replay",
            "results_path": str(self.results_path),
        }

    def print_summary(self) -> None:
        """Print a compact metric block in the autoresearch style."""
        s = self.summary()
        print("---")
        print(f"mode: {s.get('mode', 'random')}")
        print(f"starting_save_rate: {s.get('starting_save_rate', 0)}")
        print(f"ending_save_rate: {s.get('ending_save_rate', 0)}")
        print(f"best_save_rate: {s.get('best_save_rate', 0)}")
        print(f"total_save_lift: {s.get('total_save_lift', 0)}")
        print(f"starting_reward: {s.get('starting_reward', 0)}")
        print(f"ending_reward: {s.get('ending_reward', 0)}")
        if s.get("alignment_score") is not None:
            print(f"alignment_score: {s.get('alignment_score', 0)}")
        if s.get("trust_score") is not None:
            print(f"trust_score: {s.get('trust_score', 0)}")
        print(f"eval_mode: {s.get('eval_mode', 'replay')}")
        print(f"iterations: {s.get('iterations', 0)}")
        print(f"kept: {s.get('kept', 0)}")
        print(f"discarded: {s.get('discarded', 0)}")
        for name, stats in s.get("mutation_breakdown", {}).items():
            print(f"  {name}: {stats['kept']}/{stats['attempted']} kept")


# ── CLI entry point ────────────────────────────────────────────────────


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Self-improving policy optimizer (autoresearch style)"
    )
    parser.add_argument(
        "--iterations", type=int, default=20,
        help="Number of optimize-evaluate-decide cycles.",
    )
    parser.add_argument(
        "--bootstrap-traffic", type=int, default=300,
        help="Synthetic traffic to generate before optimizing. 0 to skip.",
    )
    parser.add_argument(
        "--bootstrap-save-probability", type=float, default=0.45,
        help="Base save probability for synthetic traffic.",
    )
    parser.add_argument(
        "--intensity", type=float, default=1.0,
        help="Mutation intensity. Higher = bigger changes per iteration.",
    )
    parser.add_argument(
        "--data-dir", type=str, default="optimizer_data",
        help="Directory for policy state and results.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="optimizer_output",
        help="Directory for results.tsv and run logs.",
    )
    parser.add_argument(
        "--seed", type=int, default=7,
    )
    parser.add_argument(
        "--warm-start-file", type=str, default="",
        help="Optional JSON file with historical rows to seed posteriors.",
    )
    parser.add_argument(
        "--mode", type=str, default="random", choices=["random", "agent"],
        help="Mutation proposal mode. 'random' needs no API key. 'agent' uses LLM.",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="OpenAI model for agent mode.",
    )
    parser.add_argument(
        "--eval-cohort", type=str, default="",
        help="Path to enriched JSON for simulator-based eval (the Karpathy harness).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    runtime = CancelPolicyRuntime(data_dir / "policy", seed=args.seed)

    # Optional warm-start from historical data
    if args.warm_start_file:
        warm_path = Path(args.warm_start_file)
        if warm_path.exists():
            payload = json.loads(warm_path.read_text())
            rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
            if isinstance(rows, list):
                summary = runtime.warm_start(rows, reset_state=False)
                print(f"Warm-start: {summary['rows_applied']} rows applied.")

    optimizer = PolicyOptimizer(
        runtime,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        intensity=args.intensity,
        mode=args.mode,
        openai_model=args.model,
        eval_cohort_path=args.eval_cohort,
    )

    results = optimizer.optimize(
        iterations=args.iterations,
        bootstrap_traffic=args.bootstrap_traffic,
        bootstrap_seed=args.seed + 1000,
        bootstrap_save_probability=args.bootstrap_save_probability,
    )

    optimizer.print_summary()

    # Save full summary as JSON
    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(optimizer.summary(), indent=2))
    print(f"results: {optimizer.results_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

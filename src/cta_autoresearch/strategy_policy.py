from __future__ import annotations

from functools import lru_cache
import random

from cta_autoresearch.models import Persona, StrategyCandidate


MESSAGE_ANGLES = {
    "progress_reflection": {
        "label": "Progress Reflection",
        "specificity": 0.72,
        "description": "Reflect back the learner's momentum, streaks, and real progress.",
    },
    "outcome_proof": {
        "label": "Outcome Proof",
        "specificity": 0.42,
        "description": "Lead with results, gains, and measurable outcomes.",
    },
    "mistake_recovery": {
        "label": "Mistake Recovery",
        "specificity": 0.80,
        "description": "Position the product as a coach that helps recover from mistakes fast.",
    },
    "habit_identity": {
        "label": "Habit Identity",
        "specificity": 0.60,
        "description": "Reinforce the learner identity and the value of staying consistent.",
    },
    "empathetic_exit": {
        "label": "Empathetic Exit",
        "specificity": 0.25,
        "description": "Meet the user with empathy and reduce resistance before asking for a choice.",
    },
    "feature_unlock": {
        "label": "Feature Unlock",
        "specificity": 0.36,
        "description": "Highlight underused capabilities and unrealized value.",
    },
    "momentum_protection": {
        "label": "Momentum Protection",
        "specificity": 0.68,
        "description": "Frame cancellation as a break in a valuable learning rhythm.",
    },
    "cost_value_reframe": {
        "label": "Cost / Value Reframe",
        "specificity": 0.30,
        "description": "Make the product feel worth the cost versus the alternatives.",
    },
    "goal_deadline": {
        "label": "Goal Deadline",
        "specificity": 0.74,
        "description": "Tie the choice to the learner's immediate test, exam, or deadline.",
    },
    "flexibility_relief": {
        "label": "Flexibility Relief",
        "specificity": 0.28,
        "description": "Reduce fear by showing softer paths than all-or-nothing cancellation.",
    },
    "fresh_start_reset": {
        "label": "Fresh Start Reset",
        "specificity": 0.40,
        "description": "Offer a clean restart for users who feel behind or disengaged.",
    },
}

PROOF_STYLES = {
    "none": {
        "label": "No explicit proof",
        "description": "Rely on the message angle without added evidence.",
    },
    "quantified_outcome": {
        "label": "Quantified outcome",
        "description": "Use concrete, data-backed claims or performance metrics.",
    },
    "peer_testimonial": {
        "label": "Peer testimonial",
        "description": "Use a quote or story from a similar learner.",
    },
    "similar_user_story": {
        "label": "Similar user story",
        "description": "Show a learner with a similar use case succeeding with the product.",
    },
    "expert_validation": {
        "label": "Expert validation",
        "description": "Use educator or study-science framing to create authority.",
    },
    "personal_usage_signal": {
        "label": "Personal usage signal",
        "description": "Reference the user's own study behavior or product usage.",
    },
}

BASE_OFFERS = {
    "none": {
        "label": "No offer",
        "kind": "none",
        "generosity": 0.00,
    },
    "discount_10": {
        "label": "10% discount",
        "kind": "discount",
        "generosity": 0.10,
    },
    "discount_15": {
        "label": "15% discount",
        "kind": "discount",
        "generosity": 0.15,
    },
    "discount_20": {
        "label": "20% discount",
        "kind": "discount",
        "generosity": 0.20,
    },
    "discount_25": {
        "label": "25% discount",
        "kind": "discount",
        "generosity": 0.25,
    },
    "discount_30": {
        "label": "30% discount",
        "kind": "discount",
        "generosity": 0.30,
    },
    "discount_40": {
        "label": "40% discount",
        "kind": "discount",
        "generosity": 0.40,
    },
    "discount_45": {
        "label": "45% discount",
        "kind": "discount",
        "generosity": 0.45,
    },
    "discount_50": {
        "label": "50% discount",
        "kind": "discount",
        "generosity": 0.50,
    },
    "discount_60": {
        "label": "60% discount",
        "kind": "discount",
        "generosity": 0.60,
    },
    "discount_65": {
        "label": "65% discount",
        "kind": "discount",
        "generosity": 0.65,
    },
    "discount_70": {
        "label": "70% discount",
        "kind": "discount",
        "generosity": 0.70,
    },
    "discount_75": {
        "label": "75% discount",
        "kind": "discount",
        "generosity": 0.75,
    },
    "discount_80": {
        "label": "80% discount",
        "kind": "discount",
        "generosity": 0.80,
    },
    "discount_85": {
        "label": "85% discount",
        "kind": "discount",
        "generosity": 0.85,
    },
    "discount_90": {
        "label": "90% discount",
        "kind": "discount",
        "generosity": 0.90,
    },
    "discount_95": {
        "label": "95% discount",
        "kind": "discount",
        "generosity": 0.95,
    },
    "discount_100": {
        "label": "100% discount / free month",
        "kind": "discount",
        "generosity": 1.00,
    },
    "pause_plan": {
        "label": "Pause plan",
        "kind": "pause",
        "generosity": 0.18,
    },
    "downgrade_lite": {
        "label": "Downgrade to lighter plan",
        "kind": "downgrade",
        "generosity": 0.28,
    },
    "exam_sprint": {
        "label": "Exam sprint extension",
        "kind": "extension",
        "generosity": 0.22,
    },
    "bonus_credits": {
        "label": "Bonus credits",
        "kind": "credit",
        "generosity": 0.16,
    },
    "flexible_billing": {
        "label": "Flexible billing",
        "kind": "billing",
        "generosity": 0.12,
    },
    "concierge_support": {
        "label": "Concierge support",
        "kind": "support",
        "generosity": 0.08,
    },
    "study_plan_reset": {
        "label": "Study plan reset",
        "kind": "support",
        "generosity": 0.10,
    },
    "priority_review_pack": {
        "label": "Priority review pack",
        "kind": "credit",
        "generosity": 0.18,
    },
    "deadline_extension_plus": {
        "label": "Deadline extension plus",
        "kind": "extension",
        "generosity": 0.26,
    },
    "office_hours_access": {
        "label": "Office hours access",
        "kind": "support",
        "generosity": 0.14,
    },
}

CTAS = {
    "stay_on_current_plan": {
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
    "switch_to_lite": {
        "label": "Switch to lighter plan",
        "allowed_offer_kinds": {"downgrade"},
    },
    "finish_current_goal": {
        "label": "Help me finish this goal",
        "allowed_offer_kinds": {"none", "extension", "credit", "discount", "support"},
    },
    "talk_to_learning_support": {
        "label": "Talk to learning support",
        "allowed_offer_kinds": {"support", "none"},
    },
    "tell_us_why": {
        "label": "Tell us why you're leaving",
        "allowed_offer_kinds": {"none", "support"},
    },
    "see_plan_options": {
        "label": "See my options",
        "allowed_offer_kinds": {"pause", "downgrade", "billing", "discount", "credit", "support"},
    },
    "remind_me_later": {
        "label": "Remind me later",
        "allowed_offer_kinds": {"none", "discount", "billing"},
    },
}

PERSONALIZATION_LEVELS = {
    "generic": {
        "label": "Generic",
        "intensity": 0.15,
        "description": "Broad category-level copy with little user-specific detail.",
    },
    "contextual": {
        "label": "Contextual",
        "intensity": 0.38,
        "description": "Use the user's study context or broad product usage context.",
    },
    "behavioral": {
        "label": "Behavioral",
        "intensity": 0.62,
        "description": "Reference meaningful behavior patterns or progress.",
    },
    "highly_specific": {
        "label": "Highly specific",
        "intensity": 0.88,
        "description": "Use deeply personalized phrasing that may feel powerful or creepy.",
    },
}

CONTEXTUAL_GROUNDINGS = {
    "generic": {"label": "Generic grounding", "specificity": 0.10},
    "study_goal": {"label": "Study-goal mirror", "specificity": 0.42},
    "progress_snapshot": {"label": "Progress snapshot", "specificity": 0.60},
    "unused_value": {"label": "Unused-value recap", "specificity": 0.44},
    "deadline_countdown": {"label": "Deadline countdown", "specificity": 0.66},
    "recovery_moment": {"label": "Recovery moment", "specificity": 0.58},
    "habit_streak": {"label": "Habit streak callout", "specificity": 0.64},
}

CREATIVE_TREATMENTS = {
    "plain_note": {"label": "Plain note", "boldness": 0.10},
    "feature_collage": {"label": "Feature collage", "boldness": 0.46},
    "progress_thermometer": {"label": "Progress thermometer", "boldness": 0.54},
    "comeback_plan": {"label": "Comeback plan", "boldness": 0.52},
    "social_proof_card": {"label": "Social proof card", "boldness": 0.38},
    "coach_note": {"label": "Coach note", "boldness": 0.48},
    "before_after_frame": {"label": "Before / after frame", "boldness": 0.44},
}

FRICTION_REDUCERS = {
    "none": {"label": "No friction reducer", "assist": 0.00},
    "single_tap_pause": {"label": "Single-tap pause", "assist": 0.54},
    "prefilled_downgrade": {"label": "Prefilled downgrade", "assist": 0.52},
    "smart_resume_date": {"label": "Smart resume date", "assist": 0.48},
    "concierge_setup": {"label": "Concierge setup", "assist": 0.42},
    "billing_date_shift": {"label": "Billing date shift", "assist": 0.44},
    "plan_comparison": {"label": "Plan comparison", "assist": 0.38},
}

CONTEXTUAL_GROUNDINGS.update(
    {
        "pricing_context": {"label": "Pricing context", "specificity": 0.36},
        "support_signal": {"label": "Support signal", "specificity": 0.40},
        "comeback_window": {"label": "Comeback window", "specificity": 0.44},
        "deadline_pressure": {"label": "Deadline pressure", "specificity": 0.62},
        "recent_progress": {"label": "Recent progress", "specificity": 0.56},
    }
)

CREATIVE_TREATMENTS.update(
    {
        "feature_visual": {"label": "Feature visual", "boldness": 0.46},
        "progress_snapshot": {"label": "Progress snapshot", "boldness": 0.52},
        "study_timeline": {"label": "Study timeline", "boldness": 0.44},
        "peer_story_card": {"label": "Peer story card", "boldness": 0.38},
        "coach_plan": {"label": "Coach plan", "boldness": 0.48},
        "options_table": {"label": "Options table", "boldness": 0.40},
    }
)

FRICTION_REDUCERS.update(
    {
        "one_tap_pause": {"label": "One-tap pause", "assist": 0.54},
        "one_tap_downgrade": {"label": "One-tap downgrade", "assist": 0.52},
        "billing_shift": {"label": "Billing shift", "assist": 0.44},
        "keep_history": {"label": "Keep history", "assist": 0.36},
        "guided_reset": {"label": "Guided reset", "assist": 0.40},
        "human_concierge": {"label": "Human concierge", "assist": 0.42},
    }
)


def _depth_value(settings: object | None) -> int:
    if settings is None:
        return 2
    if hasattr(settings, "depth"):
        depth = getattr(settings, "depth")
        if isinstance(depth, str):
            return {"quick": 1, "balanced": 2, "deep": 3, "max": 4}.get(depth, 2)
        return int(depth)
    return 2


def offer_catalog(settings: object | None = None) -> dict[str, dict]:
    depth = _depth_value(settings)
    discount_step = max(1, int(getattr(settings, "discount_step", 5) or 5))
    discount_floor = max(0, min(100, int(getattr(settings, "discount_floor", 0) or 0)))
    discount_ceiling = max(0, min(100, int(getattr(settings, "discount_ceiling", 100) or 100)))
    if discount_floor > discount_ceiling:
        discount_floor, discount_ceiling = discount_ceiling, discount_floor
    if depth <= 1:
        offers = {
            key: value
            for key, value in BASE_OFFERS.items()
            if key
            in {
                "none",
                "discount_10",
                "discount_20",
                "discount_40",
                "pause_plan",
                "downgrade_lite",
                "bonus_credits",
                "flexible_billing",
                "concierge_support",
                "exam_sprint",
            }
        }
    elif depth == 2:
        offers = {
            key: value
            for key, value in BASE_OFFERS.items()
            if key not in {"discount_85", "discount_90", "discount_95"}
        }
    else:
        offers = dict(BASE_OFFERS)

    filtered: dict[str, dict] = {}
    for key, value in offers.items():
        if not key.startswith("discount_"):
            filtered[key] = value
            continue
        generosity = int(round(float(value["generosity"]) * 100))
        if generosity < discount_floor or generosity > discount_ceiling:
            continue
        if generosity != 100 and generosity % discount_step != 0:
            continue
        filtered[key] = value
    return filtered


OFFERS = dict(BASE_OFFERS)


def candidate_key(candidate: StrategyCandidate) -> str:
    return "|".join(
        [
            candidate.message_angle,
            candidate.proof_style,
            candidate.offer,
            candidate.cta,
            candidate.personalization,
            candidate.contextual_grounding,
            candidate.creative_treatment,
            candidate.friction_reducer,
        ]
    )


def candidate_label(candidate: StrategyCandidate, offers: dict[str, dict] | None = None) -> str:
    active_offers = offers or OFFERS
    return " + ".join(
        [
            MESSAGE_ANGLES[candidate.message_angle]["label"],
            PROOF_STYLES[candidate.proof_style]["label"],
            active_offers[candidate.offer]["label"],
            CTAS[candidate.cta]["label"],
            PERSONALIZATION_LEVELS[candidate.personalization]["label"],
            CONTEXTUAL_GROUNDINGS[candidate.contextual_grounding]["label"],
            CREATIVE_TREATMENTS[candidate.creative_treatment]["label"],
            FRICTION_REDUCERS[candidate.friction_reducer]["label"],
        ]
    )


def _candidate_is_valid(candidate: StrategyCandidate, offers: dict[str, dict]) -> bool:
    offer_meta = offers[candidate.offer]
    cta_meta = CTAS[candidate.cta]

    if offer_meta["kind"] not in cta_meta["allowed_offer_kinds"]:
        return False
    if offer_meta["kind"] == "none" and candidate.cta == "claim_offer":
        return False
    if candidate.message_angle == "outcome_proof" and candidate.proof_style == "none":
        return False
    if candidate.message_angle == "empathetic_exit" and candidate.personalization == "highly_specific":
        return False
    if candidate.proof_style == "personal_usage_signal" and candidate.personalization == "generic":
        return False
    if candidate.message_angle == "fresh_start_reset" and candidate.offer == "discount_100":
        return False
    if candidate.offer == "concierge_support" and candidate.cta == "claim_offer":
        return False
    if candidate.offer == "pause_plan" and candidate.cta == "stay_on_current_plan":
        return False
    if candidate.offer == "downgrade_lite" and candidate.cta == "stay_on_current_plan":
        return False
    if candidate.contextual_grounding == "deadline_countdown" and candidate.message_angle not in {"goal_deadline", "momentum_protection", "progress_reflection"}:
        return False
    if candidate.contextual_grounding == "habit_streak" and candidate.personalization == "generic":
        return False
    if candidate.creative_treatment == "feature_collage" and candidate.message_angle == "empathetic_exit":
        return False
    if candidate.friction_reducer == "single_tap_pause" and offer_meta["kind"] != "pause":
        return False
    if candidate.friction_reducer == "prefilled_downgrade" and offer_meta["kind"] != "downgrade":
        return False
    if candidate.friction_reducer == "billing_date_shift" and offer_meta["kind"] not in {"billing", "discount"}:
        return False
    if candidate.friction_reducer == "concierge_setup" and offer_meta["kind"] not in {"support", "extension"}:
        return False
    return True


def valid_candidate(candidate: StrategyCandidate, settings: object | None = None) -> bool:
    return _candidate_is_valid(candidate, offer_catalog(settings))


def _candidate_priority(candidate: StrategyCandidate, offers: dict[str, dict]) -> float:
    return (
        MESSAGE_ANGLES[candidate.message_angle]["specificity"]
        + PERSONALIZATION_LEVELS[candidate.personalization]["intensity"]
        + CONTEXTUAL_GROUNDINGS[candidate.contextual_grounding]["specificity"]
        + CREATIVE_TREATMENTS[candidate.creative_treatment]["boldness"]
        + FRICTION_REDUCERS[candidate.friction_reducer]["assist"]
        + (0.08 if candidate.proof_style != "none" else 0.0)
        - 0.12 * offers[candidate.offer]["generosity"]
    )


def _default_candidate_for_offer(offer: str) -> StrategyCandidate:
    kind = BASE_OFFERS[offer]["kind"]
    cta = {
        "pause": "pause_instead",
        "downgrade": "switch_to_lite",
        "support": "talk_to_learning_support",
        "discount": "claim_offer",
        "credit": "claim_offer",
        "billing": "see_plan_options",
        "extension": "finish_current_goal",
    }.get(kind, "stay_on_current_plan")
    return StrategyCandidate(
        message_angle="progress_reflection",
        proof_style="similar_user_story",
        offer=offer,
        cta=cta,
        personalization="contextual",
        contextual_grounding="study_goal",
        creative_treatment="plain_note",
        friction_reducer="none",
    )


def _sample_candidates(settings: object | None = None, budget: int = 2000) -> list[StrategyCandidate]:
    offers = offer_catalog(settings)
    rng = random.Random(int(getattr(settings, "seed", 7)))
    selected: list[StrategyCandidate] = []
    seen: set[str] = set()

    def add(candidate: StrategyCandidate) -> None:
        key = candidate_key(candidate)
        if key in seen or not _candidate_is_valid(candidate, offers):
            return
        selected.append(candidate)
        seen.add(key)

    for offer in offers:
        add(_default_candidate_for_offer(offer))

    catalogs = [
        ("message_angle", list(MESSAGE_ANGLES)),
        ("proof_style", list(PROOF_STYLES)),
        ("offer", list(offers)),
        ("cta", list(CTAS)),
        ("personalization", list(PERSONALIZATION_LEVELS)),
        ("contextual_grounding", list(CONTEXTUAL_GROUNDINGS)),
        ("creative_treatment", list(CREATIVE_TREATMENTS)),
        ("friction_reducer", list(FRICTION_REDUCERS)),
    ]

    while len(selected) < budget:
        candidate = _default_candidate_for_offer(rng.choice(list(offers)))
        payload = dict(candidate.__dict__)
        for field, values in catalogs:
            if rng.random() < 0.72:
                payload[field] = rng.choice(values)
        add(StrategyCandidate(**payload))
        if len(seen) > budget * 12:
            break

    return selected[:budget]


@lru_cache(maxsize=8)
def _cached_candidates(depth: int) -> tuple[StrategyCandidate, ...]:
    budget = 1500 if depth <= 1 else 2500 if depth == 2 else 4000 if depth == 3 else 6000
    settings = type("Depth", (), {"depth": depth, "seed": 7})()
    return tuple(_sample_candidates(settings, budget=budget))


def all_candidates(settings: object | None = None) -> list[StrategyCandidate]:
    return list(_cached_candidates(_depth_value(settings)))


def select_candidate_pool(settings: object | None = None) -> list[StrategyCandidate]:
    budget = int(getattr(settings, "strategy_budget", 0) or 0) if settings is not None else 0
    if budget <= 0:
        return all_candidates(settings)
    candidates = _sample_candidates(settings, budget=max(budget * 3, 1200))
    if budget >= len(candidates):
        return candidates

    offers = offer_catalog(settings)
    rng = random.Random(int(getattr(settings, "seed", 7)))
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    scored = sorted(
        shuffled,
        key=lambda candidate: (_candidate_priority(candidate, offers), candidate_key(candidate)),
        reverse=True,
    )

    selected: list[StrategyCandidate] = []
    seen: set[str] = set()

    def add(candidate: StrategyCandidate) -> None:
        key = candidate_key(candidate)
        if key in seen or len(selected) >= budget:
            return
        selected.append(candidate)
        seen.add(key)

    for attr, catalog in (
        ("message_angle", MESSAGE_ANGLES),
        ("offer", offers),
        ("cta", CTAS),
        ("personalization", PERSONALIZATION_LEVELS),
        ("contextual_grounding", CONTEXTUAL_GROUNDINGS),
        ("creative_treatment", CREATIVE_TREATMENTS),
        ("friction_reducer", FRICTION_REDUCERS),
    ):
        for key in catalog:
            for candidate in scored:
                if getattr(candidate, attr) == key:
                    add(candidate)
                    break

    for candidate in scored:
        add(candidate)
        if len(selected) >= budget:
            break
    return selected


def render_message(persona: Persona, candidate: StrategyCandidate, settings: object | None = None) -> str:
    context = persona.profile.study_context
    name = persona.profile.name
    features = persona.features

    openings = {
        "progress_reflection": (
            f"{name}, you've already built real momentum in {context}. "
            f"This is exactly when continuity matters."
        ),
        "outcome_proof": (
            f"Learners who stay with Jungle AI through {context} prep usually get faster recall and stronger practice accuracy."
        ),
        "mistake_recovery": (
            f"When a hard question knocks you off balance, Jungle AI helps you recover faster instead of losing the thread."
        ),
        "habit_identity": (
            f"You're not just trying something out anymore. You've been showing up like a real learner."
        ),
        "empathetic_exit": (
            f"If something is not working for you right now, we want to make the next step easier instead of boxing you in."
        ),
        "feature_unlock": (
            f"You still have meaningful study value left unused for {context}, including guided practice and generation capacity."
        ),
        "momentum_protection": (
            f"Canceling now could break the exact learning rhythm you built when things were starting to click."
        ),
        "cost_value_reframe": (
            f"The question is not just what Jungle AI costs. It is what it saves you in time, confusion, and rework."
        ),
        "goal_deadline": (
            f"You're close enough to a real goal in {context} that this is a bad moment to lose support."
        ),
        "flexibility_relief": (
            f"You do not have to choose between paying the same way forever and walking away completely."
        ),
        "fresh_start_reset": (
            f"If the product feels stale or off-track, we can reset the experience so it feels useful again."
        ),
    }

    groundings = {
        "generic": "",
        "study_goal": f"This is especially relevant for your {context} prep.",
        "progress_snapshot": f"You've already put in {persona.profile.total_sessions} sessions and {persona.profile.time_in_app_hours:.1f} focused hours.",
        "unused_value": f"You still have {persona.profile.monthly_generations_remaining} generations left that could move this goal forward.",
        "deadline_countdown": f"The timing matters because the next milestone in {context} is close enough that a reset now creates rework.",
        "recovery_moment": "The fact that you come back after mistakes is a strong signal that the goal is still alive.",
        "habit_streak": "Your recent usage pattern looks more like a habit under pressure than a goal you've truly abandoned.",
    }

    proofs = {
        "none": "",
        "quantified_outcome": "Students who stick with structured review usually retain more and cram less at the end.",
        "peer_testimonial": "Other learners in similar study modes tell us the product starts paying off once they keep the loop going for a few more sessions.",
        "similar_user_story": "A learner using Jungle AI for the same kind of prep stayed through crunch time and turned scattered study into repeatable progress.",
        "expert_validation": "Study science consistently shows spaced practice and fast feedback work better than starting from scratch each time.",
        "personal_usage_signal": "Your own usage already shows the pattern: when you study here, you keep moving instead of stalling.",
    }

    offers = {
        "none": "",
        "discount_10": "If price is the issue, we can lower this cycle by 10%.",
        "discount_15": "If price is the issue, we can lower this cycle by 15%.",
        "discount_20": "If price is the issue, we can lower this cycle by 20%.",
        "discount_25": "If price is the issue, we can lower this cycle by 25%.",
        "discount_30": "If price is the issue, we can lower this cycle by 30%.",
        "discount_40": "If price is the issue, we can lower this cycle by 40%.",
        "discount_45": "If price is the issue, we can lower this cycle by 45% for this save attempt.",
        "discount_50": "If price is the issue, we can lower this cycle by 50%.",
        "discount_60": "If price is the issue, we can lower this cycle by 60% for this save attempt.",
        "discount_65": "If price is the issue, we can lower this cycle by 65%.",
        "discount_70": "If price is the issue, we can lower this cycle by 70%.",
        "discount_75": "If price is the issue, we can lower this cycle by 75%.",
        "discount_80": "If price is the issue, we can lower this cycle by 80%.",
        "discount_85": "If price is the issue, we can lower this cycle by 85%.",
        "discount_90": "If price is the issue, we can lower this cycle by 90%.",
        "discount_95": "If price is the issue, we can nearly wipe out the next bill so you do not lose momentum now.",
        "discount_100": "If the timing is terrible, we can cover the next cycle completely so you do not lose momentum now.",
        "pause_plan": "You can pause instead of canceling and keep your setup, history, and path back in.",
        "downgrade_lite": "You can switch to a lighter plan instead of paying for more than you need right now.",
        "exam_sprint": "We can keep you covered through this exam window so you can finish what you started.",
        "bonus_credits": "We can add bonus study credits so you can get a few more useful sessions before deciding.",
        "flexible_billing": "We can make billing more flexible so this does not land at the worst time.",
        "concierge_support": "We can help you reset your setup and make the product feel useful again with human guidance.",
        "study_plan_reset": "We can rebuild the study plan around where you are right now instead of where you started.",
        "priority_review_pack": "We can unlock a tighter review pack so the next sessions feel more targeted.",
        "deadline_extension_plus": "We can stretch the support window around your deadline so the goal does not expire on you.",
        "office_hours_access": "We can give you access to guided office-hours style help so the next step feels clearer.",
    }

    nuances = {
        "generic": "",
        "contextual": f"This is especially relevant for your {context} prep.",
        "behavioral": (
            "Your recent study behavior suggests you're not done with this goal yet."
            if features.habit_strength > 0.55
            else "Your usage suggests there is still unfinished value here."
        ),
        "highly_specific": (
            "Based on the exact way you've been using the product recently, this is one of the worst moments to step away."
        ),
    }

    treatments = {
        "plain_note": "",
        "feature_collage": "We would show this as a quick collage of the features and outputs you've already touched.",
        "progress_thermometer": "We would frame this with a visible progress meter so the remaining gap feels finite.",
        "comeback_plan": "We would show a short comeback plan for the next few sessions instead of a vague promise.",
        "social_proof_card": "We would package the proof as a card from a similar learner path.",
        "coach_note": "We would present this like a short coach note rather than a pricing negotiation.",
        "before_after_frame": "We would contrast the current momentum with what gets lost if the loop breaks here.",
    }

    reducers = {
        "none": "",
        "single_tap_pause": "The flow would offer a one-tap pause so staying feels reversible.",
        "prefilled_downgrade": "The downgrade option would be prefilled to remove setup friction.",
        "smart_resume_date": "We would suggest a smart resume date so this feels like a break, not a goodbye.",
        "concierge_setup": "A human-guided reset would be offered inline if you want help getting back on track.",
        "billing_date_shift": "We would let you shift the billing date so the timing lands better.",
        "plan_comparison": "We would show a simple plan comparison to make the lighter path easy to understand.",
    }

    groundings.update(
        {
            "pricing_context": groundings["unused_value"],
            "support_signal": groundings["recovery_moment"],
            "comeback_window": groundings["recovery_moment"],
            "deadline_pressure": groundings["deadline_countdown"],
            "recent_progress": groundings["progress_snapshot"],
        }
    )
    treatments.update(
        {
            "feature_visual": treatments["feature_collage"],
            "progress_snapshot": treatments["progress_thermometer"],
            "study_timeline": treatments["before_after_frame"],
            "peer_story_card": treatments["social_proof_card"],
            "coach_plan": treatments["coach_note"],
            "options_table": treatments["before_after_frame"],
        }
    )
    reducers.update(
        {
            "one_tap_pause": reducers["single_tap_pause"],
            "one_tap_downgrade": reducers["prefilled_downgrade"],
            "billing_shift": reducers["billing_date_shift"],
            "keep_history": reducers["plan_comparison"],
            "guided_reset": reducers["smart_resume_date"],
            "human_concierge": reducers["concierge_setup"],
        }
    )

    ctas = {
        "stay_on_current_plan": "CTA: Keep my plan",
        "claim_offer": "CTA: Claim this option",
        "pause_instead": "CTA: Pause my plan",
        "switch_to_lite": "CTA: Switch to a lighter plan",
        "finish_current_goal": "CTA: Help me finish this goal",
        "talk_to_learning_support": "CTA: Talk to learning support",
        "tell_us_why": "CTA: Tell us why you're leaving",
        "see_plan_options": "CTA: Show my options",
        "remind_me_later": "CTA: Remind me later",
    }

    message = " ".join(
        part
        for part in [
            openings[candidate.message_angle],
            nuances[candidate.personalization],
            groundings[candidate.contextual_grounding],
            proofs[candidate.proof_style],
            offers[candidate.offer],
            treatments[candidate.creative_treatment],
            reducers[candidate.friction_reducer],
            ctas[candidate.cta],
        ]
        if part
    )
    return " ".join(message.split())

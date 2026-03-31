from __future__ import annotations

import random
from dataclasses import replace

from cta_autoresearch.features import derive_features
from cta_autoresearch.models import Persona, PersonaInsights, UserProfile


ARCHETYPE_TEMPLATES: list[dict] = [
    {
        "name": "Exam Jam Priya",
        "cohort": "synthetic_template",
        "plan": "Super Learner",
        "status": "active",
        "billing_period": "monthly",
        "user_type": "Student - Undergraduate",
        "lifetime_days": 140,
        "total_sessions": 88,
        "total_events": 2650,
        "time_in_app_hours": 21.0,
        "card_sets_generated": 19,
        "monthly_generations_remaining": 2,
        "monthly_generations_total": 15,
        "chat_messages_remaining": 3,
        "answer_feedback_remaining": 2,
        "multi_device_count": 4,
        "acquisition_source": "google_organic",
        "recent_behavior": "Heavy exam week usage with multiple late-night sessions.",
        "study_context": "organic chemistry final exam",
        "retry_after_mistake": True,
        "source_context_usage": True,
        "accuracy_signal": "mixed_with_recovery",
        "dormancy_days": 0,
    },
    {
        "name": "Budget Ben",
        "cohort": "synthetic_template",
        "plan": "Starter",
        "status": "active",
        "billing_period": "monthly",
        "user_type": "Student - Graduate",
        "lifetime_days": 220,
        "total_sessions": 74,
        "total_events": 1900,
        "time_in_app_hours": 12.0,
        "card_sets_generated": 17,
        "monthly_generations_remaining": 9,
        "monthly_generations_total": 12,
        "chat_messages_remaining": 6,
        "answer_feedback_remaining": 4,
        "multi_device_count": 3,
        "acquisition_source": "referral",
        "recent_behavior": "Uses the app regularly but leaves quota on the table.",
        "study_context": "licensing exam prep",
        "retry_after_mistake": False,
        "source_context_usage": False,
        "accuracy_signal": "mixed",
        "dormancy_days": 4,
    },
    {
        "name": "Dormant Dana",
        "cohort": "synthetic_template",
        "plan": "Free",
        "status": "inactive",
        "billing_period": "none",
        "user_type": "Student - High School",
        "lifetime_days": 310,
        "total_sessions": 21,
        "total_events": 420,
        "time_in_app_hours": 2.8,
        "card_sets_generated": 5,
        "monthly_generations_remaining": 10,
        "monthly_generations_total": 10,
        "chat_messages_remaining": 8,
        "answer_feedback_remaining": 5,
        "multi_device_count": 2,
        "acquisition_source": "google_organic",
        "recent_behavior": "Generated a few sets early, then mostly disappeared.",
        "study_context": "history coursework",
        "retry_after_mistake": False,
        "source_context_usage": False,
        "accuracy_signal": "low_initial",
        "dormancy_days": 37,
    },
    {
        "name": "Working Wes",
        "cohort": "synthetic_template",
        "plan": "Starter",
        "status": "active",
        "billing_period": "monthly",
        "user_type": "Working Professional",
        "lifetime_days": 420,
        "total_sessions": 63,
        "total_events": 1420,
        "time_in_app_hours": 9.4,
        "card_sets_generated": 11,
        "monthly_generations_remaining": 4,
        "monthly_generations_total": 12,
        "chat_messages_remaining": 1,
        "answer_feedback_remaining": 3,
        "multi_device_count": 2,
        "acquisition_source": "youtube",
        "recent_behavior": "Studies in short bursts after work and disappears when schedule pressure spikes.",
        "study_context": "certification exam prep",
        "retry_after_mistake": True,
        "source_context_usage": True,
        "accuracy_signal": "mixed_with_recovery",
        "dormancy_days": 6,
    },
    {
        "name": "Skeptical Sofia",
        "cohort": "synthetic_template",
        "plan": "Starter",
        "status": "active",
        "billing_period": "monthly",
        "user_type": "Student - Graduate",
        "lifetime_days": 120,
        "total_sessions": 31,
        "total_events": 820,
        "time_in_app_hours": 5.6,
        "card_sets_generated": 7,
        "monthly_generations_remaining": 8,
        "monthly_generations_total": 12,
        "chat_messages_remaining": 4,
        "answer_feedback_remaining": 5,
        "multi_device_count": 2,
        "acquisition_source": "influencer",
        "recent_behavior": "Uses the product cautiously and needs strong proof before paying for another cycle.",
        "study_context": "MCAT biology review",
        "retry_after_mistake": False,
        "source_context_usage": False,
        "accuracy_signal": "mixed",
        "dormancy_days": 9,
    },
    {
        "name": "Overloaded Omar",
        "cohort": "synthetic_template",
        "plan": "Super Learner",
        "status": "active",
        "billing_period": "annual",
        "user_type": "Student - Undergraduate",
        "lifetime_days": 520,
        "total_sessions": 144,
        "total_events": 3380,
        "time_in_app_hours": 27.5,
        "card_sets_generated": 28,
        "monthly_generations_remaining": 5,
        "monthly_generations_total": 20,
        "chat_messages_remaining": 2,
        "answer_feedback_remaining": 1,
        "multi_device_count": 5,
        "acquisition_source": "referral",
        "recent_behavior": "Usage is high, but the learner is overwhelmed and close to burnout before a deadline.",
        "study_context": "engineering midterms",
        "retry_after_mistake": True,
        "source_context_usage": True,
        "accuracy_signal": "high_but_stressed",
        "dormancy_days": 1,
    },
    {
        "name": "Reset Riya",
        "cohort": "synthetic_template",
        "plan": "Starter",
        "status": "inactive",
        "billing_period": "monthly",
        "user_type": "Student - Undergraduate",
        "lifetime_days": 260,
        "total_sessions": 44,
        "total_events": 980,
        "time_in_app_hours": 6.1,
        "card_sets_generated": 9,
        "monthly_generations_remaining": 10,
        "monthly_generations_total": 12,
        "chat_messages_remaining": 7,
        "answer_feedback_remaining": 4,
        "multi_device_count": 3,
        "acquisition_source": "google_organic",
        "recent_behavior": "Used to rely on the product, then fell behind and now wants a cleaner restart path.",
        "study_context": "nursing school coursework",
        "retry_after_mistake": False,
        "source_context_usage": True,
        "accuracy_signal": "low_initial",
        "dormancy_days": 24,
    },
]

BEHAVIORAL_OVERLAYS = [
    "Recently opened the app in short bursts while deciding whether the goal is still worth finishing.",
    "Shows repeated rescue behavior after mistakes, which signals unfinished intent rather than lost interest.",
    "Has meaningful unused product value left, but the path back in feels heavier than it should.",
    "Keeps coming back near stressful milestones, then backs away once the workload spikes.",
    "Uses the product purposefully, but current pricing or timing is creating emotional drag.",
]

STUDY_CONTEXT_SUFFIXES = [
    "with a deadline inside the next two weeks",
    "where recall speed matters more than broad reading",
    "and still has a concrete finish line worth protecting",
    "where falling off now would create rework later",
]

RICHNESS_PROFILES = {
    "standard": {"mutation_intensity": 1.0, "archetype_span": 0.5, "blend_every": 0},
    "rich": {"mutation_intensity": 1.35, "archetype_span": 0.8, "blend_every": 4},
    "extreme": {"mutation_intensity": 1.75, "archetype_span": 1.0, "blend_every": 3},
}


def _bounded_int(value: int | float, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _mutate_profile(profile: UserProfile, rng: random.Random, name_suffix: str, richness: str) -> UserProfile:
    richness_profile = RICHNESS_PROFILES[richness]
    intensity = richness_profile["mutation_intensity"]
    sessions = _bounded_int(profile.total_sessions * rng.uniform(0.7, 1.25 + 0.10 * intensity), 3, 900)
    events = _bounded_int(profile.total_events * rng.uniform(0.75, 1.30 + 0.08 * intensity), 80, 25000)
    hours = max(0.5, round(profile.time_in_app_hours * rng.uniform(0.7, 1.3 + 0.08 * intensity), 1))
    card_sets = _bounded_int(profile.card_sets_generated * rng.uniform(0.65, 1.35 + 0.08 * intensity), 1, 100)
    dormancy = _bounded_int(profile.dormancy_days + rng.randint(-7, 18 + int(8 * intensity)), 0, 90)
    devices = _bounded_int(profile.multi_device_count + rng.randint(-2, 2 + int(intensity)), 1, 20)
    lifetime = _bounded_int(profile.lifetime_days + rng.randint(-40, 60 + int(25 * intensity)), 14, 1200)
    total_generations = _bounded_int(
        profile.monthly_generations_total + rng.randint(-2, 4 + int(2 * intensity)),
        max(profile.monthly_generations_total, 1),
        max(profile.monthly_generations_total + 8, 10),
    )
    remaining = _bounded_int(
        profile.monthly_generations_remaining + rng.randint(-3 - int(intensity), 3 + int(intensity)),
        0,
        total_generations,
    )
    behavior = profile.recent_behavior
    context = profile.study_context
    if richness != "standard":
        behavior = f"{behavior.rstrip('.')} {rng.choice(BEHAVIORAL_OVERLAYS)}"
        context = f"{context.rstrip('.')} {rng.choice(STUDY_CONTEXT_SUFFIXES)}"

    return replace(
        profile,
        name=f"{profile.name} {name_suffix}",
        cohort="synthetic_mutation",
        plan=profile.plan if intensity < 1.5 else rng.choice([profile.plan, "Starter", "Super Learner", "Free"]),
        status=profile.status if intensity < 1.25 else rng.choice([profile.status, "active", "inactive"]),
        total_sessions=sessions,
        total_events=events,
        time_in_app_hours=hours,
        card_sets_generated=card_sets,
        monthly_generations_remaining=remaining,
        monthly_generations_total=total_generations,
        multi_device_count=devices,
        lifetime_days=lifetime,
        dormancy_days=dormancy,
        recent_behavior=behavior,
        study_context=context,
        retry_after_mistake=profile.retry_after_mistake if intensity < 1.3 else rng.choice([True, False, profile.retry_after_mistake]),
        source_context_usage=profile.source_context_usage if intensity < 1.3 else rng.choice([True, False, profile.source_context_usage]),
        accuracy_signal=rng.choice([profile.accuracy_signal, "mixed", "mixed_with_recovery", "high_but_stressed", "low_initial"]),
    )


def _blend_profiles(primary: UserProfile, secondary: UserProfile, rng: random.Random, name_suffix: str) -> UserProfile:
    return replace(
        primary,
        name=f"{primary.name} {name_suffix}",
        cohort="synthetic_hybrid",
        plan=rng.choice([primary.plan, secondary.plan]),
        status=rng.choice([primary.status, secondary.status]),
        billing_period=rng.choice([primary.billing_period, secondary.billing_period]),
        user_type=rng.choice([primary.user_type, secondary.user_type]),
        lifetime_days=_bounded_int((primary.lifetime_days + secondary.lifetime_days) / 2, 14, 1200),
        total_sessions=_bounded_int((primary.total_sessions + secondary.total_sessions) / 2, 3, 900),
        total_events=_bounded_int((primary.total_events + secondary.total_events) / 2, 80, 25000),
        time_in_app_hours=round((primary.time_in_app_hours + secondary.time_in_app_hours) / 2, 1),
        card_sets_generated=_bounded_int((primary.card_sets_generated + secondary.card_sets_generated) / 2, 1, 100),
        monthly_generations_remaining=_bounded_int(
            (primary.monthly_generations_remaining + secondary.monthly_generations_remaining) / 2,
            0,
            max(primary.monthly_generations_total, secondary.monthly_generations_total),
        ),
        monthly_generations_total=max(primary.monthly_generations_total, secondary.monthly_generations_total),
        chat_messages_remaining=_bounded_int((primary.chat_messages_remaining + secondary.chat_messages_remaining) / 2, 0, 100),
        answer_feedback_remaining=_bounded_int((primary.answer_feedback_remaining + secondary.answer_feedback_remaining) / 2, 0, 100),
        multi_device_count=_bounded_int((primary.multi_device_count + secondary.multi_device_count) / 2, 1, 20),
        acquisition_source=rng.choice([primary.acquisition_source, secondary.acquisition_source]),
        recent_behavior=f"{primary.recent_behavior} {secondary.recent_behavior}",
        study_context=rng.choice([primary.study_context, secondary.study_context]),
        retry_after_mistake=rng.choice([primary.retry_after_mistake, secondary.retry_after_mistake]),
        source_context_usage=rng.choice([primary.source_context_usage, secondary.source_context_usage]),
        accuracy_signal=rng.choice([primary.accuracy_signal, secondary.accuracy_signal]),
        dormancy_days=_bounded_int((primary.dormancy_days + secondary.dormancy_days) / 2, 0, 90),
    )


def build_behavioral_dossier(persona: Persona, richness: str | int = "rich") -> dict[str, object]:
    if isinstance(richness, int):
        richness = {1: "standard", 2: "rich"}.get(richness, "extreme")
    profile = persona.profile
    features = persona.features
    signals = [
        f"{profile.total_sessions} sessions across {profile.lifetime_days} lifetime days",
        f"{profile.card_sets_generated} card sets generated with {profile.monthly_generations_remaining}/{profile.monthly_generations_total} generations remaining",
        f"{profile.time_in_app_hours:.1f} hours in product across {profile.multi_device_count} devices",
        f"{profile.dormancy_days} dormancy days and status {profile.status}",
        f"Study context: {profile.study_context}",
    ]
    motivations = [
        "Protect momentum" if features.habit_strength > 0.6 else "Rebuild a broken routine",
        "Reduce rework before the next milestone" if features.urgency > 0.6 else "Keep the learning loop light enough to continue",
        "Recover value from already-started work" if features.activation_score < 0.55 else "Convert existing engagement into a finish line",
    ]
    objections = [
        "Price feels heavy relative to current usage" if features.price_sensitivity > 0.6 else "May not need a pure discount to stay",
        "Could react badly to invasive specificity" if features.trust_sensitivity > 0.55 else "Can tolerate stronger contextual grounding",
        "Needs a lower-friction path back in" if features.friction_sensitivity > 0.55 else "Can handle a firmer CTA",
    ]
    levers = [
        "Pause, downgrade, or billing relief" if features.price_sensitivity > 0.58 else "Progress reinforcement or finish-the-goal framing",
        "Recovery coaching and concierge support" if features.support_need > 0.55 else "Evidence plus contextual usage recap",
        "Feature-value recap" if features.feature_awareness_gap > 0.48 else "Momentum-protection framing",
    ]
    narrative = (
        f"{persona.name} is a {profile.user_type.lower()} on {profile.plan} who shows {profile.recent_behavior.lower()} "
        f"The strongest save signals are urgency {features.urgency:.2f}, habit strength {features.habit_strength:.2f}, "
        f"and support need {features.support_need:.2f}."
    )
    if richness == "extreme":
        signals.extend(
            [
                f"Trust sensitivity {features.trust_sensitivity:.2f} and friction sensitivity {features.friction_sensitivity:.2f}",
                f"Proof need {features.proof_need:.2f} and discount affinity {features.discount_affinity:.2f}",
                f"Activation score {features.activation_score:.2f} and feature awareness gap {features.feature_awareness_gap:.2f}",
                f"Deadline pressure {features.deadline_pressure:.2f} and fatigue risk {features.fatigue_risk:.2f}",
                f"Value realization {features.value_realization:.2f} and rescue readiness {features.rescue_readiness:.2f}",
            ]
        )
        motivations.append("Prefers retention plays that feel like operational help rather than a hard sell")
    return {
        "narrative": narrative,
        "signals": signals,
        "motivations": motivations,
        "objections": objections,
        "save_levers": levers,
    }


def _build_persona_insights(profile: UserProfile, richness: str) -> PersonaInsights:
    persona = Persona(name=profile.name, profile=profile, features=derive_features(profile))
    dossier = build_behavioral_dossier(persona, richness)
    behavioral_trace = {
        "sessions": str(profile.total_sessions),
        "events": str(profile.total_events),
        "hours_in_app": f"{profile.time_in_app_hours:.1f}",
        "generated_sets": str(profile.card_sets_generated),
        "generation_balance": f"{profile.monthly_generations_remaining}/{profile.monthly_generations_total}",
        "dormancy_days": str(profile.dormancy_days),
        "devices": str(profile.multi_device_count),
        "acquisition_source": profile.acquisition_source,
        "retry_after_mistake": "yes" if profile.retry_after_mistake else "no",
        "source_context_usage": "yes" if profile.source_context_usage else "no",
        "accuracy_signal": profile.accuracy_signal,
    }
    return PersonaInsights(
        behavioral_trace=behavioral_trace,
        risk_factors=tuple(dossier["signals"]),
        retention_motivators=tuple(dossier["motivations"]),
        likely_objections=tuple(dossier["objections"]),
        recommended_hooks=tuple(dossier["save_levers"]),
        narrative=str(dossier["narrative"]),
    )


def generate_personas(
    seed_profiles: list[UserProfile],
    population: int = 60,
    seed: int = 7,
    richness: str = "standard",
    archetype_template_count: int | None = None,
    blend_every: int | None = None,
) -> list[Persona]:
    richness = richness if richness in RICHNESS_PROFILES else "standard"
    profile_meta = RICHNESS_PROFILES[richness]
    rng = random.Random(seed)
    synthetic_profiles = list(seed_profiles)
    default_template_count = round(len(ARCHETYPE_TEMPLATES) * profile_meta["archetype_span"])
    template_count = archetype_template_count or default_template_count
    template_count = max(3, min(len(ARCHETYPE_TEMPLATES), template_count))
    template_profiles = [UserProfile.from_dict(item) for item in ARCHETYPE_TEMPLATES[:template_count]]
    synthetic_profiles.extend(template_profiles)

    counter = 1
    blend_every = int(blend_every if blend_every is not None else profile_meta["blend_every"])
    while len(synthetic_profiles) < population:
        sources = seed_profiles + template_profiles
        if blend_every and len(sources) > 1 and counter % blend_every == 0:
            primary = rng.choice(sources)
            secondary = rng.choice([item for item in sources if item.name != primary.name])
            synthetic_profiles.append(_blend_profiles(primary, secondary, rng, f"Blend {counter}"))
        else:
            source = rng.choice(sources)
            synthetic_profiles.append(_mutate_profile(source, rng, f"Variant {counter}", richness))
        counter += 1

    personas: list[Persona] = []
    for profile in synthetic_profiles[:population]:
        features = derive_features(profile)
        personas.append(
            Persona(
                name=profile.name,
                profile=profile,
                features=features,
                insights=_build_persona_insights(profile, richness),
            )
        )
    return personas

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

from cta_autoresearch.ideation import generate_ideas
from cta_autoresearch.models import FeatureVector, Persona, StrategyCandidate, StrategyScore
from cta_autoresearch.personas import build_behavioral_dossier
from cta_autoresearch.config import ResearchSettings
from cta_autoresearch.research_config import ResearchConfig, control_payload
from cta_autoresearch.simulator import score_candidate_details
from cta_autoresearch.strategy_policy import (
    CTAS,
    CONTEXTUAL_GROUNDINGS,
    CREATIVE_TREATMENTS,
    FRICTION_REDUCERS,
    MESSAGE_ANGLES,
    OFFERS,
    PERSONALIZATION_LEVELS,
    PROOF_STYLES,
    all_candidates,
    candidate_key,
    candidate_label,
    offer_catalog,
    render_message,
    select_candidate_pool,
)


BASELINE = StrategyCandidate(
    message_angle="empathetic_exit",
    proof_style="none",
    offer="none",
    cta="tell_us_why",
    personalization="generic",
)


def _score_from_details(candidate: StrategyCandidate, details: list[dict[str, float]], baseline_average: float) -> StrategyScore:
    average_score = mean(item["score"] for item in details)
    retention_score = mean(item["retention"] for item in details)
    revenue_score = mean(item["revenue"] for item in details)
    trust_safety = mean(item["trust"] for item in details)
    component_scores = {
        key: mean(item[key] for item in details)
        for key in (
            "angle_fit",
            "proof_fit",
            "offer_fit",
            "cta_fit",
            "personalization_fit",
            "grounding_fit",
            "treatment_fit",
            "friction_fit",
            "trust_penalty",
        )
    }
    return StrategyScore(
        candidate=candidate,
        average_score=average_score,
        baseline_lift=average_score - baseline_average,
        retention_score=retention_score,
        revenue_score=revenue_score,
        trust_safety_score=trust_safety,
        component_scores=component_scores,
    )


def _feature_fields() -> list[str]:
    return [field for field in FeatureVector.__dataclass_fields__ if field != "segment"]


def _aggregate_persona(personas: list[Persona], label: str = "aggregate_cohort") -> Persona:
    base = personas[0]
    averaged = {
        key: mean(getattr(persona.features, key) for persona in personas)
        for key in _feature_fields()
    }
    return Persona(
        name=label,
        profile=base.profile,
        features=FeatureVector(segment=label, **averaged),
    )


def _select_validation_pool(
    personas: list[Persona],
    candidate_universe: list[StrategyCandidate],
    settings: ResearchSettings,
    idea_candidates: list[StrategyCandidate],
) -> list[StrategyCandidate]:
    if settings.strategy_depth == "exhaustive" or settings.effective_validation_budget >= len(candidate_universe):
        return candidate_universe

    aggregate = _aggregate_persona(personas)
    ranked = sorted(
        candidate_universe,
        key=lambda candidate: (
            score_candidate_details(aggregate, candidate)["score"],
            candidate.offer != "none",
            candidate.personalization != "generic",
        ),
        reverse=True,
    )

    selected: list[StrategyCandidate] = []
    seen: set[str] = set()

    def add(candidate: StrategyCandidate) -> None:
        key = candidate_key(candidate)
        if key in seen:
            return
        selected.append(candidate)
        seen.add(key)

    add(BASELINE)
    for candidate in idea_candidates:
        add(candidate)

    diversity_stride = max(len(ranked) // max(settings.effective_validation_budget, 1), 1)
    for index, candidate in enumerate(ranked):
        if len(selected) >= settings.effective_validation_budget:
            break
        if index < int(settings.effective_validation_budget * 0.65) or index % diversity_stride == 0:
            add(candidate)

    return selected[: settings.effective_validation_budget]


def analyze_search_space(personas: list[Persona], settings: ResearchSettings | None = None) -> dict:
    settings = settings or ResearchSettings(base_population=len(personas))
    offers = offer_catalog(settings)

    baseline_by_persona = {
        persona.name: score_candidate_details(persona, BASELINE)["score"]
        for persona in personas
    }
    baseline_average = mean(baseline_by_persona.values())

    segment_personas: dict[str, list[Persona]] = defaultdict(list)
    for persona in personas:
        segment_personas[persona.features.segment].append(persona)

    baseline_by_segment = {
        segment: mean(baseline_by_persona[persona.name] for persona in group)
        for segment, group in segment_personas.items()
    }

    candidate_universe = select_candidate_pool(settings)
    ideas, warnings = generate_ideas(personas, candidate_universe, settings)
    selected_candidates = _select_validation_pool(
        personas,
        candidate_universe,
        settings,
        [proposal.candidate for proposal in ideas],
    )

    results: list[StrategyScore] = []
    best_by_segment: dict[str, StrategyScore] = {}
    best_by_persona: dict[str, StrategyScore] = {}
    persona_lookup = {persona.name: persona for persona in personas}

    for candidate in selected_candidates:
        detail_by_persona = {
            persona.name: score_candidate_details(persona, candidate)
            for persona in personas
        }
        all_details = list(detail_by_persona.values())
        score = _score_from_details(candidate, all_details, baseline_average)
        results.append(score)

        for segment, group in segment_personas.items():
            segment_details = [detail_by_persona[persona.name] for persona in group]
            segment_score = _score_from_details(candidate, segment_details, baseline_by_segment[segment])
            current = best_by_segment.get(segment)
            if current is None or segment_score.average_score > current.average_score:
                best_by_segment[segment] = segment_score

        for persona in personas:
            detail = detail_by_persona[persona.name]
            persona_score = StrategyScore(
                candidate=candidate,
                average_score=detail["score"],
                baseline_lift=detail["score"] - baseline_by_persona[persona.name],
                retention_score=detail["retention"],
                revenue_score=detail["revenue"],
                trust_safety_score=detail["trust"],
                component_scores={
                    "angle_fit": detail["angle_fit"],
                    "proof_fit": detail["proof_fit"],
                    "offer_fit": detail["offer_fit"],
                    "cta_fit": detail["cta_fit"],
                    "personalization_fit": detail["personalization_fit"],
                    "grounding_fit": detail["grounding_fit"],
                    "treatment_fit": detail["treatment_fit"],
                    "friction_fit": detail["friction_fit"],
                    "trust_penalty": detail["trust_penalty"],
                },
            )
            current = best_by_persona.get(persona.name)
            if current is None or persona_score.average_score > current.average_score:
                best_by_persona[persona.name] = persona_score

    results.sort(key=lambda item: (item.average_score, item.trust_safety_score, item.revenue_score), reverse=True)
    return {
        "baseline_average": baseline_average,
        "results": results,
        "best_by_segment": best_by_segment,
        "best_by_persona": best_by_persona,
        "persona_lookup": persona_lookup,
        "idea_proposals": ideas,
        "warnings": warnings,
        "candidate_universe_size": len(candidate_universe),
        "validated_candidate_count": len(selected_candidates),
        "offers": offers,
        "settings": settings,
    }


def evaluate_candidates(
    personas: list[Persona],
    settings: ResearchSettings | None = None,
) -> tuple[float, list[StrategyScore]]:
    analysis = analyze_search_space(personas, settings=settings)
    return analysis["baseline_average"], analysis["results"]


def best_for_persona(persona: Persona, settings: ResearchSettings | None = None) -> StrategyScore:
    settings = settings or ResearchSettings(population=2)
    return analyze_search_space([persona], settings=settings)["best_by_persona"][persona.name]


def segment_leaders(personas: list[Persona], settings: ResearchSettings | None = None) -> dict[str, StrategyScore]:
    return analyze_search_space(personas, settings=settings)["best_by_segment"]


def score_to_dict(
    score: StrategyScore,
    sample_persona: Persona | None = None,
    settings: ResearchSettings | None = None,
) -> dict:
    offers = offer_catalog(settings)
    payload = {
        "id": candidate_key(score.candidate),
        "label": candidate_label(score.candidate, offers=offers),
        "message_angle": score.candidate.message_angle,
        "proof_style": score.candidate.proof_style,
        "offer": score.candidate.offer,
        "cta": score.candidate.cta,
        "personalization": score.candidate.personalization,
        "contextual_grounding": score.candidate.contextual_grounding,
        "creative_treatment": score.candidate.creative_treatment,
        "friction_reducer": score.candidate.friction_reducer,
        "message_angle_label": MESSAGE_ANGLES[score.candidate.message_angle]["label"],
        "proof_style_label": PROOF_STYLES[score.candidate.proof_style]["label"],
        "offer_label": offers[score.candidate.offer]["label"],
        "cta_label": CTAS[score.candidate.cta]["label"],
        "personalization_label": PERSONALIZATION_LEVELS[score.candidate.personalization]["label"],
        "contextual_grounding_label": CONTEXTUAL_GROUNDINGS[score.candidate.contextual_grounding]["label"],
        "creative_treatment_label": CREATIVE_TREATMENTS[score.candidate.creative_treatment]["label"],
        "friction_reducer_label": FRICTION_REDUCERS[score.candidate.friction_reducer]["label"],
        "average_score": round(score.average_score, 4),
        "baseline_lift": round(score.baseline_lift, 4),
        "retention_score": round(score.retention_score, 4),
        "revenue_score": round(score.revenue_score, 4),
        "trust_safety_score": round(score.trust_safety_score, 4),
        "component_scores": {key: round(value, 4) for key, value in score.component_scores.items()},
    }
    if sample_persona:
        payload["sample_message"] = render_message(sample_persona, score.candidate, settings)
    return payload


def build_report(
    personas: list[Persona],
    top_n: int = 5,
    settings: ResearchSettings | None = None,
) -> tuple[str, dict[str, str]]:
    settings = settings or ResearchSettings(base_population=len(personas))
    analysis = analyze_search_space(personas, settings=settings)
    baseline_average = analysis["baseline_average"]
    results = analysis["results"]
    leaders = results[:top_n]
    non_discount = [result for result in results if not result.candidate.offer.startswith("discount_")]
    segments = analysis["best_by_segment"]

    seed_personas = [persona for persona in personas if persona.profile.cohort == "seed_profile"]
    representative = seed_personas[0] if seed_personas else personas[0]
    seed_winners = {persona.name: analysis["best_by_persona"][persona.name] for persona in seed_personas}

    top = leaders[0]
    best_non_discount = non_discount[0]
    metrics = {
        "baseline_retention_score": f"{baseline_average:.4f}",
        "expected_retention_score": f"{top.average_score:.4f}",
        "estimated_lift": f"{top.baseline_lift:.4f}",
        "trust_safety_score": f"{top.trust_safety_score:.4f}",
        "personas_evaluated": str(len(personas)),
        "top_strategy": candidate_label(top.candidate, offers=analysis["offers"]),
        "search_space_size": str(len(results)),
        "validated_strategy_count": str(analysis["validated_candidate_count"]),
        "best_non_discount_strategy": candidate_label(best_non_discount.candidate, offers=analysis["offers"]),
        "model_name": settings.model_name,
    }

    lines = [
        "# Sample Optimization Report",
        "",
        "## Cohort Summary",
        f"- Personas evaluated: {len(personas)}",
        f"- Search space available: {analysis['candidate_universe_size']} valid strategies",
        f"- Strategies validated this run: {analysis['validated_candidate_count']}",
        f"- Baseline score: {baseline_average:.4f}",
        f"- Best composite score: {top.average_score:.4f}",
        f"- Estimated lift vs baseline: {top.baseline_lift:.4f}",
        f"- Best trust-safety score among leaders: {top.trust_safety_score:.4f}",
        f"- Best non-discount strategy: {candidate_label(best_non_discount.candidate, offers=analysis['offers'])}",
        f"- Model: {settings.model_name}",
        "",
        "## Top Strategies",
    ]

    for index, result in enumerate(leaders, start=1):
        lines.extend(
            [
                f"### {index}. {candidate_label(result.candidate, offers=analysis['offers'])}",
                f"- Composite score: {result.average_score:.4f}",
                f"- Lift: {result.baseline_lift:.4f}",
                f"- Retention score: {result.retention_score:.4f}",
                f"- Revenue score: {result.revenue_score:.4f}",
                f"- Trust-safety score: {result.trust_safety_score:.4f}",
                f"- Sample message: {render_message(representative, result.candidate, settings)}",
                "",
            ]
        )

    if analysis["idea_proposals"]:
        lines.append("## Generated Ideas")
        for proposal in analysis["idea_proposals"][: min(len(analysis["idea_proposals"]), 8)]:
            lines.extend(
                [
                    f"### {proposal.agent_role}: {proposal.label}",
                    f"- Segment: {proposal.target_segment}",
                    f"- Thesis: {proposal.thesis}",
                    f"- Rationale: {proposal.rationale}",
                    f"- Sample message: {proposal.sample_message}",
                    "",
                ]
            )

    lines.append("## Segment Leaders")
    for segment, result in sorted(segments.items()):
        lines.extend(
            [
                f"### {segment}",
                f"- Best strategy: {candidate_label(result.candidate, offers=analysis['offers'])}",
                f"- Composite score: {result.average_score:.4f}",
                f"- Lift: {result.baseline_lift:.4f}",
                "",
            ]
        )

    lines.append("## Seed Persona Playbooks")
    for persona in seed_personas:
        winner = seed_winners[persona.name]
        lines.extend(
            [
                f"### {persona.name}",
                f"- Segment: {persona.features.segment}",
                f"- Recommended strategy: {candidate_label(winner.candidate, offers=analysis['offers'])}",
                f"- Composite score: {winner.average_score:.4f}",
                f"- Retention score: {winner.retention_score:.4f}",
                f"- Revenue score: {winner.revenue_score:.4f}",
                f"- Trust-safety score: {winner.trust_safety_score:.4f}",
                f"- Lift vs baseline: {winner.baseline_lift:.4f}",
                f"- Suggested copy: {render_message(persona, winner.candidate, settings)}",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n", metrics


def build_dashboard_payload(personas: list[Persona], settings: ResearchSettings | None = None) -> dict:
    settings = settings or ResearchSettings(base_population=len(personas))
    analysis = analyze_search_space(personas, settings=settings)
    control_config = ResearchConfig.from_overrides(
        population=len(personas),
        seed=settings.seed,
        top_n=settings.top_n,
        depth_mode="extreme" if settings.depth >= 4 else "deep" if settings.depth >= 3 else "standard" if settings.depth >= 2 else "quick",
        persona_richness="extreme" if settings.persona_richness >= 3 else "rich" if settings.persona_richness >= 2 else "standard",
        validation_budget=settings.effective_validation_budget,
        ideation_rounds=settings.ideation_rounds,
        model_provider="openai" if settings.use_llm else "heuristic",
        model_name=settings.model,
    )
    results = analysis["results"]
    seed_personas = [persona for persona in personas if persona.profile.cohort == "seed_profile"]
    representative = seed_personas[0] if seed_personas else personas[0]
    top = results[: settings.top_n]
    non_discount = [result for result in results if not result.candidate.offer.startswith("discount_")]
    offers = analysis["offers"]

    message_counter = Counter(item.candidate.message_angle for item in top)
    offer_counter = Counter(item.candidate.offer for item in top)
    cta_counter = Counter(item.candidate.cta for item in top)
    treatment_counter = Counter(item.candidate.creative_treatment for item in top)

    persona_payload = []
    for persona in personas:
        best = analysis["best_by_persona"][persona.name]
        dossier = build_behavioral_dossier(persona, richness=settings.persona_richness)
        persona_payload.append(
            {
                "name": persona.name,
                "cohort": persona.profile.cohort,
                "plan": persona.profile.plan,
                "status": persona.profile.status,
                "study_context": persona.profile.study_context,
                "segment": persona.features.segment,
                "profile_summary": persona.profile.recent_behavior,
                "raw_profile": vars(persona.profile),
                "behavioral_trace": vars(persona.profile),
                "features": {
                    key: round(value, 4)
                    for key, value in vars(persona.features).items()
                    if key != "segment"
                },
                "dossier": dossier,
                "narrative": dossier["narrative"],
                "risk_factors": dossier["signals"],
                "retention_motivators": dossier["motivations"],
                "likely_objections": dossier["objections"],
                "recommended_hooks": dossier["save_levers"],
                "best_strategy": score_to_dict(best, sample_persona=persona, settings=settings),
            }
        )

    dimensions = {
        "message_angles": MESSAGE_ANGLES,
        "proof_styles": PROOF_STYLES,
        "offers": offers,
        "ctas": {
            key: {
                **value,
                "allowed_offer_kinds": sorted(value["allowed_offer_kinds"]),
            }
            for key, value in CTAS.items()
        },
        "personalization": PERSONALIZATION_LEVELS,
        "contextual_groundings": CONTEXTUAL_GROUNDINGS,
        "creative_treatments": CREATIVE_TREATMENTS,
        "friction_reducers": FRICTION_REDUCERS,
    }

    preview_candidates = results[: settings.workbench_limit]
    idea_scores = {candidate_key(score.candidate): score for score in results}

    return {
        "meta": {
            "personas_evaluated": len(personas),
            "search_space_size": len(results),
            "validated_strategy_count": analysis["validated_candidate_count"],
            "workbench_candidate_count": len(preview_candidates),
            "baseline_score": round(analysis["baseline_average"], 4),
            "top_score": round(top[0].average_score, 4),
            "top_lift": round(top[0].baseline_lift, 4),
            "top_strategy": candidate_label(top[0].candidate, offers=offers),
            "best_non_discount_strategy": candidate_label(non_discount[0].candidate, offers=offers),
            "warnings": analysis["warnings"],
            "research_settings": settings.as_dict(),
            "model_backend": settings.model_name,
        },
        "controls": control_payload(control_config),
        "research_meta": {
            "provider_status": control_config.provider_status(),
            "generated_candidates_proposed": len(analysis["idea_proposals"]),
            "validated_candidates": analysis["validated_candidate_count"],
        },
        "dimensions": dimensions,
        "top_patterns": {
            "message_angles": {MESSAGE_ANGLES[key]["label"]: count for key, count in message_counter.items()},
            "offers": {offers[key]["label"]: count for key, count in offer_counter.items()},
            "ctas": {CTAS[key]["label"]: count for key, count in cta_counter.items()},
            "creative_treatments": {CREATIVE_TREATMENTS[key]["label"]: count for key, count in treatment_counter.items()},
        },
        "top_strategies": [score_to_dict(item, sample_persona=representative, settings=settings) for item in top],
        "best_non_discount": [score_to_dict(item, sample_persona=representative, settings=settings) for item in non_discount[:10]],
        "all_candidates": [score_to_dict(item, settings=settings) for item in preview_candidates],
        "segment_leaders": {
            segment: score_to_dict(result, sample_persona=representative, settings=settings)
            for segment, result in sorted(analysis["best_by_segment"].items())
        },
        "personas": persona_payload,
        "idea_agents": [
            {
                "id": proposal.id,
                "agent_role": proposal.agent_role,
                "label": proposal.label,
                "thesis": proposal.thesis,
                "rationale": proposal.rationale,
                "target_segment": proposal.target_segment,
                "confidence": proposal.confidence,
                "sample_message": proposal.sample_message,
                "validated_score": round(idea_scores[candidate_key(proposal.candidate)].average_score, 4)
                if candidate_key(proposal.candidate) in idea_scores
                else None,
                "validated_lift": round(idea_scores[candidate_key(proposal.candidate)].baseline_lift, 4)
                if candidate_key(proposal.candidate) in idea_scores
                else None,
            }
            for proposal in analysis["idea_proposals"]
        ],
        "idea_proposals": [
            {
                "id": proposal.id,
                "agent_role": proposal.agent_role,
                "label": proposal.label,
                "thesis": proposal.thesis,
                "rationale": proposal.rationale,
                "target_segment": proposal.target_segment,
                "confidence": proposal.confidence,
                "sample_message": proposal.sample_message,
                "average_score": round(idea_scores[candidate_key(proposal.candidate)].average_score, 4)
                if candidate_key(proposal.candidate) in idea_scores
                else None,
            }
            for proposal in analysis["idea_proposals"]
        ],
    }

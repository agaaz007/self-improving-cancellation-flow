from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
import random

from cta_autoresearch.models import FeatureVector, Persona, StrategyCandidate, StrategyScore
from cta_autoresearch.openai_research import evaluate_candidates_via_api, evaluate_persona_shortlist_via_api
from cta_autoresearch.personas import build_behavioral_dossier
from cta_autoresearch.research_settings import ResearchSettings
from cta_autoresearch.simulator import OFFERS as SIMULATOR_OFFERS, score_candidate_details
from cta_autoresearch.strategy_policy import (
    CONTEXTUAL_GROUNDINGS,
    CREATIVE_TREATMENTS,
    CTAS,
    FRICTION_REDUCERS,
    MESSAGE_ANGLES,
    PERSONALIZATION_LEVELS,
    PROOF_STYLES,
    candidate_key,
    candidate_label,
    offer_catalog,
    render_message,
    valid_candidate,
)
from cta_autoresearch.swarm_ideation import generate_ideas


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
    component_keys = (
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
    component_scores = {
        key: mean(item[key] for item in details)
        for key in component_keys
        if key in details[0]
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
        insights=None,
    )


def _detail_or_fallback(
    score_map: dict[str, dict[str, dict[str, float]]] | None,
    candidate: StrategyCandidate,
    target_id: str,
    persona: Persona,
) -> dict[str, float]:
    if score_map is not None:
        details = score_map.get(candidate_key(candidate), {}).get(target_id)
        if details is not None:
            return details
    return score_candidate_details(persona, candidate)


def _candidate_universe(settings: ResearchSettings) -> list[StrategyCandidate]:
    grounding_keys = list(CONTEXTUAL_GROUNDINGS)[: settings.grounding_limit]
    treatment_keys = list(CREATIVE_TREATMENTS)[: settings.treatment_limit]
    friction_keys = list(FRICTION_REDUCERS)[: settings.friction_limit]
    offers = {
        key: value
        for key, value in offer_catalog(settings).items()
        if key in SIMULATOR_OFFERS
    }

    candidates: list[StrategyCandidate] = []
    for message_angle in MESSAGE_ANGLES:
        for proof_style in PROOF_STYLES:
            for offer in offers:
                for cta in CTAS:
                    for personalization in PERSONALIZATION_LEVELS:
                        for grounding in grounding_keys:
                            for treatment in treatment_keys:
                                for friction in friction_keys:
                                    candidate = StrategyCandidate(
                                        message_angle=message_angle,
                                        proof_style=proof_style,
                                        offer=offer,
                                        cta=cta,
                                        personalization=personalization,
                                        contextual_grounding=grounding,
                                        creative_treatment=treatment,
                                        friction_reducer=friction,
                                    )
                                    if valid_candidate(candidate, settings):
                                        candidates.append(candidate)
    return candidates


def _representative_candidate_pool(
    full_universe: list[StrategyCandidate],
    settings: ResearchSettings,
) -> list[StrategyCandidate]:
    budget = min(settings.effective_validation_budget, len(full_universe))
    if budget >= len(full_universe):
        return full_universe

    rng = random.Random(settings.seed)
    shuffled = list(full_universe)
    rng.shuffle(shuffled)
    selected: list[StrategyCandidate] = []
    seen: set[str] = set()

    def add(candidate: StrategyCandidate) -> None:
        key = candidate_key(candidate)
        if key in seen or len(selected) >= budget:
            return
        selected.append(candidate)
        seen.add(key)

    for attr in (
        "message_angle",
        "offer",
        "cta",
        "personalization",
        "contextual_grounding",
        "creative_treatment",
        "friction_reducer",
    ):
        values = []
        for candidate in shuffled:
            value = getattr(candidate, attr)
            if value not in values:
                values.append(value)
        for value in values:
            for candidate in shuffled:
                if getattr(candidate, attr) == value:
                    add(candidate)
                    break

    for candidate in shuffled:
        add(candidate)
        if len(selected) >= budget:
            break
    return selected


def _select_candidates(
    personas: list[Persona],
    settings: ResearchSettings,
    progress_callback=None,
) -> tuple[list[StrategyCandidate], list, list[str], int, int]:
    full_universe = _candidate_universe(settings)
    structural_pool = _representative_candidate_pool(full_universe, settings)
    if progress_callback:
        progress_callback(0.4, "search-space", f"Selected {len(structural_pool)} structural candidates from {len(full_universe)} possible combinations.")
    idea_proposals, warnings = generate_ideas(personas, structural_pool, settings, progress_callback=progress_callback)

    selected: list[StrategyCandidate] = []
    seen: set[str] = set()

    def add(candidate: StrategyCandidate) -> None:
        key = candidate_key(candidate)
        if key in seen:
            return
        selected.append(candidate)
        seen.add(key)

    add(BASELINE)
    for candidate in structural_pool:
        add(candidate)
    for proposal in idea_proposals:
        add(proposal.candidate)

    budget = settings.effective_validation_budget
    return selected[:budget], idea_proposals, warnings, len(structural_pool), len(full_universe)


def analyze_search_space(personas: list[Persona], settings: ResearchSettings | None = None, progress_callback=None) -> dict:
    settings = settings or ResearchSettings(population=len(personas))
    if progress_callback:
        progress_callback(0.34, "search-space", "Constructing candidate universe and selecting a representative validation pool.")
    if settings.api_only:
        return _analyze_search_space_api_only(personas, settings, progress_callback=progress_callback)
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

    selected_candidates, ideas, warnings, structural_count, full_universe_size = _select_candidates(
        personas,
        settings,
        progress_callback=progress_callback,
    )
    if progress_callback:
        progress_callback(0.46, "ideation", f"Prepared {len(ideas)} swarm proposals across {settings.ideation_agents} agents.")

    results: list[StrategyScore] = []
    best_by_segment: dict[str, StrategyScore] = {}
    best_by_persona: dict[str, StrategyScore] = {}

    total_candidates = max(len(selected_candidates), 1)
    progress_every = max(total_candidates // 8, 1)
    for index, candidate in enumerate(selected_candidates, start=1):
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
                component_scores={key: value for key, value in detail.items() if key.endswith("_fit") or key == "trust_penalty"},
            )
            current = best_by_persona.get(persona.name)
            if current is None or persona_score.average_score > current.average_score:
                best_by_persona[persona.name] = persona_score
        if progress_callback and (index == 1 or index % progress_every == 0 or index == total_candidates):
            progress = 0.5 + 0.38 * (index / total_candidates)
            progress_callback(progress, "validation", f"Validated {index} of {total_candidates} candidate combinations.")

    results.sort(key=lambda item: (item.average_score, item.trust_safety_score, item.revenue_score), reverse=True)
    if progress_callback:
        progress_callback(0.94, "ranking", "Ranked strategies and selected segment and persona winners.")
    return {
        "baseline_average": baseline_average,
        "results": results,
        "best_by_segment": best_by_segment,
        "best_by_persona": best_by_persona,
        "idea_proposals": ideas,
        "warnings": list(dict.fromkeys(warnings)),
        "candidate_universe_size": full_universe_size,
        "validated_candidate_count": len(results),
        "structural_candidate_count": structural_count,
        "offers": offers,
        "settings": settings,
        "active_backend": (
            f"hybrid:{settings.model_name}+heuristic-validation"
            if settings.openai_for_research
            else "heuristic:heuristic-simulator"
        ),
    }


def _analyze_search_space_api_only(personas: list[Persona], settings: ResearchSettings, progress_callback=None) -> dict:
    offers = offer_catalog(settings)
    segment_personas: dict[str, list[Persona]] = defaultdict(list)
    for persona in personas:
        segment_personas[persona.features.segment].append(persona)

    segment_aggregates = {
        segment: _aggregate_persona(group, label=segment)
        for segment, group in segment_personas.items()
    }
    cohort_persona = _aggregate_persona(personas, label="cohort")
    selected_candidates, ideas, warnings, structural_count, full_universe_size = _select_candidates(
        personas,
        settings,
        progress_callback=progress_callback,
    )
    if progress_callback:
        progress_callback(0.46, "ideation", f"Prepared {len(ideas)} swarm proposals across {settings.ideation_agents} agents.")

    score_map, api_warnings = evaluate_candidates_via_api(
        cohort_persona=cohort_persona,
        segment_personas=segment_aggregates,
        candidates=selected_candidates,
        settings=settings,
        progress_callback=progress_callback,
    )
    warnings.extend(api_warnings)

    baseline_detail = _detail_or_fallback(score_map, BASELINE, "cohort", cohort_persona)
    baseline_average = baseline_detail["score"]
    baseline_by_segment = {
        segment: _detail_or_fallback(score_map, BASELINE, f"segment::{segment}", aggregate)["score"]
        for segment, aggregate in segment_aggregates.items()
    }

    results: list[StrategyScore] = []
    best_by_segment: dict[str, StrategyScore] = {}
    best_by_persona: dict[str, StrategyScore] = {}

    for candidate in selected_candidates:
        cohort_detail = _detail_or_fallback(score_map, candidate, "cohort", cohort_persona)
        results.append(_score_from_details(candidate, [cohort_detail], baseline_average))

        for segment, aggregate in segment_aggregates.items():
            segment_detail = _detail_or_fallback(score_map, candidate, f"segment::{segment}", aggregate)
            segment_score = _score_from_details(candidate, [segment_detail], baseline_by_segment[segment])
            current = best_by_segment.get(segment)
            if current is None or segment_score.average_score > current.average_score:
                best_by_segment[segment] = segment_score

    shortlist_size = min(max(settings.top_n * settings.persona_shortlist_multiplier, 12), len(results))
    shortlist = [score.candidate for score in results[:shortlist_size]]
    persona_score_map, persona_warnings = evaluate_persona_shortlist_via_api(
        personas=personas,
        candidates=shortlist,
        settings=settings,
        progress_callback=progress_callback,
    )
    warnings.extend(persona_warnings)

    baseline_by_persona = {}
    if persona_score_map is not None:
        for persona in personas:
            baseline_by_persona[persona.name] = _detail_or_fallback(
                persona_score_map,
                BASELINE,
                f"persona::{persona.name}",
                persona,
            )["score"]

    for persona in personas:
        current_best: StrategyScore | None = None
        for candidate in shortlist:
            target_id = f"persona::{persona.name}"
            detail = _detail_or_fallback(persona_score_map, candidate, target_id, persona)
            baseline = baseline_by_persona.get(persona.name)
            if baseline is None:
                baseline = score_candidate_details(persona, BASELINE)["score"]
            persona_score = StrategyScore(
                candidate=candidate,
                average_score=detail["score"],
                baseline_lift=detail["score"] - baseline,
                retention_score=detail["retention"],
                revenue_score=detail["revenue"],
                trust_safety_score=detail["trust"],
                component_scores={key: value for key, value in detail.items() if key.endswith("_fit") or key == "trust_penalty"},
            )
            if current_best is None or persona_score.average_score > current_best.average_score:
                current_best = persona_score
        if current_best is not None:
            best_by_persona[persona.name] = current_best

    results.sort(key=lambda item: (item.average_score, item.trust_safety_score, item.revenue_score), reverse=True)
    if progress_callback:
        progress_callback(0.94, "ranking", "Ranked API-scored strategies and prepared persona winners.")
    active_backend = "fallback:heuristic-simulator"
    if score_map is not None:
        active_backend = f"api_only:{settings.model_name}"
    return {
        "baseline_average": baseline_average,
        "results": results,
        "best_by_segment": best_by_segment,
        "best_by_persona": best_by_persona,
        "idea_proposals": ideas,
        "warnings": list(dict.fromkeys(warnings)),
        "candidate_universe_size": full_universe_size,
        "validated_candidate_count": len(results),
        "structural_candidate_count": structural_count,
        "offers": offers,
        "settings": settings,
        "active_backend": active_backend,
    }


def evaluate_candidates(
    personas: list[Persona],
    settings: ResearchSettings | None = None,
) -> tuple[float, list[StrategyScore]]:
    analysis = analyze_search_space(personas, settings=settings)
    return analysis["baseline_average"], analysis["results"]


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


def build_report(personas: list[Persona], top_n: int = 5, settings: ResearchSettings | None = None) -> tuple[str, dict[str, str]]:
    settings = settings or ResearchSettings(population=len(personas), top_n=top_n)
    analysis = analyze_search_space(personas, settings=settings)
    leaders = analysis["results"][:top_n]
    representative = personas[0]
    top = leaders[0]
    non_discount = next(item for item in analysis["results"] if not item.candidate.offer.startswith("discount_"))

    metrics = {
        "baseline_retention_score": f"{analysis['baseline_average']:.4f}",
        "expected_retention_score": f"{top.average_score:.4f}",
        "estimated_lift": f"{top.baseline_lift:.4f}",
        "trust_safety_score": f"{top.trust_safety_score:.4f}",
        "personas_evaluated": str(len(personas)),
        "search_space_size": str(analysis["candidate_universe_size"]),
        "validated_strategy_count": str(analysis["validated_candidate_count"]),
        "top_strategy": candidate_label(top.candidate, offers=analysis["offers"]),
        "best_non_discount_strategy": candidate_label(non_discount.candidate, offers=analysis["offers"]),
        "model_name": settings.model_name,
    }

    lines = [
        "# Sample Optimization Report",
        "",
        f"- Personas evaluated: {len(personas)}",
        f"- Strategies validated: {analysis['validated_candidate_count']}",
        f"- Structural candidates considered: {analysis['structural_candidate_count']}",
        f"- Best strategy: {metrics['top_strategy']}",
        f"- Best non-discount strategy: {metrics['best_non_discount_strategy']}",
        "",
        "## Top Strategies",
    ]

    for index, result in enumerate(leaders, start=1):
        lines.extend(
            [
                f"### {index}. {candidate_label(result.candidate, offers=analysis['offers'])}",
                f"- Composite score: {result.average_score:.4f}",
                f"- Lift: {result.baseline_lift:.4f}",
                f"- Revenue score: {result.revenue_score:.4f}",
                f"- Trust-safety score: {result.trust_safety_score:.4f}",
                f"- Sample message: {render_message(representative, result.candidate, settings)}",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n", metrics


def build_dashboard_payload(personas: list[Persona], settings: ResearchSettings | None = None, progress_callback=None) -> dict:
    settings = settings or ResearchSettings(population=len(personas))
    analysis = analyze_search_space(personas, settings=settings, progress_callback=progress_callback)
    offers = analysis["offers"]
    top = analysis["results"][: settings.top_n]
    representative = personas[0]
    preview_candidates = analysis["results"][: settings.workbench_limit]

    message_counter = Counter(item.candidate.message_angle for item in top)
    offer_counter = Counter(item.candidate.offer for item in top)
    cta_counter = Counter(item.candidate.cta for item in top)

    return {
        "meta": {
            "personas_evaluated": len(personas),
            "search_space_size": analysis["candidate_universe_size"],
            "validated_strategy_count": analysis["validated_candidate_count"],
            "workbench_candidate_count": len(preview_candidates),
            "baseline_score": round(analysis["baseline_average"], 4),
            "top_score": round(top[0].average_score, 4),
            "top_lift": round(top[0].baseline_lift, 4),
            "top_strategy": candidate_label(top[0].candidate, offers=offers),
            "best_non_discount_strategy": next(
                candidate_label(item.candidate, offers=offers)
                for item in analysis["results"]
                if not item.candidate.offer.startswith("discount_")
            ),
            "warnings": analysis["warnings"],
            "research_settings": settings.as_dict(),
            "model_backend": analysis["active_backend"],
        },
        "dimensions": {
            "message_angles": MESSAGE_ANGLES,
            "proof_styles": PROOF_STYLES,
            "offers": offers,
            "ctas": {key: {**value, "allowed_offer_kinds": sorted(value["allowed_offer_kinds"])} for key, value in CTAS.items()},
            "personalization": PERSONALIZATION_LEVELS,
        },
        "top_patterns": {
            "message_angles": {MESSAGE_ANGLES[key]["label"]: count for key, count in message_counter.items()},
            "offers": {offers[key]["label"]: count for key, count in offer_counter.items()},
            "ctas": {CTAS[key]["label"]: count for key, count in cta_counter.items()},
        },
        "top_strategies": [score_to_dict(item, sample_persona=representative, settings=settings) for item in top],
        "best_non_discount": [
            score_to_dict(item, sample_persona=representative, settings=settings)
            for item in analysis["results"]
            if not item.candidate.offer.startswith("discount_")
        ][:10],
        "all_candidates": [score_to_dict(item, settings=settings) for item in preview_candidates],
        "segment_leaders": {
            segment: score_to_dict(result, sample_persona=representative, settings=settings)
            for segment, result in sorted(analysis["best_by_segment"].items())
        },
        "personas": [
            {
                "name": persona.name,
                "cohort": persona.profile.cohort,
                "plan": persona.profile.plan,
                "status": persona.profile.status,
                "study_context": persona.profile.study_context,
                "segment": persona.features.segment,
                "profile_summary": persona.profile.recent_behavior,
                "raw_profile": vars(persona.profile),
                "features": {key: round(value, 4) for key, value in vars(persona.features).items() if key != "segment"},
                "dossier": build_behavioral_dossier(persona, richness=settings.persona_richness),
                "best_strategy": score_to_dict(analysis["best_by_persona"][persona.name], sample_persona=persona, settings=settings),
            }
            for persona in personas
        ],
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
            }
            for proposal in analysis["idea_proposals"]
        ],
    }

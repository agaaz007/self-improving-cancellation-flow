from __future__ import annotations

import unittest

from cta_autoresearch.autoresearch.compiler import compile_flow_spec
from cta_autoresearch.autoresearch.schemas import FlowResearchSpec
from cta_autoresearch.features import derive_features
from cta_autoresearch.models import Persona, StrategyCandidate
from cta_autoresearch.lab_optimizer import build_dashboard_payload, build_report, evaluate_candidates
from cta_autoresearch.personas import generate_personas
from cta_autoresearch.research_settings import HEURISTIC_MODEL, build_settings
from cta_autoresearch.sample_data import load_seed_profiles
from cta_autoresearch.simulator import score_candidate_details
from cta_autoresearch.strategy_policy import offer_catalog
from cta_autoresearch.swarm_ideation import generate_ideas


class StrategyScoringTest(unittest.TestCase):
    def test_power_user_prefers_progress_and_goal_completion_over_deep_discount(self) -> None:
        kaitlyn, _ = load_seed_profiles()
        persona = Persona(name=kaitlyn.name, profile=kaitlyn, features=derive_features(kaitlyn))

        progress_candidate = StrategyCandidate(
            message_angle="progress_reflection",
            proof_style="personal_usage_signal",
            offer="exam_sprint",
            cta="finish_current_goal",
            personalization="behavioral",
        )
        deep_discount_candidate = StrategyCandidate(
            message_angle="outcome_proof",
            proof_style="quantified_outcome",
            offer="discount_100",
            cta="claim_offer",
            personalization="generic",
        )

        self.assertGreater(
            score_candidate_details(persona, progress_candidate)["score"],
            score_candidate_details(persona, deep_discount_candidate)["score"],
        )

    def test_dormant_free_user_prefers_reactivation_offer_over_generic_stay(self) -> None:
        _, lillian = load_seed_profiles()
        persona = Persona(name=lillian.name, profile=lillian, features=derive_features(lillian))

        reactivation_candidate = StrategyCandidate(
            message_angle="feature_unlock",
            proof_style="similar_user_story",
            offer="bonus_credits",
            cta="claim_offer",
            personalization="contextual",
        )
        generic_candidate = StrategyCandidate(
            message_angle="habit_identity",
            proof_style="none",
            offer="none",
            cta="stay_on_current_plan",
            personalization="generic",
        )

        self.assertGreater(
            score_candidate_details(persona, reactivation_candidate)["score"],
            score_candidate_details(persona, generic_candidate)["score"],
        )

    def test_composite_score_uses_only_retention_and_revenue(self) -> None:
        kaitlyn, _ = load_seed_profiles()
        persona = Persona(name=kaitlyn.name, profile=kaitlyn, features=derive_features(kaitlyn))
        candidate = StrategyCandidate(
            message_angle="progress_reflection",
            proof_style="personal_usage_signal",
            offer="exam_sprint",
            cta="finish_current_goal",
            personalization="behavioral",
        )

        details = score_candidate_details(persona, candidate)

        self.assertAlmostEqual(
            details["score"],
            0.60 * details["retention"] + 0.40 * details["revenue"],
        )

    def test_search_space_is_large_and_report_exposes_non_discount_recommendation(self) -> None:
        standard = build_settings({"population": 6, "strategy_depth": "standard"})
        deep = build_settings({"population": 6, "strategy_depth": "deep"})
        standard_personas = generate_personas(load_seed_profiles(), population=6, seed=7, richness="rich")
        standard_payload = build_dashboard_payload(standard_personas, settings=standard)
        deep_payload = build_dashboard_payload(standard_personas, settings=deep)
        self.assertGreater(standard_payload["meta"]["search_space_size"], 1000)
        self.assertGreater(deep_payload["meta"]["search_space_size"], standard_payload["meta"]["search_space_size"])

        personas = generate_personas(load_seed_profiles(), population=6, seed=7, richness="rich")
        baseline, results = evaluate_candidates(personas, settings=deep)
        self.assertGreater(len(results), 200)
        self.assertGreater(baseline, 0.0)

        _, metrics = build_report(personas, top_n=3, settings=standard)
        self.assertIn("best_non_discount_strategy", metrics)
        self.assertIn("validated_strategy_count", metrics)

    def test_dashboard_payload_exposes_controls_and_idea_agents(self) -> None:
        settings = build_settings(
            {
                "population": 12,
                "strategy_depth": "deep",
                "persona_richness": "rich",
                "ideation_agents": 5,
                "idea_proposals_per_agent": 2,
                "archetype_template_count": 6,
                "persona_blend_every": 2,
            }
        )
        personas = generate_personas(
            load_seed_profiles(),
            population=12,
            seed=7,
            richness="rich",
            archetype_template_count=settings.archetype_template_count,
            blend_every=settings.persona_blend_every,
        )
        payload = build_dashboard_payload(personas, settings=settings)

        self.assertIn("research_settings", payload["meta"])
        self.assertGreaterEqual(len(payload["idea_agents"]), settings.ideation_agents)
        self.assertIn("raw_profile", payload["personas"][0])
        self.assertIn("dossier", payload["personas"][0])
        self.assertEqual(payload["meta"]["payload_format_version"], 2)
        self.assertIn("research_trace", payload["idea_agents"][0])
        self.assertIn("flow_spec", payload["idea_agents"][0])
        self.assertIn("experiment_spec", payload["idea_agents"][0])

    def test_search_space_knobs_expand_candidate_universe(self) -> None:
        personas = generate_personas(load_seed_profiles(), population=6, seed=7, richness="rich")
        narrow = build_settings(
            {
                "population": 6,
                "strategy_depth": "quick",
                "grounding_limit": 1,
                "treatment_limit": 1,
                "friction_limit": 1,
                "discount_step": 10,
                "discount_floor": 0,
                "discount_ceiling": 20,
            }
        )
        wide = build_settings(
            {
                "population": 6,
                "strategy_depth": "quick",
                "grounding_limit": 4,
                "treatment_limit": 4,
                "friction_limit": 3,
                "discount_step": 5,
                "discount_floor": 0,
                "discount_ceiling": 50,
            }
        )

        narrow_payload = build_dashboard_payload(personas, settings=narrow)
        wide_payload = build_dashboard_payload(personas, settings=wide)

        self.assertGreater(wide_payload["meta"]["search_space_size"], narrow_payload["meta"]["search_space_size"])

    def test_discount_controls_constrain_offer_catalog(self) -> None:
        settings = build_settings(
            {
                "strategy_depth": "extreme",
                "discount_step": 25,
                "discount_floor": 25,
                "discount_ceiling": 50,
            }
        )

        offers = offer_catalog(settings)

        self.assertIn("discount_25", offers)
        self.assertIn("discount_50", offers)
        self.assertNotIn("discount_10", offers)
        self.assertNotIn("discount_20", offers)
        self.assertNotIn("discount_100", offers)

    def test_generate_ideas_heuristic_no_fake_specs(self) -> None:
        """Heuristic proposals are honest — no fabricated research specs."""
        personas = generate_personas(load_seed_profiles(), population=6, seed=7, richness="rich")
        settings = build_settings({"population": 6, "strategy_depth": "quick", "model_name": HEURISTIC_MODEL})
        candidate_universe = [
            StrategyCandidate(
                message_angle="flexibility_relief",
                proof_style="similar_user_story",
                offer="pause_plan",
                cta="pause_instead",
                personalization="contextual",
                contextual_grounding="generic",
                creative_treatment="plain_note",
                friction_reducer="single_tap_pause",
            ),
            StrategyCandidate(
                message_angle="cost_value_reframe",
                proof_style="quantified_outcome",
                offer="flexible_billing",
                cta="see_plan_options",
                personalization="contextual",
                contextual_grounding="generic",
                creative_treatment="plain_note",
                friction_reducer="billing_date_shift",
            ),
        ]

        ideas, warnings = generate_ideas(personas, candidate_universe, settings)

        self.assertFalse(warnings)
        self.assertGreater(len(ideas), 0)
        # Heuristic proposals should NOT carry fabricated research metadata
        self.assertIsNone(ideas[0].research_trace)
        self.assertIsNone(ideas[0].flow_spec)
        self.assertIsNone(ideas[0].experiment_spec)

    def test_compile_flow_spec_prefers_fallback_candidate(self) -> None:
        """Compiler prefers the LLM's explicit candidate over guessing."""
        settings = build_settings({"population": 6, "strategy_depth": "quick"})
        pause_candidate = StrategyCandidate(
            message_angle="flexibility_relief",
            proof_style="similar_user_story",
            offer="pause_plan",
            cta="pause_instead",
            personalization="contextual",
            contextual_grounding="generic",
            creative_treatment="plain_note",
            friction_reducer="single_tap_pause",
        )
        candidate_universe = [
            pause_candidate,
            StrategyCandidate(
                message_angle="progress_reflection",
                proof_style="similar_user_story",
                offer="none",
                cta="stay_on_current_plan",
                personalization="contextual",
            ),
        ]
        spec = FlowResearchSpec(
            id="spec-1",
            agent_role="Retention Psychologist",
            target_segment="fatigued_students",
            user_state_hypothesis="Users are overwhelmed and need a reversible break.",
            cancellation_moment_hypothesis="The user is cancelling because the commitment feels too heavy right now.",
            rescue_objective="Offer a safe pause instead of a hard cancel.",
            step_sequence=("Acknowledge burnout.", "Offer a pause path.", "Keep return friction near zero."),
            copy_blocks=("Take a short break without losing momentum.",),
            offer_logic="Pause, not discount.",
            cta_logic="Let the user pause in one tap.",
            branch_logic="If they still want to cancel, show a softer fallback.",
            confidence=0.8,
        )

        # With fallback_candidate (simulating LLM's explicit choice), compiler uses it directly
        candidate, notes = compile_flow_spec(
            spec, candidate_universe=candidate_universe, settings=settings,
            fallback_candidate=pause_candidate,
        )

        self.assertEqual(candidate.offer, "pause_plan")
        self.assertEqual(candidate.cta, "pause_instead")
        self.assertGreater(len(notes), 0)

    def test_compile_flow_spec_nearest_match_without_fallback(self) -> None:
        """Without a fallback, compiler returns first candidate in universe."""
        settings = build_settings({"population": 6, "strategy_depth": "quick"})
        candidate_universe = [
            StrategyCandidate(
                message_angle="progress_reflection",
                proof_style="similar_user_story",
                offer="none",
                cta="stay_on_current_plan",
                personalization="contextual",
            ),
        ]
        spec = FlowResearchSpec(
            id="spec-2",
            agent_role="Test",
            target_segment="test",
            user_state_hypothesis="test",
            cancellation_moment_hypothesis="test",
            rescue_objective="test",
            step_sequence=("test",),
            copy_blocks=("test",),
            offer_logic="test",
            cta_logic="test",
            branch_logic="test",
            confidence=0.5,
        )

        candidate, notes = compile_flow_spec(spec, candidate_universe=candidate_universe, settings=settings)

        self.assertIsNotNone(candidate)
        self.assertGreater(len(notes), 0)


if __name__ == "__main__":
    unittest.main()

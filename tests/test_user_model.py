from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cta_autoresearch.user_model import (
    ACTION_TO_CANDIDATE,
    ARCHETYPES,
    analyze_cohort,
    action_to_candidate,
    apply_cohort_priors,
    classify_user,
    enriched_row_to_persona,
    simulator_eval,
)


class ClassifyUserTest(unittest.TestCase):
    def test_burned_bridge(self) -> None:
        d = classify_user({
            "primary_reason": "quality_bug",
            "plan_tier": "super_learner",
            "features": {"frustration_level": 0.85, "save_openness": 0.1, "trust_risk": 0.5},
        })
        self.assertEqual(d.archetype_id, "burned_bridge")
        self.assertEqual(d.recommended_actions[0], "control_empathic_exit")

    def test_broken_workflow(self) -> None:
        d = classify_user({
            "primary_reason": "quality_bug",
            "plan_tier": "super_learner",
            "features": {
                "frustration_level": 0.7,
                "save_openness": 0.3,
                "bug_signals": ["pdf_upload_failure"],
            },
        })
        self.assertEqual(d.archetype_id, "broken_workflow")
        self.assertEqual(d.recommended_actions[0], "concierge_recovery")

    def test_free_plan_ceiling(self) -> None:
        d = classify_user({
            "primary_reason": "feature_gap",
            "plan_tier": "free",
            "features": {
                "frustration_level": 0.4,
                "save_openness": 0.5,
                "tags": ["Free Plan", "Upgrade Exploration"],
            },
        })
        self.assertEqual(d.archetype_id, "free_plan_ceiling")
        self.assertIn("targeted_discount_20", d.recommended_actions)

    def test_power_user_gap(self) -> None:
        d = classify_user({
            "primary_reason": "feature_gap",
            "plan_tier": "super_learner",
            "student_type": "Medical",
            "features": {
                "frustration_level": 0.5,
                "save_openness": 0.4,
                "feature_requests": ["image_support"],
            },
        })
        self.assertEqual(d.archetype_id, "power_user_gap")
        self.assertEqual(d.recommended_actions[0], "concierge_recovery")

    def test_missing_core_feature(self) -> None:
        d = classify_user({
            "primary_reason": "feature_gap",
            "plan_tier": "super_learner",
            "features": {
                "frustration_level": 0.4,
                "save_openness": 0.3,
                "feature_requests": ["flashcard_flip"],
            },
        })
        self.assertEqual(d.archetype_id, "missing_core_feature")

    def test_wrong_fit_teacher(self) -> None:
        d = classify_user({
            "primary_reason": "other",
            "plan_tier": "super_learner",
            "features": {
                "frustration_level": 0.3,
                "save_openness": 0.2,
                "tags": ["teacher", "settings"],
            },
        })
        self.assertEqual(d.archetype_id, "wrong_fit")
        self.assertLessEqual(d.save_potential, 0.15)

    def test_soft_churn(self) -> None:
        d = classify_user({
            "primary_reason": "other",
            "plan_tier": "super_learner",
            "features": {
                "frustration_level": 0.3,
                "save_openness": 0.45,
            },
        })
        self.assertEqual(d.archetype_id, "soft_churn")
        self.assertGreaterEqual(d.save_potential, 0.5)

    def test_reliability_erosion(self) -> None:
        d = classify_user({
            "primary_reason": "quality_bug",
            "plan_tier": "super_learner",
            "features": {
                "frustration_level": 0.65,
                "save_openness": 0.3,
                "tags": ["error messages", "technical issues"],
            },
        })
        self.assertEqual(d.archetype_id, "reliability_erosion")

    def test_csv_format_input(self) -> None:
        """Classify works with CSV-style flat dict (no nested features key)."""
        d = classify_user({
            "primary_reason": "other",
            "plan_tier": "free",
            "frustration_level": "0.25",
            "save_openness": "0.5",
            "tags_json": '["subscription"]',
        })
        self.assertEqual(d.archetype_id, "soft_churn")

    def test_to_dict(self) -> None:
        d = classify_user({
            "primary_reason": "other",
            "plan_tier": "super_learner",
            "features": {"frustration_level": 0.3, "save_openness": 0.5},
        })
        payload = d.to_dict()
        self.assertIn("archetype_id", payload)
        self.assertIn("root_cause_summary", payload)
        self.assertIsInstance(payload["signals"], list)


class AnalyzeCohortTest(unittest.TestCase):
    def _sample_rows(self) -> list[dict]:
        return [
            {"primary_reason": "quality_bug", "plan_tier": "super_learner",
             "features": {"frustration_level": 0.8, "save_openness": 0.1}},
            {"primary_reason": "feature_gap", "plan_tier": "free",
             "features": {"frustration_level": 0.3, "save_openness": 0.5,
                          "tags": ["Free Plan", "Upgrade Exploration"]}},
            {"primary_reason": "other", "plan_tier": "super_learner",
             "features": {"frustration_level": 0.2, "save_openness": 0.6}},
            {"primary_reason": "quality_bug", "plan_tier": "super_learner",
             "features": {"frustration_level": 0.7, "save_openness": 0.3,
                          "bug_signals": ["crash"]}},
        ]

    def test_cohort_produces_counts_and_priors(self) -> None:
        analysis = analyze_cohort(self._sample_rows())
        self.assertEqual(analysis.total_users, 4)
        self.assertGreater(len(analysis.context_arm_priors), 0)
        self.assertGreater(analysis.saveable_count, 0)

    def test_apply_cohort_priors_seeds_runtime(self) -> None:
        from cta_autoresearch.cancel_policy import CancelPolicyRuntime

        analysis = analyze_cohort(self._sample_rows())
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = CancelPolicyRuntime(Path(tmpdir))
            result = apply_cohort_priors(runtime, analysis)
            self.assertGreater(result["context_arms_seeded"], 0)
            self.assertIn("arms_context", runtime.state)
            self.assertGreater(len(runtime.state["arms_context"]), 0)


class ArchetypesTest(unittest.TestCase):
    def test_all_archetypes_have_required_fields(self) -> None:
        for arch_id, arch in ARCHETYPES.items():
            self.assertEqual(arch.id, arch_id)
            self.assertTrue(arch.label)
            self.assertTrue(arch.root_cause)
            self.assertGreaterEqual(arch.save_potential, 0.0)
            self.assertLessEqual(arch.save_potential, 1.0)
            self.assertTrue(arch.recommended_actions)

    def test_burned_bridge_has_lowest_save_potential(self) -> None:
        potentials = {a.id: a.save_potential for a in ARCHETYPES.values()}
        self.assertEqual(min(potentials, key=potentials.get), "burned_bridge")

    def test_soft_churn_has_highest_save_potential(self) -> None:
        potentials = {a.id: a.save_potential for a in ARCHETYPES.values()}
        self.assertEqual(max(potentials, key=potentials.get), "soft_churn")


class ActionToCandidateTest(unittest.TestCase):
    def test_all_candidates_are_valid(self) -> None:
        from cta_autoresearch.strategy_policy import valid_candidate

        for action_id, cand in ACTION_TO_CANDIDATE.items():
            self.assertTrue(
                valid_candidate(cand),
                f"{action_id} produces invalid StrategyCandidate",
            )

    def test_unknown_action_returns_control(self) -> None:
        cand = action_to_candidate("nonexistent_action")
        self.assertEqual(cand.offer, "none")
        self.assertEqual(cand.cta, "tell_us_why")

    def test_pause_maps_correctly(self) -> None:
        cand = action_to_candidate("pause_plan_relief")
        self.assertEqual(cand.offer, "pause_plan")
        self.assertEqual(cand.cta, "pause_instead")

    def test_discount_maps_correctly(self) -> None:
        cand = action_to_candidate("targeted_discount_20")
        self.assertEqual(cand.offer, "discount_20")
        self.assertEqual(cand.cta, "claim_offer")


class EnrichedRowToPersonaTest(unittest.TestCase):
    def _sample_row(self) -> dict:
        return {
            "primary_reason": "feature_gap",
            "plan_tier": "super_learner",
            "features": {
                "frustration_level": 0.5,
                "save_openness": 0.6,
                "trust_risk": 0.2,
                "churn_risk_score": 0.7,
                "feature_requests": ["image_support"],
                "bug_signals": [],
                "tags": ["Student User"],
            },
        }

    def test_produces_persona_with_nonzero_features(self) -> None:
        persona = enriched_row_to_persona(self._sample_row())
        self.assertGreater(persona.features.habit_strength, 0.0)
        self.assertGreater(persona.features.rescue_readiness, 0.0)

    def test_plan_mapping(self) -> None:
        row = self._sample_row()
        row["plan_tier"] = "free"
        persona = enriched_row_to_persona(row)
        self.assertEqual(persona.profile.plan, "free")

    def test_frustration_increases_dormancy(self) -> None:
        low = self._sample_row()
        low["features"]["frustration_level"] = 0.1
        high = self._sample_row()
        high["features"]["frustration_level"] = 0.9
        p_low = enriched_row_to_persona(low)
        p_high = enriched_row_to_persona(high)
        self.assertGreater(p_high.profile.dormancy_days, p_low.profile.dormancy_days)


class SimulatorEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        from cta_autoresearch.simulator import reset_scorer
        reset_scorer()

    def _sample_rows(self) -> list[dict]:
        return [
            {"primary_reason": "quality_bug", "plan_tier": "super_learner",
             "features": {"frustration_level": 0.8, "save_openness": 0.1,
                          "trust_risk": 0.4, "churn_risk_score": 0.9}},
            {"primary_reason": "feature_gap", "plan_tier": "free",
             "features": {"frustration_level": 0.3, "save_openness": 0.5,
                          "trust_risk": 0.2, "churn_risk_score": 0.6,
                          "tags": ["Free Plan", "Upgrade Exploration"]}},
            {"primary_reason": "other", "plan_tier": "super_learner",
             "features": {"frustration_level": 0.2, "save_openness": 0.6,
                          "trust_risk": 0.1, "churn_risk_score": 0.4}},
        ]

    def test_scores_change_with_different_actions(self) -> None:
        rows = self._sample_rows()
        result_control = simulator_eval(rows, lambda r: "control_empathic_exit")
        result_pause = simulator_eval(rows, lambda r: "pause_plan_relief")
        # Scores must differ — control (no offer) should score lower than pause
        self.assertNotAlmostEqual(
            result_control.composite_score,
            result_pause.composite_score,
            places=2,
        )

    def test_alignment_differs_by_policy(self) -> None:
        # Use rows where pause is recommended and control is not
        rows = [
            {"primary_reason": "other", "plan_tier": "super_learner",
             "features": {"frustration_level": 0.2, "save_openness": 0.6,
                          "trust_risk": 0.1, "churn_risk_score": 0.4}},
            {"primary_reason": "other", "plan_tier": "starter",
             "features": {"frustration_level": 0.3, "save_openness": 0.5,
                          "trust_risk": 0.15, "churn_risk_score": 0.45}},
        ]
        # These classify as soft_churn, where pause is recommended and control is anti
        result_control = simulator_eval(rows, lambda r: "control_empathic_exit")
        result_pause = simulator_eval(rows, lambda r: "pause_plan_relief")
        self.assertGreater(result_pause.alignment_score, result_control.alignment_score)

    def test_result_has_per_archetype_breakdown(self) -> None:
        rows = self._sample_rows()
        result = simulator_eval(rows, lambda r: "pause_plan_relief")
        self.assertGreater(len(result.per_archetype), 0)
        for arch_id, stats in result.per_archetype.items():
            self.assertIn("composite", stats)
            self.assertIn("alignment", stats)

    def test_to_dict(self) -> None:
        rows = self._sample_rows()
        result = simulator_eval(rows, lambda r: "pause_plan_relief")
        d = result.to_dict()
        self.assertIn("composite_score", d)
        self.assertIn("alignment_score", d)
        self.assertEqual(d["total_users"], 3)


if __name__ == "__main__":
    unittest.main()

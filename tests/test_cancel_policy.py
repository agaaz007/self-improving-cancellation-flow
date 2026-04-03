from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cta_autoresearch.cancel_policy import (
    CancelContextV1,
    CancelOutcomeV1,
    CancelPolicyRuntime,
    TranscriptExtractor,
)


class TranscriptExtractorTest(unittest.TestCase):
    def test_heuristic_extraction_returns_structured_fields(self) -> None:
        extractor = TranscriptExtractor(use_openai=False)
        extraction = extractor.extract(
            "This is too expensive and billing charges were unexpected. "
            "I might stay if there is a discount. I'm switching to ChatGPT."
        )

        self.assertEqual(extraction.primary_reason, "billing_confusion")
        self.assertTrue(extraction.billing_confusion_flag)
        self.assertIn("chatgpt", extraction.competitor_mentions)
        self.assertGreater(extraction.intent_strength, 0.0)
        self.assertGreater(extraction.save_openness, 0.0)


class CancelPolicyRuntimeTest(unittest.TestCase):
    def _context(self, *, session_id: str, user_id: str, reason: str = "price", discount_exposures: int = 0) -> CancelContextV1:
        return CancelContextV1.from_dict(
            {
                "session_id": session_id,
                "user_id_hash": user_id,
                "timestamp": 1_710_000_000.0,
                "plan_tier": "starter",
                "tenure_days": 220,
                "engagement_7d": 0.4,
                "engagement_30d": 0.55,
                "prior_cancel_attempts_30d": 0,
                "discount_exposures_30d": discount_exposures,
                "transcript_extraction": {
                    "primary_reason": reason,
                    "secondary_reasons": [],
                    "intent_strength": 0.8,
                    "save_openness": 0.6,
                    "frustration_level": 0.4,
                    "trust_risk": 0.3,
                    "billing_confusion_flag": reason == "billing_confusion",
                    "competitor_mentions": [],
                    "feature_requests": [],
                    "bug_signals": [],
                    "summary": "test",
                    "confidence": 0.8,
                    "extractor_version": "test-v1",
                },
            }
        )

    def test_decide_and_record_outcome_updates_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = CancelPolicyRuntime(Path(tmpdir))
            context = self._context(session_id="s1", user_id="u1")
            decision = runtime.decide(context)
            self.assertTrue(decision.action_id)
            self.assertGreater(decision.propensity, 0.0)

            outcome = CancelOutcomeV1.from_dict(
                {
                    "decision_id": decision.decision_id,
                    "session_id": "s1",
                    "saved_flag": True,
                    "cancel_completed_flag": False,
                    "support_escalation_flag": False,
                    "complaint_flag": False,
                }
            )
            result = runtime.record_outcome(outcome)
            self.assertEqual(result["status"], "recorded")
            self.assertGreater(result["reward"], 0.0)

            health = runtime.health()
            self.assertEqual(health["decisions"], 1)
            self.assertEqual(health["outcomes"], 1)

    def test_discount_cap_blocks_discount_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = CancelPolicyRuntime(Path(tmpdir), discount_cap_30d=1)
            payload = self._context(
                session_id="s2",
                user_id="u2",
                reason="price",
                discount_exposures=1,
            ).to_dict()
            payload["eligible_actions"] = [
                "targeted_discount_20",
                "targeted_discount_40",
                "control_empathic_exit",
            ]
            context = CancelContextV1.from_dict(payload)
            decision = runtime.decide(context)
            action = runtime.get_action(decision.action_id)
            self.assertFalse(action["is_discount"])
            self.assertIn("targeted_discount_20", decision.blocked_action_ids)

    def test_repeat_attempts_are_sticky_within_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = CancelPolicyRuntime(Path(tmpdir))
            first = runtime.decide(self._context(session_id="s3", user_id="sticky-user"))
            second = runtime.decide(self._context(session_id="s4", user_id="sticky-user"))
            self.assertEqual(first.action_id, second.action_id)

    def test_warm_start_replay_and_regression_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = CancelPolicyRuntime(Path(tmpdir))
            summary = runtime.warm_start(
                [
                    {
                        "action_id": "pause_plan_relief",
                        "primary_reason": "break",
                        "plan_tier": "starter",
                        "saved_flag": True,
                    }
                    for _ in range(20)
                ],
                reset_state=True,
            )
            self.assertEqual(summary["rows_applied"], 20)

            for index in range(140):
                context = self._context(session_id=f"session-{index}", user_id=f"user-{index}")
                decision = runtime.decide(context)
                saved = not decision.holdout_flag
                runtime.record_outcome(
                    CancelOutcomeV1.from_dict(
                        {
                            "decision_id": decision.decision_id,
                            "session_id": context.session_id,
                            "saved_flag": saved,
                            "cancel_completed_flag": not saved,
                            "support_escalation_flag": False,
                            "complaint_flag": False,
                        }
                    )
                )

            replay = runtime.replay()
            self.assertGreater(replay["rows"], 0)
            self.assertIn("by_action", replay)
            self.assertIn("by_reason", replay)

            regression = runtime.regression_check(
                min_treatment_samples=20,
                min_holdout_samples=5,
                min_save_lift=0.05,
                max_support_delta=0.2,
                max_complaint_delta=0.2,
            )
            self.assertTrue(regression["pass"])


if __name__ == "__main__":
    unittest.main()

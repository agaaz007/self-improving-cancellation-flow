from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cta_autoresearch.lab_dashboard import build_dashboard_dataset, write_dashboard_data
from cta_autoresearch.research_settings import DEFAULT_OPENAI_MODEL, build_settings, build_settings_catalog


class ResearchSettingsTest(unittest.TestCase):
    def test_settings_expand_depth_metadata_and_budget(self) -> None:
        settings = build_settings(
            {
                "population": 120,
                "strategy_depth": "deep",
                "persona_richness": "rich",
                "ideation_agents": 6,
                "model_name": "gpt-5.4-mini",
            }
        )

        self.assertEqual(settings.discount_step, 5)
        self.assertGreaterEqual(settings.effective_validation_budget, 320)
        self.assertEqual(len(settings.available_roles()), 6)
        self.assertTrue(settings.uses_openai)

    def test_settings_catalog_exposes_defaults(self) -> None:
        catalog = build_settings_catalog()
        self.assertIn("defaults", catalog)
        self.assertEqual(catalog["defaults"]["model_name"], DEFAULT_OPENAI_MODEL)
        self.assertIn("depth_options", catalog)

    def test_settings_tolerate_common_hosted_input_shapes(self) -> None:
        settings = build_settings(
            {
                "population": "",
                "top_n": None,
                "validation_budget": "",
                "seed": "not-a-number",
                "api_batch_size": None,
                "model_name": "gpt-5.4-mini",
            }
        )
        self.assertEqual(settings.population, 120)
        self.assertEqual(settings.top_n, 25)
        self.assertGreaterEqual(settings.validation_budget, 1)
        self.assertEqual(settings.seed, 7)

    def test_settings_expose_advanced_depth_knobs(self) -> None:
        settings = build_settings(
            {
                "idea_proposals_per_agent": 4,
                "persona_shortlist_multiplier": 6,
                "segment_focus_limit": 7,
                "archetype_template_count": 6,
                "persona_blend_every": 3,
                "openai_reasoning_effort": "high",
                "api_batch_size": 9,
            }
        )

        self.assertEqual(settings.idea_proposals_per_agent, 4)
        self.assertEqual(settings.persona_shortlist_multiplier, 6)
        self.assertEqual(settings.segment_focus_limit, 7)
        self.assertEqual(settings.archetype_template_count, 6)
        self.assertEqual(settings.persona_blend_every, 3)
        self.assertEqual(settings.openai_reasoning_effort, "high")
        self.assertEqual(settings.api_batch_size, 9)

    def test_extreme_depth_dashboard_build_does_not_crash(self) -> None:
        payload = build_dashboard_dataset(
            population=10,
            strategy_depth="extreme",
            persona_richness="standard",
            ideation_agents=2,
            validation_budget=40,
        )
        self.assertEqual(payload["meta"]["research_settings"]["strategy_depth"], "extreme")

    def test_settings_allow_none_validation_budget(self) -> None:
        settings = build_settings(
            {
                "population": 60,
                "strategy_depth": "standard",
                "validation_budget": None,
                "model_name": "gpt-5.4-mini",
            }
        )

        self.assertGreater(settings.effective_validation_budget, 0)


class DashboardMetadataTest(unittest.TestCase):
    def test_dashboard_payload_includes_controls_and_settings(self) -> None:
        payload = build_dashboard_dataset(
            population=12,
            strategy_depth="deep",
            persona_richness="rich",
            ideation_agents=5,
            validation_budget=140,
            idea_proposals_per_agent=3,
            persona_shortlist_multiplier=5,
            seed=17,
        )

        self.assertIn("controls", payload)
        self.assertIn("research_settings", payload["meta"])
        self.assertGreaterEqual(payload["meta"]["research_settings"]["validation_budget"], 140)
        self.assertEqual(payload["meta"]["research_settings"]["idea_proposals_per_agent"], 3)
        self.assertEqual(payload["meta"]["research_settings"]["persona_shortlist_multiplier"], 5)
        self.assertGreaterEqual(len(payload["idea_agents"]), 1)

    def test_write_dashboard_data_writes_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = write_dashboard_data(tmpdir, population=10, strategy_depth="quick", validation_budget=80)
            payload = json.loads(Path(output).read_text())

        self.assertEqual(payload["meta"]["research_settings"]["strategy_depth"], "quick")
        self.assertLessEqual(payload["meta"]["validated_strategy_count"], 80)

    def test_extreme_depth_builds_successfully(self) -> None:
        payload = build_dashboard_dataset(
            population=10,
            strategy_depth="extreme",
            persona_richness="extreme",
            validation_budget=120,
            seed=9,
        )

        self.assertEqual(payload["meta"]["research_settings"]["strategy_depth"], "extreme")
        self.assertGreater(payload["meta"]["search_space_size"], 0)


if __name__ == "__main__":
    unittest.main()

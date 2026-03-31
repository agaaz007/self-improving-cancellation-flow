from __future__ import annotations

import unittest

from cta_autoresearch.features import derive_features
from cta_autoresearch.sample_data import load_seed_profiles


class FeatureDerivationTest(unittest.TestCase):
    def test_engaged_seed_has_higher_habit_strength_than_dormant_seed(self) -> None:
        kaitlyn, lillian = load_seed_profiles()
        kaitlyn_features = derive_features(kaitlyn)
        lillian_features = derive_features(lillian)

        self.assertGreater(kaitlyn_features.habit_strength, lillian_features.habit_strength)
        self.assertGreater(lillian_features.feature_awareness_gap, kaitlyn_features.feature_awareness_gap)
        self.assertGreater(lillian_features.discount_affinity, kaitlyn_features.discount_affinity)


if __name__ == "__main__":
    unittest.main()

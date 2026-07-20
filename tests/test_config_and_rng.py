from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frl_v3.config import (
    load_design_config,
    load_model_config,
    load_resilience_design_config,
)
from frl_v3.rng import make_random_inputs, seed_from_parts


class ConfigAndRngTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model_config(ROOT / "configs" / "model.json")
        cls.design = load_design_config(ROOT / "configs" / "design.json")
        cls.resilience = load_resilience_design_config(
            ROOT / "configs" / "resilience_design.json"
        )

    def test_locked_configs_validate(self) -> None:
        self.assertEqual(self.model.model_version, "frl-v3.1")
        self.assertEqual(
            self.resilience.design_version, "frl-v3-resilience-1.0"
        )
        self.assertEqual(self.resilience.primary_stress_budget, 0.15)
        self.assertEqual(self.model.anchor_version, "frl-v3-anchor-1.0")
        self.assertEqual(self.design.calibration_ratios, (0.5, 1.0, 2.0))
        self.assertEqual(self.design.primary_feedback_lambdas, (0.0, 0.15))

    def test_seed_derivation_is_order_independent(self) -> None:
        first = seed_from_parts("namespace", "block-1", "market_events")
        second = seed_from_parts("namespace", "block-1", "market_events")
        changed = seed_from_parts("namespace", "block-2", "market_events")
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_population_streams_are_nested(self) -> None:
        small = replace(self.model, population=50)
        large = replace(self.model, population=200)
        small_inputs = make_random_inputs(small, "nested", "rep-1")
        large_inputs = make_random_inputs(large, "nested", "rep-1")
        np.testing.assert_array_equal(
            small_inputs.leverage_u, large_inputs.leverage_u[:50]
        )
        np.testing.assert_array_equal(
            small_inputs.cash_share_u, large_inputs.cash_share_u[:50]
        )
        np.testing.assert_array_equal(
            small_inputs.funding_z, large_inputs.funding_z[:, :50]
        )

    def test_population_cannot_exceed_locked_stream_size(self) -> None:
        invalid = replace(self.model, population=201)
        with self.assertRaises(ValueError):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frl_v2.config import load_design_config, load_model_config
from frl_v2.experiment import expected_counts, primary_conditions
from frl_v2.rng import make_random_inputs, seed_from_parts


class ConfigAndRandomStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model_config(ROOT / "configs" / "model.json")
        cls.design = load_design_config(ROOT / "configs" / "design.json")

    def test_locked_design_has_27_primary_cells(self) -> None:
        conditions = primary_conditions(self.design)
        self.assertEqual(len(conditions), 27)
        self.assertEqual(len({condition.condition_id for condition in conditions}), 27)

    def test_expected_primary_row_counts(self) -> None:
        counts = expected_counts(self.model, 27, self.design.primary_replications)
        self.assertEqual(counts["runs"], 5_400)
        self.assertEqual(counts["agent_rows"], 540_000)
        self.assertEqual(counts["daily_rows"], 162_000)

    def test_seed_derivation_is_order_sensitive_and_repeatable(self) -> None:
        first = seed_from_parts("experiment", "block-1", "market_events")
        repeated = seed_from_parts("experiment", "block-1", "market_events")
        reordered = seed_from_parts("market_events", "block-1", "experiment")
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, reordered)

    def test_random_inputs_replay_exactly(self) -> None:
        first = make_random_inputs(self.model, "test", "block-0001")
        second = make_random_inputs(self.model, "test", "block-0001")
        self.assertTrue(np.array_equal(first.leverage_u, second.leverage_u))
        self.assertTrue(np.array_equal(first.liquidity_z, second.liquidity_z))
        self.assertTrue(np.array_equal(first.market_event_u, second.market_event_u))

    def test_conditions_share_random_inputs_within_a_block(self) -> None:
        first = make_random_inputs(self.model, "test", "block-0001")
        second = make_random_inputs(self.model, "test", "block-0001")
        self.assertTrue(np.array_equal(first.cash_share_u, second.cash_share_u))
        self.assertTrue(np.array_equal(first.payment_multiplier_u, second.payment_multiplier_u))


if __name__ == "__main__":
    unittest.main()

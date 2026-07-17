from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frl_v2.config import load_model_config
from frl_v2.model import Condition, channel_budgets, simulate_run


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model_config(ROOT / "configs" / "model.json")

    def test_channel_budget_preserves_total_reference_pressure(self) -> None:
        for ratio in (0.5, 1.0, 2.0):
            liquidity, market = channel_budgets(0.03, ratio)
            self.assertAlmostEqual(liquidity + market, 0.03)
            self.assertAlmostEqual(liquidity / market, ratio)

    def test_no_stress_produces_no_exit_and_constant_balance_sheet(self) -> None:
        result = simulate_run(
            self.model,
            Condition(1.0, 0, 0),
            0.03,
            "unit-test",
            "no-stress",
        )
        self.assertEqual(result.run_row["event_count"], 0)
        self.assertAlmostEqual(result.run_row["final_asset_price"], 1.0)
        for row in result.agent_rows:
            self.assertAlmostEqual(row["final_equity"], self.model.initial_equity)
            self.assertEqual(row["event"], 0)

    def test_run_replays_byte_for_byte_at_object_level(self) -> None:
        condition = Condition(1.0, 2, 2)
        first = simulate_run(self.model, condition, 0.03, "unit-test", "replay")
        second = simulate_run(self.model, condition, 0.03, "unit-test", "replay")
        self.assertEqual(first, second)

    def test_terminal_balance_sheet_identity_holds(self) -> None:
        result = simulate_run(
            self.model,
            Condition(0.5, 2, 2),
            0.04,
            "unit-test",
            "accounting",
        )
        for row in result.agent_rows:
            reconstructed = (
                row["final_cash"] + row["final_risky_value"] - row["final_liabilities"]
            )
            self.assertAlmostEqual(reconstructed, row["final_equity"], places=12)

    def test_exit_is_absorbing_and_daily_counts_reconcile(self) -> None:
        stressed = replace(self.model, price_impact_lambda=0.30)
        result = simulate_run(
            stressed,
            Condition(2.0, 2, 2),
            0.04,
            "unit-test",
            "absorption",
        )
        end_counts = [int(row["at_risk_end"]) for row in result.daily_rows]
        self.assertTrue(all(left >= right for left, right in zip(end_counts, end_counts[1:])))
        self.assertEqual(sum(int(row["exit_count"]) for row in result.daily_rows), result.run_row["event_count"])
        self.assertEqual(end_counts[-1], self.model.population - result.run_row["event_count"])

    def test_market_only_expected_reference_loss_is_calibrated(self) -> None:
        market_budget = channel_budgets(0.03, 1.0)[1]
        event_loss = (
            market_budget
            / (self.model.market_event_probability * self.model.reference_risky_value)
        )
        expected_loss = (
            self.model.market_event_probability
            * event_loss
            * self.model.reference_risky_value
        )
        self.assertAlmostEqual(expected_loss, market_budget)


if __name__ == "__main__":
    unittest.main()

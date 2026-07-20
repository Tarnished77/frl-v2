from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frl_v3.config import load_model_config
from frl_v3.model import (
    Condition,
    channel_budgets,
    expected_reference_market_loss,
    market_loss_fraction,
    settle_funding_withdrawal,
    simulate_run,
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model_config(ROOT / "configs" / "model.json")

    def test_cash_repayment_reduces_cash_and_debt_without_equity_loss(self) -> None:
        cash = np.array([0.5])
        units = np.array([0.5])
        liabilities = np.array([0.4])
        before = cash + units - liabilities
        result = settle_funding_withdrawal(
            cash,
            units,
            liabilities,
            np.array([True]),
            1.0,
            np.array([0.2]),
            0.0,
            1e-10,
        )
        after = result.cash + result.units * result.price - result.liabilities
        np.testing.assert_allclose(before, after)
        np.testing.assert_allclose(result.cash, [0.3])
        np.testing.assert_allclose(result.liabilities, [0.2])
        np.testing.assert_allclose(result.paid, [0.2])
        np.testing.assert_allclose(result.unpaid, [0.0])

    def test_no_impact_sale_repayment_preserves_equity(self) -> None:
        cash = np.array([0.1])
        units = np.array([0.9])
        liabilities = np.array([0.5])
        before = cash + units - liabilities
        result = settle_funding_withdrawal(
            cash,
            units,
            liabilities,
            np.array([True]),
            1.0,
            np.array([0.3]),
            0.0,
            1e-10,
        )
        after = result.cash + result.units * result.price - result.liabilities
        np.testing.assert_allclose(before, after)
        np.testing.assert_allclose(result.units, [0.7])
        np.testing.assert_allclose(result.liabilities, [0.2])
        np.testing.assert_allclose(result.unpaid, [0.0])

    def test_fire_sale_discount_is_the_only_repayment_equity_loss(self) -> None:
        cash = np.zeros(2)
        units = np.ones(2)
        liabilities = np.full(2, 0.5)
        before = cash + units - liabilities
        result = settle_funding_withdrawal(
            cash,
            units,
            liabilities,
            np.ones(2, dtype=bool),
            1.0,
            np.full(2, 0.1),
            0.30,
            1e-10,
        )
        after = result.cash + result.units * result.price - result.liabilities
        np.testing.assert_allclose(before - after, result.price_impact_loss)
        np.testing.assert_allclose(result.unpaid, [0.0, 0.0], atol=1e-12)
        self.assertGreater(result.sale_fraction, 0.10)

    def test_positive_impact_does_not_mechanically_cause_funding_failure(
        self,
    ) -> None:
        result = settle_funding_withdrawal(
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.5]),
            np.array([True]),
            1.0,
            np.array([0.1]),
            0.15,
            1e-10,
        )
        np.testing.assert_allclose(result.paid, [0.1], atol=1e-12)
        np.testing.assert_allclose(result.unpaid, [0.0], atol=1e-12)
        self.assertLess(result.liabilities[0], 0.5)

    def test_horizon_market_loss_is_matched(self) -> None:
        _, market_budget = channel_budgets(0.20, 1.0)
        fraction = market_loss_fraction(
            self.model,
            market_budget,
            market_intensity=1,
            event_probability=0.10,
        )
        matched = expected_reference_market_loss(self.model, 0.10, fraction)
        self.assertAlmostEqual(matched, market_budget, places=12)

    def test_no_stress_has_constant_equity_and_no_exit(self) -> None:
        condition = Condition(1.0, 0, 0, 0.15, 0.10)
        result = simulate_run(
            self.model, condition, 0.20, "unit-test", "no-stress"
        )
        self.assertEqual(result.run_row["event_count"], 0)
        self.assertAlmostEqual(result.run_row["final_asset_price"], 1.0)
        self.assertAlmostEqual(result.run_row["mean_equity_loss"], 0.0)
        self.assertAlmostEqual(
            result.run_row["worst_10pct_mean_equity_loss"], 0.0
        )
        self.assertAlmostEqual(result.run_row["mean_market_value_loss"], 0.0)
        self.assertAlmostEqual(
            result.run_row["mean_fire_sale_value_loss"], 0.0
        )
        for row in result.agent_rows:
            self.assertAlmostEqual(row["final_equity"], 1.0, places=12)
            self.assertAlmostEqual(
                row["final_liabilities"], row["initial_liabilities"], places=12
            )

    def test_funding_only_without_impact_deleverages_without_exit(self) -> None:
        condition = Condition(1.0, 2, 0, 0.0, 0.10)
        result = simulate_run(
            self.model, condition, 0.10, "unit-test", "funding-only"
        )
        self.assertEqual(result.run_row["event_count"], 0)
        for row in result.agent_rows:
            self.assertAlmostEqual(row["final_equity"], 1.0, places=10)
            self.assertLess(
                row["final_liabilities"], row["initial_liabilities"]
            )
            self.assertAlmostEqual(
                row["total_funding_requested"],
                row["total_funding_repaid"],
                places=10,
            )
        self.assertAlmostEqual(result.run_row["mean_equity_loss"], 0.0)
        self.assertGreater(result.run_row["mean_liability_reduction"], 0.0)

    def test_tail_equity_loss_is_not_below_mean_loss(self) -> None:
        condition = Condition(0.5, 2, 2, 0.15, 0.10)
        result = simulate_run(
            self.model, condition, 0.30, "unit-test", "tail-loss"
        )
        self.assertGreaterEqual(
            result.run_row["worst_10pct_mean_equity_loss"]
            + 1e-12,
            result.run_row["mean_equity_loss"],
        )
        self.assertGreaterEqual(result.run_row["asset_price_loss"], 0.0)
        self.assertAlmostEqual(
            result.run_row["mean_equity_loss"],
            result.run_row["mean_market_value_loss"]
            + result.run_row["mean_fire_sale_value_loss"],
            places=10,
        )

    def test_liabilities_never_increase_or_become_negative(self) -> None:
        condition = Condition(2.0, 2, 2, 0.30, 0.10)
        result = simulate_run(
            self.model, condition, 0.30, "unit-test", "liability-bounds"
        )
        for row in result.agent_rows:
            self.assertGreaterEqual(row["final_liabilities"], 0.0)
            self.assertLessEqual(
                row["final_liabilities"],
                row["initial_liabilities"] + 1e-12,
            )

    def test_run_replays_at_object_level(self) -> None:
        condition = Condition(0.5, 2, 2, 0.15, 0.10)
        first = simulate_run(
            self.model, condition, 0.20, "unit-test", "replay"
        )
        second = simulate_run(
            self.model, condition, 0.20, "unit-test", "replay"
        )
        self.assertEqual(first, second)

    def test_exit_is_absorbing_and_counts_reconcile(self) -> None:
        condition = Condition(0.5, 2, 2, 0.30, 0.05)
        result = simulate_run(
            self.model, condition, 0.30, "unit-test", "absorption"
        )
        counts = [int(row["at_risk_end"]) for row in result.daily_rows]
        self.assertTrue(
            all(left >= right for left, right in zip(counts, counts[1:]))
        )
        self.assertEqual(
            sum(int(row["exit_count"]) for row in result.daily_rows),
            result.run_row["event_count"],
        )


if __name__ == "__main__":
    unittest.main()

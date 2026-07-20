from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from frl_v3.analysis import primary_estimands, sensitivity_estimands


class AnalysisTests(unittest.TestCase):
    def test_locked_contrast_formulas(self) -> None:
        rows = []
        for block in range(4):
            noise = block * 0.001
            for rho in (0.5, 1.0, 2.0):
                for impact in (0.0, 0.15):
                    for funding in (0, 1, 2):
                        for market in (0, 1, 2):
                            outcome = (
                                0.10 * funding
                                + 0.20 * market
                                + 0.05 * funding * market
                                + 0.40 * impact
                                + noise
                            )
                            rows.append(
                                {
                                    "replication_id": f"block-{block}",
                                    "calibration_ratio": rho,
                                    "price_impact_lambda": impact,
                                    "funding_intensity": funding,
                                    "market_intensity": market,
                                    "outcome": outcome,
                                }
                            )
        frame = pd.DataFrame(rows)
        estimands = primary_estimands(frame, "outcome")

        def find(family: str, identifier: str, rho: float, impact: object):
            matches = [
                item
                for item in estimands
                if item.metadata["family"] == family
                and item.metadata["estimand_id"] == identifier
                and item.metadata["calibration_ratio"] == rho
                and item.metadata["price_impact_lambda"] == impact
            ]
            self.assertEqual(len(matches), 1)
            return matches[0].values

        difference = find(
            "liquidity_minus_market",
            "funding_minus_market_0_to_1",
            1.0,
            0.15,
        )
        np.testing.assert_allclose(difference, -0.10)

        interaction = find(
            "channel_interaction",
            "interaction_funding_2_market_2",
            1.0,
            0.15,
        )
        np.testing.assert_allclose(interaction, 0.20)

        amplifications = [
            item
            for item in estimands
            if item.metadata["family"] == "fire_sale_amplification_average"
            and item.metadata["calibration_ratio"] == 1.0
        ]
        self.assertEqual(len(amplifications), 1)
        np.testing.assert_allclose(amplifications[0].values, 0.06)

    def test_sensitivity_contrasts_use_paired_scenarios(self) -> None:
        rows = []
        for block in range(3):
            for probability in (0.05, 0.10, 0.20):
                for funding in (0, 1, 2):
                    for market in (0, 1, 2):
                        rows.append(
                            {
                                "replication_id": f"block-{block}",
                                "market_event_probability": probability,
                                "funding_intensity": funding,
                                "market_intensity": market,
                                "outcome": (
                                    0.1 * funding
                                    + 0.2 * market
                                    + probability * market
                                    + block * 0.001
                                ),
                            }
                        )
        _, contrasts = sensitivity_estimands(
            pd.DataFrame(rows), "outcome", "market_frequency"
        )
        matches = [
            item
            for item in contrasts
            if item.metadata["family"] == "scenario_difference_average"
            and item.metadata["scenario"] == 0.20
        ]
        self.assertEqual(len(matches), 1)
        expected = np.mean(
            [0.10 * market for funding in (0, 1, 2) for market in (0, 1, 2)
             if funding + market > 0]
        )
        np.testing.assert_allclose(matches[0].values, expected)

    def test_budget_sensitivity_includes_locked_reference(self) -> None:
        rows = []
        for block in range(3):
            for budget in (0.10, 0.15, 0.30):
                for funding in (0, 1, 2):
                    for market in (0, 1, 2):
                        rows.append(
                            {
                                "replication_id": f"block-{block}",
                                "stress_budget": budget,
                                "funding_intensity": funding,
                                "market_intensity": market,
                                "outcome": (
                                    budget * (funding + market)
                                    + block * 0.001
                                ),
                            }
                        )
        _, contrasts = sensitivity_estimands(
            pd.DataFrame(rows), "outcome", "stress_budget"
        )
        matches = [
            item
            for item in contrasts
            if item.metadata["family"] == "scenario_difference_average"
            and item.metadata["scenario"] == 0.30
        ]
        self.assertEqual(len(matches), 1)
        expected = np.mean(
            [
                0.15 * (funding + market)
                for funding in (0, 1, 2)
                for market in (0, 1, 2)
                if funding + market > 0
            ]
        )
        np.testing.assert_allclose(matches[0].values, expected)


if __name__ == "__main__":
    unittest.main()

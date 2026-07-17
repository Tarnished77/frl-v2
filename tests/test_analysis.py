from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frl_v2.analysis import block_channel_effects


class AnalysisTests(unittest.TestCase):
    def test_block_channel_effects_use_common_outcome_scale(self) -> None:
        rows = []
        for liquidity in (0, 1, 2):
            for market in (0, 1, 2):
                rows.append(
                    {
                        "calibration_ratio": 1.0,
                        "replication_id": "block-0001",
                        "liquidity_intensity": liquidity,
                        "market_intensity": market,
                        "exit_rate": 0.10 * liquidity + 0.05 * market,
                        "rmst_days": 30.0 - 2.0 * liquidity - 3.0 * market,
                    }
                )
        effects = block_channel_effects(pd.DataFrame(rows)).set_index("outcome")
        self.assertAlmostEqual(effects.loc["exit_rate", "liquidity_effect"], 0.10)
        self.assertAlmostEqual(effects.loc["exit_rate", "market_effect"], 0.05)
        self.assertAlmostEqual(effects.loc["exit_rate", "channel_difference"], 0.05)
        self.assertAlmostEqual(effects.loc["rmst_days", "liquidity_effect"], 2.0)
        self.assertAlmostEqual(effects.loc["rmst_days", "market_effect"], 3.0)
        self.assertAlmostEqual(effects.loc["rmst_days", "channel_difference"], -1.0)

    def test_unbalanced_block_is_rejected(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "calibration_ratio": 1.0,
                    "replication_id": "block-0001",
                    "liquidity_intensity": 0,
                    "market_intensity": 0,
                    "exit_rate": 0.0,
                    "rmst_days": 30.0,
                }
            ]
        )
        with self.assertRaises(ValueError):
            block_channel_effects(frame)


if __name__ == "__main__":
    unittest.main()

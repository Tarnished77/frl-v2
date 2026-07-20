"""Tests for deterministic publication-asset selection helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import matplotlib as mpl
from matplotlib import font_manager
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_publication


class PublicationBuilderTests(unittest.TestCase):
    def test_renderer_uses_agg_and_bundled_font(self) -> None:
        build_publication._configure_matplotlib()
        bundled_root = (Path(mpl.get_data_path()) / "fonts").resolve()
        selected = Path(font_manager.findfont("DejaVu Sans")).resolve()
        self.assertEqual(mpl.get_backend().lower(), "agg")
        self.assertTrue(selected.is_relative_to(bundled_root))

    def test_interval_is_expressed_in_equity_percentage_points(self) -> None:
        row = pd.Series(
            {
                "estimate": 0.01234,
                "simultaneous_lower": 0.01001,
                "simultaneous_upper": 0.01499,
            }
        )
        self.assertEqual(
            build_publication._interval(row),
            "1.23 [1.00, 1.50]",
        )

    def test_row_requires_a_unique_match(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "outcome": "mean_equity_loss",
                    "family": "funding_local_average",
                    "calibration_ratio": 1.0,
                    "price_impact_lambda": 0.15,
                    "transition": "0_to_1",
                    "funding_intensity": None,
                    "market_intensity": None,
                }
            ]
        )
        selected = build_publication._row(
            frame,
            outcome="mean_equity_loss",
            family="funding_local_average",
            rho=1.0,
            price_impact=0.15,
            transition="0_to_1",
        )
        self.assertEqual(selected["calibration_ratio"], 1.0)
        with self.assertRaises(ValueError):
            build_publication._row(
                frame,
                outcome="mean_equity_loss",
                family="market_local_average",
                rho=1.0,
            )


if __name__ == "__main__":
    unittest.main()

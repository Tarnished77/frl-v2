from pathlib import Path
import unittest

from frl_v3.config import load_design_config
from frl_v3.design_lock import select_population
from frl_v3.pilot import select_pilot_summary


ROOT = Path(__file__).resolve().parents[1]


class SelectionRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = load_design_config(ROOT / "configs" / "design.json")

    def test_pilot_rule_uses_informativeness_then_target_then_budget(self) -> None:
        rows = [
            {
                "stress_budget": 0.10,
                "informative_cells": 8,
                "distance_to_target": 0.20,
            },
            {
                "stress_budget": 0.20,
                "informative_cells": 9,
                "distance_to_target": 0.15,
            },
            {
                "stress_budget": 0.15,
                "informative_cells": 9,
                "distance_to_target": 0.15,
            },
        ]
        selected = select_pilot_summary(rows)
        self.assertEqual(selected["stress_budget"], 0.15)

    def test_population_rule_selects_100_within_tolerances(self) -> None:
        finite_means = {
            50: {"cell": (0.20, 25.0)},
            100: {"cell": (0.25, 24.0)},
            200: {"cell": (0.27, 24.4)},
        }
        selected, max_exit, max_rmst = select_population(
            finite_means, self.design
        )
        self.assertEqual(selected, 100)
        self.assertAlmostEqual(max_exit, 0.02)
        self.assertAlmostEqual(max_rmst, 0.4)

    def test_population_rule_selects_200_if_either_threshold_fails(self) -> None:
        finite_means = {
            50: {"cell": (0.20, 25.0)},
            100: {"cell": (0.25, 24.0)},
            200: {"cell": (0.29, 24.2)},
        }
        selected, _, _ = select_population(finite_means, self.design)
        self.assertEqual(selected, 200)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import unittest

from frl_v3.config import (
    load_model_config,
    load_resilience_design_config,
)
from frl_v3.resilience_experiment import (
    budget_sensitivity_conditions,
    calibration_audit_conditions,
    cash_buffer_conditions,
    finite_size_conditions,
    frequency_conditions,
    high_impact_conditions,
    primary_conditions,
)
from frl_v3.resilience_lock import select_resilience_population


ROOT = Path(__file__).resolve().parents[1]


class ResilienceDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model_config(ROOT / "configs" / "model.json")
        cls.design = load_resilience_design_config(
            ROOT / "configs" / "resilience_design.json"
        )

    def test_condition_counts_are_locked(self) -> None:
        self.assertEqual(
            len(primary_conditions(self.model, self.design)), 54
        )
        self.assertEqual(
            len(calibration_audit_conditions(self.model, self.design)), 27
        )
        self.assertEqual(
            len(finite_size_conditions(self.model, self.design)), 9
        )
        self.assertEqual(
            len(high_impact_conditions(self.model, self.design)), 18
        )
        self.assertEqual(
            len(frequency_conditions(self.model, self.design)), 27
        )
        self.assertEqual(
            len(cash_buffer_conditions(self.model, self.design)), 27
        )
        self.assertEqual(
            len(budget_sensitivity_conditions(self.model, self.design)), 9
        )

    def test_population_rule_uses_both_continuous_outcomes(self) -> None:
        within = {
            50: {"cell": (0.10, 0.20)},
            100: {"cell": (0.11, 0.21)},
            200: {"cell": (0.115, 0.225)},
        }
        selected, mean_difference, tail_difference = (
            select_resilience_population(within, self.design)
        )
        self.assertEqual(selected, 100)
        self.assertAlmostEqual(mean_difference, 0.005)
        self.assertAlmostEqual(tail_difference, 0.015)

        outside = {
            50: {"cell": (0.10, 0.20)},
            100: {"cell": (0.11, 0.21)},
            200: {"cell": (0.115, 0.235)},
        }
        selected, _, _ = select_resilience_population(
            outside, self.design
        )
        self.assertEqual(selected, 200)


if __name__ == "__main__":
    unittest.main()

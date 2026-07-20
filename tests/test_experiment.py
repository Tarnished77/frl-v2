from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from frl_v3.config import load_design_config, load_model_config
from frl_v3.experiment import (
    cash_buffer_conditions,
    finite_size_conditions,
    frequency_conditions,
    high_impact_conditions,
    pilot_conditions,
    primary_conditions,
    replication_ids,
    run_experiment,
)
from frl_v3.io_utils import sha256_file


ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model_config(ROOT / "configs" / "model.json")
        cls.design = load_design_config(ROOT / "configs" / "design.json")

    def test_condition_factories_have_locked_sizes(self) -> None:
        self.assertEqual(len(primary_conditions(self.model, self.design)), 54)
        self.assertEqual(len(pilot_conditions(self.model, self.design)), 27)
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

    def test_output_is_immutable_and_replays_byte_for_byte(self) -> None:
        model = replace(self.model, population=5, horizon=3)
        condition = primary_conditions(model, self.design)[-1]
        replications = replication_ids("test", 1, 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first_manifest = run_experiment(
                model,
                [condition],
                0.20,
                "frl-v3-test-replay",
                replications,
                first,
                "test-replay",
            )
            second_manifest = run_experiment(
                model,
                [condition],
                0.20,
                "frl-v3-test-replay",
                replications,
                second,
                "test-replay",
            )
            self.assertEqual(first_manifest["actual_counts"]["agent_rows"], 10)
            self.assertEqual(first_manifest["actual_counts"]["daily_rows"], 6)
            self.assertEqual(first_manifest["actual_counts"]["run_rows"], 2)
            for filename in (
                "agent_survival.csv",
                "run_daily.csv",
                "run_summary.csv",
                "seed_manifest.json",
                "experiment_manifest.json",
            ):
                self.assertEqual(
                    sha256_file(first / filename),
                    sha256_file(second / filename),
                )
            with self.assertRaises(FileExistsError):
                run_experiment(
                    model,
                    [condition],
                    0.20,
                    "frl-v3-test-replay",
                    replications,
                    first,
                    "test-replay",
                )


if __name__ == "__main__":
    unittest.main()

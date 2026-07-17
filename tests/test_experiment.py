from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frl_v2.config import load_model_config
from frl_v2.experiment import run_experiment
from frl_v2.model import Condition


class ExperimentOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = replace(
            load_model_config(ROOT / "configs" / "model.json"),
            population=8,
            horizon=4,
        )

    def test_output_counts_checksums_and_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experiment"
            manifest = run_experiment(
                self.model,
                [Condition(1.0, 0, 0), Condition(1.0, 1, 1)],
                0.03,
                "unit-test-output",
                ["block-0001", "block-0002"],
                output,
                "unit-test-experiment",
            )
            self.assertEqual(manifest["actual_counts"]["agent_rows"], 32)
            self.assertEqual(manifest["actual_counts"]["daily_rows"], 16)
            self.assertEqual(manifest["actual_counts"]["run_rows"], 4)
            self.assertEqual(set(manifest["files"]), {
                "agent_survival.csv",
                "run_daily.csv",
                "run_summary.csv",
                "seed_manifest.json",
            })
            with (output / "experiment_manifest.json").open(encoding="utf-8") as handle:
                saved_manifest = json.load(handle)
            self.assertEqual(saved_manifest, manifest)
            with self.assertRaises(FileExistsError):
                run_experiment(
                    self.model,
                    [Condition(1.0, 0, 0)],
                    0.03,
                    "unit-test-output",
                    ["block-0001"],
                    output,
                    "must-not-overwrite",
                )


if __name__ == "__main__":
    unittest.main()

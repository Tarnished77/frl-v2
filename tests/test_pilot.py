from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frl_v2.config import load_design_config, load_model_config
from frl_v2.design_lock import lock_confirmatory_design
from frl_v2.pilot import run_pilot


class PilotAndDesignLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model_config(ROOT / "configs" / "model.json")
        cls.design = load_design_config(ROOT / "configs" / "design.json")

    def test_pilot_replays_locked_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            pilot_dir = temporary_root / "pilot"
            locked = run_pilot(self.model, self.design, pilot_dir)
            self.assertEqual(locked["selected_stress_budget"], 0.03)
            self.assertTrue(locked["selected_is_acceptable"])

            design_dir = temporary_root / "design"
            manifest = lock_confirmatory_design(
                self.design,
                pilot_dir,
                design_dir,
                "de7ed51",
            )
            self.assertEqual(manifest["replication_groups"]["primary_and_homogeneous"], 200)
            self.assertEqual(manifest["replication_groups"]["finite_size"], 50)
            self.assertTrue((design_dir / "formal_seed_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()

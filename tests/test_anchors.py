from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AnchorTests(unittest.TestCase):
    def test_preserved_source_checksums(self) -> None:
        anchors = ROOT / "anchors"
        manifest = json.loads(
            (anchors / "source_manifest.json").read_text(encoding="utf-8")
        )
        for source in manifest["sources"]:
            if "raw_file" not in source:
                continue
            self.assertEqual(
                sha256(anchors / source["raw_file"]), source["sha256"]
            )

    def test_parameter_anchor_values_are_locked(self) -> None:
        data = json.loads(
            (ROOT / "anchors" / "parameter_anchors.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(data["anchor_version"], "frl-v3-anchor-1.0")
        self.assertEqual(data["leverage"]["reference"], 2.3)
        self.assertEqual(data["cash_share_of_gross_assets"]["reference"], 0.06)
        self.assertEqual(
            data["pilot_stress_budgets"], [0.1, 0.15, 0.2, 0.25, 0.3]
        )
        self.assertAlmostEqual(
            data["market_loss_21_trading_days"]["loss_q95"],
            0.06342005464258038,
        )


if __name__ == "__main__":
    unittest.main()

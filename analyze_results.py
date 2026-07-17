"""Run the locked FRL v2 statistical analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from frl_v2.analysis import run_analysis
from frl_v2.config import load_design_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--homogeneous-dir", type=Path, required=True)
    parser.add_argument("--finite-50-dir", type=Path, required=True)
    parser.add_argument("--finite-200-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    design = load_design_config(ROOT / "configs" / "design.json")
    manifest = run_analysis(
        design,
        args.primary_dir,
        args.homogeneous_dir,
        {50: args.finite_50_dir, 200: args.finite_200_dir},
        args.output_dir,
        args.analysis_source_commit,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

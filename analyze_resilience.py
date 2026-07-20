"""CLI for deterministic continuous resilience analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from frl_v3.config import load_resilience_design_config
from frl_v3.resilience_analysis import (
    analyze_resilience_primary,
    analyze_resilience_sensitivity,
)


DESIGN_PATH = ROOT / "configs" / "resilience_design.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    primary = subparsers.add_parser("primary")
    primary.add_argument(
        "--input-dir", action="append", type=Path, required=True
    )
    primary.add_argument("--output-dir", type=Path, required=True)
    primary.add_argument("--locked-design", type=Path, required=True)

    sensitivity = subparsers.add_parser("sensitivity")
    sensitivity.add_argument(
        "kind",
        choices=(
            "high_impact",
            "market_frequency",
            "cash_buffer",
            "stress_budget",
        ),
    )
    sensitivity.add_argument(
        "--input-dir", action="append", type=Path, required=True
    )
    sensitivity.add_argument("--output-dir", type=Path, required=True)
    sensitivity.add_argument("--locked-design", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    design = load_resilience_design_config(DESIGN_PATH)
    if args.command == "primary":
        result = analyze_resilience_primary(
            args.input_dir,
            args.locked_design,
            design,
            args.output_dir,
        )
        print(json.dumps(result["precision_decision"], indent=2))
    elif args.command == "sensitivity":
        result = analyze_resilience_sensitivity(
            args.input_dir,
            args.locked_design,
            design,
            args.kind,
            args.output_dir,
        )
        print(json.dumps(result["sample_description"], indent=2))


if __name__ == "__main__":
    main()

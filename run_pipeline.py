"""Command-line entry point for the FRL v2 research pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from frl_v2.config import load_design_config, load_model_config
from frl_v2.design_lock import lock_confirmatory_design
from frl_v2.experiment import (
    expected_counts,
    parity_conditions,
    primary_conditions,
    replication_ids,
    run_experiment,
    with_population,
)
from frl_v2.pilot import run_pilot


MODEL_CONFIG_PATH = ROOT / "configs" / "model.json"
DESIGN_CONFIG_PATH = ROOT / "configs" / "design.json"
OUTPUT_ROOT = ROOT / "outputs"


def read_locked_budget(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        return float(json.load(handle)["selected_stress_budget"])


def dry_run() -> None:
    model = load_model_config(MODEL_CONFIG_PATH)
    design = load_design_config(DESIGN_CONFIG_PATH)
    primary = expected_counts(model, 27, design.primary_replications)
    homogeneous = expected_counts(model, 9, design.homogeneous_replications)
    finite = {
        str(population): expected_counts(
            with_population(model, population), 9, design.finite_size_replications
        )
        for population in design.finite_size_populations
    }
    print(json.dumps({"primary": primary, "homogeneous": homogeneous, "finite_size": finite}, indent=2))


def smoke(output_dir: Path) -> None:
    model = load_model_config(MODEL_CONFIG_PATH)
    design = load_design_config(DESIGN_CONFIG_PATH)
    smoke_model = replace(model, population=10, horizon=5)
    run_experiment(
        smoke_model,
        primary_conditions(design),
        stress_budget=0.03,
        namespace=f"{design.experiment_namespace}-smoke",
        replication_id_values=replication_ids("smoke", 2),
        output_dir=output_dir,
        experiment_id="frl-v2-smoke",
    )


def pilot(output_dir: Path) -> None:
    model = load_model_config(MODEL_CONFIG_PATH)
    design = load_design_config(DESIGN_CONFIG_PATH)
    locked = run_pilot(model, design, output_dir)
    print(json.dumps(locked, indent=2))


def lock_design(pilot_dir: Path, output_dir: Path, source_commit: str) -> None:
    design = load_design_config(DESIGN_CONFIG_PATH)
    manifest = lock_confirmatory_design(
        design,
        pilot_dir,
        output_dir,
        source_commit,
    )
    print(json.dumps(manifest, indent=2))


def formal(kind: str, output_dir: Path, locked_design: Path) -> None:
    model = load_model_config(MODEL_CONFIG_PATH)
    design = load_design_config(DESIGN_CONFIG_PATH)
    budget = read_locked_budget(locked_design)
    if kind == "primary":
        conditions = primary_conditions(design)
        reps = replication_ids("primary", design.primary_replications)
        config = model
    elif kind == "homogeneous":
        conditions = parity_conditions(design, "homogeneous")
        reps = replication_ids("primary", design.homogeneous_replications)
        config = model
    elif kind.startswith("finite-"):
        population = int(kind.split("-", maxsplit=1)[1])
        if population not in design.finite_size_populations:
            raise ValueError(f"Population {population} is not in the locked finite-size design")
        conditions = parity_conditions(design, "heterogeneous")
        reps = replication_ids("finite-size", design.finite_size_replications)
        config = with_population(model, population)
    else:
        raise ValueError(f"Unknown formal experiment kind: {kind}")

    run_experiment(
        config,
        conditions,
        stress_budget=budget,
        namespace=design.experiment_namespace,
        replication_id_values=reps,
        output_dir=output_dir,
        experiment_id=f"frl-v2-{kind}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dry-run")

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "smoke-v1")

    pilot_parser = subparsers.add_parser("pilot")
    pilot_parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "pilot-v1")

    lock_parser = subparsers.add_parser("lock-design")
    lock_parser.add_argument("--pilot-dir", type=Path, required=True)
    lock_parser.add_argument("--output-dir", type=Path, default=ROOT / "design")
    lock_parser.add_argument("--source-commit", required=True)

    formal_parser = subparsers.add_parser("run")
    formal_parser.add_argument(
        "kind",
        choices=("primary", "homogeneous", "finite-50", "finite-200"),
    )
    formal_parser.add_argument("--output-dir", type=Path, required=True)
    formal_parser.add_argument(
        "--locked-design",
        type=Path,
        default=ROOT / "design" / "locked_design.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "dry-run":
        dry_run()
    elif args.command == "smoke":
        smoke(args.output_dir)
    elif args.command == "pilot":
        pilot(args.output_dir)
    elif args.command == "lock-design":
        lock_design(args.pilot_dir, args.output_dir, args.source_commit)
    elif args.command == "run":
        formal(args.kind, args.output_dir, args.locked_design)


if __name__ == "__main__":
    main()

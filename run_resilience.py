"""CLI for the preregistered FRL v3 continuous resilience experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from frl_v3.config import (
    load_model_config,
    load_resilience_design_config,
)
from frl_v3.experiment import (
    expected_counts,
    replication_ids,
    run_experiment,
    with_population,
)
from frl_v3.resilience_experiment import (
    budget_sensitivity_conditions,
    calibration_audit_conditions,
    cash_buffer_conditions,
    finite_size_conditions,
    frequency_conditions,
    high_impact_conditions,
    primary_conditions,
    run_budget_sensitivity,
    run_calibration_audit,
)
from frl_v3.resilience_lock import lock_resilience_design


MODEL_CONFIG_PATH = ROOT / "configs" / "model.json"
DESIGN_CONFIG_PATH = ROOT / "configs" / "resilience_design.json"
ANCHOR_PATH = ROOT / "anchors" / "parameter_anchors.json"


def _load() -> tuple[object, object]:
    return (
        load_model_config(MODEL_CONFIG_PATH),
        load_resilience_design_config(DESIGN_CONFIG_PATH),
    )


def _read_lock(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def dry_run() -> None:
    model, design = _load()
    designs = {
        "primary_initial": expected_counts(
            model,
            len(primary_conditions(model, design)),
            design.primary_replications,
        ),
        "primary_extension": expected_counts(
            model,
            len(primary_conditions(model, design)),
            design.maximum_primary_replications
            - design.primary_replications,
        ),
        "calibration_audit_per_budget": expected_counts(
            model,
            len(calibration_audit_conditions(model, design)),
            design.calibration_audit_replications,
        ),
        "finite_each_population": expected_counts(
            model,
            len(finite_size_conditions(model, design)),
            design.finite_size_replications,
        ),
        "sensitivity_high_impact": expected_counts(
            model,
            len(high_impact_conditions(model, design)),
            design.sensitivity_replications,
        ),
        "sensitivity_frequency": expected_counts(
            model,
            len(frequency_conditions(model, design)),
            design.sensitivity_replications,
        ),
        "sensitivity_cash": expected_counts(
            model,
            len(cash_buffer_conditions(model, design)),
            design.sensitivity_replications,
        ),
        "sensitivity_budget_per_level": expected_counts(
            model,
            len(budget_sensitivity_conditions(model, design)),
            design.sensitivity_replications,
        ),
    }
    print(json.dumps(designs, indent=2))


def smoke(output_dir: Path) -> None:
    model, design = _load()
    smoke_model = replace(model, population=10, horizon=5)
    run_experiment(
        smoke_model,
        primary_conditions(smoke_model, design),
        design.primary_stress_budget,
        f"{design.experiment_namespace}-smoke",
        replication_ids("smoke", 1, 2),
        output_dir,
        "frl-v3-resilience-smoke",
    )


def finite(
    population: int, output_dir: Path
) -> None:
    model, design = _load()
    model = with_population(model, population)
    run_experiment(
        model,
        finite_size_conditions(model, design),
        design.primary_stress_budget,
        f"{design.experiment_namespace}-finite",
        replication_ids("finite", 1, design.finite_size_replications),
        output_dir,
        f"frl-v3-resilience-finite-{population}",
    )


def formal(kind: str, output_dir: Path, lock_path: Path) -> None:
    model, design = _load()
    lock = _read_lock(lock_path)
    model = with_population(model, int(lock["selected_population"]))
    budget = float(lock["primary_stress_budget"])
    if budget != design.primary_stress_budget:
        raise ValueError("Design lock and resilience configuration differ")

    if kind == "primary":
        conditions = primary_conditions(model, design)
        identifiers = replication_ids(
            "primary", 1, design.primary_replications
        )
        namespace = f"{design.experiment_namespace}-primary"
    elif kind == "primary-extension":
        conditions = primary_conditions(model, design)
        identifiers = replication_ids(
            "primary",
            design.primary_replications + 1,
            design.maximum_primary_replications
            - design.primary_replications,
        )
        namespace = f"{design.experiment_namespace}-primary"
    elif kind == "sensitivity-high-impact":
        conditions = high_impact_conditions(model, design)
        identifiers = replication_ids(
            "sensitivity", 1, design.sensitivity_replications
        )
        namespace = f"{design.experiment_namespace}-sensitivity-high-impact"
    elif kind == "sensitivity-frequency":
        conditions = frequency_conditions(model, design)
        identifiers = replication_ids(
            "sensitivity", 1, design.sensitivity_replications
        )
        namespace = f"{design.experiment_namespace}-sensitivity-frequency"
    elif kind == "sensitivity-cash":
        conditions = cash_buffer_conditions(model, design)
        identifiers = replication_ids(
            "sensitivity", 1, design.sensitivity_replications
        )
        namespace = f"{design.experiment_namespace}-sensitivity-cash"
    elif kind == "sensitivity-budget":
        identifiers = replication_ids(
            "sensitivity", 1, design.sensitivity_replications
        )
        run_budget_sensitivity(model, design, identifiers, output_dir)
        return
    else:
        raise ValueError(f"Unknown resilience experiment: {kind}")

    run_experiment(
        model,
        conditions,
        budget,
        namespace,
        identifiers,
        output_dir,
        f"frl-v3-resilience-{kind}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dry-run")

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--output-dir", type=Path, required=True)

    audit_parser = subparsers.add_parser("calibration-audit")
    audit_parser.add_argument("--output-root", type=Path, required=True)

    finite_parser = subparsers.add_parser("finite")
    finite_parser.add_argument("population", choices=("50", "100", "200"))
    finite_parser.add_argument("--output-dir", type=Path, required=True)

    lock_parser = subparsers.add_parser("lock-design")
    lock_parser.add_argument("--calibration-audit", type=Path, required=True)
    lock_parser.add_argument("--finite-50", type=Path, required=True)
    lock_parser.add_argument("--finite-100", type=Path, required=True)
    lock_parser.add_argument("--finite-200", type=Path, required=True)
    lock_parser.add_argument("--output-dir", type=Path, required=True)
    lock_parser.add_argument("--source-commit", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "kind",
        choices=(
            "primary",
            "primary-extension",
            "sensitivity-high-impact",
            "sensitivity-frequency",
            "sensitivity-cash",
            "sensitivity-budget",
        ),
    )
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--locked-design", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "dry-run":
        dry_run()
    elif args.command == "smoke":
        smoke(args.output_dir)
    elif args.command == "calibration-audit":
        model, design = _load()
        result = run_calibration_audit(model, design, args.output_root)
        print(json.dumps(result, indent=2))
    elif args.command == "finite":
        finite(int(args.population), args.output_dir)
    elif args.command == "lock-design":
        design = load_resilience_design_config(DESIGN_CONFIG_PATH)
        result = lock_resilience_design(
            design,
            args.calibration_audit,
            {
                50: args.finite_50,
                100: args.finite_100,
                200: args.finite_200,
            },
            args.output_dir,
            args.source_commit,
            ANCHOR_PATH,
            MODEL_CONFIG_PATH,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "run":
        formal(args.kind, args.output_dir, args.locked_design)


if __name__ == "__main__":
    main()

"""Lock the continuous resilience design before confirmatory outcomes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from .config import ResilienceDesignConfig
from .experiment import replication_ids
from .io_utils import require_new_directory, sha256_file, write_json
from .rng import stream_seed_manifest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cell_means(path: Path) -> dict[str, tuple[float, float]]:
    rows = _read_csv(path)
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        grouped.setdefault(row["condition_id"], []).append(
            (
                float(row["mean_equity_loss"]),
                float(row["worst_10pct_mean_equity_loss"]),
            )
        )
    return {
        condition_id: (
            sum(value[0] for value in values) / len(values),
            sum(value[1] for value in values) / len(values),
        )
        for condition_id, values in grouped.items()
    }


def select_resilience_population(
    finite_means: dict[int, dict[str, tuple[float, float]]],
    design: ResilienceDesignConfig,
) -> tuple[int, float, float]:
    if set(finite_means) != {50, 100, 200}:
        raise ValueError("finite-size results must cover 50, 100, and 200")
    if set(finite_means[100]) != set(finite_means[200]):
        raise ValueError("finite-size condition sets do not match")
    max_mean_loss = max(
        abs(finite_means[100][cell][0] - finite_means[200][cell][0])
        for cell in finite_means[100]
    )
    max_tail_loss = max(
        abs(finite_means[100][cell][1] - finite_means[200][cell][1])
        for cell in finite_means[100]
    )
    population = (
        100
        if max_mean_loss <= design.finite_size_mean_loss_tolerance
        and max_tail_loss <= design.finite_size_tail_loss_tolerance
        else 200
    )
    return population, max_mean_loss, max_tail_loss


def lock_resilience_design(
    design: ResilienceDesignConfig,
    calibration_audit_dir: Path,
    finite_dirs: dict[int, Path],
    output_dir: Path,
    source_commit: str,
    anchor_file: Path,
    model_config_file: Path,
) -> dict[str, object]:
    if not source_commit or len(source_commit) < 7:
        raise ValueError("source_commit must identify the tested revision")
    audit_manifest = json.loads(
        (calibration_audit_dir / "calibration_audit_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if not audit_manifest["passed"]:
        raise ValueError("calibration audit did not pass")
    if (
        float(audit_manifest["primary_stress_budget_locked_before_audit"])
        != design.primary_stress_budget
    ):
        raise ValueError("calibration audit did not use the locked budget")
    if set(finite_dirs) != {50, 100, 200}:
        raise ValueError("finite-size directories must cover 50, 100, and 200")
    means = {
        population: _cell_means(directory / "run_summary.csv")
        for population, directory in finite_dirs.items()
    }
    population, max_mean_loss, max_tail_loss = select_resilience_population(
        means, design
    )

    require_new_directory(output_dir)
    for filename in (
        "calibration_audit_manifest.json",
        "calibration_audit_summary.csv",
    ):
        shutil.copyfile(
            calibration_audit_dir / filename, output_dir / filename
        )
    finite_hashes: dict[str, str] = {}
    for value, directory in sorted(finite_dirs.items()):
        source = directory / "experiment_manifest.json"
        target = output_dir / f"finite_{value}_experiment_manifest.json"
        shutil.copyfile(source, target)
        finite_hashes[target.name] = sha256_file(target)

    groups = {
        "primary_initial": (
            f"{design.experiment_namespace}-primary",
            replication_ids("primary", 1, design.primary_replications),
        ),
        "primary_extension": (
            f"{design.experiment_namespace}-primary",
            replication_ids(
                "primary",
                design.primary_replications + 1,
                design.maximum_primary_replications
                - design.primary_replications,
            ),
        ),
        "sensitivity_high_impact": (
            f"{design.experiment_namespace}-sensitivity-high-impact",
            replication_ids(
                "sensitivity", 1, design.sensitivity_replications
            ),
        ),
        "sensitivity_frequency": (
            f"{design.experiment_namespace}-sensitivity-frequency",
            replication_ids(
                "sensitivity", 1, design.sensitivity_replications
            ),
        ),
        "sensitivity_cash": (
            f"{design.experiment_namespace}-sensitivity-cash",
            replication_ids(
                "sensitivity", 1, design.sensitivity_replications
            ),
        ),
        "sensitivity_budget": (
            f"{design.experiment_namespace}-sensitivity-budget",
            replication_ids(
                "sensitivity", 1, design.sensitivity_replications
            ),
        ),
    }
    formal_seed_manifest = {
        name: {
            "namespace": namespace,
            "seeds": stream_seed_manifest(namespace, identifiers),
        }
        for name, (namespace, identifiers) in groups.items()
    }
    write_json(
        output_dir / "formal_seed_manifest.json", formal_seed_manifest
    )
    locked = {
        "lock_version": "1.0",
        "status": "locked before continuous confirmatory outcomes",
        "source_commit": source_commit,
        "model_config_sha256": sha256_file(model_config_file),
        "anchor_sha256": sha256_file(anchor_file),
        "primary_stress_budget": design.primary_stress_budget,
        "selected_population": population,
        "finite_size_diagnostic": {
            "max_mean_equity_loss_difference_n100_vs_n200": max_mean_loss,
            "max_tail_equity_loss_difference_n100_vs_n200": max_tail_loss,
            "mean_loss_tolerance": design.finite_size_mean_loss_tolerance,
            "tail_loss_tolerance": design.finite_size_tail_loss_tolerance,
        },
        "design_config": design.to_dict(),
        "namespace_groups": {
            name: {
                "namespace": namespace,
                "replications": len(identifiers),
            }
            for name, (namespace, identifiers) in groups.items()
        },
        "files": {
            "formal_seed_manifest.json": sha256_file(
                output_dir / "formal_seed_manifest.json"
            ),
            "calibration_audit_summary.csv": sha256_file(
                output_dir / "calibration_audit_summary.csv"
            ),
            **finite_hashes,
        },
    }
    write_json(output_dir / "locked_resilience_design.json", locked)
    return locked

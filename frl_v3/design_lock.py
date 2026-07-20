"""Lock the v3 confirmatory design after pilot and finite-size diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from .config import DesignConfig
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
            (float(row["exit_rate"]), float(row["rmst_days"]))
        )
    return {
        condition_id: (
            sum(value[0] for value in values) / len(values),
            sum(value[1] for value in values) / len(values),
        )
        for condition_id, values in grouped.items()
    }


def select_population(
    finite_means: dict[int, dict[str, tuple[float, float]]],
    design: DesignConfig,
) -> tuple[int, float, float]:
    """Apply the pre-specified N=100 versus N=200 accuracy thresholds."""
    if set(finite_means) != {50, 100, 200}:
        raise ValueError("finite-size results must cover 50, 100, and 200")
    if set(finite_means[100]) != set(finite_means[200]):
        raise ValueError("finite-size condition sets do not match")
    max_exit = max(
        abs(finite_means[100][cell][0] - finite_means[200][cell][0])
        for cell in finite_means[100]
    )
    max_rmst = max(
        abs(finite_means[100][cell][1] - finite_means[200][cell][1])
        for cell in finite_means[100]
    )
    selected_population = (
        100
        if max_exit <= design.finite_size_exit_risk_tolerance
        and max_rmst <= design.finite_size_rmst_tolerance_days
        else 200
    )
    return selected_population, max_exit, max_rmst


def lock_confirmatory_design(
    design: DesignConfig,
    pilot_dir: Path,
    finite_dirs: dict[int, Path],
    output_dir: Path,
    source_commit: str,
    anchor_file: Path,
) -> dict[str, object]:
    if not source_commit or len(source_commit) < 7:
        raise ValueError("source_commit must identify the tested model revision")
    pilot_lock = json.loads(
        (pilot_dir / "pilot_lock.json").read_text(encoding="utf-8")
    )
    if not pilot_lock["selected_meets_minimum"]:
        raise ValueError("pilot did not meet the locked informativeness minimum")
    if set(finite_dirs) != {50, 100, 200}:
        raise ValueError("finite-size directories must cover 50, 100, and 200")

    finite_means = {
        population: _cell_means(path / "run_summary.csv")
        for population, path in finite_dirs.items()
    }
    selected_population, max_exit, max_rmst = select_population(
        finite_means, design
    )

    require_new_directory(output_dir)
    for filename in (
        "pilot_summary.csv",
        "pilot_cells.csv",
        "pilot_lock.json",
        "seed_manifest.json",
    ):
        shutil.copyfile(pilot_dir / filename, output_dir / filename)
    finite_manifest_hashes: dict[str, str] = {}
    for population, directory in sorted(finite_dirs.items()):
        source = directory / "experiment_manifest.json"
        target = output_dir / f"finite_{population}_experiment_manifest.json"
        shutil.copyfile(source, target)
        finite_manifest_hashes[target.name] = sha256_file(target)

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
    }
    seed_manifest = {
        name: {
            "namespace": namespace,
            "seeds": stream_seed_manifest(namespace, identifiers),
        }
        for name, (namespace, identifiers) in groups.items()
    }
    write_json(output_dir / "formal_seed_manifest.json", seed_manifest)

    locked = {
        "lock_version": "1.0",
        "status": "locked before confirmatory outcomes were generated",
        "source_commit": source_commit,
        "anchor_version": pilot_lock["model_config"]["anchor_version"],
        "anchor_sha256": sha256_file(anchor_file),
        "selected_stress_budget": pilot_lock["selected_stress_budget"],
        "selected_population": selected_population,
        "finite_size_diagnostic": {
            "max_exit_risk_difference_n100_vs_n200": max_exit,
            "max_rmst_difference_n100_vs_n200": max_rmst,
            "exit_risk_tolerance": design.finite_size_exit_risk_tolerance,
            "rmst_tolerance_days": design.finite_size_rmst_tolerance_days,
        },
        "model_config": pilot_lock["model_config"],
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
            **finite_manifest_hashes,
        },
    }
    write_json(output_dir / "locked_design.json", locked)
    return locked

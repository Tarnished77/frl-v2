"""Deterministic paired-block analysis for the continuous resilience design."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import (
    CELL_FIELDS,
    CONTRAST_FIELDS,
    SENSITIVITY_CELL_FIELDS,
    SENSITIVITY_CONTRAST_FIELDS,
    _bootstrap_rows,
    cell_estimands,
    load_primary_runs,
    primary_estimands,
    sensitivity_estimands,
    validate_experiment_directory,
)
from .config import ResilienceDesignConfig
from .io_utils import (
    StableCsvWriter,
    require_new_directory,
    sha256_file,
    write_json,
)


PRIMARY_OUTCOMES = {
    "mean_equity_loss": "mean_equity_loss",
    "worst_10pct_mean_equity_loss": "worst_10pct_mean_equity_loss",
}

SECONDARY_OUTCOMES = {
    "asset_price_loss": "asset_price_loss",
    "market_value_loss": "mean_market_value_loss",
    "fire_sale_value_loss": "mean_fire_sale_value_loss",
    "liability_reduction": "mean_liability_reduction",
    "forced_sale_proceeds": "mean_forced_sale_proceeds",
    "exit_risk": "exit_rate",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_resilience_frame(
    frame: pd.DataFrame,
    lock: dict[str, Any],
    require_primary_budget: bool,
) -> dict[str, float]:
    population = int(lock["selected_population"])
    if set(frame["population"].astype(int)) != {population}:
        raise ValueError("Resilience population does not match design lock")
    if require_primary_budget and not np.allclose(
        frame["stress_budget"].astype(float),
        float(lock["primary_stress_budget"]),
    ):
        raise ValueError("Primary stress budget does not match design lock")
    residual = (
        frame["mean_equity_loss"]
        - frame["mean_market_value_loss"]
        - frame["mean_fire_sale_value_loss"]
    ).abs()
    maximum_residual = float(residual.max())
    if maximum_residual > 1e-9:
        raise ValueError(
            f"Equity-loss decomposition residual is {maximum_residual}"
        )
    return {
        "maximum_equity_loss_decomposition_residual": maximum_residual,
        "maximum_mean_unpaid_funding": float(
            frame["mean_total_unpaid"].max()
        ),
        "maximum_exit_rate": float(frame["exit_rate"].max()),
    }


def analyze_resilience_primary(
    input_directories: list[Path],
    locked_design_path: Path,
    design: ResilienceDesignConfig,
    output_dir: Path,
) -> dict[str, Any]:
    require_new_directory(output_dir)
    frame, input_records = load_primary_runs(input_directories)
    lock = _read_json(locked_design_path)
    validation = _validate_resilience_frame(frame, lock, True)
    blocks = int(
        frame.groupby("condition_id")["replication_id"].nunique().iloc[0]
    )

    cell_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    outcomes = {**PRIMARY_OUTCOMES, **SECONDARY_OUTCOMES}
    for outcome, column in outcomes.items():
        cell_rows.extend(
            _bootstrap_rows(
                outcome,
                cell_estimands(frame, column),
                design.bootstrap_replications,
                f"resilience-cells-{outcome}-{blocks}",
                CELL_FIELDS,
            )
        )
        estimands = primary_estimands(frame, column)
        for family in sorted(
            {str(item.metadata["family"]) for item in estimands}
        ):
            family_estimands = [
                item
                for item in estimands
                if item.metadata["family"] == family
            ]
            contrast_rows.extend(
                _bootstrap_rows(
                    outcome,
                    family_estimands,
                    design.bootstrap_replications,
                    f"resilience-{family}-{outcome}-{blocks}",
                    CONTRAST_FIELDS,
                )
            )

    with StableCsvWriter(
        output_dir / "resilience_cells.csv", CELL_FIELDS
    ) as writer:
        writer.write_many(cell_rows)
    with StableCsvWriter(
        output_dir / "resilience_contrasts.csv", CONTRAST_FIELDS
    ) as writer:
        writer.write_many(contrast_rows)

    precision: dict[str, dict[str, float | bool]] = {}
    thresholds = {
        "mean_equity_loss": (
            design.monte_carlo_mean_loss_halfwidth_tolerance
        ),
        "worst_10pct_mean_equity_loss": (
            design.monte_carlo_tail_loss_halfwidth_tolerance
        ),
    }
    for outcome, threshold in thresholds.items():
        rows = [
            row
            for row in contrast_rows
            if row["outcome"] == outcome
            and row["family"] == "liquidity_minus_market"
        ]
        maximum = max(
            (
                float(row["simultaneous_upper"])
                - float(row["simultaneous_lower"])
            )
            / 2.0
            for row in rows
        )
        precision[outcome] = {
            "maximum_simultaneous_halfwidth": maximum,
            "tolerance": threshold,
            "passes": maximum <= threshold,
        }
    extension_required = blocks < design.maximum_primary_replications and any(
        not result["passes"] for result in precision.values()
    )
    precision_decision = {
        "decision_version": "1.0",
        "rule": (
            "Use the largest 95% family-wise max-t half-width in the "
            "funding-minus-market local-effect family across calibration "
            "ratios, feedback states, and both local transitions. If either "
            "primary outcome exceeds its preregistered tolerance, append "
            "locked blocks 201--400 to every primary condition."
        ),
        "independent_monte_carlo_unit": "replication block",
        "blocks_analyzed": blocks,
        "precision_by_outcome": precision,
        "extension_required": extension_required,
    }
    write_json(output_dir / "precision_decision.json", precision_decision)
    write_json(output_dir / "mechanism_validation.json", validation)

    files = {
        filename: sha256_file(output_dir / filename)
        for filename in (
            "resilience_cells.csv",
            "resilience_contrasts.csv",
            "precision_decision.json",
            "mechanism_validation.json",
        )
    }
    manifest = {
        "manifest_version": "1.0",
        "analysis_version": "frl-v3-resilience-primary-1.0",
        "locked_design_sha256": sha256_file(locked_design_path),
        "inputs": input_records,
        "sample_description": {
            "conditions": int(frame["condition_id"].nunique()),
            "replication_blocks": blocks,
            "institutions_per_condition_block": int(
                lock["selected_population"]
            ),
            "agent_condition_trajectories": int(
                len(frame) * int(lock["selected_population"])
            ),
            "independent_monte_carlo_unit": "replication block",
        },
        "primary_outcomes": list(PRIMARY_OUTCOMES),
        "secondary_outcomes": list(SECONDARY_OUTCOMES),
        "bootstrap": {
            "method": "paired replication-block nonparametric bootstrap",
            "replications": design.bootstrap_replications,
            "pointwise_level": 0.95,
            "simultaneous_method": "family-wise bootstrap max-t",
            "simultaneous_level": 0.95,
        },
        "mechanism_validation": validation,
        "precision_decision": precision_decision,
        "files": files,
    }
    write_json(output_dir / "results_manifest.json", manifest)
    return manifest


def _load_sensitivity_inputs(
    directories: list[Path],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for directory in directories:
        manifest = validate_experiment_directory(directory)
        frame = pd.read_csv(directory / "run_summary.csv")
        frames.append(frame)
        records.append(
            {
                "directory": directory.as_posix(),
                "experiment_manifest_sha256": sha256_file(
                    directory / "experiment_manifest.json"
                ),
                "run_summary_sha256": sha256_file(
                    directory / "run_summary.csv"
                ),
                "actual_counts": manifest["actual_counts"],
                "exit_events": manifest["exit_events"],
            }
        )
    if not frames:
        raise ValueError("At least one sensitivity input is required")
    return pd.concat(frames, ignore_index=True), records


def analyze_resilience_sensitivity(
    input_directories: list[Path],
    locked_design_path: Path,
    design: ResilienceDesignConfig,
    sensitivity: str,
    output_dir: Path,
) -> dict[str, Any]:
    require_new_directory(output_dir)
    frame, input_records = _load_sensitivity_inputs(input_directories)
    duplicate_key = ["replication_id", "condition_id"]
    if sensitivity == "stress_budget":
        duplicate_key.append("stress_budget")
    if frame.duplicated(duplicate_key).any():
        raise ValueError("Duplicate sensitivity run rows")
    lock = _read_json(locked_design_path)
    validation = _validate_resilience_frame(frame, lock, False)
    counts = frame.groupby(
        ["condition_id"]
        + (["stress_budget"] if sensitivity == "stress_budget" else [])
    )["replication_id"].nunique()
    if counts.nunique() != 1:
        raise ValueError("Sensitivity cells have unequal block counts")
    blocks = int(counts.iloc[0])

    cell_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    outcomes = {**PRIMARY_OUTCOMES, **SECONDARY_OUTCOMES}
    for outcome, column in outcomes.items():
        cells, estimands = sensitivity_estimands(
            frame, column, sensitivity
        )
        cell_rows.extend(
            _bootstrap_rows(
                outcome,
                cells,
                design.bootstrap_replications,
                f"resilience-sensitivity-{sensitivity}-cells-{outcome}",
                SENSITIVITY_CELL_FIELDS,
            )
        )
        for family in sorted(
            {str(item.metadata["family"]) for item in estimands}
        ):
            family_estimands = [
                item
                for item in estimands
                if item.metadata["family"] == family
            ]
            contrast_rows.extend(
                _bootstrap_rows(
                    outcome,
                    family_estimands,
                    design.bootstrap_replications,
                    (
                        f"resilience-sensitivity-{sensitivity}-{family}-"
                        f"{outcome}"
                    ),
                    SENSITIVITY_CONTRAST_FIELDS,
                )
            )

    with StableCsvWriter(
        output_dir / "sensitivity_cells.csv", SENSITIVITY_CELL_FIELDS
    ) as writer:
        writer.write_many(cell_rows)
    with StableCsvWriter(
        output_dir / "sensitivity_contrasts.csv",
        SENSITIVITY_CONTRAST_FIELDS,
    ) as writer:
        writer.write_many(contrast_rows)
    files = {
        filename: sha256_file(output_dir / filename)
        for filename in (
            "sensitivity_cells.csv",
            "sensitivity_contrasts.csv",
        )
    }
    manifest = {
        "manifest_version": "1.0",
        "analysis_version": "frl-v3-resilience-sensitivity-1.0",
        "sensitivity": sensitivity,
        "locked_design_sha256": sha256_file(locked_design_path),
        "inputs": input_records,
        "sample_description": {
            "conditions": int(frame["condition_id"].nunique()),
            "scenario_cells": int(len(counts)),
            "replication_blocks": blocks,
            "institutions_per_condition_block": int(
                lock["selected_population"]
            ),
            "agent_condition_trajectories": int(
                len(frame) * int(lock["selected_population"])
            ),
            "independent_monte_carlo_unit": "replication block",
        },
        "mechanism_validation": validation,
        "files": files,
    }
    write_json(output_dir / "results_manifest.json", manifest)
    return manifest

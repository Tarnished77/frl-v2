"""Outcome-informativeness pilot that is blind to channel ranking."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import DesignConfig, ModelConfig
from .experiment import pilot_conditions, replication_ids
from .io_utils import StableCsvWriter, require_new_directory, write_json
from .model import simulate_run
from .rng import stream_seed_manifest


SUMMARY_FIELDS = [
    "stress_budget",
    "replications",
    "conditions",
    "informative_cells",
    "floor_cells",
    "ceiling_cells",
    "central_exit_rate",
    "central_rmst_days",
    "distance_to_target",
    "meets_minimum",
]

CELL_FIELDS = [
    "stress_budget",
    "calibration_ratio",
    "funding_intensity",
    "market_intensity",
    "agents",
    "events",
    "mean_exit_rate",
    "mean_rmst_days",
    "informative",
]


def select_pilot_summary(
    summary_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Apply the pre-specified outcome-informativeness rule."""
    if not summary_rows:
        raise ValueError("pilot summary cannot be empty")
    return min(
        summary_rows,
        key=lambda row: (
            -int(row["informative_cells"]),
            float(row["distance_to_target"]),
            float(row["stress_budget"]),
        ),
    )


def run_pilot(
    config: ModelConfig,
    design: DesignConfig,
    output_dir: Path,
) -> dict[str, object]:
    require_new_directory(output_dir)
    conditions = pilot_conditions(config, design)
    identifiers = replication_ids("pilot", 1, design.pilot_replications)
    namespace = f"{design.experiment_namespace}-pilot"
    summary_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []

    for stress_budget in design.pilot_stress_budgets:
        budget_cells: list[dict[str, object]] = []
        for condition in conditions:
            exit_rates: list[float] = []
            rmst_values: list[float] = []
            events = 0
            for replication_id in identifiers:
                result = simulate_run(
                    config,
                    condition,
                    stress_budget,
                    namespace,
                    replication_id,
                )
                exit_rates.append(float(result.run_row["exit_rate"]))
                rmst_values.append(float(result.run_row["rmst_days"]))
                events += int(result.run_row["event_count"])
            mean_exit = float(np.mean(exit_rates))
            informative = (
                design.pilot_informative_exit_low
                <= mean_exit
                <= design.pilot_informative_exit_high
            )
            row = {
                "stress_budget": stress_budget,
                "calibration_ratio": condition.calibration_ratio,
                "funding_intensity": condition.funding_intensity,
                "market_intensity": condition.market_intensity,
                "agents": design.pilot_replications * config.population,
                "events": events,
                "mean_exit_rate": mean_exit,
                "mean_rmst_days": float(np.mean(rmst_values)),
                "informative": int(informative),
            }
            budget_cells.append(row)
            cell_rows.append(row)

        stressed = [
            row
            for row in budget_cells
            if int(row["funding_intensity"]) + int(row["market_intensity"]) > 0
        ]
        central = next(
            row
            for row in budget_cells
            if float(row["calibration_ratio"]) == 1.0
            and int(row["funding_intensity"]) == 1
            and int(row["market_intensity"]) == 1
        )
        informative_cells = sum(int(row["informative"]) for row in stressed)
        summary_rows.append(
            {
                "stress_budget": stress_budget,
                "replications": design.pilot_replications,
                "conditions": len(conditions),
                "informative_cells": informative_cells,
                "floor_cells": sum(
                    float(row["mean_exit_rate"])
                    < design.pilot_informative_exit_low
                    for row in stressed
                ),
                "ceiling_cells": sum(
                    float(row["mean_exit_rate"])
                    > design.pilot_informative_exit_high
                    for row in stressed
                ),
                "central_exit_rate": central["mean_exit_rate"],
                "central_rmst_days": central["mean_rmst_days"],
                "distance_to_target": abs(
                    float(central["mean_exit_rate"])
                    - design.pilot_target_exit_rate
                ),
                "meets_minimum": int(
                    informative_cells >= design.pilot_min_informative_cells
                ),
            }
        )

    selected = select_pilot_summary(summary_rows)
    with StableCsvWriter(
        output_dir / "pilot_summary.csv", SUMMARY_FIELDS
    ) as writer:
        writer.write_many(summary_rows)
    with StableCsvWriter(output_dir / "pilot_cells.csv", CELL_FIELDS) as writer:
        writer.write_many(cell_rows)
    write_json(
        output_dir / "seed_manifest.json",
        stream_seed_manifest(namespace, identifiers),
    )

    lock = {
        "lock_version": "1.0",
        "selection_rule": (
            "Maximize the number of stressed cells with mean exit risk in the "
            "pre-specified informative interval; break ties by central-cell "
            "distance to 0.30, then by the smaller stress budget. Channel "
            "rankings are not used."
        ),
        "pilot_namespace": namespace,
        "pilot_replication_ids": identifiers,
        "selected_stress_budget": float(selected["stress_budget"]),
        "selected_informative_cells": int(selected["informative_cells"]),
        "selected_central_exit_rate": float(selected["central_exit_rate"]),
        "selected_central_rmst_days": float(selected["central_rmst_days"]),
        "selected_meets_minimum": bool(selected["meets_minimum"]),
        "model_config": config.to_dict(),
        "design_config": design.to_dict(),
    }
    write_json(output_dir / "pilot_lock.json", lock)
    return lock

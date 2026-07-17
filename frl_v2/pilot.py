"""Outcome-blind-to-ranking calibration of the common stress budget."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import DesignConfig, ModelConfig
from .io_utils import StableCsvWriter, require_new_directory, write_json
from .model import Condition, simulate_run


PILOT_FIELDS = [
    "stress_budget",
    "replications",
    "agents",
    "events",
    "mean_exit_rate",
    "mean_rmst_days",
    "acceptable",
    "distance_to_target",
]


def run_pilot(
    config: ModelConfig,
    design: DesignConfig,
    output_dir: Path,
) -> dict[str, object]:
    require_new_directory(output_dir)
    condition = Condition(1.0, 1, 1, "heterogeneous")
    replication_ids = [
        f"pilot-{index:04d}" for index in range(1, design.pilot_replications + 1)
    ]
    rows: list[dict[str, object]] = []

    for stress_budget in design.pilot_stress_budgets:
        exit_rates: list[float] = []
        rmst_values: list[float] = []
        events = 0
        for replication_id in replication_ids:
            result = simulate_run(
                config,
                condition,
                stress_budget,
                f"{design.experiment_namespace}-pilot",
                replication_id,
            )
            exit_rates.append(float(result.run_row["exit_rate"]))
            rmst_values.append(float(result.run_row["rmst_days"]))
            events += int(result.run_row["event_count"])
        mean_exit_rate = float(np.mean(exit_rates))
        acceptable = (
            design.pilot_acceptable_exit_rate_low
            <= mean_exit_rate
            <= design.pilot_acceptable_exit_rate_high
        )
        rows.append(
            {
                "stress_budget": stress_budget,
                "replications": design.pilot_replications,
                "agents": design.pilot_replications * config.population,
                "events": events,
                "mean_exit_rate": mean_exit_rate,
                "mean_rmst_days": float(np.mean(rmst_values)),
                "acceptable": int(acceptable),
                "distance_to_target": abs(mean_exit_rate - design.pilot_target_exit_rate),
            }
        )

    acceptable_rows = [row for row in rows if row["acceptable"] == 1]
    candidates = acceptable_rows or rows
    selected = min(
        candidates,
        key=lambda row: (float(row["distance_to_target"]), float(row["stress_budget"])),
    )

    with StableCsvWriter(output_dir / "pilot_results.csv", PILOT_FIELDS) as writer:
        writer.write_many(rows)

    locked = {
        "lock_version": "1.0",
        "selection_rule": (
            "Choose the acceptable candidate closest to the target exit rate; "
            "if none is acceptable, choose the closest candidate. Break ties "
            "toward the smaller common stress budget."
        ),
        "pilot_namespace": f"{design.experiment_namespace}-pilot",
        "pilot_condition": asdict(condition),
        "pilot_replication_ids": replication_ids,
        "selected_stress_budget": float(selected["stress_budget"]),
        "selected_mean_exit_rate": float(selected["mean_exit_rate"]),
        "selected_mean_rmst_days": float(selected["mean_rmst_days"]),
        "selected_is_acceptable": bool(selected["acceptable"]),
        "model_config": config.to_dict(),
        "design_config": design.to_dict(),
    }
    write_json(output_dir / "locked_design.json", locked)
    return locked

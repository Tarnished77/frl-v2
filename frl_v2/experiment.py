"""Experiment construction, execution, and immutable output manifests."""

from __future__ import annotations

from dataclasses import replace
import itertools
from pathlib import Path
from typing import Iterable

from .config import DesignConfig, ModelConfig
from .io_utils import StableCsvWriter, require_new_directory, sha256_file, write_json
from .model import Condition, SimulationResult, simulate_run
from .rng import stream_seed_manifest


AGENT_FIELDS = [
    "run_id",
    "replication_id",
    "condition_id",
    "population_design",
    "calibration_ratio",
    "liquidity_intensity",
    "market_intensity",
    "agent_id",
    "event",
    "exit_day",
    "survival_time",
    "exit_reason",
    "initial_equity",
    "initial_leverage",
    "initial_cash_share",
    "initial_cash",
    "initial_risky_value",
    "initial_liabilities",
    "final_cash",
    "final_risky_value",
    "final_liabilities",
    "final_equity",
    "total_payment",
    "total_unpaid",
    "total_forced_sale_proceeds",
]

DAILY_FIELDS = [
    "run_id",
    "replication_id",
    "condition_id",
    "population_design",
    "calibration_ratio",
    "liquidity_intensity",
    "market_intensity",
    "day",
    "at_risk_start",
    "at_risk_end",
    "exit_count",
    "liquidity_exit_count",
    "insolvency_exit_count",
    "joint_exit_count",
    "market_event",
    "market_loss_fraction",
    "price_impact_loss",
    "asset_price",
    "preimpact_forced_sale_value",
    "forced_sale_proceeds",
    "total_obligation",
    "total_unpaid",
    "mean_active_equity",
    "mean_active_cash",
]

RUN_FIELDS = [
    "run_id",
    "replication_id",
    "condition_id",
    "population_design",
    "calibration_ratio",
    "liquidity_intensity",
    "market_intensity",
    "population",
    "horizon",
    "stress_budget",
    "liquidity_budget",
    "market_budget",
    "market_event_loss_fraction",
    "event_count",
    "exit_rate",
    "rmst_days",
    "final_asset_price",
    "mean_final_equity",
    "mean_final_cash",
    "mean_total_unpaid",
    "mean_forced_sale_proceeds",
]


def primary_conditions(design: DesignConfig, population_design: str = "heterogeneous") -> list[Condition]:
    return [
        Condition(rho, liquidity, market, population_design)
        for rho, liquidity, market in itertools.product(
            design.calibration_ratios,
            design.liquidity_intensities,
            design.market_intensities,
        )
    ]


def parity_conditions(design: DesignConfig, population_design: str) -> list[Condition]:
    return [
        Condition(1.0, liquidity, market, population_design)
        for liquidity, market in itertools.product(
            design.liquidity_intensities,
            design.market_intensities,
        )
    ]


def replication_ids(prefix: str, count: int) -> list[str]:
    return [f"{prefix}-{index:04d}" for index in range(1, count + 1)]


def expected_counts(config: ModelConfig, conditions: int, replications: int) -> dict[str, int]:
    runs = conditions * replications
    return {
        "conditions": conditions,
        "runs": runs,
        "agent_rows": runs * config.population,
        "daily_rows": runs * config.horizon,
        "run_rows": runs,
    }


def run_experiment(
    config: ModelConfig,
    conditions: Iterable[Condition],
    stress_budget: float,
    namespace: str,
    replication_id_values: list[str],
    output_dir: Path,
    experiment_id: str,
) -> dict[str, object]:
    condition_values = sorted(list(conditions))
    if not condition_values:
        raise ValueError("At least one condition is required")
    require_new_directory(output_dir)

    agent_path = output_dir / "agent_survival.csv"
    daily_path = output_dir / "run_daily.csv"
    run_path = output_dir / "run_summary.csv"
    seed_path = output_dir / "seed_manifest.json"

    exit_events = 0
    with (
        StableCsvWriter(agent_path, AGENT_FIELDS) as agent_writer,
        StableCsvWriter(daily_path, DAILY_FIELDS) as daily_writer,
        StableCsvWriter(run_path, RUN_FIELDS) as run_writer,
    ):
        for condition in condition_values:
            for replication_id in replication_id_values:
                result: SimulationResult = simulate_run(
                    config,
                    condition,
                    stress_budget,
                    namespace,
                    replication_id,
                )
                agent_writer.write_many(result.agent_rows)
                daily_writer.write_many(result.daily_rows)
                run_writer.write(result.run_row)
                exit_events += int(result.run_row["event_count"])
            print(
                f"[{experiment_id}] completed {condition.condition_id} "
                f"({len(replication_id_values)} replications)",
                flush=True,
            )

        row_counts = {
            "agent_rows": agent_writer.rows_written,
            "daily_rows": daily_writer.rows_written,
            "run_rows": run_writer.rows_written,
        }

    write_json(seed_path, stream_seed_manifest(namespace, replication_id_values))
    expected = expected_counts(config, len(condition_values), len(replication_id_values))
    if any(row_counts[key] != expected[key] for key in row_counts):
        raise RuntimeError(f"Output row counts do not match design: {row_counts} vs {expected}")

    manifest = {
        "manifest_version": "1.0",
        "experiment_id": experiment_id,
        "experiment_namespace": namespace,
        "model_config": config.to_dict(),
        "stress_budget": stress_budget,
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "calibration_ratio": condition.calibration_ratio,
                "liquidity_intensity": condition.liquidity_intensity,
                "market_intensity": condition.market_intensity,
                "population_design": condition.population_design,
            }
            for condition in condition_values
        ],
        "replication_ids": replication_id_values,
        "expected_counts": expected,
        "actual_counts": row_counts,
        "exit_events": exit_events,
        "files": {
            "agent_survival.csv": sha256_file(agent_path),
            "run_daily.csv": sha256_file(daily_path),
            "run_summary.csv": sha256_file(run_path),
            "seed_manifest.json": sha256_file(seed_path),
        },
    }
    write_json(output_dir / "experiment_manifest.json", manifest)
    return manifest


def with_population(config: ModelConfig, population: int) -> ModelConfig:
    updated = replace(config, population=population)
    updated.validate()
    return updated

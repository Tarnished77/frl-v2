"""Experiment construction, execution, and immutable output manifests."""

from __future__ import annotations

from dataclasses import replace
import itertools
from pathlib import Path
from typing import Iterable

from .config import DesignConfig, ModelConfig
from .io_utils import (
    StableCsvWriter,
    require_new_directory,
    sha256_file,
    write_json,
)
from .model import Condition, SimulationResult, simulate_run
from .rng import stream_seed_manifest


AGENT_FIELDS = [
    "run_id",
    "replication_id",
    "condition_id",
    "calibration_ratio",
    "funding_intensity",
    "market_intensity",
    "price_impact_lambda",
    "market_event_probability",
    "cash_scenario",
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
    "total_funding_requested",
    "total_funding_repaid",
    "total_unpaid",
    "total_forced_sale_proceeds",
    "total_market_value_loss",
    "total_fire_sale_value_loss",
]

DAILY_FIELDS = [
    "run_id",
    "replication_id",
    "condition_id",
    "calibration_ratio",
    "funding_intensity",
    "market_intensity",
    "price_impact_lambda",
    "market_event_probability",
    "cash_scenario",
    "day",
    "at_risk_start",
    "at_risk_end",
    "exit_count",
    "funding_exit_count",
    "insolvency_exit_count",
    "joint_exit_count",
    "market_event",
    "market_loss_fraction",
    "price_impact_loss",
    "forced_sale_fraction",
    "asset_price",
    "preimpact_forced_sale_value",
    "forced_sale_proceeds",
    "total_funding_requested",
    "total_funding_repaid",
    "total_unpaid",
    "total_liabilities",
    "mean_active_equity",
    "mean_active_cash",
]

RUN_FIELDS = [
    "run_id",
    "replication_id",
    "condition_id",
    "calibration_ratio",
    "funding_intensity",
    "market_intensity",
    "price_impact_lambda",
    "market_event_probability",
    "cash_scenario",
    "population",
    "horizon",
    "stress_budget",
    "funding_budget",
    "market_budget",
    "market_event_loss_fraction",
    "target_cumulative_funding",
    "target_cumulative_market_loss",
    "event_count",
    "exit_rate",
    "rmst_days",
    "final_asset_price",
    "asset_price_loss",
    "mean_final_equity",
    "mean_equity_loss",
    "p90_equity_loss",
    "worst_10pct_mean_equity_loss",
    "mean_final_cash",
    "mean_final_liabilities",
    "mean_liability_reduction",
    "mean_total_funding_requested",
    "mean_total_funding_repaid",
    "mean_total_unpaid",
    "mean_forced_sale_proceeds",
    "mean_market_value_loss",
    "mean_fire_sale_value_loss",
]


def grid_conditions(
    ratios: Iterable[float],
    funding_intensities: Iterable[int],
    market_intensities: Iterable[int],
    lambdas: Iterable[float],
    event_probabilities: Iterable[float],
    cash_scenarios: Iterable[str],
) -> list[Condition]:
    conditions = [
        Condition(rho, funding, market, impact, probability, cash)
        for rho, funding, market, impact, probability, cash in itertools.product(
            ratios,
            funding_intensities,
            market_intensities,
            lambdas,
            event_probabilities,
            cash_scenarios,
        )
    ]
    identifiers = [condition.condition_id for condition in conditions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("condition identifiers are not unique")
    return sorted(conditions)


def primary_conditions(
    config: ModelConfig, design: DesignConfig
) -> list[Condition]:
    return grid_conditions(
        design.calibration_ratios,
        design.funding_intensities,
        design.market_intensities,
        design.primary_feedback_lambdas,
        (config.market_event_probability,),
        ("baseline",),
    )


def pilot_conditions(
    config: ModelConfig, design: DesignConfig
) -> list[Condition]:
    return grid_conditions(
        design.calibration_ratios,
        design.funding_intensities,
        design.market_intensities,
        (config.benchmark_price_impact_lambda,),
        (config.market_event_probability,),
        ("baseline",),
    )


def finite_size_conditions(
    config: ModelConfig, design: DesignConfig
) -> list[Condition]:
    return grid_conditions(
        (1.0,),
        design.funding_intensities,
        design.market_intensities,
        (config.benchmark_price_impact_lambda,),
        (config.market_event_probability,),
        ("baseline",),
    )


def high_impact_conditions(
    config: ModelConfig, design: DesignConfig
) -> list[Condition]:
    return grid_conditions(
        (1.0,),
        design.funding_intensities,
        design.market_intensities,
        (
            config.benchmark_price_impact_lambda,
            config.high_price_impact_lambda,
        ),
        (config.market_event_probability,),
        ("baseline",),
    )


def frequency_conditions(
    config: ModelConfig, design: DesignConfig
) -> list[Condition]:
    probabilities = tuple(
        sorted(
            {
                config.market_event_probability,
                *design.market_frequency_sensitivity,
            }
        )
    )
    return grid_conditions(
        (1.0,),
        design.funding_intensities,
        design.market_intensities,
        (config.benchmark_price_impact_lambda,),
        probabilities,
        ("baseline",),
    )


def cash_buffer_conditions(
    config: ModelConfig, design: DesignConfig
) -> list[Condition]:
    return grid_conditions(
        (1.0,),
        design.funding_intensities,
        design.market_intensities,
        (config.benchmark_price_impact_lambda,),
        (config.market_event_probability,),
        ("low", "reference", "high"),
    )


def replication_ids(prefix: str, start: int, count: int) -> list[str]:
    if start < 1 or count < 1:
        raise ValueError("replication start and count must be positive")
    return [
        f"{prefix}-{index:04d}" for index in range(start, start + count)
    ]


def expected_counts(
    config: ModelConfig, conditions: int, replications: int
) -> dict[str, int]:
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
                f"({len(replication_id_values)} blocks)",
                flush=True,
            )

        row_counts = {
            "agent_rows": agent_writer.rows_written,
            "daily_rows": daily_writer.rows_written,
            "run_rows": run_writer.rows_written,
        }

    write_json(seed_path, stream_seed_manifest(namespace, replication_id_values))
    expected = expected_counts(
        config, len(condition_values), len(replication_id_values)
    )
    if any(row_counts[key] != expected[key] for key in row_counts):
        raise RuntimeError(
            f"Output row counts do not match design: {row_counts} vs {expected}"
        )

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
                "funding_intensity": condition.funding_intensity,
                "market_intensity": condition.market_intensity,
                "price_impact_lambda": condition.price_impact_lambda,
                "market_event_probability": condition.market_event_probability,
                "cash_scenario": condition.cash_scenario,
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

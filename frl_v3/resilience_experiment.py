"""Experiment design and numerical audit for continuous resilience outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelConfig, ResilienceDesignConfig
from .experiment import (
    grid_conditions,
    replication_ids,
    run_experiment,
)
from .io_utils import (
    StableCsvWriter,
    require_new_directory,
    sha256_file,
    write_json,
)
from .model import Condition


AUDIT_FIELDS = [
    "stress_budget",
    "conditions",
    "replication_blocks",
    "minimum_mean_equity_loss",
    "maximum_mean_equity_loss",
    "maximum_worst_10pct_mean_equity_loss",
    "maximum_asset_price_loss",
    "maximum_exit_rate",
    "nonfinite_values",
    "negative_liability_rows",
    "negative_unpaid_rows",
    "status",
]


def primary_conditions(
    config: ModelConfig, design: ResilienceDesignConfig
) -> list[Condition]:
    return grid_conditions(
        design.calibration_ratios,
        design.funding_intensities,
        design.market_intensities,
        design.primary_feedback_lambdas,
        (config.market_event_probability,),
        ("baseline",),
    )


def calibration_audit_conditions(
    config: ModelConfig, design: ResilienceDesignConfig
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
    config: ModelConfig, design: ResilienceDesignConfig
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
    config: ModelConfig, design: ResilienceDesignConfig
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
    config: ModelConfig, design: ResilienceDesignConfig
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
    config: ModelConfig, design: ResilienceDesignConfig
) -> list[Condition]:
    return grid_conditions(
        (1.0,),
        design.funding_intensities,
        design.market_intensities,
        (config.benchmark_price_impact_lambda,),
        (config.market_event_probability,),
        ("low", "reference", "high"),
    )


def budget_sensitivity_conditions(
    config: ModelConfig, design: ResilienceDesignConfig
) -> list[Condition]:
    return finite_size_conditions(config, design)


def _budget_token(value: float) -> str:
    return format(value, ".2f").replace(".", "p")


def run_calibration_audit(
    config: ModelConfig,
    design: ResilienceDesignConfig,
    output_root: Path,
) -> dict[str, object]:
    require_new_directory(output_root)
    identifiers = replication_ids(
        "audit", 1, design.calibration_audit_replications
    )
    namespace = f"{design.experiment_namespace}-calibration-audit"
    conditions = calibration_audit_conditions(config, design)
    summaries: list[dict[str, object]] = []
    child_manifests: dict[str, dict[str, object]] = {}
    checked_columns = [
        "mean_equity_loss",
        "worst_10pct_mean_equity_loss",
        "asset_price_loss",
        "mean_final_liabilities",
        "mean_total_unpaid",
    ]

    for budget in design.calibration_audit_budgets:
        child = output_root / f"budget-{_budget_token(budget)}"
        manifest = run_experiment(
            config,
            conditions,
            budget,
            namespace,
            identifiers,
            child,
            f"frl-v3-resilience-audit-{_budget_token(budget)}",
        )
        frame = pd.read_csv(child / "run_summary.csv")
        values = frame[checked_columns].to_numpy(dtype=float)
        nonfinite = int((~np.isfinite(values)).sum())
        negative_liabilities = int(
            (frame["mean_final_liabilities"] < -config.exit_tolerance).sum()
        )
        negative_unpaid = int(
            (frame["mean_total_unpaid"] < -config.exit_tolerance).sum()
        )
        status = (
            "passed"
            if nonfinite == 0
            and negative_liabilities == 0
            and negative_unpaid == 0
            else "failed"
        )
        summaries.append(
            {
                "stress_budget": budget,
                "conditions": len(conditions),
                "replication_blocks": len(identifiers),
                "minimum_mean_equity_loss": float(
                    frame["mean_equity_loss"].min()
                ),
                "maximum_mean_equity_loss": float(
                    frame["mean_equity_loss"].max()
                ),
                "maximum_worst_10pct_mean_equity_loss": float(
                    frame["worst_10pct_mean_equity_loss"].max()
                ),
                "maximum_asset_price_loss": float(
                    frame["asset_price_loss"].max()
                ),
                "maximum_exit_rate": float(frame["exit_rate"].max()),
                "nonfinite_values": nonfinite,
                "negative_liability_rows": negative_liabilities,
                "negative_unpaid_rows": negative_unpaid,
                "status": status,
            }
        )
        child_manifests[str(budget)] = {
            "directory": child.as_posix(),
            "experiment_manifest_sha256": sha256_file(
                child / "experiment_manifest.json"
            ),
            "run_summary_sha256": manifest["files"]["run_summary.csv"],
        }

    with StableCsvWriter(
        output_root / "calibration_audit_summary.csv", AUDIT_FIELDS
    ) as writer:
        writer.write_many(summaries)
    passed = all(row["status"] == "passed" for row in summaries)
    audit_manifest = {
        "manifest_version": "1.0",
        "design_version": design.design_version,
        "purpose": (
            "Numerical and range audit only; no outcome is used to select "
            "the primary stress budget."
        ),
        "primary_stress_budget_locked_before_audit": (
            design.primary_stress_budget
        ),
        "namespace": namespace,
        "replication_ids": identifiers,
        "budgets": list(design.calibration_audit_budgets),
        "child_manifests": child_manifests,
        "summary_sha256": sha256_file(
            output_root / "calibration_audit_summary.csv"
        ),
        "passed": passed,
    }
    write_json(output_root / "calibration_audit_manifest.json", audit_manifest)
    if not passed:
        raise RuntimeError("Continuous-outcome calibration audit failed")
    return audit_manifest


def run_budget_sensitivity(
    config: ModelConfig,
    design: ResilienceDesignConfig,
    replication_id_values: list[str],
    output_root: Path,
) -> dict[str, object]:
    require_new_directory(output_root)
    namespace = f"{design.experiment_namespace}-sensitivity-budget"
    conditions = budget_sensitivity_conditions(config, design)
    children: dict[str, dict[str, object]] = {}
    budgets = (
        design.stress_budget_sensitivity[0],
        design.primary_stress_budget,
        design.stress_budget_sensitivity[1],
    )
    for budget in budgets:
        child = output_root / f"budget-{_budget_token(budget)}"
        manifest = run_experiment(
            config,
            conditions,
            budget,
            namespace,
            replication_id_values,
            child,
            f"frl-v3-resilience-budget-{_budget_token(budget)}",
        )
        children[str(budget)] = {
            "directory": child.name,
            "experiment_manifest_sha256": sha256_file(
                child / "experiment_manifest.json"
            ),
            "files": manifest["files"],
        }
    root_manifest = {
        "manifest_version": "1.0",
        "design_version": design.design_version,
        "namespace": namespace,
        "replication_ids": replication_id_values,
        "budgets": list(budgets),
        "children": children,
    }
    write_json(output_root / "budget_sensitivity_manifest.json", root_manifest)
    return root_manifest

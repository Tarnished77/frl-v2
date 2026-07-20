"""Paired-block analysis for the locked FRL v3 experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import DesignConfig
from .io_utils import (
    StableCsvWriter,
    require_new_directory,
    sha256_file,
    write_json,
)
from .rng import seed_from_parts


CELL_FIELDS = [
    "outcome",
    "calibration_ratio",
    "price_impact_lambda",
    "funding_intensity",
    "market_intensity",
    "blocks",
    "estimate",
    "block_se",
    "pointwise_lower",
    "pointwise_upper",
    "simultaneous_lower",
    "simultaneous_upper",
]

CONTRAST_FIELDS = [
    "outcome",
    "family",
    "estimand_id",
    "calibration_ratio",
    "price_impact_lambda",
    "transition",
    "conditioning_intensity",
    "funding_intensity",
    "market_intensity",
    "blocks",
    "estimate",
    "block_se",
    "pointwise_lower",
    "pointwise_upper",
    "simultaneous_lower",
    "simultaneous_upper",
]

SENSITIVITY_CELL_FIELDS = [
    "sensitivity",
    "outcome",
    "scenario",
    "funding_intensity",
    "market_intensity",
    "blocks",
    "estimate",
    "block_se",
    "pointwise_lower",
    "pointwise_upper",
    "simultaneous_lower",
    "simultaneous_upper",
]

SENSITIVITY_CONTRAST_FIELDS = [
    "sensitivity",
    "outcome",
    "family",
    "estimand_id",
    "scenario",
    "reference_scenario",
    "transition",
    "funding_intensity",
    "market_intensity",
    "blocks",
    "estimate",
    "block_se",
    "pointwise_lower",
    "pointwise_upper",
    "simultaneous_lower",
    "simultaneous_upper",
]

KEY_COLUMNS = [
    "calibration_ratio",
    "price_impact_lambda",
    "funding_intensity",
    "market_intensity",
]


@dataclass(frozen=True)
class VectorEstimand:
    metadata: dict[str, Any]
    values: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_experiment_directory(path: Path) -> dict[str, Any]:
    manifest_path = path / "experiment_manifest.json"
    manifest = _read_json(manifest_path)
    for filename, expected_hash in manifest["files"].items():
        actual_hash = sha256_file(path / filename)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Experiment file hash mismatch for {path / filename}"
            )
    run_rows = len(pd.read_csv(path / "run_summary.csv"))
    if run_rows != int(manifest["actual_counts"]["run_rows"]):
        raise ValueError(f"Run-row count mismatch in {path}")
    return manifest


def load_primary_runs(
    input_directories: Iterable[Path],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    input_records: list[dict[str, Any]] = []
    condition_sets: list[set[str]] = []
    for path in input_directories:
        manifest = validate_experiment_directory(path)
        frame = pd.read_csv(path / "run_summary.csv")
        frames.append(frame)
        condition_sets.append(set(frame["condition_id"]))
        input_records.append(
            {
                "directory": path.as_posix(),
                "experiment_id": manifest["experiment_id"],
                "experiment_manifest_sha256": sha256_file(
                    path / "experiment_manifest.json"
                ),
                "run_summary_sha256": sha256_file(
                    path / "run_summary.csv"
                ),
                "actual_counts": manifest["actual_counts"],
                "exit_events": manifest["exit_events"],
            }
        )
    if not frames:
        raise ValueError("At least one primary experiment directory is required")
    if any(values != condition_sets[0] for values in condition_sets[1:]):
        raise ValueError("Primary experiment condition sets do not match")
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["replication_id", "condition_id"]).any():
        raise ValueError("Duplicate replication-condition run rows")
    counts = combined.groupby("condition_id")["replication_id"].nunique()
    if counts.nunique() != 1:
        raise ValueError("Primary conditions do not have equal block counts")
    required = {
        "replication_id",
        "condition_id",
        *KEY_COLUMNS,
        "horizon",
        "exit_rate",
        "rmst_days",
    }
    missing = required.difference(combined.columns)
    if missing:
        raise ValueError(f"Primary run data are missing columns: {sorted(missing)}")
    horizons = set(combined["horizon"].astype(int))
    if len(horizons) != 1:
        raise ValueError("Primary inputs do not share one horizon")
    combined["rmst_loss_days"] = (
        float(next(iter(horizons))) - combined["rmst_days"].astype(float)
    )
    return combined, input_records


def _condition_vectors(
    frame: pd.DataFrame, outcome_column: str
) -> tuple[list[str], dict[tuple[float, float, int, int], np.ndarray]]:
    ordered = frame.sort_values(["replication_id", *KEY_COLUMNS])
    pivot = ordered.pivot(
        index="replication_id",
        columns=KEY_COLUMNS,
        values=outcome_column,
    ).sort_index()
    if pivot.isna().any().any():
        raise ValueError("Primary grid is incomplete")
    identifiers = [str(value) for value in pivot.index]
    vectors = {
        (
            float(column[0]),
            float(column[1]),
            int(column[2]),
            int(column[3]),
        ): pivot[column].to_numpy(dtype=float)
        for column in pivot.columns
    }
    return identifiers, vectors


def _scenario_vectors(
    frame: pd.DataFrame,
    outcome_column: str,
    scenario_column: str,
) -> dict[tuple[Any, int, int], np.ndarray]:
    ordered = frame.sort_values(
        ["replication_id", scenario_column, "funding_intensity", "market_intensity"]
    )
    pivot = ordered.pivot(
        index="replication_id",
        columns=[scenario_column, "funding_intensity", "market_intensity"],
        values=outcome_column,
    ).sort_index()
    if pivot.isna().any().any():
        raise ValueError("Sensitivity grid is incomplete")
    return {
        (column[0], int(column[1]), int(column[2])): pivot[column].to_numpy(
            dtype=float
        )
        for column in pivot.columns
    }


def cell_estimands(
    frame: pd.DataFrame, outcome_column: str
) -> list[VectorEstimand]:
    _, vectors = _condition_vectors(frame, outcome_column)
    return [
        VectorEstimand(
            {
                "calibration_ratio": key[0],
                "price_impact_lambda": key[1],
                "funding_intensity": key[2],
                "market_intensity": key[3],
            },
            values,
        )
        for key, values in sorted(vectors.items())
    ]


def primary_estimands(
    frame: pd.DataFrame, outcome_column: str
) -> list[VectorEstimand]:
    _, values = _condition_vectors(frame, outcome_column)
    ratios = sorted({key[0] for key in values})
    lambdas = sorted({key[1] for key in values})
    funding_levels = sorted({key[2] for key in values})
    market_levels = sorted({key[3] for key in values})
    if funding_levels != [0, 1, 2] or market_levels != [0, 1, 2]:
        raise ValueError("Primary intensity grid must contain 0, 1, and 2")
    if lambdas != [0.0, 0.15]:
        raise ValueError("Primary feedback grid must contain 0 and 0.15")

    estimands: list[VectorEstimand] = []

    def add(family: str, identifier: str, vector: np.ndarray, **metadata: Any) -> None:
        estimands.append(
            VectorEstimand(
                {
                    "family": family,
                    "estimand_id": identifier,
                    "calibration_ratio": "",
                    "price_impact_lambda": "",
                    "transition": "",
                    "conditioning_intensity": "",
                    "funding_intensity": "",
                    "market_intensity": "",
                    **metadata,
                },
                vector,
            )
        )

    for rho in ratios:
        for impact in lambdas:
            funding_averages: dict[str, np.ndarray] = {}
            market_averages: dict[str, np.ndarray] = {}
            for lower in (0, 1):
                transition = f"{lower}_to_{lower + 1}"
                funding_vectors = []
                for market in market_levels:
                    vector = (
                        values[(rho, impact, lower + 1, market)]
                        - values[(rho, impact, lower, market)]
                    )
                    funding_vectors.append(vector)
                    add(
                        "funding_local_cell",
                        f"funding_{transition}_given_market_{market}",
                        vector,
                        calibration_ratio=rho,
                        price_impact_lambda=impact,
                        transition=transition,
                        conditioning_intensity=market,
                        market_intensity=market,
                    )
                funding_average = np.mean(funding_vectors, axis=0)
                funding_averages[transition] = funding_average
                add(
                    "funding_local_average",
                    f"funding_{transition}_average_market",
                    funding_average,
                    calibration_ratio=rho,
                    price_impact_lambda=impact,
                    transition=transition,
                )

                market_vectors = []
                for funding in funding_levels:
                    vector = (
                        values[(rho, impact, funding, lower + 1)]
                        - values[(rho, impact, funding, lower)]
                    )
                    market_vectors.append(vector)
                    add(
                        "market_local_cell",
                        f"market_{transition}_given_funding_{funding}",
                        vector,
                        calibration_ratio=rho,
                        price_impact_lambda=impact,
                        transition=transition,
                        conditioning_intensity=funding,
                        funding_intensity=funding,
                    )
                market_average = np.mean(market_vectors, axis=0)
                market_averages[transition] = market_average
                add(
                    "market_local_average",
                    f"market_{transition}_average_funding",
                    market_average,
                    calibration_ratio=rho,
                    price_impact_lambda=impact,
                    transition=transition,
                )

                add(
                    "liquidity_minus_market",
                    f"funding_minus_market_{transition}",
                    funding_average - market_average,
                    calibration_ratio=rho,
                    price_impact_lambda=impact,
                    transition=transition,
                )
                axis_funding = (
                    values[(rho, impact, lower + 1, 0)]
                    - values[(rho, impact, lower, 0)]
                )
                axis_market = (
                    values[(rho, impact, 0, lower + 1)]
                    - values[(rho, impact, 0, lower)]
                )
                add(
                    "axis_liquidity_minus_market",
                    f"axis_funding_minus_market_{transition}",
                    axis_funding - axis_market,
                    calibration_ratio=rho,
                    price_impact_lambda=impact,
                    transition=transition,
                )

            funding_slope = np.mean(
                [
                    (
                        values[(rho, impact, 2, market)]
                        - values[(rho, impact, 0, market)]
                    )
                    / 2.0
                    for market in market_levels
                ],
                axis=0,
            )
            market_slope = np.mean(
                [
                    (
                        values[(rho, impact, funding, 2)]
                        - values[(rho, impact, funding, 0)]
                    )
                    / 2.0
                    for funding in funding_levels
                ],
                axis=0,
            )
            add(
                "zero_to_two_summary",
                "funding_0_to_2_average_slope",
                funding_slope,
                calibration_ratio=rho,
                price_impact_lambda=impact,
                transition="0_to_2_average",
            )
            add(
                "zero_to_two_summary",
                "market_0_to_2_average_slope",
                market_slope,
                calibration_ratio=rho,
                price_impact_lambda=impact,
                transition="0_to_2_average",
            )
            add(
                "zero_to_two_channel_difference",
                "funding_minus_market_0_to_2_average_slope",
                funding_slope - market_slope,
                calibration_ratio=rho,
                price_impact_lambda=impact,
                transition="0_to_2_average",
            )

            for funding in (1, 2):
                for market in (1, 2):
                    interaction = (
                        values[(rho, impact, funding, market)]
                        - values[(rho, impact, funding, 0)]
                        - values[(rho, impact, 0, market)]
                        + values[(rho, impact, 0, 0)]
                    )
                    add(
                        "channel_interaction",
                        f"interaction_funding_{funding}_market_{market}",
                        interaction,
                        calibration_ratio=rho,
                        price_impact_lambda=impact,
                        funding_intensity=funding,
                        market_intensity=market,
                    )

        amplification_vectors = []
        for funding in funding_levels:
            for market in market_levels:
                if funding == 0 and market == 0:
                    continue
                vector = (
                    values[(rho, 0.15, funding, market)]
                    - values[(rho, 0.0, funding, market)]
                )
                amplification_vectors.append(vector)
                add(
                    "fire_sale_amplification_cell",
                    f"fire_sale_funding_{funding}_market_{market}",
                    vector,
                    calibration_ratio=rho,
                    funding_intensity=funding,
                    market_intensity=market,
                )
        add(
            "fire_sale_amplification_average",
            "fire_sale_average_stressed_cells",
            np.mean(amplification_vectors, axis=0),
            calibration_ratio=rho,
        )

    return estimands


def sensitivity_estimands(
    frame: pd.DataFrame,
    outcome_column: str,
    sensitivity: str,
) -> tuple[list[VectorEstimand], list[VectorEstimand]]:
    specifications = {
        "high_impact": ("price_impact_lambda", 0.15, (0.30,)),
        "market_frequency": (
            "market_event_probability",
            0.10,
            (0.05, 0.20),
        ),
        "cash_buffer": ("cash_scenario", "reference", ("low", "high")),
        "stress_budget": ("stress_budget", 0.15, (0.10, 0.30)),
    }
    if sensitivity not in specifications:
        raise ValueError(f"Unknown sensitivity: {sensitivity}")
    scenario_column, reference, comparisons = specifications[sensitivity]
    values = _scenario_vectors(frame, outcome_column, scenario_column)
    scenarios = sorted(
        {key[0] for key in values},
        key=lambda value: str(value),
    )
    expected_scenarios = {reference, *comparisons}
    if set(scenarios) != expected_scenarios:
        raise ValueError(
            f"{sensitivity} scenarios do not match the locked design"
        )

    cells = [
        VectorEstimand(
            {
                "sensitivity": sensitivity,
                "scenario": scenario,
                "funding_intensity": funding,
                "market_intensity": market,
            },
            vector,
        )
        for (scenario, funding, market), vector in sorted(
            values.items(), key=lambda item: tuple(str(value) for value in item[0])
        )
    ]
    contrasts: list[VectorEstimand] = []

    def add(
        family: str,
        identifier: str,
        vector: np.ndarray,
        **metadata: Any,
    ) -> None:
        contrasts.append(
            VectorEstimand(
                {
                    "sensitivity": sensitivity,
                    "family": family,
                    "estimand_id": identifier,
                    "scenario": "",
                    "reference_scenario": "",
                    "transition": "",
                    "funding_intensity": "",
                    "market_intensity": "",
                    **metadata,
                },
                vector,
            )
        )

    stressed = [
        (funding, market)
        for funding in (0, 1, 2)
        for market in (0, 1, 2)
        if funding + market > 0
    ]
    for comparison in comparisons:
        difference_vectors = []
        for funding, market in stressed:
            vector = (
                values[(comparison, funding, market)]
                - values[(reference, funding, market)]
            )
            difference_vectors.append(vector)
            add(
                "scenario_difference_cell",
                f"{comparison}_minus_{reference}_funding_{funding}_market_{market}",
                vector,
                scenario=comparison,
                reference_scenario=reference,
                funding_intensity=funding,
                market_intensity=market,
            )
        add(
            "scenario_difference_average",
            f"{comparison}_minus_{reference}_average_stressed",
            np.mean(difference_vectors, axis=0),
            scenario=comparison,
            reference_scenario=reference,
        )

    for scenario in scenarios:
        for lower in (0, 1):
            transition = f"{lower}_to_{lower + 1}"
            funding_average = np.mean(
                [
                    values[(scenario, lower + 1, market)]
                    - values[(scenario, lower, market)]
                    for market in (0, 1, 2)
                ],
                axis=0,
            )
            market_average = np.mean(
                [
                    values[(scenario, funding, lower + 1)]
                    - values[(scenario, funding, lower)]
                    for funding in (0, 1, 2)
                ],
                axis=0,
            )
            add(
                "sensitivity_channel_local",
                f"funding_{transition}",
                funding_average,
                scenario=scenario,
                transition=transition,
            )
            add(
                "sensitivity_channel_local",
                f"market_{transition}",
                market_average,
                scenario=scenario,
                transition=transition,
            )
            add(
                "sensitivity_channel_difference",
                f"funding_minus_market_{transition}",
                funding_average - market_average,
                scenario=scenario,
                transition=transition,
            )
    return cells, contrasts


def _bootstrap_rows(
    outcome: str,
    estimands: list[VectorEstimand],
    bootstrap_replications: int,
    seed_label: str,
    output_fields: list[str],
) -> list[dict[str, Any]]:
    if not estimands:
        return []
    block_counts = {len(estimand.values) for estimand in estimands}
    if len(block_counts) != 1:
        raise ValueError("Estimands do not share one block count")
    blocks = next(iter(block_counts))
    matrix = np.column_stack([estimand.values for estimand in estimands])
    estimates = matrix.mean(axis=0)
    bootstrap = np.empty((bootstrap_replications, matrix.shape[1]))
    generator = np.random.default_rng(
        seed_from_parts("frl-v3-paired-bootstrap", seed_label)
    )
    batch = 100
    for start in range(0, bootstrap_replications, batch):
        count = min(batch, bootstrap_replications - start)
        indices = generator.integers(0, blocks, size=(count, blocks))
        bootstrap[start : start + count] = matrix[indices].mean(axis=1)

    standard_errors = bootstrap.std(axis=0, ddof=1)
    pointwise = np.quantile(
        bootstrap, (0.025, 0.975), axis=0, method="linear"
    )
    valid = standard_errors > np.finfo(float).eps
    max_statistics = np.zeros(bootstrap_replications)
    if np.any(valid):
        max_statistics = np.max(
            np.abs(
                (bootstrap[:, valid] - estimates[valid])
                / standard_errors[valid]
            ),
            axis=1,
        )
    critical = float(
        np.quantile(max_statistics, 0.95, method="linear")
    )
    simultaneous_halfwidth = critical * standard_errors

    rows: list[dict[str, Any]] = []
    for index, estimand in enumerate(estimands):
        row = {
            "outcome": outcome,
            **estimand.metadata,
            "blocks": blocks,
            "estimate": float(estimates[index]),
            "block_se": float(standard_errors[index]),
            "pointwise_lower": float(pointwise[0, index]),
            "pointwise_upper": float(pointwise[1, index]),
            "simultaneous_lower": float(
                estimates[index] - simultaneous_halfwidth[index]
            ),
            "simultaneous_upper": float(
                estimates[index] + simultaneous_halfwidth[index]
            ),
        }
        rows.append({field: row.get(field, "") for field in output_fields})
    return rows


def analyze_primary(
    input_directories: list[Path],
    locked_design_path: Path,
    design: DesignConfig,
    output_dir: Path,
) -> dict[str, Any]:
    require_new_directory(output_dir)
    frame, input_records = load_primary_runs(input_directories)
    lock = _read_json(locked_design_path)
    expected_population = int(lock["selected_population"])
    if set(frame["population"].astype(int)) != {expected_population}:
        raise ValueError("Primary population does not match the design lock")
    expected_budget = float(lock["selected_stress_budget"])
    if not np.allclose(frame["stress_budget"].astype(float), expected_budget):
        raise ValueError("Primary stress budget does not match the design lock")
    blocks = int(frame.groupby("condition_id")["replication_id"].nunique().iloc[0])

    cell_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    outcome_columns = {
        "day30_exit_risk": "exit_rate",
        "rmst_loss_days": "rmst_loss_days",
    }
    for outcome, column in outcome_columns.items():
        cells = cell_estimands(frame, column)
        cell_rows.extend(
            _bootstrap_rows(
                outcome,
                cells,
                design.bootstrap_replications,
                f"primary-cells-{outcome}-{blocks}",
                CELL_FIELDS,
            )
        )
        contrasts = primary_estimands(frame, column)
        families = sorted(
            {str(item.metadata["family"]) for item in contrasts}
        )
        for family in families:
            family_estimands = [
                item
                for item in contrasts
                if item.metadata["family"] == family
            ]
            contrast_rows.extend(
                _bootstrap_rows(
                    outcome,
                    family_estimands,
                    design.bootstrap_replications,
                    f"primary-{family}-{outcome}-{blocks}",
                    CONTRAST_FIELDS,
                )
            )

    with StableCsvWriter(
        output_dir / "primary_cells.csv", CELL_FIELDS
    ) as writer:
        writer.write_many(cell_rows)
    with StableCsvWriter(
        output_dir / "primary_contrasts.csv", CONTRAST_FIELDS
    ) as writer:
        writer.write_many(contrast_rows)

    precision_rows = [
        row
        for row in contrast_rows
        if row["family"] == "liquidity_minus_market"
    ]
    precision_by_outcome: dict[str, dict[str, float]] = {}
    for outcome in outcome_columns:
        rows = [row for row in precision_rows if row["outcome"] == outcome]
        maximum_halfwidth = max(
            (
                float(row["simultaneous_upper"])
                - float(row["simultaneous_lower"])
            )
            / 2.0
            for row in rows
        )
        tolerance = (
            design.monte_carlo_exit_halfwidth_tolerance
            if outcome == "day30_exit_risk"
            else design.monte_carlo_rmst_halfwidth_tolerance
        )
        precision_by_outcome[outcome] = {
            "maximum_simultaneous_halfwidth": maximum_halfwidth,
            "tolerance": tolerance,
            "passes": maximum_halfwidth <= tolerance,
        }
    extension_required = blocks < design.maximum_primary_replications and any(
        not values["passes"] for values in precision_by_outcome.values()
    )
    precision_decision = {
        "decision_version": "1.0",
        "rule": (
            "Use the largest 95% simultaneous confidence-interval half-width "
            "within the pre-specified liquidity-minus-market local-effect "
            "family across calibration ratios, feedback states, and 0-to-1 "
            "or 1-to-2 transitions. If either outcome exceeds its tolerance, "
            "append locked blocks 201--400 to every primary condition."
        ),
        "independent_monte_carlo_unit": "replication block",
        "blocks_analyzed": blocks,
        "precision_by_outcome": precision_by_outcome,
        "extension_required": extension_required,
    }
    write_json(output_dir / "precision_decision.json", precision_decision)

    files = {
        filename: sha256_file(output_dir / filename)
        for filename in (
            "primary_cells.csv",
            "primary_contrasts.csv",
            "precision_decision.json",
        )
    }
    manifest = {
        "manifest_version": "1.0",
        "analysis_version": "frl-v3-primary-1.0",
        "locked_design_sha256": sha256_file(locked_design_path),
        "inputs": input_records,
        "sample_description": {
            "conditions": int(frame["condition_id"].nunique()),
            "replication_blocks": blocks,
            "institutions_per_condition_block": expected_population,
            "agent_condition_trajectories": int(len(frame) * expected_population),
            "independent_monte_carlo_unit": "replication block",
        },
        "bootstrap": {
            "method": "paired replication-block nonparametric bootstrap",
            "replications": design.bootstrap_replications,
            "pointwise_level": 0.95,
            "simultaneous_method": "family-wise bootstrap max-t",
            "simultaneous_level": 0.95,
            "seed_derivation": (
                "SHA-256 of frl-v3-paired-bootstrap and the fixed family label"
            ),
        },
        "precision_decision": precision_decision,
        "files": files,
    }
    write_json(output_dir / "results_manifest.json", manifest)
    return manifest


def analyze_sensitivity(
    input_directory: Path,
    locked_design_path: Path,
    design: DesignConfig,
    sensitivity: str,
    output_dir: Path,
) -> dict[str, Any]:
    require_new_directory(output_dir)
    manifest = validate_experiment_directory(input_directory)
    frame = pd.read_csv(input_directory / "run_summary.csv")
    if frame.duplicated(["replication_id", "condition_id"]).any():
        raise ValueError("Duplicate replication-condition sensitivity rows")
    horizon_values = set(frame["horizon"].astype(int))
    if len(horizon_values) != 1:
        raise ValueError("Sensitivity input does not share one horizon")
    frame["rmst_loss_days"] = (
        float(next(iter(horizon_values))) - frame["rmst_days"].astype(float)
    )
    lock = _read_json(locked_design_path)
    if set(frame["population"].astype(int)) != {
        int(lock["selected_population"])
    }:
        raise ValueError("Sensitivity population does not match design lock")
    if not np.allclose(
        frame["stress_budget"].astype(float),
        float(lock["selected_stress_budget"]),
    ):
        raise ValueError("Sensitivity stress budget does not match design lock")
    counts = frame.groupby("condition_id")["replication_id"].nunique()
    if counts.nunique() != 1:
        raise ValueError("Sensitivity conditions have unequal block counts")
    blocks = int(counts.iloc[0])

    cell_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for outcome, column in {
        "day30_exit_risk": "exit_rate",
        "rmst_loss_days": "rmst_loss_days",
    }.items():
        cells, contrasts = sensitivity_estimands(
            frame, column, sensitivity
        )
        cell_rows.extend(
            _bootstrap_rows(
                outcome,
                cells,
                design.bootstrap_replications,
                f"sensitivity-{sensitivity}-cells-{outcome}-{blocks}",
                SENSITIVITY_CELL_FIELDS,
            )
        )
        for family in sorted(
            {str(item.metadata["family"]) for item in contrasts}
        ):
            family_estimands = [
                item
                for item in contrasts
                if item.metadata["family"] == family
            ]
            contrast_rows.extend(
                _bootstrap_rows(
                    outcome,
                    family_estimands,
                    design.bootstrap_replications,
                    (
                        f"sensitivity-{sensitivity}-{family}-"
                        f"{outcome}-{blocks}"
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
    result_manifest = {
        "manifest_version": "1.0",
        "analysis_version": "frl-v3-sensitivity-1.0",
        "sensitivity": sensitivity,
        "locked_design_sha256": sha256_file(locked_design_path),
        "input": {
            "directory": input_directory.as_posix(),
            "experiment_manifest_sha256": sha256_file(
                input_directory / "experiment_manifest.json"
            ),
            "run_summary_sha256": sha256_file(
                input_directory / "run_summary.csv"
            ),
            "actual_counts": manifest["actual_counts"],
            "exit_events": manifest["exit_events"],
        },
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
        "bootstrap": {
            "method": "paired replication-block nonparametric bootstrap",
            "replications": design.bootstrap_replications,
            "simultaneous_method": "family-wise bootstrap max-t",
        },
        "files": files,
    }
    write_json(output_dir / "results_manifest.json", result_manifest)
    return result_manifest

"""Pre-specified outcome analysis for the FRL v2 experiments."""

from __future__ import annotations

import json
from pathlib import Path
import platform
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from patsy import dmatrix

from .config import DesignConfig
from .io_utils import StableCsvWriter, require_new_directory, sha256_file, write_json
from .rng import seed_from_parts


OUTCOMES = ("exit_rate", "rmst_days")


def read_and_validate_experiment(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = path / "experiment_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for filename, expected_hash in manifest["files"].items():
        actual_hash = sha256_file(path / filename)
        if actual_hash != expected_hash:
            raise ValueError(f"Checksum mismatch for {path / filename}")

    runs = pd.read_csv(path / "run_summary.csv")
    expected_rows = int(manifest["actual_counts"]["run_rows"])
    if len(runs) != expected_rows:
        raise ValueError(f"Run-summary row count mismatch in {path}")
    return manifest, runs


def _bootstrap_indices(
    blocks: int,
    bootstrap_replications: int,
    namespace: str,
) -> np.ndarray:
    generator = np.random.default_rng(
        seed_from_parts(namespace, "paired-block-bootstrap")
    )
    return generator.integers(
        0,
        blocks,
        size=(bootstrap_replications, blocks),
    )


def _interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, (0.025, 0.975))
    return float(low), float(high)


def summarize_cells(
    runs: pd.DataFrame,
    design: DesignConfig,
    namespace: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["calibration_ratio", "liquidity_intensity", "market_intensity"]
    for key_values, group in runs.groupby(keys, sort=True):
        blocks = group["replication_id"].nunique()
        if blocks != len(group):
            raise ValueError(f"Duplicate replication block in cell {key_values}")
        for outcome in OUTCOMES:
            values = group.sort_values("replication_id")[outcome].to_numpy(dtype=float)
            indices = _bootstrap_indices(
                len(values),
                design.bootstrap_replications,
                f"{namespace}|cell|{key_values}|{outcome}",
            )
            bootstrap_means = values[indices].mean(axis=1)
            ci_low, ci_high = _interval(bootstrap_means)
            rows.append(
                {
                    "calibration_ratio": float(key_values[0]),
                    "liquidity_intensity": int(key_values[1]),
                    "market_intensity": int(key_values[2]),
                    "outcome": outcome,
                    "estimate": float(values.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "replication_blocks": blocks,
                    "agents": int(group["population"].sum()),
                    "events": int(group["event_count"].sum()),
                }
            )
    return pd.DataFrame(rows)


def block_channel_effects(runs: pd.DataFrame) -> pd.DataFrame:
    required = {
        "calibration_ratio",
        "replication_id",
        "liquidity_intensity",
        "market_intensity",
        *OUTCOMES,
    }
    if not required.issubset(runs.columns):
        raise ValueError(f"Missing analysis columns: {sorted(required - set(runs.columns))}")

    rows: list[dict[str, Any]] = []
    for ratio, ratio_frame in runs.groupby("calibration_ratio", sort=True):
        for replication_id, block in ratio_frame.groupby("replication_id", sort=True):
            if len(block) != 9:
                raise ValueError(
                    f"Block {replication_id} at rho={ratio} does not contain nine cells"
                )
            for outcome in OUTCOMES:
                pivot = block.pivot(
                    index="liquidity_intensity",
                    columns="market_intensity",
                    values=outcome,
                )
                if pivot.shape != (3, 3) or pivot.isna().any().any():
                    raise ValueError(f"Unbalanced stress grid in block {replication_id}")
                liquidity_effect = float(
                    np.mean([(pivot.loc[2, m] - pivot.loc[0, m]) / 2 for m in (0, 1, 2)])
                )
                market_effect = float(
                    np.mean([(pivot.loc[l, 2] - pivot.loc[l, 0]) / 2 for l in (0, 1, 2)])
                )
                if outcome == "rmst_days":
                    liquidity_effect *= -1.0
                    market_effect *= -1.0
                rows.append(
                    {
                        "calibration_ratio": float(ratio),
                        "replication_id": replication_id,
                        "outcome": outcome,
                        "liquidity_effect": liquidity_effect,
                        "market_effect": market_effect,
                        "channel_difference": liquidity_effect - market_effect,
                    }
                )
    return pd.DataFrame(rows)


def summarize_channel_effects(
    block_effects: pd.DataFrame,
    design: DesignConfig,
    namespace: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (ratio, outcome), group in block_effects.groupby(
        ["calibration_ratio", "outcome"], sort=True
    ):
        ordered = group.sort_values("replication_id")
        blocks = ordered["replication_id"].nunique()
        if blocks != len(ordered):
            raise ValueError("Duplicate block-level channel contrast")
        indices = _bootstrap_indices(
            blocks,
            design.bootstrap_replications,
            f"{namespace}|channel|{ratio}|{outcome}",
        )
        for estimand, column in (
            ("liquidity_effect", "liquidity_effect"),
            ("market_effect", "market_effect"),
            ("liquidity_minus_market", "channel_difference"),
        ):
            values = ordered[column].to_numpy(dtype=float)
            bootstrap_means = values[indices].mean(axis=1)
            ci_low, ci_high = _interval(bootstrap_means)
            estimate = float(values.mean())
            if estimand == "liquidity_minus_market":
                if ci_low > 0:
                    evidence = "liquidity_larger"
                elif ci_high < 0:
                    evidence = "market_larger"
                else:
                    evidence = "uncertain"
            else:
                evidence = "not_applicable"
            rows.append(
                {
                    "calibration_ratio": float(ratio),
                    "outcome": outcome,
                    "estimand": estimand,
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "replication_blocks": blocks,
                    "evidence": evidence,
                }
            )
    return pd.DataFrame(rows)


def finite_size_diagnostic(
    primary_runs: pd.DataFrame,
    finite_runs: dict[int, pd.DataFrame],
    design: DesignConfig,
) -> pd.DataFrame:
    keys = ["liquidity_intensity", "market_intensity"]
    baseline = (
        primary_runs[primary_runs["calibration_ratio"] == 1.0]
        .groupby(keys, sort=True)[list(OUTCOMES)]
        .mean()
    )
    rows: list[dict[str, Any]] = []
    for population, runs in sorted(finite_runs.items()):
        candidate = runs.groupby(keys, sort=True)[list(OUTCOMES)].mean()
        if not candidate.index.equals(baseline.index):
            raise ValueError(f"Finite-size grid mismatch for N={population}")
        for index in baseline.index:
            exit_difference = float(candidate.loc[index, "exit_rate"] - baseline.loc[index, "exit_rate"])
            rmst_difference = float(candidate.loc[index, "rmst_days"] - baseline.loc[index, "rmst_days"])
            rows.append(
                {
                    "population": population,
                    "liquidity_intensity": int(index[0]),
                    "market_intensity": int(index[1]),
                    "exit_rate": float(candidate.loc[index, "exit_rate"]),
                    "baseline_exit_rate_n100": float(baseline.loc[index, "exit_rate"]),
                    "exit_rate_difference": exit_difference,
                    "rmst_days": float(candidate.loc[index, "rmst_days"]),
                    "baseline_rmst_days_n100": float(baseline.loc[index, "rmst_days"]),
                    "rmst_difference_days": rmst_difference,
                    "within_tolerance": int(
                        abs(exit_difference) <= design.finite_size_exit_risk_tolerance
                        and abs(rmst_difference) <= design.finite_size_rmst_tolerance_days
                    ),
                }
            )
    return pd.DataFrame(rows)


def homogeneous_sensitivity(
    primary_runs: pd.DataFrame,
    homogeneous_runs: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["liquidity_intensity", "market_intensity"]
    heterogeneous = primary_runs[primary_runs["calibration_ratio"] == 1.0][
        ["replication_id", *keys, *OUTCOMES]
    ]
    homogeneous = homogeneous_runs[["replication_id", *keys, *OUTCOMES]]
    merged = heterogeneous.merge(
        homogeneous,
        on=["replication_id", *keys],
        suffixes=("_heterogeneous", "_homogeneous"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for key_values, group in merged.groupby(keys, sort=True):
        rows.append(
            {
                "liquidity_intensity": int(key_values[0]),
                "market_intensity": int(key_values[1]),
                "heterogeneous_exit_rate": float(group["exit_rate_heterogeneous"].mean()),
                "homogeneous_exit_rate": float(group["exit_rate_homogeneous"].mean()),
                "homogeneous_minus_heterogeneous_exit_rate": float(
                    (group["exit_rate_homogeneous"] - group["exit_rate_heterogeneous"]).mean()
                ),
                "heterogeneous_rmst_days": float(group["rmst_days_heterogeneous"].mean()),
                "homogeneous_rmst_days": float(group["rmst_days_homogeneous"].mean()),
                "homogeneous_minus_heterogeneous_rmst_days": float(
                    (group["rmst_days_homogeneous"] - group["rmst_days_heterogeneous"]).mean()
                ),
                "replication_blocks": group["replication_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def summarize_exit_reasons(agent_outcomes: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "calibration_ratio",
        "liquidity_intensity",
        "market_intensity",
        "exit_reason",
    ]
    summary = (
        agent_outcomes.groupby(keys, sort=True, observed=True)
        .size()
        .rename("agents")
        .reset_index()
    )
    totals = agent_outcomes.groupby(
        ["calibration_ratio", "liquidity_intensity", "market_intensity"],
        sort=True,
        observed=True,
    ).size()
    summary["cell_agents"] = [
        int(totals.loc[(row.calibration_ratio, row.liquidity_intensity, row.market_intensity)])
        for row in summary.itertuples(index=False)
    ]
    summary["share_of_cell"] = summary["agents"] / summary["cell_agents"]
    return summary


def fit_cloglog_time_models(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    formula = (
        "bs(day, df=5, include_intercept=False) + liquidity_intensity + "
        "market_intensity + liquidity_intensity:market_intensity + "
        "liquidity_intensity:log_day + market_intensity:log_day"
    )
    for ratio, ratio_frame in daily.groupby("calibration_ratio", sort=True):
        frame = ratio_frame[
            (ratio_frame["at_risk_start"] > 0)
            & (ratio_frame["liquidity_intensity"] > 0)
            & (ratio_frame["market_intensity"] > 0)
        ].copy()
        frame["log_day"] = np.log(frame["day"])
        design_matrix = dmatrix(formula, frame, return_type="dataframe")
        response = np.column_stack(
            [
                frame["exit_count"].to_numpy(),
                (frame["at_risk_start"] - frame["exit_count"]).to_numpy(),
            ]
        )
        fitted = sm.GLM(
            response,
            design_matrix,
            family=sm.families.Binomial(link=sm.families.links.CLogLog()),
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": frame["replication_id"]},
            maxiter=200,
        )
        if not fitted.converged or not np.isfinite(fitted.params).all():
            raise RuntimeError(f"Cloglog time model failed for rho={ratio}")

        confidence = fitted.conf_int(alpha=0.05)
        for name in design_matrix.columns:
            coefficient_rows.append(
                {
                    "calibration_ratio": float(ratio),
                    "term": name,
                    "coefficient": float(fitted.params[name]),
                    "standard_error_clustered": float(fitted.bse[name]),
                    "ci_low": float(confidence.loc[name, 0]),
                    "ci_high": float(confidence.loc[name, 1]),
                    "p_value": float(fitted.pvalues[name]),
                }
            )

        diagnostic_rows.append(
            {
                "calibration_ratio": float(ratio),
                "scope": "interior stressed cells (liquidity and market intensities 1 or 2)",
                "aggregated_run_day_rows": len(frame),
                "replication_clusters": frame["replication_id"].nunique(),
                "at_risk_agent_days": int(frame["at_risk_start"].sum()),
                "exit_events": int(frame["exit_count"].sum()),
                "converged": int(fitted.converged),
                "deviance": float(fitted.deviance),
                "liquidity_log_time_coefficient": float(
                    fitted.params["liquidity_intensity:log_day"]
                ),
                "liquidity_log_time_p_value": float(
                    fitted.pvalues["liquidity_intensity:log_day"]
                ),
                "market_log_time_coefficient": float(
                    fitted.params["market_intensity:log_day"]
                ),
                "market_log_time_p_value": float(
                    fitted.pvalues["market_intensity:log_day"]
                ),
            }
        )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(diagnostic_rows)


def mean_daily_survival(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["survival_probability"] = frame["at_risk_end"] / frame.groupby(
        ["run_id"]
    )["at_risk_start"].transform("first")
    keys = [
        "calibration_ratio",
        "liquidity_intensity",
        "market_intensity",
        "day",
    ]
    return (
        frame.groupby(keys, sort=True)["survival_probability"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_survival", "std": "sd_survival", "count": "runs"})
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    with StableCsvWriter(path, list(frame.columns)) as writer:
        writer.write_many(frame.to_dict(orient="records"))


def _manifest_hash(path: Path) -> str:
    return sha256_file(path / "experiment_manifest.json")


def run_analysis(
    design: DesignConfig,
    primary_dir: Path,
    homogeneous_dir: Path,
    finite_dirs: dict[int, Path],
    output_dir: Path,
    analysis_source_commit: str,
) -> dict[str, Any]:
    require_new_directory(output_dir)
    primary_manifest, primary_runs = read_and_validate_experiment(primary_dir)
    homogeneous_manifest, homogeneous_runs = read_and_validate_experiment(homogeneous_dir)
    finite_inputs = {
        population: read_and_validate_experiment(path)
        for population, path in finite_dirs.items()
    }

    daily = pd.read_csv(primary_dir / "run_daily.csv")
    if len(daily) != int(primary_manifest["actual_counts"]["daily_rows"]):
        raise ValueError("Primary run-day row count does not match its manifest")

    cell_summary = summarize_cells(
        primary_runs,
        design,
        design.experiment_namespace,
    )
    block_effects = block_channel_effects(primary_runs)
    channel_summary = summarize_channel_effects(
        block_effects,
        design,
        design.experiment_namespace,
    )
    finite_summary = finite_size_diagnostic(
        primary_runs,
        {population: value[1] for population, value in finite_inputs.items()},
        design,
    )
    homogeneous_summary = homogeneous_sensitivity(primary_runs, homogeneous_runs)
    homogeneous_block_effects = block_channel_effects(homogeneous_runs)
    homogeneous_channel_summary = summarize_channel_effects(
        homogeneous_block_effects,
        design,
        f"{design.experiment_namespace}|homogeneous",
    )
    cloglog_coefficients, cloglog_diagnostics = fit_cloglog_time_models(daily)
    daily_survival = mean_daily_survival(daily)
    agent_outcomes = pd.read_csv(
        primary_dir / "agent_survival.csv",
        usecols=[
            "calibration_ratio",
            "liquidity_intensity",
            "market_intensity",
            "exit_reason",
        ],
    )
    if len(agent_outcomes) != int(primary_manifest["actual_counts"]["agent_rows"]):
        raise ValueError("Primary agent row count does not match its manifest")
    exit_reason_summary = summarize_exit_reasons(agent_outcomes)

    outputs = {
        "cell_summary.csv": cell_summary,
        "block_channel_effects.csv": block_effects,
        "channel_contrasts.csv": channel_summary,
        "finite_size_diagnostic.csv": finite_summary,
        "homogeneous_sensitivity.csv": homogeneous_summary,
        "homogeneous_channel_contrasts.csv": homogeneous_channel_summary,
        "exit_reason_summary.csv": exit_reason_summary,
        "cloglog_coefficients.csv": cloglog_coefficients,
        "cloglog_diagnostics.csv": cloglog_diagnostics,
        "daily_survival.csv": daily_survival,
    }
    for filename, frame in outputs.items():
        _write_frame(output_dir / filename, frame)

    comparison_rows = channel_summary[
        channel_summary["estimand"] == "liquidity_minus_market"
    ]
    evidence_patterns = {
        outcome: list(group.sort_values("calibration_ratio")["evidence"])
        for outcome, group in comparison_rows.groupby("outcome")
    }
    homogeneous_evidence = list(
        homogeneous_channel_summary[
            homogeneous_channel_summary["estimand"] == "liquidity_minus_market"
        ]
        .sort_values("outcome")["evidence"]
    )
    calibration_dependent = any(
        len(set(pattern)) > 1 or "uncertain" in pattern
        for pattern in evidence_patterns.values()
    ) or len({tuple(value) for value in evidence_patterns.values()}) > 1

    finite_pass = bool((finite_summary["within_tolerance"] == 1).all())
    contrast_half_widths = {
        outcome: float(
            ((group["ci_high"] - group["ci_low"]) / 2.0).max()
        )
        for outcome, group in channel_summary.groupby("outcome")
    }
    files = {filename: sha256_file(output_dir / filename) for filename in outputs}
    manifest = {
        "manifest_version": "1.0",
        "analysis_source_commit": analysis_source_commit,
        "analysis_namespace": design.experiment_namespace,
        "primary_sample": {
            "conditions": len(primary_manifest["conditions"]),
            "runs": int(primary_manifest["actual_counts"]["run_rows"]),
            "agents": int(primary_manifest["actual_counts"]["agent_rows"]),
            "events": int(primary_manifest["exit_events"]),
            "horizon_days": int(primary_manifest["model_config"]["horizon"]),
            "replication_blocks_per_cell": design.primary_replications,
        },
        "sensitivity_samples": {
            "homogeneous": {
                "runs": int(homogeneous_manifest["actual_counts"]["run_rows"]),
                "agents": int(homogeneous_manifest["actual_counts"]["agent_rows"]),
                "events": int(homogeneous_manifest["exit_events"]),
            },
            **{
                f"finite_{population}": {
                    "runs": int(value[0]["actual_counts"]["run_rows"]),
                    "agents": int(value[0]["actual_counts"]["agent_rows"]),
                    "events": int(value[0]["exit_events"]),
                }
                for population, value in finite_inputs.items()
            },
        },
        "primary_estimands": {
            "exit_risk": "day-30 exit-risk difference per one intensity step",
            "rmst": "30-day RMST loss in days per one intensity step",
            "interval": (
                f"95% percentile interval from {design.bootstrap_replications} "
                "paired replication-block bootstrap draws"
            ),
        },
        "channel_comparison_evidence": evidence_patterns,
        "homogeneous_channel_comparison_evidence_at_rho_1": homogeneous_evidence,
        "calibration_or_estimand_dependent": calibration_dependent,
        "finite_size_diagnostic_passed": finite_pass,
        "maximum_primary_contrast_ci_half_width": contrast_half_widths,
        "cell_saturation": {
            "mean_exit_rate_at_least_0_95": int(
                (
                    cell_summary[
                        cell_summary["outcome"] == "exit_rate"
                    ]["estimate"]
                    >= 0.95
                ).sum()
            ),
            "mean_exit_rate_at_most_0_05": int(
                (
                    cell_summary[
                        cell_summary["outcome"] == "exit_rate"
                    ]["estimate"]
                    <= 0.05
                ).sum()
            ),
        },
        "cloglog_scope": (
            "Interior stressed cells only; structural no-stress cells have zero "
            "events and were excluded to avoid separation. Day baseline uses a "
            "five-degree-of-freedom B-spline; standard errors cluster by block."
        ),
        "input_manifests": {
            "primary": _manifest_hash(primary_dir),
            "homogeneous": _manifest_hash(homogeneous_dir),
            **{
                f"finite_{population}": _manifest_hash(path)
                for population, path in finite_dirs.items()
            },
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "files": files,
    }
    write_json(output_dir / "results_manifest.json", manifest)
    return manifest

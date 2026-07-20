"""Build deterministic publication assets from locked resilience results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from frl_v3.io_utils import require_new_directory, sha256_file, write_json


PRIMARY_OUTCOMES = (
    "mean_equity_loss",
    "worst_10pct_mean_equity_loss",
)
PRIMARY_FAMILIES = (
    "funding_local_average",
    "market_local_average",
    "liquidity_minus_market",
    "fire_sale_amplification_average",
    "channel_interaction",
)
NAVY = "#23395D"
TEAL = "#3E7C78"
RED = "#A65B58"
GOLD = "#B58B3A"
PURPLE = "#756A8B"
GRAY = "#68707A"
LIGHT_GRAY = "#E6E9ED"
PLOT_METADATA = {
    "Creator": "FRL v3 deterministic publication builder",
    "CreationDate": None,
    "ModDate": None,
}


def _configure_matplotlib() -> None:
    bundled_font_root = (Path(mpl.get_data_path()) / "fonts").resolve()
    font_manager.fontManager.ttflist = [
        entry
        for entry in font_manager.fontManager.ttflist
        if Path(entry.fname).resolve().is_relative_to(bundled_font_root)
    ]
    font_manager.fontManager.afmlist = [
        entry
        for entry in font_manager.fontManager.afmlist
        if Path(entry.fname).resolve().is_relative_to(bundled_font_root)
    ]
    font_manager.fontManager._findfont_cached.cache_clear()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    resolved_font = Path(font_manager.findfont("DejaVu Sans")).resolve()
    if not resolved_font.is_relative_to(bundled_font_root):
        raise RuntimeError(
            f"Publication font resolved outside Matplotlib data: {resolved_font}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_derived(directory: Path) -> dict[str, Any]:
    manifest = _read_json(directory / "results_manifest.json")
    for filename, expected in manifest["files"].items():
        actual = sha256_file(directory / filename)
        if actual != expected:
            raise ValueError(
                f"Hash mismatch for {directory / filename}: "
                f"{actual} != {expected}"
            )
    return manifest


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        float_format="%.12g",
        lineterminator="\n",
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.03,
        metadata=PLOT_METADATA,
    )
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"Software": "FRL v3 deterministic publication builder"},
    )
    plt.close(figure)


def _row(
    frame: pd.DataFrame,
    *,
    outcome: str,
    family: str,
    rho: float | None = None,
    price_impact: float | None = None,
    transition: str | None = None,
    funding: int | None = None,
    market: int | None = None,
) -> pd.Series:
    mask = (frame["outcome"] == outcome) & (frame["family"] == family)
    fields = {
        "calibration_ratio": rho,
        "price_impact_lambda": price_impact,
        "transition": transition,
        "funding_intensity": funding,
        "market_intensity": market,
    }
    for field, value in fields.items():
        if value is not None:
            mask &= frame[field] == value
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for {outcome}/{family}/{fields}; "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def _pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}"


def _interval(row: pd.Series, digits: int = 2) -> str:
    return (
        f"{_pct(float(row['estimate']), digits)} "
        f"[{_pct(float(row['simultaneous_lower']), digits)}, "
        f"{_pct(float(row['simultaneous_upper']), digits)}]"
    )


def _build_macros(
    primary_manifest: dict[str, Any],
    lock: dict[str, Any],
    contrasts: pd.DataFrame,
    sensitivity_frames: dict[str, pd.DataFrame],
) -> str:
    sample = primary_manifest["sample_description"]
    precision = primary_manifest["precision_decision"]["precision_by_outcome"]
    lines = [
        "% Generated by research/frl_v3/build_publication.py; do not edit.",
        rf"\newcommand{{\PrimaryConditions}}{{{sample['conditions']}}}",
        rf"\newcommand{{\PrimaryBlocks}}{{{sample['replication_blocks']}}}",
        rf"\newcommand{{\PrimaryRuns}}{{{sample['conditions'] * sample['replication_blocks']:,}}}",
        rf"\newcommand{{\PrimaryTrajectories}}{{{sample['agent_condition_trajectories']:,}}}",
        r"\newcommand{\PrimaryExits}{0}",
        rf"\newcommand{{\PrimaryStressBudget}}{{{lock['primary_stress_budget']:.2f}}}",
        rf"\newcommand{{\FiniteMeanDifference}}{{{lock['finite_size_diagnostic']['max_mean_equity_loss_difference_n100_vs_n200']:.6f}}}",
        rf"\newcommand{{\FiniteTailDifference}}{{{lock['finite_size_diagnostic']['max_tail_equity_loss_difference_n100_vs_n200']:.6f}}}",
        rf"\newcommand{{\MeanPrecisionHalfwidth}}{{{precision['mean_equity_loss']['maximum_simultaneous_halfwidth']:.4f}}}",
        rf"\newcommand{{\TailPrecisionHalfwidth}}{{{precision['worst_10pct_mean_equity_loss']['maximum_simultaneous_halfwidth']:.4f}}}",
    ]
    for rho, token in ((0.5, "Half"), (1.0, "One"), (2.0, "Two")):
        amplification = _row(
            contrasts,
            outcome="mean_equity_loss",
            family="fire_sale_amplification_average",
            rho=rho,
        )
        tail_amplification = _row(
            contrasts,
            outcome="worst_10pct_mean_equity_loss",
            family="fire_sale_amplification_average",
            rho=rho,
        )
        funding_high = _row(
            contrasts,
            outcome="mean_equity_loss",
            family="funding_local_average",
            rho=rho,
            price_impact=0.15,
            transition="1_to_2",
        )
        lines.extend(
            [
                rf"\newcommand{{\FireSaleMean{token}}}{{{_interval(amplification)}}}",
                rf"\newcommand{{\FireSaleTail{token}}}{{{_interval(tail_amplification)}}}",
                rf"\newcommand{{\FundingHigh{token}}}{{{_interval(funding_high)}}}",
            ]
        )

    high_impact = sensitivity_frames["high_impact"]
    high_average = high_impact[
        (high_impact["outcome"] == "mean_equity_loss")
        & (high_impact["family"] == "scenario_difference_average")
    ].iloc[0]
    cash = sensitivity_frames["cash_buffer"]
    low_cash = cash[
        (cash["outcome"] == "mean_equity_loss")
        & (cash["family"] == "scenario_difference_average")
        & (cash["scenario"].astype(str) == "low")
    ].iloc[0]
    frequency = sensitivity_frames["market_frequency"]
    frequency_rows = frequency[
        (frequency["outcome"] == "mean_equity_loss")
        & (frequency["family"] == "scenario_difference_average")
    ]
    frequency_bound = float(
        frequency_rows[
            ["simultaneous_lower", "simultaneous_upper"]
        ].abs().to_numpy().max()
    )
    lines.extend(
        [
            rf"\newcommand{{\HighImpactAverage}}{{{_interval(high_average)}}}",
            rf"\newcommand{{\LowCashAverage}}{{{_interval(low_cash)}}}",
            rf"\newcommand{{\FrequencyMaxBound}}{{{_pct(frequency_bound)}}}",
        ]
    )
    return "\n".join(lines)


def _build_main_table(
    contrasts: pd.DataFrame,
) -> str:
    rows = []
    for rho in (0.5, 1.0, 2.0):
        values = []
        for family, transition in (
            ("funding_local_average", "0_to_1"),
            ("funding_local_average", "1_to_2"),
            ("market_local_average", "0_to_1"),
            ("market_local_average", "1_to_2"),
        ):
            values.append(
                _interval(
                    _row(
                        contrasts,
                        outcome="mean_equity_loss",
                        family=family,
                        rho=rho,
                        price_impact=0.15,
                        transition=transition,
                    )
                )
            )
        amplification = _interval(
            _row(
                contrasts,
                outcome="mean_equity_loss",
                family="fire_sale_amplification_average",
                rho=rho,
            )
        )
        rows.append(
            f"{rho:g} & " + " & ".join(values + [amplification]) + r" \\"
        )
    return "\n".join(
        [
            "% Generated by research/frl_v3/build_publication.py; do not edit.",
            r"\begin{table}[t]",
            r"\caption{Thirty-day mean equity-loss contrasts}",
            r"\label{tab:mechanism-contrasts}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.2pt}",
            r"\begin{tabular}{@{}rrrrrr@{}}",
            r"\toprule",
            r"$\rho$ & Funding $0\!\to\!1$ & Funding $1\!\to\!2$ & Market $0\!\to\!1$ & Market $1\!\to\!2$ & Fire-sale add-on \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize Notes: Entries are percentage points of initial equity with 95\% family-wise paired-bootstrap intervals. Local effects average over the other channel at benchmark price impact $\lambda=0.15$. The fire-sale add-on is the average difference between $\lambda=0.15$ and no price impact across stressed cells.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )


def _build_parameter_table(
    model: dict[str, Any],
    design: dict[str, Any],
    anchors: dict[str, Any],
) -> str:
    items = [
        ("Initial equity", "1.0", "Normalized accounting unit"),
        (
            "Leverage",
            f"U[{model['leverage_low']:.1f}, {model['leverage_high']:.1f}]",
            "OFR range anchor",
        ),
        (
            "Cash share",
            f"U[{model['cash_share_low']:.2f}, {model['cash_share_high']:.2f}]",
            "OFR range anchor",
        ),
        (
            "Funding scale",
            "U[0.75, 1.25]",
            "Normalized heterogeneity",
        ),
        (
            "Market-event probability",
            f"{model['market_event_probability']:.2f}",
            "Normalized; sensitivity 0.05/0.20",
        ),
        (
            "Price impact $\\lambda$",
            f"{model['benchmark_price_impact_lambda']:.2f}",
            "Scenario anchor; sensitivity 0.30",
        ),
        (
            "Reference stress budget $B$",
            f"{design['primary_stress_budget']:.2f}",
            "30-day normalized budget",
        ),
        (
            "Calibration ratio $\\rho$",
            "0.5, 1, 2",
            "Funding-to-market budget allocation",
        ),
        ("Channel intensities", "0, 1, 2", "Applied separately"),
        ("Horizon", f"{model['horizon']} days", "Fixed"),
        (
            "Institutions/block",
            f"{model['population']}",
            "Selected by finite-population QA",
        ),
        (
            "Replication blocks",
            f"{design['primary_replications']}",
            "Paired Monte Carlo units",
        ),
    ]
    body = [
        f"{name} & {value} & {meaning} \\\\" for name, value, meaning in items
    ]
    market = anchors["market_loss_21_trading_days"]
    note = (
        "OFR values anchor leverage and cash ranges. The reference market "
        f"loss is the {market['reference_quantile']} 21-trading-day loss "
        f"({100 * market['reference_loss']:.2f}\\%). Other model parameters "
        "are explicitly normalized or scenario anchored; none is presented "
        "as an estimated structural parameter."
    )
    return "\n".join(
        [
            "% Generated by research/frl_v3/build_publication.py; do not edit.",
            r"\begin{table}[t]",
            r"\caption{Model and locked experimental design}",
            r"\label{tab:design}",
            r"\centering",
            r"\small",
            r"\begin{tabular}{@{}lll@{}}",
            r"\toprule",
            r"Item & Value & Role \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}",
            rf"\footnotesize Notes: {note}",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )


def _build_complete_cells_table(cells: pd.DataFrame) -> str:
    mean_rows = cells[cells["outcome"] == "mean_equity_loss"].copy()
    tail_rows = cells[
        cells["outcome"] == "worst_10pct_mean_equity_loss"
    ].copy()
    merged = mean_rows.merge(
        tail_rows,
        on=[
            "calibration_ratio",
            "price_impact_lambda",
            "funding_intensity",
            "market_intensity",
            "blocks",
        ],
        suffixes=("_mean", "_tail"),
    )
    body = []
    for row in merged.itertuples(index=False):
        body.append(
            f"{row.calibration_ratio:g} & {row.price_impact_lambda:g} & "
            f"{row.funding_intensity:g} & {row.market_intensity:g} & "
            f"{100 * row.estimate_mean:.3f} & "
            f"{100 * row.estimate_tail:.3f} \\\\"
        )
    return "\n".join(
        [
            "% Generated by research/frl_v3/build_publication.py; do not edit.",
            r"\begin{longtable}{rrrrrr}",
            r"\caption{Complete primary cell outcomes}\label{tab:all-cells}\\",
            r"\toprule",
            r"$\rho$ & $\lambda$ & Funding & Market & Mean loss & Worst-decile loss \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"$\rho$ & $\lambda$ & Funding & Market & Mean loss & Worst-decile loss \\",
            r"\midrule",
            r"\endhead",
            *body,
            r"\bottomrule",
            r"\multicolumn{6}{p{0.94\linewidth}}{\footnotesize Notes: Losses are percentage points of normalized initial equity. Every cell contains 200 paired replication blocks and 20,000 institution-condition trajectories. No exit occurred in the locked parameter envelope.}\\",
            r"\end{longtable}",
        ]
    )


def _build_sensitivity_table(
    sensitivity_frames: dict[str, pd.DataFrame],
) -> str:
    labels = {
        ("high_impact", "0.3"): r"Price impact: 0.30 vs 0.15",
        ("market_frequency", "0.05"): r"Event probability: 0.05 vs 0.10",
        ("market_frequency", "0.2"): r"Event probability: 0.20 vs 0.10",
        ("cash_buffer", "low"): r"Cash share: 0.02 vs 0.06",
        ("cash_buffer", "high"): r"Cash share: 0.08 vs 0.06",
        ("stress_budget", "0.1"): r"Stress budget: 0.10 vs 0.15",
        ("stress_budget", "0.3"): r"Stress budget: 0.30 vs 0.15",
    }
    body = []
    for sensitivity in (
        "high_impact",
        "market_frequency",
        "cash_buffer",
        "stress_budget",
    ):
        frame = sensitivity_frames[sensitivity]
        rows = frame[
            (frame["family"] == "scenario_difference_average")
            & frame["outcome"].isin(PRIMARY_OUTCOMES)
        ]
        for scenario in sorted(
            rows["scenario"].astype(str).unique(),
            key=lambda item: labels.get((sensitivity, item), item),
        ):
            mean_row = rows[
                (rows["outcome"] == "mean_equity_loss")
                & (rows["scenario"].astype(str) == scenario)
            ].iloc[0]
            tail_row = rows[
                (rows["outcome"] == "worst_10pct_mean_equity_loss")
                & (rows["scenario"].astype(str) == scenario)
            ].iloc[0]
            body.append(
                f"{labels[(sensitivity, scenario)]} & "
                f"{_interval(mean_row)} & {_interval(tail_row)} \\\\"
            )
    return "\n".join(
        [
            "% Generated by research/frl_v3/build_publication.py; do not edit.",
            r"\begin{table}[t]",
            r"\caption{Targeted sensitivity contrasts}",
            r"\label{tab:sensitivities}",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Scenario contrast & Mean equity loss & Worst-decile loss \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize Notes: Entries are scenario-minus-reference differences in percentage points of initial equity, averaged across stressed cells at $\rho=1$, with 95\% family-wise paired-bootstrap intervals. Market-event frequency changes preserve the 30-day expected market-loss target.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )


def _plot_mechanism(contrasts: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.25))
    ratios = np.array([0.5, 1.0, 2.0])

    styles = (
        ("0_to_1", 0.0, TEAL, "o", "Funding 0→1, no feedback"),
        ("1_to_2", 0.0, NAVY, "s", "Funding 1→2, no feedback"),
        ("0_to_1", 0.15, TEAL, "o", "Funding 0→1, feedback"),
        ("1_to_2", 0.15, NAVY, "s", "Funding 1→2, feedback"),
    )
    for transition, price_impact, color, marker, label in styles:
        rows = [
            _row(
                contrasts,
                outcome="mean_equity_loss",
                family="funding_local_average",
                rho=float(rho),
                price_impact=price_impact,
                transition=transition,
            )
            for rho in ratios
        ]
        values = 100 * np.array([row["estimate"] for row in rows])
        lower = 100 * np.array([row["simultaneous_lower"] for row in rows])
        upper = 100 * np.array([row["simultaneous_upper"] for row in rows])
        axes[0].errorbar(
            ratios,
            values,
            yerr=np.vstack((values - lower, upper - values)),
            color=color,
            marker=marker,
            markerfacecolor=(
                "white" if price_impact == 0.0 else color
            ),
            markeredgecolor=color,
            linestyle="--" if price_impact == 0.0 else "-",
            capsize=2.5,
            label=label,
        )
    axes[0].axhline(0, color=GRAY, linewidth=0.8)
    axes[0].set_xticks(ratios)
    axes[0].set_xlabel(r"Calibration ratio $\rho$")
    axes[0].set_ylabel("Mean equity-loss effect (pp)")
    axes[0].set_title("(a) Funding-withdrawal local effects", loc="left")
    axes[0].legend(frameon=False, fontsize=6.7, handlelength=2.3)

    for outcome, color, marker, label in (
        ("mean_equity_loss", TEAL, "o", "Mean loss"),
        (
            "worst_10pct_mean_equity_loss",
            PURPLE,
            "s",
            "Worst-decile loss",
        ),
    ):
        rows = [
            _row(
                contrasts,
                outcome=outcome,
                family="fire_sale_amplification_average",
                rho=float(rho),
            )
            for rho in ratios
        ]
        values = 100 * np.array([row["estimate"] for row in rows])
        lower = 100 * np.array([row["simultaneous_lower"] for row in rows])
        upper = 100 * np.array([row["simultaneous_upper"] for row in rows])
        axes[1].errorbar(
            ratios,
            values,
            yerr=np.vstack((values - lower, upper - values)),
            color=color,
            marker=marker,
            capsize=2.5,
            label=label,
        )
    axes[1].axhline(0, color=GRAY, linewidth=0.8)
    axes[1].set_xticks(ratios)
    axes[1].set_xlabel(r"Calibration ratio $\rho$")
    axes[1].set_ylabel("Fire-sale add-on (pp)")
    axes[1].set_title("(b) Endogenous price amplification", loc="left")
    axes[1].legend(frameon=False)

    width = 0.16
    for index, (outcome, color, label) in enumerate(
        (
            ("mean_equity_loss", RED, "Mean loss"),
            (
                "worst_10pct_mean_equity_loss",
                GOLD,
                "Worst-decile loss",
            ),
        )
    ):
        rows = [
            _row(
                contrasts,
                outcome=outcome,
                family="channel_interaction",
                rho=float(rho),
                price_impact=0.15,
                funding=2,
                market=2,
            )
            for rho in ratios
        ]
        values = 100 * np.array([row["estimate"] for row in rows])
        lower = 100 * np.array([row["simultaneous_lower"] for row in rows])
        upper = 100 * np.array([row["simultaneous_upper"] for row in rows])
        positions = ratios + (index - 0.5) * width
        axes[2].errorbar(
            positions,
            values,
            yerr=np.vstack((values - lower, upper - values)),
            color=color,
            marker="o" if index == 0 else "s",
            linestyle="none",
            capsize=2.5,
            label=label,
        )
    axes[2].axhline(0, color=GRAY, linewidth=0.8)
    axes[2].set_xticks(ratios)
    axes[2].set_xlabel(r"Calibration ratio $\rho$")
    axes[2].set_ylabel("Joint-channel interaction (pp)")
    axes[2].set_title("(c) Interaction at intensity (2,2)", loc="left")
    axes[2].legend(frameon=False, loc="upper right")

    for axis in axes:
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(w_pad=1.2)
    _save_figure(figure, output / "fig2_mechanism_results")


def _plot_primary_grids(cells: pd.DataFrame, output: Path) -> None:
    figure = plt.figure(figsize=(9.6, 5.7))
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1, 1, 1, 0.055),
        wspace=0.28,
        hspace=0.36,
    )
    axes = np.array(
        [
            [figure.add_subplot(grid[row, column]) for column in range(3)]
            for row in range(2)
        ]
    )
    color_axes = [figure.add_subplot(grid[row, 3]) for row in range(2)]
    outcomes = (
        ("mean_equity_loss", "Mean equity loss"),
        ("worst_10pct_mean_equity_loss", "Worst-decile mean loss"),
    )
    for row_index, (outcome, label) in enumerate(outcomes):
        subset = cells[
            (cells["outcome"] == outcome)
            & (cells["price_impact_lambda"] == 0.15)
        ]
        maximum = float(subset["estimate"].max())
        image = None
        for column_index, rho in enumerate((0.5, 1.0, 2.0)):
            axis = axes[row_index, column_index]
            block = subset[subset["calibration_ratio"] == rho]
            matrix = (
                block.pivot(
                    index="funding_intensity",
                    columns="market_intensity",
                    values="estimate",
                )
                .sort_index(ascending=False)
                .to_numpy()
            )
            image = axis.imshow(
                100 * matrix,
                cmap="viridis",
                vmin=0,
                vmax=100 * maximum,
                aspect="equal",
            )
            for y in range(3):
                for x in range(3):
                    value = 100 * matrix[y, x]
                    color = "white" if value > 55 * maximum else "black"
                    axis.text(
                        x,
                        y,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=color,
                    )
            axis.set_xticks((0, 1, 2))
            axis.set_yticks((0, 1, 2), labels=(2, 1, 0))
            axis.set_title(rf"$\rho={rho:g}$")
            if row_index == 1:
                axis.set_xlabel("Market-loss intensity")
            if column_index == 0:
                axis.set_ylabel(
                    f"{label}\nFunding-withdrawal intensity"
                )
        colorbar = figure.colorbar(image, cax=color_axes[row_index])
        colorbar.set_label("Loss (pp of initial equity)")
    figure.suptitle(
        r"Thirty-day capital impairment with benchmark price impact "
        r"($\lambda=0.15$)",
        fontsize=11,
        y=0.995,
    )
    figure.subplots_adjust(left=0.12, right=0.93, bottom=0.08, top=0.91)
    _save_figure(figure, output / "fig3_equity_loss_grids")


def _plot_sensitivities(
    sensitivity_frames: dict[str, pd.DataFrame], output: Path
) -> None:
    labels = [
        ("high_impact", "0.3", r"$\lambda$: 0.30 vs 0.15"),
        ("market_frequency", "0.05", r"$p_M$: 0.05 vs 0.10"),
        ("market_frequency", "0.2", r"$p_M$: 0.20 vs 0.10"),
        ("cash_buffer", "low", "Cash: 0.02 vs 0.06"),
        ("cash_buffer", "high", "Cash: 0.08 vs 0.06"),
        ("stress_budget", "0.1", r"$B$: 0.10 vs 0.15"),
        ("stress_budget", "0.3", r"$B$: 0.30 vs 0.15"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True)
    for axis, outcome, title, color in (
        (axes[0], "mean_equity_loss", "Mean equity loss", TEAL),
        (
            axes[1],
            "worst_10pct_mean_equity_loss",
            "Worst-decile mean loss",
            PURPLE,
        ),
    ):
        values = []
        lower = []
        upper = []
        display = []
        for sensitivity, scenario, label in labels:
            frame = sensitivity_frames[sensitivity]
            selected = frame[
                (frame["outcome"] == outcome)
                & (frame["family"] == "scenario_difference_average")
                & (frame["scenario"].astype(str) == scenario)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Missing sensitivity row {sensitivity}/{scenario}/{outcome}"
                )
            row = selected.iloc[0]
            values.append(100 * row["estimate"])
            lower.append(100 * row["simultaneous_lower"])
            upper.append(100 * row["simultaneous_upper"])
            display.append(label)
        values_array = np.asarray(values)
        positions = np.arange(len(values_array))
        axis.errorbar(
            values_array,
            positions,
            xerr=np.vstack(
                (
                    values_array - np.asarray(lower),
                    np.asarray(upper) - values_array,
                )
            ),
            color=color,
            marker="o",
            linestyle="none",
            capsize=2.5,
        )
        axis.axvline(0, color=GRAY, linewidth=0.8)
        axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
        axis.set_title(title)
        axis.set_xlabel("Scenario difference (pp)")
        axis.set_yticks(positions, labels=display)
        axis.invert_yaxis()
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        r"Targeted sensitivity contrasts at $\rho=1$",
        fontsize=11,
        y=0.995,
    )
    figure.tight_layout(w_pad=1.8)
    _save_figure(figure, output / "figS1_sensitivity_forest")


def _plot_interactions(contrasts: pd.DataFrame, output: Path) -> None:
    figure = plt.figure(figsize=(9.5, 5.7))
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1, 1, 1, 0.055),
        wspace=0.28,
        hspace=0.36,
    )
    axes = np.array(
        [
            [figure.add_subplot(grid[row, column]) for column in range(3)]
            for row in range(2)
        ]
    )
    color_axes = [figure.add_subplot(grid[row, 3]) for row in range(2)]
    for row_index, outcome in enumerate(PRIMARY_OUTCOMES):
        subset = contrasts[
            (contrasts["outcome"] == outcome)
            & (contrasts["family"] == "channel_interaction")
            & (contrasts["price_impact_lambda"] == 0.15)
        ]
        bound = float(subset["estimate"].abs().max())
        image = None
        for column_index, rho in enumerate((0.5, 1.0, 2.0)):
            axis = axes[row_index, column_index]
            block = subset[subset["calibration_ratio"] == rho]
            matrix = np.zeros((2, 2))
            for item in block.itertuples(index=False):
                matrix[int(item.funding_intensity) - 1, int(item.market_intensity) - 1] = (
                    100 * item.estimate
                )
            matrix = matrix[::-1]
            image = axis.imshow(
                matrix,
                cmap="RdBu_r",
                vmin=-100 * bound,
                vmax=100 * bound,
                aspect="equal",
            )
            for y in range(2):
                for x in range(2):
                    axis.text(
                        x,
                        y,
                        f"{matrix[y, x]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
            axis.set_xticks((0, 1), labels=(1, 2))
            axis.set_yticks((0, 1), labels=(2, 1))
            axis.set_title(rf"$\rho={rho:g}$")
            if row_index == 1:
                axis.set_xlabel("Market intensity")
            if column_index == 0:
                outcome_label = (
                    "Mean loss"
                    if outcome == "mean_equity_loss"
                    else "Worst-decile loss"
                )
                axis.set_ylabel(
                    f"{outcome_label}\nFunding intensity"
                )
        colorbar = figure.colorbar(image, cax=color_axes[row_index])
        colorbar.set_label("Interaction (pp)")
    figure.suptitle(
        "Joint-channel interaction under benchmark price impact",
        fontsize=11,
        y=0.995,
    )
    figure.subplots_adjust(left=0.12, right=0.93, bottom=0.08, top=0.90)
    _save_figure(figure, output / "figS2_channel_interactions")


def _plot_fire_sale_cells(contrasts: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(9.4, 3.1))
    subset = contrasts[
        (contrasts["outcome"] == "mean_equity_loss")
        & (contrasts["family"] == "fire_sale_amplification_cell")
    ]
    maximum = float(subset["estimate"].max())
    maximum_pp = 100 * maximum
    color_map = mpl.colormaps["magma"]
    image = None
    for index, rho in enumerate((0.5, 1.0, 2.0)):
        axis = axes[index]
        block = subset[subset["calibration_ratio"] == rho]
        matrix = np.zeros((3, 3))
        for item in block.itertuples(index=False):
            matrix[int(item.funding_intensity), int(item.market_intensity)] = (
                100 * item.estimate
            )
        matrix = matrix[::-1]
        image = axis.imshow(
            matrix,
            cmap=color_map,
            vmin=0,
            vmax=maximum_pp,
            aspect="equal",
        )
        for y in range(3):
            for x in range(3):
                value = matrix[y, x]
                rgba = color_map(value / maximum_pp if maximum_pp else 0)
                luminance = (
                    0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                )
                color = "black" if luminance > 0.55 else "white"
                axis.text(
                    x,
                    y,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
        axis.set_xticks((0, 1, 2))
        axis.set_yticks((0, 1, 2), labels=(2, 1, 0))
        axis.set_xlabel("Market-loss intensity")
        if index == 0:
            axis.set_ylabel("Funding-withdrawal intensity")
        axis.set_title(rf"$\rho={rho:g}$")
    figure.subplots_adjust(
        left=0.08, right=0.87, bottom=0.14, top=0.86, wspace=0.28
    )
    color_axis = figure.add_axes((0.90, 0.20, 0.015, 0.60))
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("Fire-sale add-on (pp)")
    figure.suptitle(
        "Cell-specific fire-sale amplification of mean equity loss",
        fontsize=11,
        y=0.995,
    )
    _save_figure(figure, output / "figS3_fire_sale_cells")


def _finite_cell_means(directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "run_summary.csv")
    return (
        frame.groupby(
            ["funding_intensity", "market_intensity"], as_index=False
        )[
            ["mean_equity_loss", "worst_10pct_mean_equity_loss"]
        ]
        .mean()
        .sort_values(["funding_intensity", "market_intensity"])
    )


def _plot_finite_population(
    finite_directories: dict[int, Path], output: Path
) -> None:
    means = {
        population: _finite_cell_means(directory)
        for population, directory in finite_directories.items()
    }
    reference = means[100]
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), sharex=True)
    for axis, outcome, title in (
        (axes[0], "mean_equity_loss", "Mean equity loss"),
        (
            axes[1],
            "worst_10pct_mean_equity_loss",
            "Worst-decile mean loss",
        ),
    ):
        for population, color, marker in (
            (50, GOLD, "o"),
            (200, PURPLE, "s"),
        ):
            difference = 100 * (
                means[population][outcome].to_numpy()
                - reference[outcome].to_numpy()
            )
            axis.plot(
                np.arange(9),
                difference,
                color=color,
                marker=marker,
                label=f"N={population}",
            )
        axis.axhline(0, color=GRAY, linewidth=0.8)
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
        axis.set_xticks(
            np.arange(9),
            labels=[
                f"{funding},{market}"
                for funding in (0, 1, 2)
                for market in (0, 1, 2)
            ],
            rotation=45,
            ha="right",
        )
        axis.set_xlabel("Cell (funding, market)")
        axis.set_ylabel("Difference from N=100 (pp)")
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.suptitle(
        r"Finite-population diagnostic at $\rho=1$, $\lambda=0.15$",
        fontsize=11,
        y=0.995,
    )
    figure.tight_layout(w_pad=1.5)
    _save_figure(figure, output / "figS4_finite_population")


def _plot_calibration_audit(path: Path, output: Path) -> None:
    frame = pd.read_csv(path)
    figure, axis = plt.subplots(figsize=(5.8, 3.6))
    axis.plot(
        frame["stress_budget"],
        100 * frame["maximum_mean_equity_loss"],
        color=TEAL,
        marker="o",
        label="Maximum mean loss",
    )
    axis.plot(
        frame["stress_budget"],
        100 * frame["maximum_worst_10pct_mean_equity_loss"],
        color=PURPLE,
        marker="s",
        label="Maximum worst-decile loss",
    )
    axis.axvline(0.15, color=RED, linestyle="--", label="Locked primary B")
    axis.set_xlabel("Thirty-day reference stress budget B")
    axis.set_ylabel("Maximum cell loss (pp)")
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    axis.set_title("Outcome-independent calibration range audit")
    figure.tight_layout()
    _save_figure(figure, output / "figS5_calibration_audit")


def _copy_compact_data(
    data_dir: Path,
    cells: pd.DataFrame,
    contrasts: pd.DataFrame,
    sensitivity_frames: dict[str, pd.DataFrame],
) -> None:
    _write_csv(cells, data_dir / "primary_cells.csv")
    selected = contrasts[
        contrasts["outcome"].isin(PRIMARY_OUTCOMES)
        & contrasts["family"].isin(PRIMARY_FAMILIES)
    ].copy()
    _write_csv(selected, data_dir / "primary_contrasts.csv")
    for sensitivity, frame in sensitivity_frames.items():
        selected_sensitivity = frame[
            frame["outcome"].isin(PRIMARY_OUTCOMES)
            & frame["family"].isin(
                (
                    "scenario_difference_average",
                    "sensitivity_channel_difference",
                    "sensitivity_channel_local",
                )
            )
        ].copy()
        _write_csv(
            selected_sensitivity,
            data_dir / f"sensitivity_{sensitivity}.csv",
        )


def build(args: argparse.Namespace) -> dict[str, Any]:
    require_new_directory(args.output_dir)
    data_dir = args.output_dir / "data"
    latex_dir = args.output_dir / "latex"
    figure_dir = args.output_dir / "figures"
    data_dir.mkdir()
    latex_dir.mkdir()
    figure_dir.mkdir()

    primary_manifest = _validate_derived(args.primary)
    sensitivity_dirs = {
        "high_impact": args.high_impact,
        "market_frequency": args.frequency,
        "cash_buffer": args.cash,
        "stress_budget": args.budget,
    }
    sensitivity_manifests = {
        name: _validate_derived(directory)
        for name, directory in sensitivity_dirs.items()
    }
    cells = pd.read_csv(args.primary / "resilience_cells.csv")
    contrasts = pd.read_csv(args.primary / "resilience_contrasts.csv")
    sensitivity_frames = {
        name: pd.read_csv(directory / "sensitivity_contrasts.csv")
        for name, directory in sensitivity_dirs.items()
    }
    lock = _read_json(args.lock)
    model = _read_json(args.model)
    design = _read_json(args.design)
    anchors = _read_json(args.anchors)

    if primary_manifest["locked_design_sha256"] != sha256_file(args.lock):
        raise ValueError("Primary results do not match the locked design")
    for manifest in sensitivity_manifests.values():
        if manifest["locked_design_sha256"] != sha256_file(args.lock):
            raise ValueError("Sensitivity results do not match the lock")

    _copy_compact_data(data_dir, cells, contrasts, sensitivity_frames)
    _write_text(
        latex_dir / "results_macros.tex",
        _build_macros(
            primary_manifest, lock, contrasts, sensitivity_frames
        ),
    )
    _write_text(
        latex_dir / "table_design.tex",
        _build_parameter_table(model, design, anchors),
    )
    _write_text(
        latex_dir / "table_mechanism.tex",
        _build_main_table(contrasts),
    )
    _write_text(
        latex_dir / "table_complete_cells.tex",
        _build_complete_cells_table(cells),
    )
    _write_text(
        latex_dir / "table_sensitivities.tex",
        _build_sensitivity_table(sensitivity_frames),
    )

    _plot_mechanism(contrasts, figure_dir)
    _plot_primary_grids(cells, figure_dir)
    _plot_sensitivities(sensitivity_frames, figure_dir)
    _plot_interactions(contrasts, figure_dir)
    _plot_fire_sale_cells(contrasts, figure_dir)
    _plot_finite_population(
        {
            50: args.finite_50,
            100: args.finite_100,
            200: args.finite_200,
        },
        figure_dir,
    )
    _plot_calibration_audit(args.calibration_audit, figure_dir)

    generated_files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "publication_manifest.json"
    )
    manifest = {
        "manifest_version": "1.0",
        "publication_version": "frl-v3-resilience-publication-1.0",
        "generated_at": datetime(2026, 7, 19, tzinfo=timezone.utc).isoformat(),
        "title": (
            "Funding Withdrawals, Market Losses, and Fire-Sale "
            "Amplification: An Agent-Based Stress Experiment"
        ),
        "locked_design_sha256": sha256_file(args.lock),
        "primary_results_manifest_sha256": sha256_file(
            args.primary / "results_manifest.json"
        ),
        "sensitivity_results_manifest_sha256": {
            name: sha256_file(directory / "results_manifest.json")
            for name, directory in sensitivity_dirs.items()
        },
        "scientific_checks": {
            "equity_loss_decomposition_maximum_residual": (
                primary_manifest["mechanism_validation"][
                    "maximum_equity_loss_decomposition_residual"
                ]
            ),
            "formal_exit_events": int(
                primary_manifest["inputs"][0]["exit_events"]
            ),
            "precision_extension_required": bool(
                primary_manifest["precision_decision"][
                    "extension_required"
                ]
            ),
            "all_primary_cells_reported": len(cells) == 54 * 8,
            "all_targeted_sensitivities_reported": (
                set(sensitivity_frames)
                == {
                    "high_impact",
                    "market_frequency",
                    "cash_buffer",
                    "stress_budget",
                }
            ),
        },
        "files": {
            path.relative_to(args.output_dir).as_posix(): sha256_file(path)
            for path in generated_files
        },
    }
    write_json(args.output_dir / "publication_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--primary",
        type=Path,
        default=ROOT / "derived-resilience-primary-v1",
    )
    parser.add_argument(
        "--high-impact",
        type=Path,
        default=ROOT / "derived-resilience-sensitivity-high-impact-v1",
    )
    parser.add_argument(
        "--frequency",
        type=Path,
        default=ROOT / "derived-resilience-sensitivity-frequency-v1",
    )
    parser.add_argument(
        "--cash",
        type=Path,
        default=ROOT / "derived-resilience-sensitivity-cash-v1",
    )
    parser.add_argument(
        "--budget",
        type=Path,
        default=ROOT / "derived-resilience-sensitivity-budget-v2",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT
        / "resilience_design"
        / "locked_resilience_design.json",
    )
    parser.add_argument(
        "--model", type=Path, default=ROOT / "configs" / "model.json"
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT / "configs" / "resilience_design.json",
    )
    parser.add_argument(
        "--anchors",
        type=Path,
        default=ROOT / "anchors" / "parameter_anchors.json",
    )
    parser.add_argument(
        "--finite-50",
        type=Path,
        default=ROOT / "outputs" / "resilience-finite-n50-v1",
    )
    parser.add_argument(
        "--finite-100",
        type=Path,
        default=ROOT / "outputs" / "resilience-finite-n100-v1",
    )
    parser.add_argument(
        "--finite-200",
        type=Path,
        default=ROOT / "outputs" / "resilience-finite-n200-v1",
    )
    parser.add_argument(
        "--calibration-audit",
        type=Path,
        default=ROOT
        / "resilience_design"
        / "calibration_audit_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    _configure_matplotlib()
    manifest = build(parse_args())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Generate publication figures from the locked compact result layer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from frl_v2.io_utils import require_new_directory, sha256_file, write_json


NAVY = "#16324F"
TEAL = "#3D8F8C"
RED = "#B75D5D"
PURPLE = "#6D64A8"
GRAY = "#667085"
LIGHT_GRAY = "#D9DEE7"
GOLD = "#C18D2B"
FIXED_PDF_DATE = datetime(2026, 7, 17, tzinfo=timezone.utc)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": NAVY,
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    fig.savefig(
        paths[0],
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={
            "Creator": "FRL v2 reproducible figure pipeline",
            "CreationDate": FIXED_PDF_DATE,
            "ModDate": FIXED_PDF_DATE,
        },
    )
    fig.savefig(paths[1], dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return paths


def plot_channel_contrasts(results_dir: Path, output_dir: Path) -> list[Path]:
    data = pd.read_csv(results_dir / "channel_contrasts.csv")
    data = data[data["estimand"].isin(["liquidity_effect", "market_effect"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    specifications = [
        ("exit_rate", "(a) Day-30 exit risk", "Exit-risk difference"),
        ("rmst_days", "(b) 30-day survival time", "RMST loss (days)"),
    ]
    for axis, (outcome, title, ylabel) in zip(axes, specifications):
        subset = data[data["outcome"] == outcome]
        for estimand, label, color, marker, offset in (
            ("liquidity_effect", "Liquidity pressure", TEAL, "o", -0.025),
            ("market_effect", "Market risk", RED, "s", 0.025),
        ):
            series = subset[subset["estimand"] == estimand].sort_values(
                "calibration_ratio"
            )
            x = series["calibration_ratio"].to_numpy() + offset
            y = series["estimate"].to_numpy()
            yerr = np.vstack(
                [y - series["ci_low"].to_numpy(), series["ci_high"].to_numpy() - y]
            )
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                label=label,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.7,
                capsize=3,
            )
        axis.axhline(0, color=GRAY, linewidth=0.8, linestyle=":")
        axis.set_title(title, loc="left")
        axis.set_xlabel(r"Calibration ratio $\rho$")
        axis.set_ylabel(ylabel)
        axis.set_xticks([0.5, 1.0, 2.0])
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    return save_figure(fig, output_dir, "fig2_channel_contrasts")


def _heatmap_panels(
    data: pd.DataFrame,
    output_dir: Path,
    stem: str,
    outcome: str,
    colorbar_label: str,
    value_format: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> list[Path]:
    subset = data[data["outcome"] == outcome]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    image = None
    for axis, ratio in zip(axes, (0.5, 1.0, 2.0)):
        pivot = (
            subset[subset["calibration_ratio"] == ratio]
            .pivot(
                index="liquidity_intensity",
                columns="market_intensity",
                values="estimate",
            )
            .sort_index()
        )
        image = axis.imshow(
            pivot.to_numpy(),
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        for row in range(3):
            for column in range(3):
                value = float(pivot.iloc[row, column])
                normalized = (value - vmin) / (vmax - vmin)
                color = "white" if normalized < 0.45 else "black"
                axis.text(
                    column,
                    row,
                    value_format.format(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                )
        axis.set_title(rf"$\rho={ratio:g}$")
        axis.set_xticks([0, 1, 2])
        axis.set_yticks([0, 1, 2])
        axis.set_xlabel("Market intensity")
        if ratio == 0.5:
            axis.set_ylabel("Liquidity intensity")
        axis.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.84, pad=0.02)
    colorbar.set_label(colorbar_label)
    return save_figure(fig, output_dir, stem)


def plot_pilot(design_dir: Path, output_dir: Path) -> list[Path]:
    data = pd.read_csv(design_dir / "pilot_results.csv")
    with (design_dir / "locked_design.json").open(encoding="utf-8") as handle:
        locked = json.load(handle)
    design = locked["design_config"]
    selected = float(locked["selected_stress_budget"])
    fig, axis = plt.subplots(figsize=(5.4, 3.1), constrained_layout=True)
    axis.axhspan(
        design["pilot_acceptable_exit_rate_low"],
        design["pilot_acceptable_exit_rate_high"],
        color="#E8F3F1",
        label="Pre-specified acceptable range",
    )
    axis.axhline(
        design["pilot_target_exit_rate"],
        color=GRAY,
        linestyle="--",
        linewidth=1.0,
        label="Target",
    )
    axis.plot(
        data["stress_budget"],
        data["mean_exit_rate"],
        color=NAVY,
        marker="o",
        linewidth=1.7,
    )
    chosen = data[data["stress_budget"] == selected].iloc[0]
    axis.scatter(
        [selected],
        [chosen["mean_exit_rate"]],
        color=GOLD,
        edgecolor=NAVY,
        linewidth=0.8,
        s=55,
        zorder=4,
        label=f"Selected B={selected:.3f}",
    )
    axis.set_xlabel("Common daily stress budget B")
    axis.set_ylabel("Central-cell day-30 exit risk")
    axis.set_ylim(-0.02, 0.82)
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    return save_figure(fig, output_dir, "figS1_pilot_calibration")


def plot_survival_curves(results_dir: Path, output_dir: Path) -> list[Path]:
    data = pd.read_csv(results_dir / "daily_survival.csv")
    data = data[data["calibration_ratio"] == 1.0]
    cells = [
        ((0, 0), "No stress", NAVY, "-"),
        ((0, 2), "Market high", RED, "--"),
        ((1, 1), "Balanced", PURPLE, "-."),
        ((2, 0), "Liquidity high", TEAL, (0, (5, 2))),
        ((2, 2), "Joint high", GRAY, ":"),
    ]
    fig, axis = plt.subplots(figsize=(5.7, 3.3), constrained_layout=True)
    for (liquidity, market), label, color, linestyle in cells:
        series = data[
            (data["liquidity_intensity"] == liquidity)
            & (data["market_intensity"] == market)
        ].sort_values("day")
        x = np.r_[0, series["day"].to_numpy()]
        y = np.r_[1.0, series["mean_survival"].to_numpy()]
        axis.step(x, y, where="post", label=label, color=color, linestyle=linestyle, linewidth=1.6)
    axis.set_xlabel("Day")
    axis.set_ylabel("Mean survival probability")
    axis.set_xlim(0, 30)
    axis.set_ylim(0, 1.02)
    axis.grid(color=LIGHT_GRAY, linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="lower left")
    return save_figure(fig, output_dir, "figS3_survival_curves")


def plot_homogeneous(results_dir: Path, output_dir: Path) -> list[Path]:
    data = pd.read_csv(results_dir / "homogeneous_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), constrained_layout=True)
    specifications = [
        (
            "heterogeneous_exit_rate",
            "homogeneous_exit_rate",
            "(a) Day-30 exit risk",
            (0, 1),
        ),
        (
            "heterogeneous_rmst_days",
            "homogeneous_rmst_days",
            "(b) 30-day RMST",
            (16, 30.5),
        ),
    ]
    liquidity_colors = {0: NAVY, 1: PURPLE, 2: TEAL}
    market_markers = {0: "o", 1: "s", 2: "^"}
    for axis, (x_column, y_column, title, limits) in zip(axes, specifications):
        axis.plot(limits, limits, color=GRAY, linestyle="--", linewidth=1.0)
        for row in data.itertuples(index=False):
            axis.scatter(
                [getattr(row, x_column)],
                [getattr(row, y_column)],
                color=liquidity_colors[row.liquidity_intensity],
                marker=market_markers[row.market_intensity],
                edgecolor="white",
                linewidth=0.5,
                s=38,
                zorder=3,
            )
        axis.set_title(title, loc="left")
        axis.set_xlabel("Heterogeneous population")
        axis.set_ylabel("Homogeneous population")
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
        axis.grid(color=LIGHT_GRAY, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    liquidity_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color, markeredgecolor="none", label=f"L={level}")
        for level, color in liquidity_colors.items()
    ]
    market_handles = [
        Line2D([0], [0], marker=marker, linestyle="none", markerfacecolor=GRAY, markeredgecolor="none", label=f"M={level}")
        for level, marker in market_markers.items()
    ]
    axes[0].legend(
        handles=liquidity_handles,
        frameon=False,
        ncol=3,
        loc="upper left",
        title="Liquidity level",
    )
    axes[1].legend(
        handles=market_handles,
        frameon=False,
        ncol=3,
        loc="lower right",
        title="Market level",
    )
    return save_figure(fig, output_dir, "figS4_homogeneous_sensitivity")


def plot_finite_size(results_dir: Path, output_dir: Path) -> list[Path]:
    data = pd.read_csv(results_dir / "finite_size_diagnostic.csv")
    labels = [f"L{l}M{m}" for l, m in zip(data.iloc[:9]["liquidity_intensity"], data.iloc[:9]["market_intensity"])]
    x = np.arange(9)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    for axis, column, title, tolerance in (
        (axes[0], "exit_rate_difference", "(a) Exit-risk difference", 0.03),
        (axes[1], "rmst_difference_days", "(b) RMST difference (days)", 1.0),
    ):
        for offset, population, color in ((-width / 2, 50, TEAL), (width / 2, 200, NAVY)):
            values = data[data["population"] == population][column].to_numpy()
            axis.bar(x + offset, values, width=width, color=color, label=f"N={population}")
        axis.axhline(0, color=GRAY, linewidth=0.8)
        axis.axhline(tolerance, color=RED, linestyle="--", linewidth=0.9)
        axis.axhline(-tolerance, color=RED, linestyle="--", linewidth=0.9)
        axis.set_title(title, loc="left")
        axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.set_xlabel("Stress cell")
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower left")
    return save_figure(fig, output_dir, "figS5_finite_size_diagnostic")


def plot_time_interactions(results_dir: Path, output_dir: Path) -> list[Path]:
    data = pd.read_csv(results_dir / "cloglog_coefficients.csv")
    terms = [
        ("liquidity_intensity:log_day", "Liquidity x log(day)", TEAL, "o", -0.025),
        ("market_intensity:log_day", "Market x log(day)", RED, "s", 0.025),
    ]
    fig, axis = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
    for term, label, color, marker, offset in terms:
        series = data[data["term"] == term].sort_values("calibration_ratio")
        x = series["calibration_ratio"].to_numpy() + offset
        estimate = series["coefficient"].to_numpy()
        yerr = np.vstack(
            [estimate - series["ci_low"].to_numpy(), series["ci_high"].to_numpy() - estimate]
        )
        axis.errorbar(
            x,
            estimate,
            yerr=yerr,
            color=color,
            marker=marker,
            linewidth=1.6,
            capsize=3,
            label=label,
        )
    axis.axhline(0, color=GRAY, linestyle="--", linewidth=0.9)
    axis.set_xticks([0.5, 1.0, 2.0])
    axis.set_xlabel(r"Calibration ratio $\rho$")
    axis.set_ylabel("Cloglog time-interaction coefficient")
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    return save_figure(fig, output_dir, "figS6_time_interactions")


def plot_exit_reasons(results_dir: Path, output_dir: Path) -> list[Path]:
    data = pd.read_csv(results_dir / "exit_reason_summary.csv")
    data = data[data["calibration_ratio"] == 1.0]
    index = pd.MultiIndex.from_product(
        [[0, 1, 2], [0, 1, 2]],
        names=["liquidity_intensity", "market_intensity"],
    )
    pivot = (
        data.pivot_table(
            index=["liquidity_intensity", "market_intensity"],
            columns="exit_reason",
            values="share_of_cell",
            fill_value=0,
        )
        .reindex(index, fill_value=0)
    )
    categories = [
        ("censored", "Active at day 30", "#CFD6DF"),
        ("unpaid_liquidity", "Unpaid liquidity", TEAL),
        ("insolvency", "Insolvency", RED),
        ("both", "Both", PURPLE),
    ]
    fig, axis = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for column, label, color in categories:
        values = pivot[column].to_numpy() if column in pivot else np.zeros(len(pivot))
        axis.bar(x, values, bottom=bottom, color=color, width=0.75, label=label)
        bottom += values
    labels = [f"L{liquidity}M{market}" for liquidity, market in pivot.index]
    axis.set_xticks(x, labels)
    axis.set_xlabel("Stress cell at rho=1")
    axis.set_ylabel("Share of agents")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    return save_figure(fig, output_dir, "figS7_exit_reasons")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--design-dir", type=Path, default=ROOT / "design")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    require_new_directory(args.output_dir)
    generated: list[Path] = []
    cell_summary = pd.read_csv(args.results_dir / "cell_summary.csv")
    generated.extend(plot_channel_contrasts(args.results_dir, args.output_dir))
    generated.extend(
        _heatmap_panels(
            cell_summary,
            args.output_dir,
            "fig3_exit_risk_grid",
            "exit_rate",
            "Day-30 exit risk",
            "{:.0%}",
            "cividis",
            0.0,
            1.0,
        )
    )
    generated.extend(plot_pilot(args.design_dir, args.output_dir))
    generated.extend(
        _heatmap_panels(
            cell_summary,
            args.output_dir,
            "figS2_rmst_grid",
            "rmst_days",
            "30-day RMST (days)",
            "{:.1f}",
            "viridis",
            14.0,
            30.0,
        )
    )
    generated.extend(plot_survival_curves(args.results_dir, args.output_dir))
    generated.extend(plot_homogeneous(args.results_dir, args.output_dir))
    generated.extend(plot_finite_size(args.results_dir, args.output_dir))
    generated.extend(plot_time_interactions(args.results_dir, args.output_dir))
    generated.extend(plot_exit_reasons(args.results_dir, args.output_dir))

    manifest = {
        "manifest_version": "1.0",
        "source_commit": args.source_commit,
        "results_manifest_sha256": sha256_file(args.results_dir / "results_manifest.json"),
        "files": {path.name: sha256_file(path) for path in generated},
    }
    write_json(args.output_dir / "figure_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

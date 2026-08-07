"""Plot the paper-facing knowledge-assimilation figures from cached CSV files."""
from __future__ import annotations

import os
BASE = os.path.dirname(os.path.abspath(__file__))


import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from knowledge_assimilation import DEFAULT_OUTPUT_DIR


TARGET_ORDER = [
    "LOI",
    "UL94_Rating",
    "pHRR",
    "THR",
    "TSP",
    "Flexural_Strength",
]
DISPLAY = {
    "LOI": "LOI",
    "UL94_Rating": "UL-94",
    "pHRR": "pHRR",
    "THR": "THR",
    "TSP": "TSP",
    "Flexural_Strength": "Flexural strength",
}
COLORS = {
    target: color
    for target, color in zip(
        TARGET_ORDER,
        ["#0072B2", "#E69F00", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"],
    )
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )


def _read(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _ordered_targets(values: Sequence[str]) -> list[str]:
    observed = set(values)
    return [target for target in TARGET_ORDER if target in observed] + sorted(
        observed - set(TARGET_ORDER)
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in [".png", ".pdf", ".svg"]:
        path = directory / f"{stem}{suffix}"
        fig.savefig(path, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def build_figure3(output_dir: str | Path) -> list[Path]:
    """Assimilation curve + sample-level experimental data value."""
    output = Path(output_dir)
    dose = _read(output / "Dose" / "assimilation_curve_summary.csv")
    values = _read(output / "DataValue" / "experimental_data_value.csv")
    _style()

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # A: target-wise assimilation curves. Aggregate only across completed seeds.
    dose_records = []
    for (target, dose_percent), group in dose.groupby(
        ["target", "dose_percent"]
    ):
        n_seeds = int(group["seed"].nunique())
        if n_seeds == 1:
            lower = float(group["utility_ci_low"].iloc[0])
            upper = float(group["utility_ci_high"].iloc[0])
        else:
            lower = float(group["utility_mean"].quantile(0.025))
            upper = float(group["utility_mean"].quantile(0.975))
        dose_records.append(
            {
                "target": target,
                "dose_percent": dose_percent,
                "utility": float(group["utility_mean"].mean()),
                "lower": lower,
                "upper": upper,
                "n_seeds": n_seeds,
            }
        )
    dose_grouped = pd.DataFrame(dose_records)
    for target in _ordered_targets(dose_grouped["target"]):
        group = dose_grouped[dose_grouped["target"] == target].sort_values(
            "dose_percent"
        )
        color = COLORS.get(target, "#555555")
        ax_a.plot(
            group["dose_percent"],
            group["utility"],
            marker="o",
            linewidth=1.7,
            markersize=4,
            color=color,
            label=DISPLAY.get(target, target),
        )
        ax_a.fill_between(
            group["dose_percent"],
            group["lower"],
            group["upper"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    ax_a.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax_a.set(
        xlabel="Experimental dose (%)",
        ylabel="Update utility (Δ primary metric)",
        title="Assimilation dose–response",
    )
    ax_a.legend(frameon=False, ncol=2)
    _panel_label(ax_a, "A")

    # B: exact LOEO values.
    numeric_value_columns = [
        "marginal_value",
        "descriptor_novelty_percentile",
        "label_surprise",
        "local_compatibility",
    ]
    present_numeric = [
        column for column in numeric_value_columns if column in values.columns
    ]
    values_plot = (
        values.groupby(
            ["target", "experiment_sample_id"], as_index=False
        )[present_numeric]
        .mean()
    )
    targets = _ordered_targets(values_plot["target"])
    positions = {target: index for index, target in enumerate(targets)}
    rng = np.random.default_rng(42)
    for target in targets:
        group = values_plot[values_plot["target"] == target]
        jitter = rng.uniform(-0.17, 0.17, len(group))
        ax_b.scatter(
            np.full(len(group), positions[target]) + jitter,
            group["marginal_value"],
            s=20,
            alpha=0.78,
            color=COLORS.get(target, "#555555"),
            edgecolor="white",
            linewidth=0.35,
        )
        ax_b.hlines(
            group["marginal_value"].median(),
            positions[target] - 0.25,
            positions[target] + 0.25,
            color="black",
            linewidth=1.6,
        )
    ax_b.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax_b.set_xticks(range(len(targets)))
    ax_b.set_xticklabels(
        [DISPLAY.get(target, target) for target in targets],
        rotation=30,
        ha="right",
    )
    ax_b.set(
        ylabel="LOEO marginal value",
        title="Unequal value of individual experiments",
    )
    _panel_label(ax_b, "B")

    # C/D: data-value hypotheses.
    for ax, predictor, xlabel, panel in [
        (
            ax_c,
            "descriptor_novelty_percentile",
            "Descriptor novelty percentile",
            "C",
        ),
        (ax_d, "label_surprise", "Label surprise", "D"),
    ]:
        if predictor not in values_plot.columns:
            ax.text(
                0.5,
                0.5,
                f"{predictor}\nnot available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            for target in targets:
                group = values_plot[values_plot["target"] == target]
                ax.scatter(
                    group[predictor],
                    group["marginal_value"],
                    s=22,
                    alpha=0.78,
                    color=COLORS.get(target, "#555555"),
                    label=DISPLAY.get(target, target),
                    edgecolor="white",
                    linewidth=0.35,
                )
        ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set(
            xlabel=xlabel,
            ylabel="LOEO marginal value",
            title=(
                "Does geometric novelty predict value?"
                if panel == "C"
                else "Does label surprise predict value?"
            ),
        )
        _panel_label(ax, panel)

    fig.suptitle(
        "Experimental knowledge assimilation and sample-level data value",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    return _save_figure(
        fig,
        output / "Figures",
        "Figure3_Assimilation_and_DataValue",
    )


def _representative_shap_panel(
    ax: plt.Axes,
    phase: pd.DataFrame,
    shap_values: pd.DataFrame,
) -> None:
    if phase.empty or shap_values.empty:
        ax.text(
            0.5,
            0.5,
            "SHAP feature values not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    representative = phase.sort_values("plasticity", ascending=False).iloc[0]
    target = representative["target"]
    seed = int(representative["seed"])
    subset = shap_values[
        (shap_values["target"] == target) & (shap_values["seed"] == seed)
    ].copy()
    before = subset[subset["track"] == "before"].nsmallest(6, "rank")
    after = subset[subset["track"] == "after"].nsmallest(6, "rank")
    feature_order = list(
        dict.fromkeys(
            list(before.sort_values("rank")["feature"])
            + list(after.sort_values("rank")["feature"])
        )
    )[:10]
    pivot = (
        subset[subset["feature"].isin(feature_order)]
        .pivot_table(
            index="feature",
            columns="track",
            values="normalized_mean_abs_shap",
            aggfunc="first",
        )
        .reindex(feature_order[::-1])
        .fillna(0.0)
    )
    y = np.arange(len(pivot))
    ax.barh(
        y - 0.17,
        -pivot.get("before", pd.Series(0, index=pivot.index)),
        height=0.32,
        color="#5DA5DA",
        label="Before",
    )
    ax.barh(
        y + 0.17,
        pivot.get("after", pd.Series(0, index=pivot.index)),
        height=0.32,
        color="#C91511",
        label="After",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    limit = max(
        float(pivot.to_numpy().max(initial=0.0)) * 1.15,
        0.01,
    )
    ax.set_xlim(-limit, limit)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(value):.2f}"))
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set(
        xlabel="Normalized mean |SHAP|",
        title=f"Representative decision-structure change\n{DISPLAY.get(target, target)}, seed {seed}",
    )
    ax.legend(frameon=False, loc="lower right")


def build_figure4(output_dir: str | Path) -> list[Path]:
    """Knowledge propagation + utility-plasticity phase map."""
    output = Path(output_dir)
    samples = _read(output / "Propagation" / "propagation_samples.csv")
    regions = _read(output / "Propagation" / "propagation_regions.csv")
    phase = _read(output / "Plasticity" / "utility_plasticity_phase_map.csv")
    shap_values = _read(
        output / "Plasticity" / "shap_feature_values.csv", required=False
    )
    _style()

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # A: per-test-sample propagation.
    for target in _ordered_targets(samples["target"]):
        group = samples[samples["target"] == target]
        ax_a.scatter(
            group["distance_percentile"],
            group["loss_improvement"],
            s=10,
            alpha=0.28,
            color=COLORS.get(target, "#555555"),
            label=DISPLAY.get(target, target),
            rasterized=True,
        )
        if len(group) >= 9:
            group = group.assign(
                distance_bin=pd.qcut(
                    group["distance_percentile"].rank(method="first"),
                    q=min(8, len(group)),
                    duplicates="drop",
                )
            )
            smooth = (
                group.groupby("distance_bin", observed=True)
                .agg(
                    x=("distance_percentile", "mean"),
                    y=("loss_improvement", "mean"),
                )
                .sort_values("x")
            )
            ax_a.plot(
                smooth["x"],
                smooth["y"],
                color=COLORS.get(target, "#555555"),
                linewidth=1.5,
            )
    ax_a.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax_a.set(
        xlabel="Distance-to-experiment percentile",
        ylabel="Per-sample loss improvement",
        title="Spatial propagation across the fixed test domain",
    )
    ax_a.legend(frameon=False, ncol=2)
    _panel_label(ax_a, "A")

    # B: near/middle/far comparison.
    region_order = ["near", "middle", "far"]
    for target in _ordered_targets(regions["target"]):
        group = (
            regions[regions["target"] == target]
            .groupby("neighborhood", as_index=False)["mean_loss_improvement"]
            .mean()
            .set_index("neighborhood")
            .reindex(region_order)
        )
        ax_b.plot(
            region_order,
            group["mean_loss_improvement"],
            marker="o",
            linewidth=1.6,
            color=COLORS.get(target, "#555555"),
            label=DISPLAY.get(target, target),
        )
    ax_b.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax_b.set(
        xlabel="Experiment neighborhood",
        ylabel="Mean loss improvement",
        title="Local versus distributed effects",
    )
    _panel_label(ax_b, "B")

    # C: pre-declared utility-plasticity map.
    phase_display = (
        phase.groupby("target", as_index=False)
        .agg(
            utility=("utility", "mean"),
            utility_sd=("utility", "std"),
            plasticity=("plasticity", "mean"),
            plasticity_sd=("plasticity", "std"),
            n_seeds=("seed", "nunique"),
            plasticity_threshold=("plasticity_threshold", "first"),
        )
        .fillna({"utility_sd": 0.0, "plasticity_sd": 0.0})
    )
    threshold = float(phase_display["plasticity_threshold"].dropna().iloc[0])
    x_min = min(float(phase_display["utility"].min()), -0.02)
    x_max = max(float(phase_display["utility"].max()), 0.02)
    ax_c.axhspan(0, threshold, color="#999999", alpha=0.06)
    ax_c.axhspan(threshold, 1, color="#E69F00", alpha=0.06)
    ax_c.axvline(0, color="#555555", linewidth=0.9)
    ax_c.axhline(threshold, color="#555555", linewidth=0.9, linestyle=":")
    for _, row in phase_display.iterrows():
        target = row["target"]
        ax_c.errorbar(
            row["utility"],
            row["plasticity"],
            xerr=row["utility_sd"] if row["n_seeds"] > 1 else None,
            yerr=row["plasticity_sd"] if row["n_seeds"] > 1 else None,
            fmt="o",
            markersize=7,
            color=COLORS.get(target, "#555555"),
            markeredgecolor="white",
            markeredgewidth=0.7,
            capsize=2,
            elinewidth=0.8,
            zorder=3,
        )
        ax_c.annotate(
            DISPLAY.get(target, target),
            (row["utility"], row["plasticity"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax_c.text(
        x_max,
        threshold * 0.5,
        "Stable assimilation",
        ha="right",
        va="center",
        fontsize=7,
        color="#555555",
    )
    ax_c.text(
        x_max,
        threshold + (1 - threshold) * 0.5,
        "Plastic assimilation",
        ha="right",
        va="center",
        fontsize=7,
        color="#555555",
    )
    ax_c.text(
        x_min,
        threshold + (1 - threshold) * 0.5,
        "Destructive\ninterference",
        ha="left",
        va="center",
        fontsize=7,
        color="#555555",
    )
    ax_c.text(
        0,
        0.04,
        "Inert update\n(near-zero utility)",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    ax_c.set(
        xlim=(x_min - 0.03 * (x_max - x_min), x_max + 0.03 * (x_max - x_min)),
        ylim=(-0.03, 1.03),
        xlabel="Update utility (Δ primary metric)",
        ylabel="Plasticity (1 − Top-10 Jaccard)",
        title="Utility–plasticity phase map",
    )
    _panel_label(ax_c, "C")

    _representative_shap_panel(ax_d, phase, shap_values)
    _panel_label(ax_d, "D")

    fig.suptitle(
        "Knowledge propagation and decision-structure plasticity",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    return _save_figure(
        fig,
        output / "Figures",
        "Figure4_Propagation_and_UtilityPlasticity",
    )


def build_all_figures(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    output = Path(output_dir)
    paths = []
    figure3_inputs = [
        output / "Dose" / "assimilation_curve_summary.csv",
        output / "DataValue" / "experimental_data_value.csv",
    ]
    figure4_inputs = [
        output / "Propagation" / "propagation_samples.csv",
        output / "Propagation" / "propagation_regions.csv",
        output / "Plasticity" / "utility_plasticity_phase_map.csv",
    ]
    if all(path.is_file() for path in figure3_inputs):
        paths.extend(build_figure3(output))
    if all(path.is_file() for path in figure4_inputs):
        paths.extend(build_figure4(output))
    if not paths:
        raise FileNotFoundError(
            "No complete Figure 3 or Figure 4 input set is available."
        )
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build knowledge-assimilation figures from cached CSV outputs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Knowledge_Assimilation result directory.",
    )
    args = parser.parse_args(argv)
    paths = build_all_figures(args.output)
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
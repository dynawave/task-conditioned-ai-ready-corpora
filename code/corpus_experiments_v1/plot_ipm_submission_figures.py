#!/usr/bin/env python3
"""Generate the frozen-result IPM submission figures and source audit payload.

This script is intentionally read-only with respect to formal experiment outputs.
It verifies every manifested source file before plotting and writes only the
requested figure artifacts, a compact report, and a temporary JSON payload used
to create ``figure_source_audit.csv`` with the spreadsheet runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAG = ROOT / "results/rq1/rag"
RAG_V2 = RAG / "budget_final_v2"
RQ2 = ROOT / "results/rq2/final_robustness"
REVIEWER2 = ROOT / "results/rq2/reviewer2"
SOURCE_NB = ROOT / "results/rq2/source_suitability"
C2 = ROOT / "results/rq3/full_grid"
C3 = ROOT / "results/rq3/scale_extension"
PHI = ROOT / "results/rq3/phi"
QWEN7B = ROOT / "results/rq3/qwen7b"

TASKS = ("T1a_ACRONYM", "T2_NUMERIC", "T3_CLAIM")
TASK_LABELS = {
    "T1a_ACRONYM": "T1a",
    "T1b_EXPLICIT_DEFINITION": "T1b",
    "T2_NUMERIC": "T2",
    "T3_CLAIM": "T3",
}
VIEW_LABELS = {"B1": "Fixed", "B2": "Atomic", "B3": "SEP"}
COLORS = {
    "B1": "#0072B2",
    "B2": "#D55E00",
    "B3": "#009E73",
    "task_specific": "#0072B2",
    "common": "#E69F00",
    "random": "#A0A0A0",
    "length_only": "#595959",
    "Qwen": "#0072B2",
    "Qwen7B": "#009E73",
    "Phi": "#D55E00",
    "T1a_ACRONYM": "#0072B2",
    "T1b_EXPLICIT_DEFINITION": "#CC79A7",
    "T2_NUMERIC": "#E69F00",
    "T3_CLAIM": "#56B4E9",
}
MARKERS = {"B1": "o", "B2": "s", "B3": "^"}


def set_ipm_plot_style() -> None:
    """Apply the shared print-oriented style to every figure."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.8,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(directory: Path, filename: str) -> str:
    manifest = directory / "checksums.sha256"
    entries: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.strip().split(maxsplit=1)
        entries[relative.replace("\\", "/")] = expected.lower()
    if filename not in entries:
        raise RuntimeError(f"Missing checksum entry: {manifest} :: {filename}")
    actual = sha256(directory / filename)
    if actual != entries[filename]:
        raise RuntimeError(f"Checksum mismatch: {directory / filename}")
    return actual


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def assert_ratio(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.between(0.0, 1.0).all():
            raise RuntimeError(f"Out-of-range ratio in {column}")


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=9.5)


def format_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    if grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(direction="out")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    metadata = {"Creator": "Frozen-result IPM figure generator", "Title": stem}
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.035, metadata=metadata)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035, metadata=metadata)
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.035, metadata={"Software": "Matplotlib"})
    plt.close(fig)


def make_fig2(topk: pd.DataFrame, topk_ci: pd.DataFrame, budget: pd.DataFrame, output_dir: Path) -> None:
    views = ["B1", "B2", "B3"]
    metrics = ["ces@5", "hit@5", "recall@5", "mrr"]
    metric_labels = ["CES@5", "Hit@5", "Recall@5", "MRR"]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.05), gridspec_kw={"width_ratios": [1.12, 1.0]})
    ax = axes[0]
    offsets = {"B1": -0.17, "B2": 0.0, "B3": 0.17}
    for view in views:
        points = []
        for metric in metrics:
            row = topk[(topk.view == view) & (topk.metric == metric)].iloc[0]
            points.append(float(row.new_value))
        xs = np.arange(len(metrics), dtype=float) + offsets[view]
        ax.scatter(xs, points, s=28, marker=MARKERS[view], facecolor="white", edgecolor=COLORS[view], linewidth=1.25, zorder=3, label=VIEW_LABELS[view])
        ci = topk_ci[(topk_ci.row_type == "view_estimate") & (topk_ci.metric == "ces@5") & (topk_ci.view_a == view)].iloc[0]
        point = points[0]
        ax.errorbar(xs[0], point, yerr=[[point - float(ci.ci_low)], [float(ci.ci_high) - point]], fmt="none", ecolor=COLORS[view], elinewidth=1.0, capsize=2.2, zorder=2)
    ax.set_xticks(range(len(metrics)), metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Retrieval metric")
    ax.set_title("Fixed Top-5", pad=7)
    panel_label(ax, "(a)")
    format_axes(ax)

    ax = axes[1]
    budgets = [512, 1024, 2048]
    for view in views:
        row = budget[budget.view == view].iloc[0]
        values = np.array([float(row[f"BudgetCES@{item}"]) for item in budgets])
        lows = np.array([float(row[f"BudgetCES@{item}_ci_low"]) for item in budgets])
        highs = np.array([float(row[f"BudgetCES@{item}_ci_high"]) for item in budgets])
        ax.errorbar(
            budgets,
            values,
            yerr=np.vstack([values - lows, highs - values]),
            color=COLORS[view],
            marker=MARKERS[view],
            markerfacecolor="white",
            markeredgewidth=1.15,
            capsize=2.2,
            elinewidth=0.9,
            label=VIEW_LABELS[view],
        )
    ax.set_xticks(budgets, ["512", "1,024", "2,048"])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Token budget")
    ax.set_ylabel("BudgetCES")
    ax.set_title("Fixed token budget", pad=7)
    panel_label(ax, "(b)")
    format_axes(ax)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=3, frameon=False, handletextpad=0.5, columnspacing=1.5)
    fig.subplots_adjust(top=0.80, bottom=0.19, left=0.09, right=0.985, wspace=0.31)
    save_figure(fig, output_dir, "fig2_rag_representation_tradeoff")


def make_fig3(curves: pd.DataFrame, length_data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 5.05), sharex=True, sharey=True)
    all_max = float(curves.capture_fraction.max())
    ymax = min(1.0, math.ceil((all_max + 0.035) * 10) / 10)
    styles = {
        "task_specific": dict(color=COLORS["task_specific"], marker="o", linestyle="-", label="Task-specific"),
        "common": dict(color=COLORS["common"], marker="s", linestyle="--", label="Unified"),
        "random": dict(color=COLORS["random"], marker="^", linestyle=(0, (2, 2)), label="Random"),
    }
    panel_names = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    budget_types = ["paper_count", "source_token"]
    for row_index, budget_type in enumerate(budget_types):
        for col_index, task in enumerate(TASKS):
            ax = axes[row_index, col_index]
            data = curves[(curves.task == task) & (curves.budget_type == budget_type)]
            for strategy in ("task_specific", "common", "random"):
                subset = data[data.strategy == strategy].sort_values("budget_fraction")
                style = styles[strategy]
                ax.plot(
                    subset.budget_fraction * 100.0,
                    subset.capture_fraction,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    label=style["label"],
                )
            if budget_type == "paper_count":
                marker = length_data[
                    (length_data.task == task)
                    & (length_data.budget_type == "paper_count")
                    & np.isclose(length_data.budget_fraction, 0.20)
                    & (length_data.strategy == "length_only")
                ]
                if len(marker) != 1:
                    raise RuntimeError(f"Expected one 20% length baseline for {task}")
                ax.scatter(
                    [20], [float(marker.iloc[0].capture_fraction)],
                    s=34, marker="D", facecolor="white", edgecolor=COLORS["length_only"], linewidth=1.2, zorder=5,
                )
            ax.axvline(20, color="#D0D0D0", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
            ax.set_xlim(4, 51)
            ax.set_ylim(0, ymax)
            ax.set_xticks([5, 10, 20, 30, 40, 50])
            panel_label(ax, panel_names[row_index * 3 + col_index], x=-0.14, y=1.04)
            format_axes(ax)
            if row_index == 0:
                ax.set_title(TASK_LABELS[task], pad=6)
            if row_index == 1:
                ax.set_xlabel("Budget (%)")
    axes[0, 0].set_ylabel("Yield capture fraction\n(Paper-count budget)")
    axes[1, 0].set_ylabel("Yield capture fraction\n(Source-token budget)")
    handles = [
        Line2D([0], [0], **{**styles["task_specific"], "markerfacecolor": "white", "markeredgewidth": 1.0}),
        Line2D([0], [0], **{**styles["common"], "markerfacecolor": "white", "markeredgewidth": 1.0}),
        Line2D([0], [0], **{**styles["random"], "markerfacecolor": "white", "markeredgewidth": 1.0}),
        Line2D([0], [0], color=COLORS["length_only"], marker="D", linestyle="none", markerfacecolor="white", markeredgewidth=1.1, label="Length baseline (20% only)"),
        Line2D([0], [0], color="#D0D0D0", linestyle=(0, (3, 3)), linewidth=0.9, label="Primary point (20%)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False, handlelength=2.3, columnspacing=1.25)
    fig.subplots_adjust(top=0.85, bottom=0.11, left=0.105, right=0.99, hspace=0.27, wspace=0.16)
    save_figure(fig, output_dir, "fig3_source_prioritization_budget_curves")


def make_forest(nb: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    significant = nb[(nb.variant == "main") & (nb.holm_adjusted_p < 0.05)].copy()
    task_order = ["T1a_ACRONYM", "T1b_EXPLICIT_DEFINITION", "T2_NUMERIC", "T3_CLAIM"]
    predictor_order = {
        "discussion_presence": 0,
        "results_presence": 1,
        "conclusion_presence": 2,
        "methods_presence": 3,
        "section_count": 4,
        "average_paragraph_length": 5,
        "figure_density_per_10k": 6,
        "suspected_word_split_rate": 7,
        "reference_like_rate": 8,
    }
    significant["task_order"] = significant.task.map({task: index for index, task in enumerate(task_order)})
    significant["predictor_order"] = significant.predictor.map(predictor_order)
    significant = significant.sort_values(["task_order", "predictor_order"]).reset_index(drop=True)
    if len(significant) != 14:
        raise RuntimeError(f"Expected 14 Holm-significant NB2 associations, found {len(significant)}")
    predictor_labels = {
        "results_presence": "Results present",
        "discussion_presence": "Discussion present",
        "conclusion_presence": "Conclusion present",
        "methods_presence": "Methods present",
        "section_count": "Section count (z)",
        "average_paragraph_length": "Average paragraph length (z)",
        "figure_density_per_10k": "Figure density per 10k (z)",
        "suspected_word_split_rate": "Suspected word-split rate (z)",
        "reference_like_rate": "Reference-like rate (z)",
    }
    # Long predictor labels otherwise cause tight-bbox export to shrink the
    # final vector width below a full IPM page; this source width yields a
    # cropped output close to 175 mm while keeping 8-10 pt typography.
    fig, ax = plt.subplots(figsize=(9.25, 5.25))
    y = np.arange(len(significant))[::-1]
    labels = [f"{TASK_LABELS[row.task]}   {predictor_labels[row.predictor]}" for row in significant.itertuples()]
    for yi, row in zip(y, significant.itertuples()):
        ax.errorbar(
            float(row.irr), yi,
            xerr=[[float(row.irr) - float(row.ci_lower)], [float(row.ci_upper) - float(row.irr)]],
            fmt="o", color=COLORS[row.task], markerfacecolor="white", markeredgewidth=1.1,
            elinewidth=1.0, capsize=2.1, markersize=5.0,
        )
    ax.axvline(1.0, color="#6F6F6F", linewidth=0.9, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlim(0.5, 2.65)
    ticks = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
    ax.set_xticks(ticks, ["0.50", "0.75", "1.00", "1.50", "2.00", "2.50"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_yticks(y, labels)
    ax.set_xlabel("Incidence rate ratio (log scale)")
    ax.set_ylim(-0.8, len(significant) - 0.2)
    ax.grid(axis="x", color="#DEDEDE", linewidth=0.55)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    # Separate task blocks without decorative boxes.
    cumulative = 0
    counts = significant.groupby("task", sort=False).size().tolist()
    for count in counts[:-1]:
        cumulative += count
        ax.axhline(len(significant) - cumulative - 0.5, color="#BEBEBE", linewidth=0.65)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=COLORS[task], label=TASK_LABELS[task])
        for task in task_order
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=4, columnspacing=1.1, handletextpad=0.35)
    fig.subplots_adjust(left=0.43, right=0.985, top=0.985, bottom=0.11)
    save_figure(fig, output_dir, "figS_source_feature_IRR_forest")
    return significant


def make_fig4(c2_metrics: pd.DataFrame, c3_metrics: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    curves = {
        "T1a_ACRONYM": c2_metrics[c2_metrics.task == "T1a_ACRONYM"].sort_values("scale"),
        "T2_NUMERIC": c3_metrics[c3_metrics.task == "T2_NUMERIC"].sort_values("scale"),
        "T3_CLAIM": c3_metrics[c3_metrics.task == "T3_CLAIM"].sort_values("scale"),
    }
    nstars = {"T1a_ACRONYM": 1000, "T2_NUMERIC": 3000}
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 3.05), sharey=True)
    panel_names = ["(a)", "(b)", "(c)"]
    for index, task in enumerate(TASKS):
        ax = axes[index]
        data = curves[task]
        ax.plot(data.scale, data.NEM_mean, color=COLORS["Qwen"], marker="o", markerfacecolor="white", markeredgewidth=1.15)
        ax.set_xscale("log")
        ax.set_ylim(0.25, 1.0)
        ax.set_title(TASK_LABELS[task], pad=7)
        panel_label(ax, panel_names[index])
        format_axes(ax)
        max_scale = int(data.scale.max())
        if task == "T1a_ACRONYM":
            ticks = [100, 250, 500, 1000, 2000]
        elif task == "T2_NUMERIC":
            ticks = [100, 500, 1000, 2000, max_scale]
        else:
            # N=2,000 and N=2,286 are too close on a log axis to label both at
            # final print width; both points remain plotted, while the observed
            # maximum is labeled explicitly.
            ticks = [100, 500, 1000, max_scale]
        ax.set_xticks(ticks, [f"{value:,}" for value in ticks])
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlabel("Training instances, N\n(log scale)")
        if task in nstars:
            scale = nstars[task]
            point = float(data[data.scale == scale].iloc[0].NEM_mean)
            ax.scatter([scale], [point], s=60, facecolor="white", edgecolor="#222222", linewidth=1.0, zorder=5)
            ax.annotate(f"n* = {scale:,}", xy=(scale, point), xytext=(0, -18), textcoords="offset points", ha="center", va="top", fontsize=8.0)
        else:
            point = float(data[data.scale == max_scale].iloc[0].NEM_mean)
            ax.annotate(
                "n* not reached within\nobserved range",
                xy=(max_scale, point), xytext=(-4, -25), textcoords="offset points",
                ha="right", va="top", fontsize=7.8,
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.75, shrinkA=2, shrinkB=3),
            )
    axes[0].set_ylabel("NEM")
    fig.subplots_adjust(top=0.90, bottom=0.23, left=0.08, right=0.99, wspace=0.18)
    save_figure(fig, output_dir, "fig4_qwen_training_scale_curves")
    return curves


def make_fig5(comparison: pd.DataFrame, qwen7b: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.75), sharey=True)
    panel_names = ["(a)", "(b)", "(c)"]
    model_specs = [
        ("Qwen2.5-3B", "Qwen", "o"),
        ("Qwen2.5-7B", "Qwen7B", "D"),
        ("Phi-4-mini", "Phi", "s"),
    ]
    interval_differences: dict[str, dict[str, float]] = {}
    all_differences: list[float] = []

    for task in TASKS:
        row = comparison[comparison.task == task].iloc[0]
        small_scale = int(row.small_scale)
        large_scale = int(row.large_scale)
        qwen7b_small = float(
            qwen7b[(qwen7b.task == task) & (qwen7b.scale == small_scale)].iloc[0].NEM_mean
        )
        qwen7b_large = float(
            qwen7b[(qwen7b.task == task) & (qwen7b.scale == large_scale)].iloc[0].NEM_mean
        )
        differences = {
            "Qwen": float(row.Qwen_large_mean) - float(row.Qwen_small_mean),
            "Qwen7B": qwen7b_large - qwen7b_small,
            "Phi": float(row.Phi_large_mean) - float(row.Phi_small_mean),
        }
        if not math.isclose(differences["Qwen"], float(row.Qwen_difference), abs_tol=5e-13):
            raise RuntimeError(f"Qwen2.5-3B endpoint/difference mismatch for {task}")
        if not math.isclose(differences["Phi"], float(row.Phi_difference), abs_tol=5e-13):
            raise RuntimeError(f"Phi endpoint/difference mismatch for {task}")
        interval_differences[task] = differences
        all_differences.extend(differences.values())

    tick_step = 0.02
    y_min = math.floor((min(all_differences) - 0.005) / tick_step) * tick_step
    y_max = math.ceil((max(all_differences) + 0.005) / tick_step) * tick_step
    y_ticks = np.arange(y_min, y_max + tick_step / 2, tick_step)

    for index, task in enumerate(TASKS):
        ax = axes[index]
        row = comparison[comparison.task == task].iloc[0]
        ax.axhline(0, color="#777777", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        for model_index, (label, color_key, marker) in enumerate(model_specs):
            ax.scatter(
                model_index,
                interval_differences[task][color_key],
                s=34,
                color=COLORS[color_key],
                marker=marker,
                facecolor="white",
                linewidth=1.25,
                label=label,
                zorder=3,
            )
        ax.set_xticks([0, 1, 2], ["3B", "7B", "Phi"])
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
        ax.set_title(
            f"{panel_names[index]} {TASK_LABELS[task]}",
            loc="left",
            pad=16,
            fontweight="bold",
        )
        ax.text(
            1.0,
            1.015,
            rf"$N = {int(row.small_scale):,} \rightarrow {int(row.large_scale):,}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.7,
        )
        format_axes(ax, grid=False)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.55, zorder=0)
    axes[0].set_ylabel(r"Change in NEM ($\Delta$NEM)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        handletextpad=0.45,
        columnspacing=1.3,
    )
    fig.subplots_adjust(top=0.76, bottom=0.17, left=0.09, right=0.99, wspace=0.17)
    save_figure(fig, output_dir, "fig5_cross_model_key_intervals")


def audit_row(
    figure_id: str,
    panel: str,
    metric: str,
    source_file: str,
    source_columns: str,
    aggregation: str,
    n: int | str,
    notes: str,
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "panel": panel,
        "metric": metric,
        "source_file": source_file,
        "source_columns": source_columns,
        "aggregation": aggregation,
        "n": n,
        "notes": notes,
    }


def build_audit_rows(significant: pd.DataFrame) -> list[dict[str, Any]]:
    rows = [
        audit_row("fig2", "(a)", "CES@5", f"{rel(RAG_V2 / 'topk_regression.csv')}; {rel(RAG / 'bootstrap.csv')}", "metric; new_value; row_type; estimate; ci_low; ci_high", "Frozen point estimate; question-level bootstrap percentile 95% CI", 1292, "10,000 bootstrap replicates; B1/B2/B3 labels mapped to Fixed/Atomic/SEP."),
        audit_row("fig2", "(a)", "Hit@5; Recall@5; MRR", f"{rel(RAG_V2 / 'topk_regression.csv')}; {rel(RAG_V2 / 'summary.json')}", "metric; new_value; mrr_definition", "Frozen point estimates; no CI plotted", 1292, "MRR uses the first gold hit in the complete persisted paper-scoped ranking; no Holm adjustment applies to plotted estimates."),
        audit_row("fig2", "(b)", "BudgetCES@512/1024/2048", rel(RAG_V2 / "budget_metrics.csv"), "view; formal_question_count; BudgetCES@B; BudgetCES@B_ci_low; BudgetCES@B_ci_high", "Frozen point estimate; question-level bootstrap percentile 95% CI", 1292, "Budget prefixes are complete; 10,000 paired bootstrap is used for formal pairwise inference, not displayed as significance marks."),
    ]
    panel_lookup = {
        ("paper_count", "T1a_ACRONYM"): "(a)", ("paper_count", "T2_NUMERIC"): "(b)", ("paper_count", "T3_CLAIM"): "(c)",
        ("source_token", "T1a_ACRONYM"): "(d)", ("source_token", "T2_NUMERIC"): "(e)", ("source_token", "T3_CLAIM"): "(f)",
    }
    total_instances = {"T1a_ACRONYM": 13547, "T2_NUMERIC": 16605, "T3_CLAIM": 7791}
    for budget_type in ("paper_count", "source_token"):
        for task in TASKS:
            rows.append(
                audit_row(
                    "fig3", panel_lookup[(budget_type, task)], "machine-verified yield capture fraction",
                    rel(RQ2 / "rq2_budget_curves.csv"),
                    "task; budget_type; budget_fraction; strategy; capture_fraction; ci_low; ci_high; specific_gain_over_common; gain_ci_low; gain_ci_high",
                    "Frozen OOF ranking capture across 5/10/20/30/40/50% budgets",
                    1898,
                    f"Source papers=1,898; {TASK_LABELS[task]} machine-verified instances={total_instances[task]:,}; random uses 10,000 permutations; 20% difference CI is paper bootstrap; machine-verified is not human expert annotation.",
                )
            )
    for task, panel in zip(TASKS, ["(a)", "(b)", "(c)"]):
        rows.append(
            audit_row("fig3", panel, "length-only capture at 20% paper budget", rel(REVIEWER2 / "rq2_length_only_baseline.csv"), "task; budget_type; budget_fraction; strategy; capture_fraction", "Single independent 20% marker from descending source length", 1898, f"Auxiliary reviewer-check result; direct file SHA-256={sha256(REVIEWER2 / 'rq2_length_only_baseline.csv')}; no token-budget marker plotted.")
        )
    for task in ["T1a_ACRONYM", "T1b_EXPLICIT_DEFINITION", "T2_NUMERIC", "T3_CLAIM"]:
        count = int((significant.task == task).sum())
        rows.append(
            audit_row("figS", TASK_LABELS[task], "Negative Binomial IRR", rel(SOURCE_NB / "nb_results.csv"), "task; predictor; irr; ci_lower; ci_upper; variant; holm_adjusted_p", "Main NB2 associations filtered to task-wise Holm-adjusted p<0.05", 1898, f"{count} association(s) shown; source_tokens enters as log offset; predictive association, not causal effect.")
        )
    rows.extend(
        [
            audit_row("fig4", "(a)", "Qwen TEMPLATE_0 NEM", f"{rel(C2 / 'full_grid_scale_metrics.csv')}; {rel(C2 / 'n_star_summary.csv')}", "task; scale; NEM_mean; n_star; n_star_status; delta_nem", "Mean across three frozen training seeds at each observed scale", 3, "No endpoint CI exists in the formal scale-metrics file; NEM_sd is not plotted as CI; delta_NEM=0.02 and n*=1,000."),
            audit_row("fig4", "(b)", "Qwen TEMPLATE_0 NEM", f"{rel(C3 / 'combined_scale_metrics.csv')}; {rel(C3 / 'n_star_summary.csv')}", "task; scale; NEM_mean; n_star; reference_scale; delta_nem", "Mean across three frozen training seeds at each observed scale", 3, "No endpoint CI plotted; delta_NEM=0.02 and n*=3,000 relative to N=3,678."),
            audit_row("fig4", "(c)", "Qwen TEMPLATE_0 NEM", f"{rel(C3 / 'combined_scale_metrics.csv')}; {rel(C3 / 'n_star_summary.csv')}", "task; scale; NEM_mean; n_star_status; reference_scale; delta_nem", "Mean across three frozen training seeds at each observed scale", 3, "No endpoint CI plotted; maximum N=2,286; REFERENCE_ONLY_NOT_IDENTIFIED / plateau not identified within available pool."),
        ]
    )
    for panel, task in zip(["(a)", "(b)", "(c)"], TASKS):
        rows.append(
            audit_row("fig5", panel, "Qwen3B/Qwen7B/Phi TEMPLATE_0 change in NEM", f"{rel(PHI / 'cross_model_comparison.csv')}; {rel(PHI / 'bootstrap.csv')}; {rel(QWEN7B / 'qwen7b_key_interval_summary.csv')}", "task; scale; NEM_mean; small_scale; large_scale; Qwen_difference; Phi_difference", "Large-scale minus small-scale three-seed mean NEM", 3, "Point estimates only: Qwen one-sided UCB0.95 and Phi two-sided 95% CI have different inferential meanings and are not mixed as error bars.")
        )
    return rows


def write_report(
    output_dir: Path,
    source_hashes: dict[str, str],
    topk: pd.DataFrame,
    budget: pd.DataFrame,
    curves: pd.DataFrame,
    length_data: pd.DataFrame,
    qwen_curves: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    qwen7b: pd.DataFrame,
) -> None:
    lines = [
        "# Figure generation report",
        "",
        "## Sources and frozen consistency",
        "",
        "- Figure 2: `results/rq1/rag/budget_final_v2/topk_regression.csv`, `results/rq1/rag/bootstrap.csv`, and `results/rq1/rag/budget_final_v2/budget_metrics.csv`.",
        "- Figure 3: `results/rq2/final_robustness/rq2_budget_curves.csv`; paper-budget 20% length markers use `results/rq2/reviewer2/rq2_length_only_baseline.csv`.",
        "- Figure S1: `results/rq2/source_suitability/nb_results.csv`, restricted to main NB2 rows with task-wise Holm-adjusted p < 0.05.",
        "- Figure 4: `results/rq3/full_grid/` inputs for T1a and `results/rq3/scale_extension/` inputs for T2/T3.",
        "- Figure 5: `results/rq3/phi/` and `results/rq3/qwen7b/` key-interval inputs; the Phi comparison file references the frozen Qwen2.5-3B C2/C3 endpoints.",
        f"- Manifest verification: PASS for {len(source_hashes)} formal input files (0 mismatches). The auxiliary length-baseline SHA-256 is `{sha256(REVIEWER2 / 'rq2_length_only_baseline.csv')}`.",
        "- Metric/statistical checks: PASS. Ratios remain in [0,1]; SD was not presented as CI; UCB95 was not drawn as a two-sided CI; machine-verified yield was not relabeled as human annotation; IRRs are described as predictive associations, not causal effects.",
        "",
        "## Automated numeric audit",
        "",
        "### Figure 2",
        "",
    ]
    for view in ["B1", "B2", "B3"]:
        ces = float(topk[(topk.view == view) & (topk.metric == "ces@5")].iloc[0].new_value)
        row = budget[budget.view == view].iloc[0]
        values = [float(row[f"BudgetCES@{item}"]) for item in [512, 1024, 2048]]
        lines.append(f"- {VIEW_LABELS[view]}: CES@5={ces:.6f}; BudgetCES 512/1024/2048={values[0]:.6f}/{values[1]:.6f}/{values[2]:.6f}.")
    lines.extend(["", "### Figure 3 (20% budget)", ""])
    for task in TASKS:
        for budget_type in ["paper_count", "source_token"]:
            subset = curves[(curves.task == task) & (curves.budget_type == budget_type) & np.isclose(curves.budget_fraction, 0.20)]
            values = {row.strategy: float(row.capture_fraction) for row in subset.itertuples()}
            specific = subset[subset.strategy == "task_specific"].iloc[0]
            length_text = ""
            if budget_type == "paper_count":
                length_row = length_data[(length_data.task == task) & (length_data.budget_type == "paper_count") & np.isclose(length_data.budget_fraction, 0.20) & (length_data.strategy == "length_only")].iloc[0]
                length_text = f"; length={float(length_row.capture_fraction):.6f}"
            lines.append(
                f"- {TASK_LABELS[task]} {budget_type}: random={values['random']:.6f}; common={values['common']:.6f}; specific={values['task_specific']:.6f}{length_text}; specific-common={float(specific.specific_gain_over_common):+.6f} [{float(specific.gain_ci_low):+.6f}, {float(specific.gain_ci_high):+.6f}]."
            )
    lines.extend(["", "### Figure 4", ""])
    for task in TASKS:
        data = qwen_curves[task]
        max_scale = int(data.scale.max())
        if task == "T1a_ACRONYM":
            status = "n*=1,000 at delta_NEM=0.02"
        elif task == "T2_NUMERIC":
            status = "n*=3,000 at delta_NEM=0.02"
        else:
            status = "n* not reached; PLATEAU_NOT_IDENTIFIED_WITHIN_AVAILABLE_POOL"
        lines.append(f"- {TASK_LABELS[task]}: maximum N={max_scale:,}; {status}.")
    lines.extend(["", "### Figure 5", ""])
    for row in comparison.itertuples():
        seven_small = float(qwen7b[(qwen7b.task == row.task) & (qwen7b.scale == int(row.small_scale))].iloc[0].NEM_mean)
        seven_large = float(qwen7b[(qwen7b.task == row.task) & (qwen7b.scale == int(row.large_scale))].iloc[0].NEM_mean)
        lines.append(
            f"- {TASK_LABELS[row.task]}: Qwen2.5-3B delta={float(row.Qwen_difference):+.6f}; Qwen2.5-7B delta={seven_large - seven_small:+.6f}; Phi delta={float(row.Phi_difference):+.6f}."
        )
    lines.extend(
        [
            "",
            "## Manuscript consistency and placement",
            "",
            "- Current-paper number mismatch: NOT ASSESSED. No Word manuscript values were read or used, by design; figures follow local frozen result files only.",
            "- Main-text recommendation: Figures 2, 3, 4, and 5.",
            "- Supplement recommendation: `figS_source_feature_IRR_forest`.",
            "",
        ]
    )
    (output_dir / "figure_generation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    set_ipm_plot_style()

    manifested = [
        (RAG_V2, "topk_regression.csv"),
        (RAG, "bootstrap.csv"),
        (RAG_V2, "budget_metrics.csv"),
        (RAG_V2, "budget_bootstrap.csv"),
        (RQ2, "rq2_budget_curves.csv"),
        (SOURCE_NB, "nb_results.csv"),
        (C2, "full_grid_scale_metrics.csv"),
        (C2, "n_star_summary.csv"),
        (C3, "combined_scale_metrics.csv"),
        (C3, "n_star_summary.csv"),
        (PHI, "scale_metrics.csv"),
        (PHI, "bootstrap.csv"),
        (PHI, "cross_model_comparison.csv"),
        (QWEN7B, "qwen7b_key_interval_summary.csv"),
    ]
    source_hashes = {rel(directory / filename): verify_manifest(directory, filename) for directory, filename in manifested}

    topk = pd.read_csv(RAG_V2 / "topk_regression.csv")
    topk_ci = pd.read_csv(RAG / "bootstrap.csv")
    budget = pd.read_csv(RAG_V2 / "budget_metrics.csv")
    curves = pd.read_csv(RQ2 / "rq2_budget_curves.csv")
    length_data = pd.read_csv(REVIEWER2 / "rq2_length_only_baseline.csv")
    nb = pd.read_csv(SOURCE_NB / "nb_results.csv")
    c2_metrics = pd.read_csv(C2 / "full_grid_scale_metrics.csv")
    c3_metrics = pd.read_csv(C3 / "combined_scale_metrics.csv")
    c2_nstar = pd.read_csv(C2 / "n_star_summary.csv")
    c3_nstar = pd.read_csv(C3 / "n_star_summary.csv")
    comparison = pd.read_csv(PHI / "cross_model_comparison.csv")
    qwen7b = pd.read_csv(QWEN7B / "qwen7b_key_interval_summary.csv")
    phi_bootstrap = pd.read_csv(PHI / "bootstrap.csv")

    if set(topk.view) != {"B1", "B2", "B3"} or set(topk.metric) != {"ces@5", "hit@5", "recall@5", "mrr"}:
        raise RuntimeError("Unexpected RAG top-k schema/membership")
    if not np.allclose(topk.absolute_difference, 0.0) or set(topk.status) != {"PASS"}:
        raise RuntimeError("RAG top-k regression is not exact")
    if set(budget.formal_question_count) != {1292}:
        raise RuntimeError("Unexpected formal RAG question count")
    assert_ratio(budget, [column for column in budget.columns if column.startswith("Budget")])
    if len(curves) != 108 or set(curves.task) != set(TASKS) or set(curves.strategy) != {"task_specific", "common", "random"}:
        raise RuntimeError("Unexpected RQ2 budget-curve grid")
    assert_ratio(curves, ["budget_fraction", "actual_token_fraction", "capture_fraction", "ci_low", "ci_high"])
    if set(c2_nstar.delta_nem) != {0.02} or set(c3_nstar.delta_nem) != {0.02}:
        raise RuntimeError("Primary delta_NEM drift")
    if c2_nstar[c2_nstar.task == "T1a_ACRONYM"].iloc[0].n_star != 1000:
        raise RuntimeError("T1a n* drift")
    if c3_nstar[c3_nstar.task == "T2_NUMERIC"].iloc[0].n_star != 3000:
        raise RuntimeError("T2 n* drift")
    if c3_nstar[c3_nstar.task == "T3_CLAIM"].iloc[0].n_star_status != "REFERENCE_ONLY_NOT_IDENTIFIED":
        raise RuntimeError("T3 n* status drift")
    if len(comparison) != 3 or len(phi_bootstrap) != 3:
        raise RuntimeError("Incomplete cross-model comparison")
    expected_qwen7b = {
        ("T1a_ACRONYM", 1000), ("T1a_ACRONYM", 2000),
        ("T2_NUMERIC", 3000), ("T2_NUMERIC", 3678),
        ("T3_CLAIM", 2000), ("T3_CLAIM", 2286),
    }
    if set(zip(qwen7b.task, qwen7b.scale.astype(int))) != expected_qwen7b or set(qwen7b.model) != {"Qwen2.5-7B-Instruct"}:
        raise RuntimeError("Unexpected frozen Qwen2.5-7B key-interval summary")
    merged = comparison.merge(phi_bootstrap[["task", "point_difference", "ci95_low", "ci95_high"]], on="task", how="left")
    if not np.allclose(merged.Phi_difference, merged.point_difference) or not np.allclose(merged.Phi_95CI_low, merged.ci95_low) or not np.allclose(merged.Phi_95CI_high, merged.ci95_high):
        raise RuntimeError("Phi bootstrap mismatch")

    make_fig2(topk, topk_ci, budget, output_dir)
    make_fig3(curves, length_data, output_dir)
    significant = make_forest(nb, output_dir)
    qwen_curves = make_fig4(c2_metrics, c3_metrics, output_dir)
    make_fig5(comparison, qwen7b, output_dir)
    audit_rows = build_audit_rows(significant)
    (output_dir / "_figure_source_audit_rows.json").write_text(json.dumps(audit_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir, source_hashes, topk, budget, curves, length_data, qwen_curves, comparison, qwen7b)

    print("FIGURE_SOURCE_CHECKSUMS: PASS (14/14 formal inputs; 0 mismatches)")
    print((output_dir / "figure_generation_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

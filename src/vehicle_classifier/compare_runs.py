"""Compare finished training runs using only their summary.json files.

Contents
--------
1. Settings          : folders, file names, plot options
2. Loading           : load_summaries
3. Table             : build_comparison_table, find_best_models, round_for_display
4. Plot              : plot_comparison
5. Runner            : main

For every folder in outputs/runs/ that holds a summary.json, this script reads
the validation result of the best checkpoint and creates two files in
outputs/reports/model_comparison/:
    model_comparison.png : grouped bars (accuracy and macro F1) per model,
                           a star marks the best model of each metric
    model_comparison.csv : the table behind the figure (values in percent,
                           rounded for display)

Both files are overwritten on every run.

The best model of each metric is found on the raw, unrounded values.
Rounding is applied only to what is shown or saved.

Usage
-----
    python -m vehicle_classifier.compare_runs

Rule: everything here uses validation data. The test set is never touched.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from vehicle_classifier.data import PROJECT_ROOT
from vehicle_classifier.utils import (
    PLOT_PALETTE,
    Colors,
    colorize,
    load_json,
    print_section,
)

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------

RUNS_ROOT = PROJECT_ROOT / "outputs" / "runs"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "model_comparison"
FIGURE_NAME = "model_comparison.png"
TABLE_NAME = "model_comparison.csv"

Y_MIN = 0  # lower end of the percent axis; raise it (for example 50) to zoom in
STAR_COLOR = "#D4A017"
LEGEND_OFFSET = -0.28  # how far below the axes the legend sits (more negative = lower)

# (column in the table, legend label, bar color)
PLOTTED_METRICS = [
    ("accuracy", "Accuracy", PLOT_PALETTE[4]),
    ("macro_f1", "Macro F1", PLOT_PALETTE[0]),
]

# Decimals used when values are shown or saved.
DISPLAY_DECIMALS = {
    "accuracy": 2,
    "macro_precision": 2,
    "macro_recall": 2,
    "macro_f1": 2,
    "val_loss": 4,
}


# ---------------------------------------------------------------------------
# 2. Loading
# ---------------------------------------------------------------------------


def load_summaries(runs_root=RUNS_ROOT):
    """Read summary.json from every run folder.

    Returns a dict {run folder name: summary dict}. Folders without a
    summary.json (an unfinished or failed run) are skipped with a warning.
    """
    if not runs_root.exists():
        raise FileNotFoundError(f"No runs folder at {runs_root}")

    summaries = {}
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            print(colorize(f"Skipped {run_dir.name}: no summary.json", Colors.YELLOW))
            continue
        summaries[run_dir.name] = load_json(summary_path)

    if not summaries:
        raise FileNotFoundError(f"No finished runs found in {runs_root}")
    return summaries


# ---------------------------------------------------------------------------
# 3. Table
# ---------------------------------------------------------------------------


def build_comparison_table(summaries):
    """Build one row per model from the best-checkpoint validation results.

    Scores are in percent and are NOT rounded, so comparisons stay exact.
    Rows are sorted by macro F1, best model first.
    """
    rows = []
    for name, summary in summaries.items():
        metrics = summary["val_metrics"]
        rows.append(
            {
                "model": name,
                "best_epoch": summary["best_epoch"],
                "accuracy": metrics["accuracy"] * 100,
                "macro_precision": metrics["macro_precision"] * 100,
                "macro_recall": metrics["macro_recall"] * 100,
                "macro_f1": metrics["macro_f1"] * 100,
                "lowest_recall_class": metrics["lowest_recall_class"],
                "lowest_precision_class": metrics["lowest_precision_class"],
                "val_loss": summary["val_loss"],
                "trainable_params": summary["trainable_params"],
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def find_best_models(table):
    """Return {metric column: name of the model with the highest raw value}.

    Example: {"accuracy": "full_ablation", "macro_f1": "full_ablation"}.
    If two models tie exactly, the first one in the table wins.
    """
    return {
        column: table.loc[table[column].idxmax(), "model"]
        for column, _label, _color in PLOTTED_METRICS
    }


def round_for_display(table):
    """Return a copy of the table rounded for printing and saving."""
    return table.round(DISPLAY_DECIMALS)


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------


def plot_comparison(table, best_models, save_path):
    """Grouped bar chart: accuracy and macro F1 (in percent) for every model.

    Every bar has its rounded value written above it. The bar of the best
    model of each metric (from best_models) gets a star above its value.
    """
    n_models = len(table)
    x = np.arange(n_models)
    width = 0.38

    # The figure grows with the number of models so labels never touch.
    fig, ax = plt.subplots(figsize=(max(8, 1.9 * n_models), 6.5))

    for k, (column, label, color) in enumerate(PLOTTED_METRICS):
        values = table[column].to_numpy()
        positions = x + (k - 0.5) * width
        ax.bar(positions, values, width, label=label, color=color)

        decimals = DISPLAY_DECIMALS[column]
        for model_name, position, value in zip(table["model"], positions, values):
            ax.annotate(
                f"{value:.{decimals}f}",
                (position, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            if model_name == best_models[column]:
                ax.annotate(
                    "\u2605",  # black star
                    (position, value),
                    xytext=(0, 17),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    color=STAR_COLOR,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(table["model"], rotation=25, ha="right")
    ax.set_yticks(range(Y_MIN, 101, 10))
    ax.set_ylim(Y_MIN, 112)  # headroom above 100 for labels and stars
    ax.set_ylabel("Score (%)")
    ax.set_title(
        "Model comparison (validation, best checkpoint)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=STAR_COLOR,
            markersize=15,
        )
    )
    labels.append("Best model in metric")
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_OFFSET),
        ncol=len(handles),
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")  # replaces any older file
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Runner
# ---------------------------------------------------------------------------


def main():
    print_section("Model comparison")

    summaries = load_summaries()
    table = build_comparison_table(summaries)  # raw values
    best_models = find_best_models(table)  # decided on raw values
    shown = round_for_display(table)  # rounded copy for display only

    print(shown.to_string(index=False))
    print()
    for column, label, _color in PLOTTED_METRICS:
        print(f"Best model by {label}: {colorize(best_models[column], Colors.GREEN)}")

    figure_path = REPORT_DIR / FIGURE_NAME
    table_path = REPORT_DIR / TABLE_NAME

    plot_comparison(table, best_models, figure_path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    shown.to_csv(table_path, index=False, encoding="utf-8-sig")

    print()
    print(colorize(f"Saved: {figure_path}", Colors.GRAY))
    print(colorize(f"Saved: {table_path}", Colors.GRAY))


if __name__ == "__main__":
    main()

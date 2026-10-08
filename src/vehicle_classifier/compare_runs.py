"""Compare finished training runs using only their summary.json files.

Contents
--------
1. Settings          : folders, file names, plot options, TOP_N
2. Loading           : load_summaries
3. Table             : build_comparison_table, find_best_models, select_top_models,
                       round_for_display
4. Plot              : plot_comparison
5. Runner            : save_report, main
6. Per-class radars  : load_per_class_tables, build_metric_dict, macro_score,
                       ask_model_selection, plot_radar_by_model,
                       plot_radar_by_class, unique_path, run_radar_comparison

For every folder in outputs/runs/ that holds a summary.json, this script reads
the validation result of the best checkpoint and creates four files in
outputs/reports/model_comparison/:
    model_comparison.png      : grouped bars (accuracy and macro F1) for ALL models,
                                a star marks the best model of each metric
    model_comparison.csv      : the table behind that figure (values in percent,
                                rounded for display)
    model_comparison_topN.png : the same figure for the TOP_N models only
    model_comparison_topN.csv : the same table for the TOP_N models only

The top models are ranked by macro F1 (the metric used to pick best checkpoints).
These four files are overwritten on every run.

The best model of each metric is found once, on the raw, unrounded values of ALL
models. The top-N figure reuses it, so a star always means "best among all
models". If the best model of a metric is not among the top N by macro F1,
that metric simply has no star in the top-N figure.
Rounding is applied only to what is shown or saved.

After that, section 6 compares the models class by class. It reads
outputs/runs/<run>/analysis/per_class_metrics.csv (written by analyze.py),
asks which models to draw, and creates two more files in the same folder:
    radar_by_model.png : one radar per model; the 8 classes are the axes and
                         the lines are precision, recall and F1
    radar_by_class.png : one radar per class; precision, recall and F1 are the
                         axes and every model is one line
These two files are never overwritten: if a name is already taken, " (1)",
" (2)", ... is added before the extension.

Usage
-----
    python -m vehicle_classifier.compare_runs

Rule: everything here uses validation data. The test set is never touched.
"""

import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from vehicle_classifier.data import CLASS_NAMES, PROJECT_ROOT
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

TOP_N = 5  # number of best models (by macro F1) in the second figure and table
TOP_FIGURE_NAME = f"model_comparison_top{TOP_N}.png"
TOP_TABLE_NAME = f"model_comparison_top{TOP_N}.csv"

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


def select_top_models(table, top_n=TOP_N):
    """Return the top_n models with the highest macro F1, best first.

    If there are fewer than top_n models, all of them are returned.
    """
    ranked = table.sort_values("macro_f1", ascending=False)
    return ranked.head(top_n).reset_index(drop=True)


def round_for_display(table):
    """Return a copy of the table rounded for printing and saving."""
    return table.round(DISPLAY_DECIMALS)


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------


def plot_comparison(
    table,
    best_models,
    save_path,
    title="Model comparison (validation, best checkpoint)",
):
    """Grouped bar chart: accuracy and macro F1 (in percent) for every model.

    Every bar has its rounded value written above it. The bar of the best
    model of each metric (from best_models) gets a star above its value.
    A model in best_models that is not in the table simply gets no star.
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
    ax.set_title(title, fontsize=12, fontweight="bold")
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
    labels.append("Best model in metric (among all models)")
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


def save_report(table, best_models, figure_path, table_path, title):
    """Print the table, then save its figure and its CSV (rounded for display)."""
    shown = round_for_display(table)
    print(shown.to_string(index=False))

    plot_comparison(table, best_models, figure_path, title)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    shown.to_csv(table_path, index=False, encoding="utf-8-sig")

    print()
    print(colorize(f"Saved: {figure_path}", Colors.GRAY))
    print(colorize(f"Saved: {table_path}", Colors.GRAY))


# ---------------------------------------------------------------------------
# 6. Per-class radar charts
# ---------------------------------------------------------------------------

ANALYSIS_DIR_NAME = "analysis"  # sub-folder of a run that analyze.py creates
PER_CLASS_FILE_NAME = "per_class_metrics.csv"

BY_MODEL_FIGURE_NAME = "radar_by_model.png"
BY_CLASS_FIGURE_NAME = "radar_by_class.png"

# Lowest radial limit in percent. None = chosen automatically from the data:
# one shared limit per figure, so every radar of a figure uses the same scale.
# Set a number (for example 50) to force the same zoom in every figure.
RADAR_MIN_PERCENT = None

# (column in per_class_metrics.csv, label, color, marker)
# The colors match the per-class bar chart of analyze.py.
RADAR_METRICS = [
    ("precision", "Precision", PLOT_PALETTE[1], "o"),
    ("recall", "Recall", PLOT_PALETTE[5], "s"),
    ("f1", "F1", PLOT_PALETTE[2], "^"),
]

# Styles for the models in radar_by_class. Okabe-Ito colors without yellow
# (hard to see on white). Model k gets color k % 7 and marker k % 7; after
# seven models the line style changes, so up to 28 models stay distinguishable.
MODEL_COLORS = [PLOT_PALETTE[i] for i in (4, 5, 2, 0, 6, 1, 7)]
MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
MODEL_LINESTYLES = ["-", "--", "-.", ":"]


# -- 6.1 Loading and collecting ---------------------------------------------


def load_per_class_tables(runs_root=RUNS_ROOT):
    """Read analysis/per_class_metrics.csv of every run.

    Returns a dict {run folder name: DataFrame}. Each DataFrame has one row per
    class (index "class") and the columns precision, recall, f1, support, with
    values between 0 and 1. A run without this file (analyze.py was not run for
    it) or with different classes is skipped with a warning.
    """
    if not runs_root.exists():
        raise FileNotFoundError(f"No runs folder at {runs_root}")

    tables = {}
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        csv_path = run_dir / ANALYSIS_DIR_NAME / PER_CLASS_FILE_NAME
        if not csv_path.exists():
            print(
                colorize(
                    f"Skipped {run_dir.name}: no {ANALYSIS_DIR_NAME}/"
                    f"{PER_CLASS_FILE_NAME} (run analyze.py for it first)",
                    Colors.YELLOW,
                )
            )
            continue

        table = pd.read_csv(csv_path, index_col="class", encoding="utf-8-sig")
        missing = [name for name in CLASS_NAMES if name not in table.index]
        if missing:
            print(
                colorize(
                    f"Skipped {run_dir.name}: classes {missing} are missing in its table",
                    Colors.YELLOW,
                )
            )
            continue
        tables[run_dir.name] = table

    if not tables:
        raise FileNotFoundError(f"No {PER_CLASS_FILE_NAME} found in {runs_root}")
    return tables


def build_metric_dict(tables):
    """Collect all per-class scores of all models into one nested dict.

    Structure (all scores in percent, like the table at the top of this script):
        data[metric][class_name][model_name] = score
    for the metrics precision, recall and f1, the 8 classes, and every model.
    """
    return {
        metric: {
            class_name: {
                model: float(table.loc[class_name, metric]) * 100
                for model, table in tables.items()
            }
            for class_name in CLASS_NAMES
        }
        for metric, _label, _color, _marker in RADAR_METRICS
    }


def macro_score(data, model, metric):
    """Macro average of one model: the mean of its per-class scores, in percent."""
    return float(np.mean([data[metric][name][model] for name in CLASS_NAMES]))


def _all_scores(data, models):
    """Every score of the chosen models (all metrics, all classes) as one flat list."""
    return [
        data[metric][name][model]
        for metric in data
        for name in CLASS_NAMES
        for model in models
    ]


# -- 6.2 Model selection ------------------------------------------------------


def ask_model_selection(data, models):
    """Ask which models are drawn on the radar charts and return their names.

    Press Enter to draw all models. Type n to choose: the models are listed
    with their macro precision and macro recall, best recall first, and you type
    the numbers of the wanted models separated by , or - or . (for example
    1,3,4 or 1-3-4 or 1.3.4). Note that 1-3 means model 1 and model 3, not a range.

    The returned names are always in list order (best recall first), so colors
    and positions in the figures do not depend on the typing order.
    """
    ranked = sorted(
        models, key=lambda model: macro_score(data, model, "recall"), reverse=True
    )

    print_section("Radar charts: choose the models")
    while True:
        answer = (
            input(
                f"Draw all {len(ranked)} models? Press Enter to confirm, "
                "or type n to choose them yourself: "
            )
            .strip()
            .lower()
        )
        if answer in ("", "y", "yes"):
            return ranked
        if answer in ("n", "no"):
            break
        print(colorize("Press Enter, or type y or n.", Colors.YELLOW))

    width = max(len(model) for model in ranked)
    print("\nMacro scores in percent, best recall first:")
    print(f"{'No.':>4}  {'Model':<{width}}  {'Precision':>10}  {'Recall':>8}")
    for number, model in enumerate(ranked, start=1):
        precision = macro_score(data, model, "precision")
        recall = macro_score(data, model, "recall")
        print(f"{number:>4}  {model:<{width}}  {precision:>10.2f}  {recall:>8.2f}")

    while True:
        text = input("\nModel numbers (separated by , or - or .): ")
        numbers = [
            int(part) for part in re.findall(r"\d+", text)
        ]  # any separator works

        if not numbers:
            print(colorize("No numbers found, try again.", Colors.YELLOW))
            continue
        invalid = [n for n in numbers if not 1 <= n <= len(ranked)]
        if invalid:
            print(
                colorize(
                    f"Out of range: {invalid}. Use numbers from 1 to {len(ranked)}.",
                    Colors.YELLOW,
                )
            )
            continue

        return [ranked[n - 1] for n in sorted(set(numbers))]


# -- 6.3 Radar helpers --------------------------------------------------------


def _closed(values):
    """Repeat the first value at the end so a radar line returns to its start."""
    values = list(values)
    return values + values[:1]


def _radar_min(scores):
    """Lowest radial limit (a multiple of 10) that leaves a margin below the smallest score."""
    if RADAR_MIN_PERCENT is not None:
        return RADAR_MIN_PERCENT
    return max(0, math.floor((min(scores) - 5) / 10) * 10)


def _radial_ticks(r_min):
    """Ring values between r_min and 100, counted down from 100. r_min itself is the center."""
    span = 100 - r_min
    step = 5 if span <= 20 else 10 if span <= 50 else 20
    ticks, value = [], 100
    while value > r_min:
        ticks.append(value)
        value -= step
    return ticks[::-1]


def _model_style(index):
    """Line style of the model at this position in the chosen list (same in every radar)."""
    return {
        "color": MODEL_COLORS[index % len(MODEL_COLORS)],
        "marker": MODEL_MARKERS[index % len(MODEL_MARKERS)],
        "linestyle": MODEL_LINESTYLES[
            (index // len(MODEL_COLORS)) % len(MODEL_LINESTYLES)
        ],
        "linewidth": 1.8,
        "markersize": 6,
        "alpha": 0.9,
    }


def _style_radar_axes(ax, labels, r_min, title):
    """Give one polar axes the radar look: first axis on top, clockwise, zoomed rings."""
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ax.set_theta_offset(np.pi / 2)  # first axis points up
    ax.set_theta_direction(-1)  # the others follow clockwise
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="x", pad=8)

    ticks = _radial_ticks(r_min)
    ax.set_ylim(r_min, 100)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}%" for t in ticks], fontsize=7.5, color="#666666")
    ax.set_rlabel_position(180 / len(labels))  # ring labels sit between two axes

    ax.grid(True, color="#D0D0D0", linewidth=0.8)
    ax.spines["polar"].set_color("#BBBBBB")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=22)


def _save_radar_figure(fig, path):
    """Save a radar figure as PNG and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def unique_path(path):
    """Return path, or "name (1).ext", "name (2).ext", ... if the file already exists.

    This is the Windows way of keeping both files, so older images are never
    overwritten.
    """
    if not path.exists():
        return path
    number = 1
    while True:
        candidate = path.with_name(f"{path.stem} ({number}){path.suffix}")
        if not candidate.exists():
            return candidate
        number += 1


# -- 6.4 Plots ----------------------------------------------------------------


def plot_radar_by_model(data, models, save_path):
    """One radar per model: the classes are the axes, the lines are precision, recall, F1.

    Shows, for each model separately, which classes are strong or weak. All
    radars share the same radial scale, so they can be compared by eye.
    """
    angles = np.linspace(0, 2 * np.pi, len(CLASS_NAMES), endpoint=False)
    r_min = _radar_min(_all_scores(data, models))

    ncols = min(len(models), 3)
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.4 * ncols, 6.4 * nrows),
        subplot_kw={"projection": "polar"},
        squeeze=False,
        layout="constrained",
    )

    for ax, model in zip(axes.flat, models):
        for metric, label, color, marker in RADAR_METRICS:
            scores = [data[metric][name][model] for name in CLASS_NAMES]
            ax.plot(
                _closed(angles),
                _closed(scores),
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.8,
                label=label,
            )
        macro_f1 = macro_score(data, model, "f1")
        _style_radar_axes(ax, CLASS_NAMES, r_min, f"{model}\nmacro F1 {macro_f1:.2f}%")

    for ax in axes.flat[len(models) :]:  # hide empty cells of the grid
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        "Per-class precision, recall and F1 of each model (validation)",
        fontsize=14,
        fontweight="bold",
    )
    _save_radar_figure(fig, save_path)


def plot_radar_by_class(data, models, save_path):
    """One radar per class: precision, recall and F1 are the axes, each model is a line.

    Shows, for each class separately, which model handles it best. All radars
    share the same radial scale, and a model keeps the same color, marker and
    line style in every radar.
    """
    metric_labels = [label for _metric, label, _color, _marker in RADAR_METRICS]
    angles = np.linspace(0, 2 * np.pi, len(RADAR_METRICS), endpoint=False)
    r_min = _radar_min(_all_scores(data, models))

    ncols = 4
    nrows = math.ceil(len(CLASS_NAMES) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.6 * ncols, 5.6 * nrows),
        subplot_kw={"projection": "polar"},
        squeeze=False,
        layout="constrained",
    )

    for ax, name in zip(axes.flat, CLASS_NAMES):
        for k, model in enumerate(models):
            scores = [data[metric][name][model] for metric, *_ in RADAR_METRICS]
            ax.plot(_closed(angles), _closed(scores), label=model, **_model_style(k))
        _style_radar_axes(ax, metric_labels, r_min, name)

    for ax in axes.flat[len(CLASS_NAMES) :]:  # hide empty cells of the grid
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(len(handles), 4),
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Precision, recall and F1 of each model, class by class (validation)",
        fontsize=14,
        fontweight="bold",
    )
    _save_radar_figure(fig, save_path)


# -- 6.5 Runner ---------------------------------------------------------------


def run_radar_comparison(runs_root=RUNS_ROOT, report_dir=REPORT_DIR):
    """Load the per-class tables, ask which models to draw, and save both radar figures."""
    print_section("Per-class radar charts")
    tables = load_per_class_tables(runs_root)
    data = build_metric_dict(tables)
    models = ask_model_selection(data, list(tables))
    print(f"\nDrawing {len(models)} model(s): {', '.join(models)}\n")

    figures = [
        (BY_MODEL_FIGURE_NAME, plot_radar_by_model),
        (BY_CLASS_FIGURE_NAME, plot_radar_by_class),
    ]
    for file_name, plot_function in figures:
        save_path = unique_path(report_dir / file_name)
        plot_function(data, models, save_path)
        print(colorize(f"Saved: {save_path}", Colors.GRAY))


def build_config_table(runs_root=RUNS_ROOT):
    """Read config.json from every run and build one flat DataFrame.

    Nested dictionaries are flattened so every nested key becomes its own
    column. For example:
        {"model": {"model_name": "resnet18", "mode": "full"}}

    becomes:
        model_model_name | model_mode

    Rows are indexed by run name.
    """
    rows = []

    if not runs_root.exists():
        raise FileNotFoundError(f"No runs folder at {runs_root}")

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        config_path = run_dir / "config.json"

        if not config_path.exists():
            print(
                colorize(
                    f"Skipped {run_dir.name}: no config.json",
                    Colors.YELLOW,
                )
            )
            continue

        config = load_json(config_path)

        # Flatten nested dictionaries such as "model" and "scheduler_params".
        row = pd.json_normalize(config, sep="_").iloc[0].to_dict()

        # Use the run folder name as the model/run identifier.
        row["run_name"] = run_dir.name

        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No config.json files found in {runs_root}")

    table = pd.DataFrame(rows)

    # Put run_name as the first column.
    columns = ["run_name"] + [
        column for column in table.columns if column != "run_name"
    ]
    table = table[columns]

    return table


def save_config_table(
    runs_root=RUNS_ROOT,
    report_dir=REPORT_DIR,
):
    """Build the configuration table and save it as a CSV."""
    table = build_config_table(runs_root)

    report_dir.mkdir(parents=True, exist_ok=True)

    save_path = report_dir / "all_configs.csv"

    table.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig",
    )

    print_section("Model configurations")
    print(table.to_string(index=False))
    print()
    print(colorize(f"Saved: {save_path}", Colors.GRAY))

    return table


def main():
    summaries = load_summaries()
    table = build_comparison_table(summaries)  # raw values, all models
    best_models = find_best_models(table)  # decided on raw values of all models

    # Save configuration of every run.
    save_config_table()

    table = build_comparison_table(summaries)
    best_models = find_best_models(table)

    # -----------
    # All models
    # -----------

    print_section(f"Model comparison: all {len(table)} models")
    save_report(
        table,
        best_models,
        REPORT_DIR / FIGURE_NAME,
        REPORT_DIR / TABLE_NAME,
        title="Model comparison (validation, best checkpoint)",
    )
    print()
    for column, label, _color in PLOTTED_METRICS:
        print(f"Best model by {label}: {colorize(best_models[column], Colors.GREEN)}")

    # -----------
    # Top N models only
    # -----------

    top_table = select_top_models(table, top_n=5)
    print_section(f"Model comparison: top {len(top_table)} models by macro F1")
    save_report(
        top_table,
        best_models,
        REPORT_DIR / TOP_FIGURE_NAME,
        REPORT_DIR / TOP_TABLE_NAME,
        title=f"Top {len(top_table)} models by macro F1 (validation, best checkpoint)",
    )

    # -----------
    # Per-class radar charts (asks which models to draw)
    # -----------

    run_radar_comparison()


if __name__ == "__main__":
    main()

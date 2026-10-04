"""Analysis of one finished training run (validation data only).

This script never trains and never loads the model. It only reads the files
that train.py saved in outputs/runs/<run_name>/ and turns them into tables
and figures, so it can be repeated as often as needed in a few seconds.

Contents
--------
1. Settings         : defaults
2. Loading          : load_run
3. Plots            : plot_training_curves, plot_confusion_matrices,
                      plot_per_class_metrics, plot_misclassified, plot_selective
4. Tables           : top_confusions, build_selective_table, suggest_threshold
5. Analysis runner  : analyze, main

Inputs  (written by train.py): history.json, summary.json, val_predictions.npz
Outputs (written here)       : outputs/runs/<run_name>/analysis/
    training_curves.png, per_class_metrics.png, confusion_matrix.png,
    misclassified_gallery.png, review_threshold_curve.png,
    per_class_metrics.csv, top_confusions.csv, confused_pairs.csv,
    misclassified.csv, review_threshold_table.csv, suggested_threshold.json

Usage
-----
    python -m vehicle_classifier.analyze
    python -m vehicle_classifier.analyze --run baseline_cnn --top-k 16
    python -m vehicle_classifier.analyze --run avg_pool --target-accuracy 0.97

Rule: everything here uses validation data. The test set is never touched,
and the review threshold suggested here is chosen on validation data only.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from vehicle_classifier.data import CLASS_NAMES, PROJECT_ROOT
from vehicle_classifier.metrics import (
    compute_confusion_matrix,
    compute_metrics,
    find_misclassified,
    metrics_to_dataframe,
    print_metrics_report,
    rank_confused_pairs,
    selective_metrics,
)
from vehicle_classifier.utils import (
    CURVE_COLORS,
    PLOT_PALETTE,
    Colors,
    colorize,
    load_json,
    print_section,
    print_subsection,
    save_json,
)

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------

OUTPUT_ROOT = "outputs/runs"  # relative to PROJECT_ROOT, same as train.py
DEFAULT_RUN_NAME = "full_ablation"
DEFAULT_TOP_K = 12  # number of misclassified images shown in the gallery
DEFAULT_TARGET_ACCURACY = 0.95  # accuracy wanted on the automatically accepted images
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _print_item(label, value):
    """Print one 'label : value' line with a colored label."""
    print(f"{colorize(label.ljust(24), Colors.CYAN)}: {value}")


def _style_axes(ax, title, xlabel=None, ylabel=None):
    """Same look for every plot: bold title, light grid, no top/right frame."""
    ax.set_title(title, fontsize=11, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save_figure(fig, path):
    """Tidy the layout, save the figure as PNG, and close it."""
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(colorize(f"Saved: {path}", Colors.GRAY))


def _save_table(table, path, index=True):
    """Save a DataFrame as CSV (utf-8-sig so Excel on Windows opens it correctly)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=index, encoding="utf-8-sig")
    print(colorize(f"Saved: {path}", Colors.GRAY))


# ---------------------------------------------------------------------------
# 2. Loading
# ---------------------------------------------------------------------------


def load_run(run_name):
    """Read the files that train.py saved for one run.

    Returns a dict with:
        "run_dir"  : Path of the run folder
        "history"  : DataFrame, one row per epoch (from history.json)
        "summary"  : dict from summary.json
        "arrays"   : dict of NumPy arrays from val_predictions.npz
                     (y_true, y_pred, confidence, probs, and paths if saved)
    """
    runs_root = PROJECT_ROOT / OUTPUT_ROOT
    run_dir = runs_root / run_name

    if not run_dir.exists():
        available = (
            sorted(p.name for p in runs_root.iterdir() if p.is_dir())
            if runs_root.exists()
            else []
        )
        raise FileNotFoundError(
            f"No run folder at {run_dir}. Available runs: {available}"
        )

    required = ["history.json", "summary.json", "val_predictions.npz"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{run_dir} is missing {missing}. Did the training run finish?"
        )

    history = pd.DataFrame(load_json(run_dir / "history.json"))
    summary = load_json(run_dir / "summary.json")
    with np.load(run_dir / "val_predictions.npz") as npz:
        arrays = {key: npz[key] for key in npz.files}

    if arrays["probs"].shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"The run has {arrays['probs'].shape[1]} classes but CLASS_NAMES "
            f"in data.py has {len(CLASS_NAMES)}."
        )

    return {
        "run_dir": run_dir,
        "history": history,
        "summary": summary,
        "arrays": arrays,
    }


# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------


def plot_training_curves(history, best_epoch, save_path):
    """Four panels: loss, accuracy, validation F1, and the loss gap.

    The dashed line marks the epoch whose weights were saved as best.pt.
    The gap panel (validation loss minus train loss) shows overfitting: it
    grows when the model memorizes the training set.
    """
    epochs = history["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_loss, ax_acc, ax_f1, ax_gap = axes.flat

    # Loss
    ax_loss.plot(
        epochs, history["train_loss"], color=CURVE_COLORS["train"], label="train"
    )
    ax_loss.plot(epochs, history["val_loss"], color=CURVE_COLORS["val"], label="val")
    _style_axes(ax_loss, "Loss", "epoch", "loss")

    # Accuracy
    ax_acc.plot(
        epochs, history["train_acc"], color=CURVE_COLORS["train"], label="train"
    )
    ax_acc.plot(epochs, history["val_acc"], color=CURVE_COLORS["val"], label="val")
    _style_axes(ax_acc, "Accuracy", "epoch", "accuracy")

    # Validation precision / recall / F1
    ax_f1.plot(
        epochs,
        history["val_macro_precision"],
        color=PLOT_PALETTE[1],
        lw=1,
        alpha=0.8,
        label="macro precision",
    )
    ax_f1.plot(
        epochs,
        history["val_macro_recall"],
        color=PLOT_PALETTE[0],
        lw=1,
        alpha=0.8,
        label="macro recall",
    )
    ax_f1.plot(
        epochs,
        history["val_macro_f1"],
        color=CURVE_COLORS["val"],
        lw=2,
        label="macro F1",
    )
    best_f1 = history.loc[history["epoch"] == best_epoch, "val_macro_f1"]
    if not best_f1.empty:
        ax_f1.scatter(
            [best_epoch],
            [best_f1.iloc[0]],
            color="black",
            zorder=5,
            marker="*",
            s=140,
            label=f"best ({best_f1.iloc[0]:.4f})",
        )
    _style_axes(ax_f1, "Validation macro metrics", "epoch", "score")

    # Gap
    ax_gap.plot(epochs, history["gap"], color=PLOT_PALETTE[6])
    ax_gap.axhline(0, color="gray", lw=1)
    _style_axes(ax_gap, "Generalization gap (val loss - train loss)", "epoch", "gap")

    # Best-epoch marker on every panel (label only once, on the loss panel)
    for ax in axes.flat:
        label = f"best epoch ({best_epoch})" if ax is ax_loss else None
        ax.axvline(best_epoch, color="gray", ls="--", lw=1, alpha=0.7, label=label)
    for ax in (ax_loss, ax_acc, ax_f1):
        ax.legend(frameon=False)

    _save_figure(fig, save_path)


def plot_confusion_matrices(cm, cm_normalized, class_names, save_path):
    """Confusion matrix twice: raw counts, and row-normalized (diagonal = recall)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    panels = [
        (axes[0], cm, "Confusion matrix (counts)", lambda v: f"{int(v)}", None),
        (
            axes[1],
            cm_normalized,
            "Row-normalized (diagonal = recall)",
            lambda v: f"{v:.2f}",
            1.0,
        ),
    ]

    ticks = np.arange(len(class_names))
    for ax, data, title, fmt, vmax in panels:
        image = ax.imshow(data, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title, fontsize=11, fontweight="bold")

        half = data.max() / 2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] > 0:  # leave empty cells blank, easier to read
                    ax.text(
                        j,
                        i,
                        fmt(data[i, j]),
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if data[i, j] > half else "black",
                    )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    _save_figure(fig, save_path)


def plot_per_class_metrics(metrics, save_path):
    """Grouped bars of precision, recall and F1 for every class."""
    table = metrics_to_dataframe(metrics).drop(index="macro avg")
    x = np.arange(len(table))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5.5))
    series = [
        ("precision", PLOT_PALETTE[1]),
        ("recall", PLOT_PALETTE[5]),
        ("f1", PLOT_PALETTE[2]),
    ]
    for k, (column, color) in enumerate(series):
        ax.bar(x + (k - 1) * width, table[column], width, label=column, color=color)

    ax.axhline(
        metrics["macro_f1"],
        color="black",
        ls="--",
        lw=1,
        label=f"macro F1 ({metrics['macro_f1']:.3f})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{name}\n(n={int(n)})" for name, n in zip(table.index, table["support"])]
    )
    ax.set_ylim(0, 1.05)
    _style_axes(ax, "Per-class metrics (validation)", ylabel="score")
    ax.legend(frameon=False, ncol=4, loc="lower center")

    _save_figure(fig, save_path)


def plot_misclassified(errors, save_path, ncols=4):
    """Show the original images of the most confident mistakes."""
    n = len(errors)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 3.7 * nrows), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")

    for ax, (_, row) in zip(axes.flat, errors.iterrows()):
        try:
            ax.imshow(Image.open(row["path"]).convert("RGB"))
        except OSError:
            ax.text(
                0.5,
                0.5,
                "image not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title(
            f"true: {row['true_class']}\n"
            f"pred: {row['pred_class']} ({row['confidence']:.2f})",
            fontsize=9,
            color=PLOT_PALETTE[5],
        )

    fig.suptitle("Most confident mistakes (validation)", fontsize=13, fontweight="bold")
    _save_figure(fig, save_path)


def plot_selective(table, target_accuracy, save_path):
    """Coverage, accuracy of accepted predictions, and share of errors caught."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    series = [
        ("coverage", "coverage (decided automatically)", PLOT_PALETTE[1]),
        ("accuracy_accepted", "accuracy on accepted", PLOT_PALETTE[2]),
        ("error_capture_rate", "errors sent to review", PLOT_PALETTE[5]),
    ]
    for column, label, color in series:
        ax.plot(table["threshold"], table[column], marker="o", color=color, label=label)

    ax.axhline(
        target_accuracy,
        color="gray",
        ls=":",
        lw=1.2,
        label=f"target accuracy ({target_accuracy:.2f})",
    )
    ax.set_ylim(0, 1.05)
    _style_axes(
        ax,
        "Review threshold: automation vs accuracy (validation)",
        "confidence threshold",
        "share",
    )
    ax.legend(frameon=False, loc="lower left")

    _save_figure(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Tables
# ---------------------------------------------------------------------------


def top_confusions(cm, cm_normalized, class_names, top_k=10):
    """List the most frequent mistakes as (true class -> predicted class).

    Unlike rank_confused_pairs, this keeps the direction: "kamyun predicted as
    kamyunet" and "kamyunet predicted as kamyun" are separate rows.
    """
    rows = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                rows.append(
                    {
                        "true_class": true_name,
                        "pred_class": pred_name,
                        "count": int(cm[i, j]),
                        "share_of_true_class": float(cm_normalized[i, j]),
                    }
                )
    columns = ["true_class", "pred_class", "count", "share_of_true_class"]
    table = pd.DataFrame(rows, columns=columns)
    return (
        table.sort_values("count", ascending=False).head(top_k).reset_index(drop=True)
    )


def build_selective_table(y_true, y_pred, confidence, thresholds=THRESHOLDS):
    """Run selective_metrics at several thresholds and collect one row each."""
    rows = [selective_metrics(y_true, y_pred, confidence, t) for t in thresholds]
    table = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")  # None -> NaN
    return table[
        [
            "threshold",
            "coverage",
            "review_rate",
            "num_review",
            "accuracy_accepted",
            "accuracy_review",
            "error_capture_rate",
        ]
    ]


def suggest_threshold(table, target_accuracy):
    """Return the row with the lowest threshold that reaches the target accuracy.

    A lower threshold means fewer images go to a human, so the lowest threshold
    that still meets the target gives the most automation. Returns None if no
    threshold in the table reaches the target.
    """
    reached = table[table["accuracy_accepted"] >= target_accuracy]
    if reached.empty:
        return None
    return reached.sort_values("threshold").iloc[0]


# ---------------------------------------------------------------------------
# 5. Analysis runner
# ---------------------------------------------------------------------------


def analyze(run_name, top_k=DEFAULT_TOP_K, target_accuracy=DEFAULT_TARGET_ACCURACY):
    """Run the whole analysis for one run and save everything to <run>/analysis/."""
    run = load_run(run_name)
    history, summary, arrays = run["history"], run["summary"], run["arrays"]
    out_dir = run["run_dir"] / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = arrays["y_true"], arrays["y_pred"]
    confidence = arrays["confidence"]
    paths = arrays.get("paths", None)
    best_epoch = summary["best_epoch"]

    print_section(f"Analysis: {run_name}")
    _print_item("Run folder", run["run_dir"])
    _print_item("Loss type", summary["config"]["loss_type"])
    _print_item("Epochs run", len(history))
    _print_item(
        "Best epoch",
        f"{best_epoch} ({summary['monitor']} = {summary['best_score']:.4f})",
    )
    _print_item("Validation samples", len(y_true))

    # -- 1. Training curves ---------------------------------------------------
    print_subsection("1. Training curves")
    last = history.iloc[-1]
    _print_item(
        "Last epoch",
        f"train acc {last['train_acc']:.4f} | val acc {last['val_acc']:.4f}",
    )
    swing = history["val_loss"].diff().abs().mean()
    _print_item("Val loss swing", f"{swing:.4f} (mean change between epochs)")
    plot_training_curves(history, best_epoch, out_dir / "training_curves.png")

    # -- 2. Metrics -----------------------------------------------------------
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    saved_f1 = summary["val_metrics"]["macro_f1"]
    print_metrics_report(metrics, title="2. Validation metrics (best checkpoint)")
    if abs(saved_f1 - metrics["macro_f1"]) > 1e-6:
        print(
            colorize(
                f"Warning: macro F1 recomputed here ({metrics['macro_f1']:.6f}) differs "
                f"from summary.json ({saved_f1:.6f}).",
                Colors.YELLOW,
            )
        )
    _save_table(metrics_to_dataframe(metrics), out_dir / "per_class_metrics.csv")
    plot_per_class_metrics(metrics, out_dir / "per_class_metrics.png")

    # -- 3. Confusion matrix --------------------------------------------------
    print_section("3. Confusion analysis")
    cm, cm_normalized = compute_confusion_matrix(y_true, y_pred, len(CLASS_NAMES))
    plot_confusion_matrices(
        cm, cm_normalized, CLASS_NAMES, out_dir / "confusion_matrix.png"
    )

    print_subsection("Most common mistakes (true -> predicted)")
    confusions = top_confusions(cm, cm_normalized, CLASS_NAMES, top_k=10)
    print(confusions.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    _save_table(confusions, out_dir / "top_confusions.csv", index=False)

    print_subsection("Most confused class pairs (both directions added)")
    pairs = rank_confused_pairs(cm_normalized, CLASS_NAMES, top_k=8)
    print(pairs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    _save_table(pairs, out_dir / "confused_pairs.csv", index=False)

    # -- 4. Misclassified images ----------------------------------------------
    print_section("4. Misclassified images")
    errors = find_misclassified(y_true, y_pred, confidence, CLASS_NAMES, paths=paths)
    _print_item("Total mistakes", f"{len(errors)} of {len(y_true)}")
    if len(errors) > 0:
        print_subsection("Most confident mistakes first")
        print(errors.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        _save_table(errors, out_dir / "misclassified.csv", index=False)

        if "path" in errors.columns:
            plot_misclassified(
                errors.head(top_k), out_dir / "misclassified_gallery.png"
            )
        else:
            print(
                colorize(
                    "No image paths in val_predictions.npz, so no gallery was made.",
                    Colors.YELLOW,
                )
            )

    # -- 5. Human-review threshold --------------------------------------------
    print_section("5. Review threshold")
    selective = build_selective_table(y_true, y_pred, confidence)
    print(selective.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    _save_table(selective, out_dir / "review_threshold_table.csv", index=False)
    plot_selective(selective, target_accuracy, out_dir / "review_threshold_curve.png")

    print_subsection(f"Suggested threshold (target accuracy {target_accuracy:.2f})")
    chosen = suggest_threshold(selective, target_accuracy)
    if chosen is None:
        print(
            colorize(
                "No threshold in the table reaches the target accuracy. "
                "Lower the target or improve the model first.",
                Colors.YELLOW,
            )
        )
    else:
        suggestion = {
            "run_name": run_name,
            "target_accuracy": target_accuracy,
            "threshold": float(chosen["threshold"]),
            "coverage": float(chosen["coverage"]),
            "review_rate": float(chosen["review_rate"]),
            "accuracy_accepted": float(chosen["accuracy_accepted"]),
            "error_capture_rate": float(chosen["error_capture_rate"]),
            "chosen_on": "validation",
        }
        print(
            colorize(
                f"Threshold {suggestion['threshold']:.2f}: "
                f"{suggestion['coverage']:.1%} decided automatically, "
                f"accuracy on those {suggestion['accuracy_accepted']:.1%}, "
                f"{suggestion['error_capture_rate']:.1%} of all errors sent to review.",
                Colors.GREEN,
            )
        )
        save_json(suggestion, out_dir / "suggested_threshold.json")
        print(colorize(f"Saved: {out_dir / 'suggested_threshold.json'}", Colors.GRAY))

    print_section("Analysis finished")
    print(f"All files are in: {out_dir}")


# %%


def main():
    parser = argparse.ArgumentParser(description="Analyze one finished training run.")
    parser.add_argument(
        "--run",
        default=DEFAULT_RUN_NAME,
        help=f"run_name of the run to analyze (default: {DEFAULT_RUN_NAME})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="number of mistakes shown in the image gallery",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=DEFAULT_TARGET_ACCURACY,
        help="accuracy wanted on automatically accepted predictions",
    )
    args = parser.parse_args()
    analyze(args.run, top_k=args.top_k, target_accuracy=args.target_accuracy)


if __name__ == "__main__":
    main()

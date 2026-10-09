"""Final test evaluation of ONE frozen model, ONCE.

The model is chosen with validation data only. It is then run a single time on
the frozen cleaned test split. The result is only reported: it never changes
the model, the review threshold, or any other decision.

Steps
-----
A. Choose the run (best validation macro-F1, CE and BCE runs both count) and
   read the review threshold (from validation analysis, or given by hand).
B. Load the checkpoint of that run (best validation epoch).
C. Build the test data from the folder and check it (classes, leakage).
D. Run the model on the test set and save the predictions immediately.
E. Report: metrics, validation vs test, confidence intervals, confusion
   matrices, mistakes, review threshold.
F. Save everything to outputs/final_test/test_by_<run_name>/.

If test_predictions.npz already exists in that folder, the model is NOT run
again: the report is rebuilt from the saved predictions.

Usage
-----
    python -m vehicle_classifier.final_test

Before the first run, run analyze.py for the chosen run so that
analysis/suggested_threshold.json exists (or set MANUAL_THRESHOLD below).
"""

import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vehicle_classifier.analyze import (
    plot_confusion_matrices,
    plot_misclassified,
    plot_per_class_metrics,
    top_confusions,
)
from vehicle_classifier.compare_runs import (
    RUNS_ROOT,
    build_comparison_table,
    load_summaries,
)
from vehicle_classifier.data import (
    CLASS_NAMES,
    PROJECT_ROOT,
    ManifestDataset,
    build_transform,
    load_manifest,
)
from vehicle_classifier.metrics import (
    compute_confusion_matrix,
    compute_metrics,
    find_misclassified,
    metrics_to_dataframe,
    print_metrics_report,
    selective_metrics,
)
from vehicle_classifier.model import build_model, get_device
from vehicle_classifier.resnet18 import TransferResNet
from vehicle_classifier.train import build_criterion, evaluate
from vehicle_classifier.utils import (
    Colors,
    load_json,
    print_section,
    save_json,
    set_seed,
)

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------

TEST_DIR = PROJECT_ROOT / "dataset" / "TestingData" / "test4"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_mentors_exam"
PREDICTIONS_FILE = "mentors_exam_predictions.npz"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

BATCH_SIZE = 32
NUM_WORKERS = 4
N_RESAMPLES = 1000  # bootstrap repetitions
BOOTSTRAP_SEED = 42  # fixed, so the confidence interval is the same on every run
GALLERY_SIZE = 12  # misclassified images shown (fewer if there are fewer mistakes)

RUN_NAME_BY_HAND = (
    None  # for example "resnet18_finetune"; None = best run on validation
)
MANUAL_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------


def format_value(value):
    """Format a number for printing. None (an undefined metric) becomes 'n/a'."""
    return "n/a" if value is None else f"{value:.3f}"


def file_fingerprint(path):
    """Return the SHA-256 fingerprint of the exact bytes of a file.

    Identical files give identical fingerprints. This catches exact copies;
    near-duplicates were already handled in the data audit.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 3. A. Choose the model and the threshold (validation data only)
# ---------------------------------------------------------------------------


def choose_best_run():
    """Return (run name, its summary) for the best validation macro-F1.

    CE and BCE runs are both in the table, so both can win.
    """
    summaries = load_summaries()
    table = build_comparison_table(summaries)  # sorted by macro-F1, best first

    run_name = (
        RUN_NAME_BY_HAND if RUN_NAME_BY_HAND is not None else table.loc[0, "model"]
    )
    if run_name not in summaries:
        raise ValueError(f"No finished run named {run_name}")

    val_metrics = summaries[run_name]["val_metrics"]
    print(f"Chosen run          : {run_name}")
    print(f"Validation accuracy : {val_metrics['accuracy']:.4f}")
    print(f"Validation macro-F1 : {val_metrics['macro_f1']:.4f}")
    return run_name, summaries[run_name]


def read_validation_threshold(run_name, manual_threshold=None):
    """Return the review threshold chosen on VALIDATION data.

    With manual_threshold=None it is read from the validation analysis of the
    run (analysis/suggested_threshold.json). Otherwise the given number is used;
    it must come from a validation analysis too, never from test results.

    It is read before the test set is touched: if it is missing or invalid we
    want to know BEFORE the one and only test run, not after it.
    """
    if manual_threshold is not None:
        threshold = manual_threshold
    else:
        path = RUNS_ROOT / run_name / "analysis" / "suggested_threshold.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run analyze.py for {run_name} first, "
                "or set MANUAL_THRESHOLD."
            )
        threshold = load_json(path)["threshold"]

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    return threshold


# ---------------------------------------------------------------------------
# 4. B. Load the frozen model
# ---------------------------------------------------------------------------


def load_checkpoint(run_name):
    """Load best.pt (best validation epoch) and check the things we rely on."""
    # weights_only=False: the file also holds plain Python objects (config, class
    # names). This is safe because we wrote the file ourselves.
    checkpoint = torch.load(
        RUNS_ROOT / run_name / "best.pt", map_location="cpu", weights_only=False
    )
    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("Class names in the checkpoint differ from CLASS_NAMES")
    if checkpoint["normalization"] is None:
        raise ValueError("The checkpoint has no normalization statistics")
    return checkpoint


def build_model_from_checkpoint(checkpoint, device):
    """Rebuild the architecture, fill it with the trained weights, set eval mode."""
    model_config = checkpoint["model_config"]
    architecture = model_config["model_name"]

    if architecture == "resnet18":
        model = TransferResNet(
            model_config["mode"], model_config["num_classes"], model_config["dropout_p"]
        )
    elif architecture == "baseline_cnn":
        model = build_model(model_config)
    else:
        raise ValueError(f"Unknown architecture {architecture!r}")

    model.load_state_dict(checkpoint["model_state_dict"])  # trained weights
    model.to(device)
    model.eval()  # no dropout, fixed BatchNorm statistics
    return model


# ---------------------------------------------------------------------------
# 5. C. Test data
# ---------------------------------------------------------------------------


def build_test_dataframe(class_to_idx):
    """Turn the test folder into a table with one row per image.

    The columns (path, filename, class, label) are the same as those of
    load_manifest, so ManifestDataset can be used as it is.
    """
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    # The sub-folders must be exactly the known classes (none missing, none extra)
    folder_names = sorted(p.name for p in TEST_DIR.iterdir() if p.is_dir())
    if folder_names != sorted(CLASS_NAMES):
        raise ValueError(f"Test folders {folder_names} are not the expected classes")

    rows, skipped = [], []
    for class_name in CLASS_NAMES:
        for path in sorted((TEST_DIR / class_name).iterdir()):  # sorted = stable order
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "path": path,
                        "filename": path.name,
                        "class": class_name,
                        "label": class_to_idx[class_name],
                    }
                )
            else:
                skipped.append(path.name)

    if skipped:
        print(f"Skipped {len(skipped)} non-image files, for example {skipped[:3]}")
    if not rows:
        raise ValueError(f"No images found in {TEST_DIR}")
    return pd.DataFrame(rows)


def check_no_leakage(test_df):
    """Stop if any test image is an exact copy of a train or validation image."""
    train_val_paths = list(load_manifest("train")["path"]) + list(
        load_manifest("val")["path"]
    )
    known = {file_fingerprint(path): path for path in train_val_paths}

    leaks = []
    for path in test_df["path"]:
        fingerprint = file_fingerprint(path)
        if fingerprint in known:
            leaks.append((path, known[fingerprint]))

    if leaks:
        raise RuntimeError(
            f"{len(leaks)} test images also exist in train/val, first: {leaks[0]}"
        )
    print("Leakage check passed: no test image matches a train/validation image")


# ---------------------------------------------------------------------------
# 6. D. Predict (the only time the model runs)
# ---------------------------------------------------------------------------


def predict_test_set(checkpoint, out_dir):
    """Build the test loader, run the model once, save and return the predictions."""
    class_names = checkpoint["class_names"]
    loss_type = checkpoint["loss_type"]
    image_size = tuple(checkpoint["image_size"])  # (height, width)
    mean = checkpoint["normalization"]["mean"]  # the SAME statistics as in training
    std = checkpoint["normalization"]["std"]

    set_seed(checkpoint["seed"])
    device = get_device()

    # Data and its checks first: they are cheap and must pass before the model runs
    test_df = build_test_dataframe(checkpoint["class_to_idx"])
    print(test_df["class"].value_counts().sort_index())  # compare with the audit report
    print(f"Total test images: {len(test_df)}")
    check_no_leakage(test_df)

    # Evaluation transform: letterbox + normalize, no augmentation
    _train_transform, eval_transform = build_transform(
        mean, std, image_size=image_size, with_aug=False
    )
    test_dataset = ManifestDataset(test_df, transform=eval_transform)
    # shuffle=False: image i of the loader is row i of test_df
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model_from_checkpoint(checkpoint, device)
    criterion = build_criterion(loss_type)
    result = evaluate(model, test_loader, criterion, device, loss_type, class_names)

    predictions = {
        "y_true": result["y_true"],
        "y_pred": result["y_pred"],
        "confidence": result["confidence"],
        "probs": result["probs"],
        "paths": np.array([str(p) for p in test_dataset.paths]),
        "loss": np.array(result["loss"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)  # created only after all checks passed
    np.savez(out_dir / PREDICTIONS_FILE, **predictions)
    return predictions


def load_saved_predictions(path):
    """Read the arrays saved by predict_test_set."""
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


# ---------------------------------------------------------------------------
# 7. E. Report helpers
# ---------------------------------------------------------------------------


def bootstrap_ci(y_true, y_pred, class_names, seed):
    """95% confidence interval for accuracy and macro-F1 by resampling predictions.

    Each repetition draws n images with replacement (the same image may appear
    several times), computes the metrics, and the 2.5th and 97.5th percentiles
    of all repetitions form the interval. The model is not run again.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accuracies, f1_scores = [], []

    for _ in range(N_RESAMPLES):
        picks = rng.integers(0, n, size=n)
        metrics = compute_metrics(y_true[picks], y_pred[picks], class_names)
        accuracies.append(metrics["accuracy"])
        f1_scores.append(metrics["macro_f1"])

    return {
        "accuracy": np.percentile(accuracies, [2.5, 97.5]).tolist(),
        "macro_f1": np.percentile(f1_scores, [2.5, 97.5]).tolist(),
    }


def make_report(
    predictions, summary, checkpoint, run_name, threshold, threshold_source, out_dir
):
    """E + F: build every table and figure from the saved predictions, then save."""
    class_names = checkpoint["class_names"]
    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    confidence = predictions["confidence"]
    paths = [str(p) for p in predictions["paths"]]

    # E1. Metrics
    metrics = compute_metrics(y_true, y_pred, class_names)
    print_metrics_report(metrics, title="TEST metrics")
    metrics_to_dataframe(metrics).to_csv(
        out_dir / "per_class_metrics.csv", encoding="utf-8-sig"
    )
    plot_per_class_metrics(
        metrics, out_dir / "per_class_metrics.png", title="Per-class metrics (test)"
    )

    # E2. The same checkpoint on validation and on test, side by side
    print_section("Validation vs test")
    val_metrics = summary["val_metrics"]
    comparison = {}
    for name in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
        difference = metrics[name] - val_metrics[name]  # negative = test is lower
        comparison[name] = {
            "validation": val_metrics[name],
            "test": metrics[name],
            "difference": difference,
        }
        print(
            f"{name:<16} val {val_metrics[name]:.4f} | "
            f"test {metrics[name]:.4f} | diff {difference:+.4f}"
        )

    # E3. How much can the test numbers move by chance?
    intervals = bootstrap_ci(y_true, y_pred, class_names, BOOTSTRAP_SEED)
    print(
        f"Accuracy 95% interval: [{intervals['accuracy'][0]:.4f}, {intervals['accuracy'][1]:.4f}]"
    )
    print(
        f"Macro-F1 95% interval: [{intervals['macro_f1'][0]:.4f}, {intervals['macro_f1'][1]:.4f}]"
    )

    # E4. Confusion matrices (counts and row-normalized)
    cm, cm_normalized = compute_confusion_matrix(y_true, y_pred, len(class_names))
    plot_confusion_matrices(
        cm, cm_normalized, class_names, out_dir / "confusion_matrix.png"
    )

    # E5. The mistakes, most confident first
    print_section("Mistakes")
    errors = find_misclassified(y_true, y_pred, confidence, class_names, paths=paths)
    errors.to_csv(out_dir / "misclassified.csv", index=False, encoding="utf-8-sig")
    print(f"Mistakes: {len(errors)} of {len(y_true)}")
    if len(errors) > 0:
        if len(errors) < GALLERY_SIZE:
            print(f"Only {len(errors)} mistakes, so the gallery shows all of them")
        plot_misclassified(
            errors.head(GALLERY_SIZE),
            out_dir / "misclassified_gallery.png",
            title="Most confident mistakes (test)",
        )
    # Describes the errors only. Merge decisions are made on validation data.
    top_confusions(cm, cm_normalized, class_names, top_k=10).to_csv(
        out_dir / "top_confusions.csv", index=False, encoding="utf-8-sig"
    )

    # E6. The validation threshold, applied unchanged
    print_section("Review threshold")
    selective = selective_metrics(y_true, y_pred, confidence, threshold)
    print(f"Threshold {threshold} ({threshold_source})")
    print(f"  coverage (decided automatically) : {format_value(selective['coverage'])}")
    print(
        f"  accuracy of accepted predictions : {format_value(selective['accuracy_accepted'])}"
    )
    print(
        f"  share of errors sent to review   : {format_value(selective['error_capture_rate'])}"
    )
    print(
        f"  number of errors sent to review   : {format_value(selective['num_review'])}"
    )
    print(f"  number of total errors   : {format_value(selective['num_errors'])}")

    # F. Save
    test_summary = {
        "run_name": run_name,
        "checkpoint": str(RUNS_ROOT / run_name / "best.pt"),
        "evaluated_at": datetime.now(ZoneInfo("Asia/Tehran")).isoformat(),
        "loss_type": checkpoint["loss_type"],
        "seed": checkpoint["seed"],
        "image_size": checkpoint["image_size"],
        "normalization": checkpoint["normalization"],
        "num_test_images": len(y_true),
        "images_per_class": {
            name: int((y_true == i).sum()) for i, name in enumerate(class_names)
        },
        "test_loss": float(predictions["loss"]),
        "test_metrics": metrics,
        "validation_vs_test": comparison,
        "confidence_intervals_95": intervals,
        "review_threshold": threshold,
        "review_threshold_source": threshold_source,
        "threshold_results": selective,
        "confusion_matrix_counts": cm,
        "confusion_matrix_row_normalized": cm_normalized,
    }
    save_json(test_summary, out_dir / "test_summary.json")


# ---------------------------------------------------------------------------
# 2. Main flow
# ---------------------------------------------------------------------------


def main():
    print_section("Mentors Exam Test...")

    run_name, summary = choose_best_run()
    threshold = read_validation_threshold(run_name, MANUAL_THRESHOLD)
    threshold_source = (
        "manual"
        if MANUAL_THRESHOLD is not None
        else "suggested_threshold.json (validation)"
    )
    out_dir = OUTPUT_ROOT / f"mentor_exam_on_{run_name}"

    # B. Load the frozen model's checkpoint
    checkpoint = load_checkpoint(run_name)

    # C + D. Run the model on the test set only if that was never done before
    predictions_path = out_dir / PREDICTIONS_FILE
    if predictions_path.exists():
        print(
            f"\n{Colors.MAGENTA}{Colors.BOLD}{predictions_path.name}{Colors.RESET} {Colors.MAGENTA}already exists: the model is NOT run again, "
            f"the report is rebuilt from the saved predictions.{Colors.MAGENTA}"
        )
        predictions = load_saved_predictions(predictions_path)
    else:
        predictions = predict_test_set(checkpoint, out_dir)

    # E + F. Report and save
    make_report(
        predictions, summary, checkpoint, run_name, threshold, threshold_source, out_dir
    )
    print(f"\nEverything is saved in {out_dir}")


if __name__ == "__main__":
    main()

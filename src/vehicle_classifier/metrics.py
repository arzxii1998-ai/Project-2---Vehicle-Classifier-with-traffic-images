# %%
"""Metrics for the project."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from vehicle_classifier.data import CLASS_NAMES
from vehicle_classifier.utils import (
    Colors,
    colorize,
    print_section,
    print_subsection,
    set_seed,
)

# ------------------
# 1. Predictions
# ------------------

LOSS_TYPES = ("ce", "bce")


def logits_to_predictions(logits: torch.Tensor, loss_type="ce"):
    """Convert raw model outputs to predicted classes, probabilities, and confidence.

    Parameters
    ----------
    logits : torch.Tensor, shape [N, num_classes]
        Raw model outputs (no softmax or sigmoid applied).
    loss_type : {"ce", "bce"}
        "ce"  : CrossEntropyLoss, scores are softmax probabilities that sum to 1.
        "bce" : BCEWithLogitsLoss, scores are independent sigmoid values that
                do not have to sum to 1.

    Returns
    -------
    y_pred : torch.Tensor, shape [N], int64
        argmax of the logits. Both loss types use the same rule.
    probs : torch.Tensor, shape [N, num_classes], float
        Softmax (ce) or sigmoid (bce) scores.
    confidence : torch.Tensor, shape [N], float
        The score of the predicted class.
    """

    if loss_type not in LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {loss_type!r}")
    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [N, num_classes], got {tuple(logits.shape)}"
        )

    logits = logits.detach()

    if loss_type == "ce":
        probs = torch.softmax(logits, dim=1)
    else:
        probs = torch.sigmoid(logits)

    y_pred = torch.argmax(logits, dim=1)
    confidence = probs.gather(1, y_pred.unsqueeze(1)).squeeze(1)
    return y_pred, probs, confidence


# ------------------
# 2. Classification metrics
# ------------------


def compute_metrics(y_true, y_pred, class_names: list = CLASS_NAMES):
    """Compute accuracy, macro and per-class precision / recall / F1.

    Parameters
    ----------
    y_true, y_pred : array-like of int, shape [N]
        True and predicted class indices (index i means class_names[i]).
    class_names : list of str
        Class names in index order.

    Returns
    -------
    dict with flat keys that train.py can monitor:
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "per_class"              -> {class_name: {"precision", "recall", "f1", "support"}},
        "lowest_recall_class", "lowest_precision_class", "num_samples".

    Every class always appears in the output, even if it is missing from the
    predictions (its precision is then 0 instead of an error).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must be 1-D with equal length, "
            f"got {y_true.shape} and {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty set of predictions")

    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "lowest_recall_class": class_names[int(np.argmin(recall))],
        "lowest_precision_class": class_names[int(np.argmin(precision))],
        "num_samples": int(y_true.size),
    }


# ------------------
# 3. Confusion matrices
# ------------------


def compute_confusion_matrix(y_true, y_pred, num_classes: int = len(CLASS_NAMES)):
    """Return the count confusion matrix and its row-normalized version.

    Rows are true classes and columns are predicted classes, so cm[i, j] is the
    number of images of class i that were predicted as class j.

    Returns
    -------
    cm : np.ndarray, shape [num_classes, num_classes], int
        Counts.
    cm_normalized : np.ndarray, shape [num_classes, num_classes], float
        Each row divided by its sum, so the diagonal is the per-class recall.
        A class with no true samples gets a row of zeros instead of NaN.
    """
    labels = list(range(num_classes))
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=labels)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros(cm.shape, dtype=np.float64),
        where=row_sums > 0,
    )
    return cm, cm_normalized


# ------------------
# 4. Running average
# ------------------


class AverageMeter:
    """Running weighted average, used for the loss over one epoch.

    The last batch of an epoch is usually smaller than the others, so a plain
    mean of batch losses would give it too much weight. Each batch loss is
    therefore weighted by the number of images in that batch.

    Usage:
        meter = AverageMeter()
        for images, labels in loader:
            loss = ...
            meter.update(loss.item(), n=images.size(0))
        epoch_loss = meter.average
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Start a new epoch."""
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        """Add one batch: value is the mean loss of the batch, n its size."""
        self.total += float(value) * n
        self.count += n

    @property
    def average(self):
        """Weighted mean of all values added since the last reset."""
        if self.count == 0:
            raise ValueError(
                "AverageMeter is empty: call update() before reading average"
            )
        return self.total / self.count


# ------------------
# 5. Reporting
# ------------------


def metrics_to_dataframe(metrics):
    """Turn the dict from compute_metrics into a per-class table.

    Returns a DataFrame with one row per class plus a final "macro avg" row,
    and the columns precision, recall, f1, support. For the macro row, support
    is the total number of samples.
    """
    columns = ["precision", "recall", "f1", "support"]

    per_class = pd.DataFrame.from_dict(metrics["per_class"], orient="index")[columns]
    macro = pd.DataFrame(
        [
            {
                "precision": metrics["macro_precision"],
                "recall": metrics["macro_recall"],
                "f1": metrics["macro_f1"],
                "support": metrics["num_samples"],
            }
        ],
        index=["macro avg"],
    )

    table = pd.concat([per_class, macro])
    table.index.name = "class"
    return table


def print_metrics_report(metrics, title="Metrics"):
    """Print overall scores, the per-class table, and the weakest classes."""
    print_section(title)

    print(
        f"Accuracy: {metrics['accuracy']:.4f}   "
        f"Macro precision: {metrics['macro_precision']:.4f}   "
        f"Macro recall: {metrics['macro_recall']:.4f}   "
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )

    print_subsection("Per-class metrics")
    table = metrics_to_dataframe(metrics)
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    print_subsection("Weakest classes")
    low_recall = metrics["lowest_recall_class"]
    low_precision = metrics["lowest_precision_class"]
    recall_value = metrics["per_class"][low_recall]["recall"]
    precision_value = metrics["per_class"][low_precision]["precision"]
    print(
        colorize(f"Lowest recall    : {low_recall} ({recall_value:.4f})", Colors.YELLOW)
    )
    print(
        colorize(
            f"Lowest precision : {low_precision} ({precision_value:.4f})", Colors.YELLOW
        )
    )


# ------------------
# 6. Human review (low-confidence predictions)
# ------------------


def _to_numpy(values):
    """Convert a torch tensor (any device) or array-like to a NumPy array."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def apply_review_threshold(confidence, threshold) -> bool:
    """Flag predictions whose confidence is below the threshold.

    Parameters
    ----------
    confidence : tensor or array-like, shape [N]
        Confidence of each prediction (from logits_to_predictions).
    threshold : float in [0, 1]
        Predictions with confidence strictly below it need human review.

    Returns
    -------
    needs_review : np.ndarray of bool, shape [N]

    The threshold must be chosen with validation data only, never with test data.
    This function can also be applied to images of an unseen class (such as
    "neysan"): the share of True values is then the share of unknown vehicles
    that the system correctly refuses to label.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return _to_numpy(confidence) < threshold


def selective_metrics(y_true, y_pred, confidence, threshold):
    """Measure the trade-off between automation and accuracy at one threshold.

    Predictions with confidence below the threshold go to a human; the rest are
    accepted automatically.

    Returns
    -------
    dict with:
        "threshold"
        "num_samples", "num_accepted", "num_review"
        "coverage"            : share of samples decided automatically
        "review_rate"         : share of samples sent to a human (1 - coverage)
        "accuracy_accepted"   : accuracy on the automatic decisions
        "accuracy_review"     : accuracy the model would have had on the reviewed samples
        "error_capture_rate"  : share of all model errors that were sent to review

    A value is None when its denominator is zero (for example, no sample is
    accepted, or the model made no errors).
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    needs_review = apply_review_threshold(confidence, threshold)
    if not (y_true.shape == y_pred.shape == needs_review.shape) or y_true.ndim != 1:
        raise ValueError("y_true, y_pred, and confidence must be 1-D with equal length")

    correct = y_true == y_pred
    accepted = ~needs_review
    num_samples = int(y_true.size)
    num_accepted = int(accepted.sum())
    num_review = int(needs_review.sum())
    num_errors = int((~correct).sum())

    return {
        "threshold": float(threshold),
        "num_samples": num_samples,
        "num_accepted": num_accepted,
        "num_review": num_review,
        "coverage": num_accepted / num_samples if num_samples else None,
        "review_rate": num_review / num_samples if num_samples else None,
        "accuracy_accepted": float(correct[accepted,].mean()) if num_accepted else None,
        "accuracy_review": float(correct[needs_review].mean()) if num_review else None,
        "error_capture_rate": (
            int((~correct & needs_review).sum()) / num_errors if num_errors else None
        ),
    }


# ------------------
# 7. Class relationships
# ------------------


def rank_confused_pairs(cm_normalized, class_names: list = CLASS_NAMES, top_k=None):
    """Rank pairs of classes by how much they are confused with each other.

    For every pair (i, j) the score is
        pair_confusion(i, j) = C_normalized[i, j] + C_normalized[j, i]
    where C_normalized is the row-normalized confusion matrix: the share of
    class i predicted as j, plus the share of class j predicted as i.

    Returns
    -------
    pandas.DataFrame, sorted from the most to the least confused pair, with columns:
        class_a, class_b, a_as_b (C[i, j]), b_as_a (C[j, i]), pair_confusion
    top_k keeps only the first top_k rows.

    The score only suggests which pairs to inspect. Merging classes also needs
    a look at the actual errors and a domain reason.
    """
    cm_normalized = np.asarray(cm_normalized)
    num_classes = len(class_names)
    if cm_normalized.shape != (num_classes, num_classes):
        raise ValueError(
            f"cm_normalized must have shape ({num_classes}, {num_classes}), "
            f"got {cm_normalized.shape}"
        )

    rows_idx, cols_idx = np.triu_indices(num_classes, k=1)
    pairs = pd.DataFrame(
        {
            "class_a": [class_names[i] for i in rows_idx],
            "class_b": [class_names[j] for j in cols_idx],
            "a_as_b": cm_normalized[rows_idx, cols_idx],
            "b_as_a": cm_normalized[cols_idx, rows_idx],
        }
    )
    pairs["pair_confusion"] = pairs["a_as_b"] + pairs["b_as_a"]
    pairs = pairs.sort_values("pair_confusion", ascending=False).reset_index(drop=True)

    return pairs if top_k is None else pairs.head(top_k)


# ------------------
# 8. Error analysis
# ------------------


def find_misclassified(y_true, y_pred, confidence, class_names, paths=None, top_k=None):
    """List the misclassified samples, most confident mistakes first.

    Parameters
    ----------
    y_true, y_pred : array-like of int, shape [N]
    confidence : tensor or array-like, shape [N]
        Confidence of each prediction.
    class_names : list of str
    paths : sequence of str, optional
        Image path of every sample, in the same order (for example
        val_loader.dataset.paths). If given, a "path" column is added.
    top_k : int, optional
        Keep only the first top_k rows.

    Returns
    -------
    pandas.DataFrame with columns sample_index, [path], true_class, pred_class,
    confidence. sample_index is the position of the sample in the evaluated
    set, so the image can be found again. Rows are sorted by confidence from
    high to low: confident mistakes are the most informative ones.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    confidence = _to_numpy(confidence)
    if not (y_true.shape == y_pred.shape == confidence.shape) or y_true.ndim != 1:
        raise ValueError("y_true, y_pred, and confidence must be 1-D with equal length")
    if paths is not None:
        paths = list(paths)
        if len(paths) != y_true.size:
            raise ValueError(f"paths has {len(paths)} items, expected {y_true.size}")

    wrong = np.flatnonzero(y_true != y_pred)
    table = pd.DataFrame(
        {
            "sample_index": wrong,
            "true_class": [class_names[i] for i in y_true[wrong]],
            "pred_class": [class_names[i] for i in y_pred[wrong]],
            "confidence": confidence[wrong],
        }
    )
    if paths is not None:
        table.insert(1, "path", [str(paths[i]) for i in wrong])

    table = table.sort_values("confidence", ascending=False).reset_index(drop=True)
    return table if top_k is None else table.head(top_k)


# ------------------
# 9. Self-test
# ------------------


def _raises(function, exception_type):
    """Return True if calling function() raises exception_type."""
    try:
        function()
    except exception_type:
        return True
    return False


def main():
    from sklearn.metrics import classification_report

    set_seed(42)
    print_section("metrics.py self-test")

    class_names = ["a", "b", "c", "d"]
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 0])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 0, 3, 0])
    confidence = np.array([0.90, 0.50, 0.95, 0.80, 0.99, 0.40, 0.60, 0.55, 0.90, 0.85])

    # -- logits_to_predictions ------------------------------------------------
    print_subsection("logits_to_predictions")
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 3.0, 1.0]], requires_grad=True)

    pred_ce, probs_ce, conf_ce = logits_to_predictions(logits, "ce")
    print(
        f"CE  predictions {pred_ce.tolist()}, row sums {probs_ce.sum(dim=1).tolist()}"
    )
    assert pred_ce.tolist() == [0, 1]
    assert torch.allclose(probs_ce.sum(dim=1), torch.ones(2))
    assert torch.allclose(conf_ce, probs_ce.max(dim=1).values)
    assert not probs_ce.requires_grad

    pred_bce, probs_bce, conf_bce = logits_to_predictions(logits, "bce")
    print(
        f"BCE predictions {pred_bce.tolist()}, row sums {probs_bce.sum(dim=1).tolist()}"
    )
    assert pred_bce.tolist() == pred_ce.tolist()
    assert torch.allclose(probs_bce, torch.sigmoid(logits.detach()))
    assert not torch.allclose(probs_bce.sum(dim=1), torch.ones(2))
    assert torch.allclose(conf_bce, probs_bce.max(dim=1).values)

    assert _raises(lambda: logits_to_predictions(logits, "mse"), ValueError)
    assert _raises(lambda: logits_to_predictions(logits[0], "ce"), ValueError)
    print("Both loss types give the same classes; only the scores differ")

    # -- compute_metrics ------------------------------------------------------
    print_subsection("compute_metrics")
    result = compute_metrics(y_true, y_pred, class_names)
    assert abs(result["accuracy"] - 0.6) < 1e-12
    assert result["lowest_recall_class"] == "d"
    assert result["lowest_precision_class"] == "a"

    perfect = compute_metrics(y_true, y_true, class_names)
    assert perfect["macro_f1"] == 1.0 and perfect["accuracy"] == 1.0

    missing = compute_metrics([0, 1, 2, 3], [0, 1, 2, 2], class_names)
    assert missing["per_class"]["d"]["precision"] == 0.0
    print("Hand-computed example, perfect predictions, and a never-predicted class: OK")

    rng = np.random.default_rng(42)
    names8 = [f"class{i}" for i in range(8)]
    true8 = rng.integers(0, 8, size=400)
    pred8 = np.where(rng.random(400) < 0.7, true8, rng.integers(0, 8, size=400))
    ours = compute_metrics(true8, pred8, names8)
    ref = classification_report(
        true8,
        pred8,
        labels=list(range(8)),
        target_names=names8,
        output_dict=True,
        zero_division=0,
    )
    for key, ref_key in (
        ("macro_precision", "precision"),
        ("macro_recall", "recall"),
        ("macro_f1", "f1-score"),
    ):
        assert abs(ours[key] - ref["macro avg"][ref_key]) < 1e-12
    for name in names8:
        assert abs(ours["per_class"][name]["f1"] - ref[name]["f1-score"]) < 1e-12
        assert ours["per_class"][name]["support"] == ref[name]["support"]
    print("Matches sklearn classification_report on 400 random predictions")

    assert _raises(lambda: compute_metrics([], [], class_names), ValueError)
    assert _raises(lambda: compute_metrics([0, 1], [0], class_names), ValueError)

    # -- compute_confusion_matrix ---------------------------------------------
    print_subsection("compute_confusion_matrix")
    cm, cm_normalized = compute_confusion_matrix(y_true, y_pred, 4)
    print(cm)
    assert cm.sum() == len(y_true)
    assert np.allclose(cm_normalized.sum(axis=1), 1.0)
    recalls = [result["per_class"][n]["recall"] for n in class_names]
    assert np.allclose(np.diag(cm_normalized), recalls)
    _, empty_row = compute_confusion_matrix([0, 1], [0, 1], 4)
    assert np.allclose(empty_row[2:], 0.0)
    print(
        "Counts sum to N, rows sum to 1, diagonal equals recall, empty class row is zeros"
    )

    # -- AverageMeter ---------------------------------------------------------
    print_subsection("AverageMeter")
    meter = AverageMeter()
    meter.update(1.0, n=32)
    meter.update(0.5, n=8)
    print(f"Weighted average: {meter.average:.4f} (plain mean would be 0.7500)")
    assert abs(meter.average - 0.9) < 1e-12
    meter.reset()
    assert _raises(lambda: meter.average, ValueError)

    # -- reporting ------------------------------------------------------------
    print_subsection("metrics_to_dataframe and print_metrics_report")
    table = metrics_to_dataframe(result)
    assert list(table.index) == class_names + ["macro avg"]
    assert table.loc["macro avg", "support"] == len(y_true)
    print_metrics_report(result, title="Example report")

    # -- review threshold -----------------------------------------------------
    print_subsection("apply_review_threshold and selective_metrics")
    flags = apply_review_threshold(confidence, 0.7)
    assert flags.tolist() == [
        False,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    ]
    selective = selective_metrics(y_true, y_pred, confidence, 0.7)
    print(selective)
    assert selective["num_review"] == 4
    assert selective["accuracy_accepted"] == 1.0
    assert selective["accuracy_review"] == 0.0
    assert selective["error_capture_rate"] == 1.0
    assert selective_metrics(y_true, y_pred, confidence, 0.0)["review_rate"] == 0.0
    assert selective_metrics(y_true, y_pred, confidence, 1.0)["coverage"] == 0.0
    assert (
        selective_metrics([0, 1], [0, 1], [0.9, 0.9], 0.5)["error_capture_rate"] is None
    )
    assert _raises(lambda: apply_review_threshold(confidence, 1.5), ValueError)
    print(
        "Hand-computed example and edge cases (threshold 0, threshold 1, no errors): OK"
    )

    # -- rank_confused_pairs --------------------------------------------------
    print_subsection("rank_confused_pairs")
    pairs = rank_confused_pairs(cm_normalized, class_names, top_k=3)
    print(pairs.to_string(float_format=lambda x: f"{x:.3f}"))
    assert pairs.loc[0, "class_a"] == "a" and pairs.loc[0, "class_b"] == "d"
    assert abs(pairs.loc[0, "pair_confusion"] - 2 / 3) < 1e-12
    assert len(rank_confused_pairs(cm_normalized, class_names)) == 6
    assert _raises(
        lambda: rank_confused_pairs(cm_normalized[:3], class_names), ValueError
    )

    # -- find_misclassified ---------------------------------------------------
    print_subsection("find_misclassified")
    fake_paths = [f"img_{i}.jpg" for i in range(len(y_true))]
    wrong = find_misclassified(
        y_true, y_pred, confidence, class_names, paths=fake_paths
    )
    print(wrong.to_string(float_format=lambda x: f"{x:.2f}"))
    assert len(wrong) == 4
    assert wrong["confidence"].is_monotonic_decreasing
    assert wrong.loc[0, "path"] == "img_6.jpg"
    assert len(find_misclassified(y_true, y_true, confidence, class_names)) == 0
    assert (
        len(find_misclassified(y_true, y_pred, confidence, class_names, top_k=2)) == 2
    )

    print_section("All metrics checks passed")


if __name__ == "__main__":
    main()

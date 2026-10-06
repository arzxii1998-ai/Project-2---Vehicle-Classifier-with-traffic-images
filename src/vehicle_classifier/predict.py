"""Predict the class of one vehicle image.

Usage
-----
    python -m vehicle_classifier.predict                 # interactive session
    python -m vehicle_classifier.predict path/to/img.jpg # JSON on stdout
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from vehicle_classifier.compare_runs import (
    RUNS_ROOT,  # noqa: F401
    build_comparison_table,
    load_summaries,
)
from vehicle_classifier.data import PROJECT_ROOT, build_transform
from vehicle_classifier.final_test import (
    build_model_from_checkpoint,
    load_checkpoint,
    read_validation_threshold,
)
from vehicle_classifier.metrics import apply_review_threshold, logits_to_predictions
from vehicle_classifier.model import get_device
from vehicle_classifier.utils import Colors, colorize

# ----------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
PRODUCTION_FILE = MODELS_DIR / "vehicle_classifier.pt"  # kept out of Git
DECIMALS = 4
BAR_WIDTH = 30

_production_predictor = None  # cache: the production file is read only once


# ----------------------------------------------------------------------
# 2. PART 1 - EXPORT THE PRODUCTION MODEL (validation data only)
# ----------------------------------------------------------------------
def list_runs():
    """All finished runs (best validation macro-F1 first), plus a loss_type column."""
    summaries = load_summaries()
    table = build_comparison_table(summaries)
    table["loss_type"] = [
        summaries[name]["config"]["loss_type"] for name in table["model"]
    ]
    return table


def choose_best_ce_run(table):
    """Name of the cross-entropy run with the best validation macro-F1."""
    ce_table = table[table["loss_type"] == "ce"]
    if len(ce_table) == 0:
        raise ValueError("There is no finished cross-entropy run")
    return ce_table.iloc[0]["model"]


def export_production_model(run_name, manual_threshold=None):
    """Write ONE self-contained file that holds everything predict needs."""
    threshold = read_validation_threshold(run_name, manual_threshold)
    checkpoint = load_checkpoint(run_name)

    if checkpoint["loss_type"] != "ce":
        raise ValueError("Only cross-entropy models can become the production model")

    checkpoint["review_threshold"] = threshold
    checkpoint["selected_from_run"] = run_name

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, PRODUCTION_FILE)
    return checkpoint


# ----------------------------------------------------------------------
# 3. PART 2 - PREDICT
# ----------------------------------------------------------------------
def make_predictor(checkpoint):
    """Build everything predict needs, once: model, device, transform, names, threshold."""
    if checkpoint["review_threshold"] is None:
        raise ValueError("This checkpoint has no review threshold")

    device = get_device()
    model = build_model_from_checkpoint(checkpoint, device)

    mean = checkpoint["normalization"]["mean"]
    std = checkpoint["normalization"]["std"]
    image_size = tuple(checkpoint["image_size"])  # (height, width)
    _train_transform, eval_transform = build_transform(
        mean, std, image_size=image_size, with_aug=False
    )

    return {
        "model": model,
        "device": device,
        "transform": eval_transform,
        "class_names": checkpoint["class_names"],
        "loss_type": checkpoint["loss_type"],
        "threshold": checkpoint["review_threshold"],
        "run_name": checkpoint.get("selected_from_run", "unknown"),
        "val_metrics": checkpoint["val_metrics"],
    }


def get_production_predictor():
    """Return the production predictor; create the model file if it is missing."""
    global _production_predictor
    if _production_predictor is not None:
        return _production_predictor

    if not PRODUCTION_FILE.exists():
        print("Production [CE] model is being chosen...", end="\r", file=sys.stderr)
        run_name = choose_best_ce_run(list_runs())
        export_production_model(run_name)
        print(" " * 60, end="\r", file=sys.stderr)
        print(
            "Production model created from: "
            + colorize(run_name, Colors.BOLD, Colors.GREEN),
            file=sys.stderr,
        )

    checkpoint = torch.load(PRODUCTION_FILE, map_location="cpu", weights_only=False)
    if checkpoint["loss_type"] != "ce":
        raise ValueError("The production file is not a cross-entropy model")
    _production_predictor = make_predictor(checkpoint)
    return _production_predictor


def predict(image_path, predictor=None):
    """Classify ONE image. Returns a dict that can be written as JSON.

    Never asks questions and never prints: it returns the result or raises.
    """
    if predictor is None:
        predictor = get_production_predictor()

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")
    try:
        image = Image.open(path).convert("RGB")
    except OSError as error:
        raise ValueError(f"{path} is not a readable image") from error

    tensor = predictor["transform"](image)  # (3, H, W)
    batch = tensor.unsqueeze(0).to(predictor["device"])  # (1, 3, H, W)

    with torch.no_grad():
        logits = predictor["model"](batch)  # (1, 8)

    index_tensor, probs, confidence = logits_to_predictions(
        logits.cpu(), predictor["loss_type"]
    )
    class_names = predictor["class_names"]
    predicted_class = class_names[int(index_tensor[0])]

    # decide on the RAW confidence (same rule as in the validation analysis)
    needs_review = bool(apply_review_threshold(confidence, predictor["threshold"])[0])

    probabilities = {
        name: round(float(p), DECIMALS) for name, p in zip(class_names, probs[0])
    }
    return {
        "predicted_class": predicted_class,
        "confidence": round(float(confidence[0]), DECIMALS),
        "probabilities": probabilities,
        "needs_review": needs_review,
    }


# ----------------------------------------------------------------------
# 4. SCREEN OUTPUT (interactive session only)
# ----------------------------------------------------------------------
def show_model_header(predictor):
    """One line: which model is active and how good it was on validation."""
    metrics = predictor["val_metrics"]
    print(
        colorize("Model: ", Colors.CYAN)
        + colorize(predictor["run_name"], Colors.BOLD, Colors.GREEN)
        + f" ({predictor['loss_type'].upper()})"
        + f" | val accuracy {metrics['accuracy']:.3f}"
        + f" | val macro-F1 {metrics['macro_f1']:.3f}"
        + f" | review threshold {predictor['threshold']:.2f}"
    )


def show_result(result, predictor):
    """Print the result in color, then the same result as JSON."""
    print()
    print(
        colorize("Prediction: ", Colors.CYAN)
        + colorize(result["predicted_class"], Colors.BOLD, Colors.GREEN)
    )
    print(colorize("Confidence: ", Colors.CYAN) + f"{result['confidence']:.4f}")
    if result["needs_review"]:
        print(
            colorize(
                "NEEDS HUMAN REVIEW (confidence below the threshold)",
                Colors.BOLD,
                Colors.YELLOW,
            )
        )
    else:
        print(colorize("Accepted automatically", Colors.GREEN))

    if predictor["loss_type"] == "bce":
        print(
            colorize(
                "BCE model: the scores are independent, they do NOT sum to 1.",
                Colors.YELLOW,
            )
        )

    print(colorize("Scores:", Colors.CYAN))
    for name, p in sorted(
        result["probabilities"].items(), key=lambda item: item[1], reverse=True
    ):
        bar = "█" * round(p * BAR_WIDTH)
        print(f"  {name:<10} {p:.4f} {bar}")

    print(colorize("JSON:", Colors.GRAY))
    print(colorize(json.dumps(result, indent=2), Colors.GRAY))


# ----------------------------------------------------------------------
# 5. INTERACTIVE SESSION
# ----------------------------------------------------------------------
def ask_threshold_by_hand():
    """Ask for a review threshold. Return a number in [0, 1], or None to cancel."""
    while True:
        text = input(
            "No validation threshold found for this run. "
            "Type one (0-1) or press Enter to cancel: "
        ).strip()
        if text == "":
            return None
        try:
            value = float(text)
        except ValueError:
            print(colorize("That is not a number.", Colors.YELLOW))
            continue
        if 0.0 <= value <= 1.0:
            return value
        print(colorize("The threshold must be between 0 and 1.", Colors.YELLOW))


def choose_other_model(current_predictor):
    """Option a: show every run and let the user pick one for this session."""
    table = list_runs()
    print(colorize("\nAll models (best macro-F1 first):", Colors.CYAN))
    for number, row in enumerate(table.itertuples(), start=1):
        print(
            f"  {number:>2}. {row.model} ({row.loss_type.upper()})  "
            f"accuracy {row.accuracy:.2f}  macro-F1 {row.macro_f1:.2f}"
        )

    text = input("Number of the model (Enter to cancel): ").strip()
    if not text.isdigit() or not 1 <= int(text) <= len(table):
        print("No change.")
        return current_predictor
    run_name = table.iloc[int(text) - 1]["model"]
    loss_type = table.iloc[int(text) - 1]["loss_type"]

    try:
        threshold = read_validation_threshold(run_name)
    except FileNotFoundError:
        threshold = ask_threshold_by_hand()
        if threshold is None:
            print("No change.")
            return current_predictor

    checkpoint = load_checkpoint(run_name)
    checkpoint["review_threshold"] = threshold  # in memory only
    checkpoint["selected_from_run"] = run_name
    new_predictor = make_predictor(checkpoint)

    if loss_type == "ce":
        answer = input("Save it as the production model? (y/n): ").strip().lower()
        if answer == "y":
            export_production_model(run_name, manual_threshold=threshold)
            print(colorize("Saved as the production model.", Colors.GREEN))
    else:
        print(
            colorize(
                "BCE models can be tried in this session, "
                "but are never saved as the production model.",
                Colors.YELLOW,
            )
        )
    return new_predictor


def interactive_session():
    """Title, production model, then a loop that reads image paths."""
    print(colorize(":::  Vehicle Classifier  :::", Colors.BOLD, Colors.CYAN))
    predictor = get_production_predictor()
    show_model_header(predictor)

    try:
        while True:
            print()
            text = input("Image path  |  a = change model  |  q = quit  >  ")
            text = text.strip().strip('"').strip("'")
            if text == "":
                continue
            if text.lower() == "q":
                break
            if text.lower() == "a":
                predictor = choose_other_model(predictor)
                show_model_header(predictor)
                continue

            try:
                result = predict(text, predictor)
            except (FileNotFoundError, ValueError) as error:
                print(
                    colorize(
                        f"Warning: {error}. Please enter a correct path.",
                        Colors.YELLOW,
                    )
                )
                continue
            show_result(result, predictor)
    except (KeyboardInterrupt, EOFError):
        print()
    print("Bye.")


# ----------------------------------------------------------------------
# 6. PART 3 - COMMAND LINE
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Classify one vehicle image.")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="image to classify; without it the interactive session starts",
    )
    args = parser.parse_args()

    if args.image_path is None:
        interactive_session()
        return

    # With a path: no questions, no colors. JSON on stdout, errors on stderr.
    try:
        result = predict(args.image_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

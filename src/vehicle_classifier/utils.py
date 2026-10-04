"""Shared helpers for the vehicle classifier project.

Contents
--------
1. Colors          : ANSI console colors and a plot palette.
2. Console output  : print_section, print_subsection, colorize.
3. Reproducibility : set_seed.
4. Small helpers   : format_duration, save_json, load_json.

This module has no imports from the rest of the project, so every other
module (data, model, metrics, train, predict) can import it safely.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1. Colors
# ---------------------------------------------------------------------------


class Colors:
    """ANSI escape codes for colored console text."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    DIM = "\033[2m"


# Set the environment variable NO_COLOR to switch console colors off.
USE_COLOR = "NO_COLOR" not in os.environ

# On Windows, an empty system call switches the console to ANSI mode.
if USE_COLOR and os.name == "nt":
    os.system("")

# Okabe-Ito palette: eight colors that stay distinguishable for color-blind readers.
PLOT_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# Fixed colors for training curves, so every figure uses the same convention.
CURVE_COLORS = {
    "train": "#0072B2",
    "val": "#D55E00",
    "test": "#009E73",
}


def get_class_color_map(class_names):
    """Map each class name to one palette color (same order as class_names)."""
    return {
        name: PLOT_PALETTE[i % len(PLOT_PALETTE)] for i, name in enumerate(class_names)
    }


# ---------------------------------------------------------------------------
# 2. Console output
# ---------------------------------------------------------------------------


def colorize(text, *styles):
    """Wrap text in ANSI styles, e.g. colorize("ok", Colors.GREEN, Colors.BOLD)."""
    if not USE_COLOR or not styles:
        return str(text)
    return "".join(styles) + str(text) + Colors.RESET


def print_section(title, width=40):
    """Print a major section header between two thick lines."""
    line = "=" * width
    print()
    print(colorize(line, Colors.BLUE))
    print(colorize(f" {title}", Colors.GREEN, Colors.BOLD))
    print(colorize(line, Colors.BLUE))


def print_subsection(title, width=40):
    """Print a minor header: a title followed by a thin line."""
    print()
    print(colorize(title, Colors.BLUE, Colors.BOLD))
    print(colorize("-" * min(width, max(len(title), 20)), Colors.CYAN, Colors.DIM))


# ---------------------------------------------------------------------------
# 3. Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed=42):
    """Seed every random number generator the project uses.

    Seeds Python's random module, NumPy, PyTorch (CPU and all CUDA devices),
    and the hash seed, then puts cuDNN in deterministic mode with benchmarking
    off. DataLoader workers are seeded separately (seed_worker in data.py).
    Returns the seed so it can be stored in a run configuration.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


# ---------------------------------------------------------------------------
# 4. Small helpers
# ---------------------------------------------------------------------------


def format_duration(seconds):
    """Format a duration in seconds as a short readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def _json_default(obj):
    """Convert types that json cannot handle (NumPy, torch, Path) to plain Python."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(obj, path):
    """Write obj to a JSON file, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    return path


def load_json(path):
    """Read a JSON file and return its content."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def main():
    import tempfile

    print_section("utils.py self-test")

    print_subsection("Console helpers")
    print(colorize("green text", Colors.GREEN), colorize("red text", Colors.RED))

    print_subsection("set_seed reproducibility")
    set_seed(42)
    a = torch.rand(3), np.random.rand(3), random.random()
    set_seed(42)
    b = torch.rand(3), np.random.rand(3), random.random()
    same = torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    print(f"Same seed gives identical numbers: {same}")
    assert same

    print_subsection("Duration formatting")
    for s in (5.64, 125, 3723):
        print(f"{s:>8} s -> {format_duration(s)}")

    print_subsection("JSON round trip with NumPy and torch values")
    data = {
        "f1": np.float32(0.5),
        "counts": np.array([1, 2, 3]),
        "t": torch.tensor([0.1, 0.2]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        out = save_json(data, Path(tmp) / "sub" / "test.json")
        loaded = load_json(out)
    print(loaded)
    assert loaded["counts"] == [1, 2, 3]

    print_section("All utils checks passed")


if __name__ == "__main__":
    main()

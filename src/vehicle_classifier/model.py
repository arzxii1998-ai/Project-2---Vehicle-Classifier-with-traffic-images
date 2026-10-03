"""Model definitions for the Project."""

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from vehicle_classifier.data import (
    IMAGE_SIZE,
    get_class_mapping,
    get_dataloaders,
    load_manifest,
    load_or_compute_norm_stats,
    print_section,
    print_subsection,
    set_seed,
)

# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------
PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)  # src/vehicle_classifier/model.py -> project root
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

DEFAULT_SEED = (
    42  # seed for weight initialization / training (separate from the split seed)
)

# Architecture knobs only.
DEFAULT_BASELINE_CONFIG = {
    "model_name": "baseline_cnn",
    "in_channels": 3,
    "channels": (
        32,
        64,
        128,
        256,
    ),  # one entry per conv block -> its length is the depth
    "pool_type": "max",  # "max" or "avg" for
    "dropout_p": 0.0,  # reference value; ablations use 0.3 / 0.5
}


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def _make_conv_block(
    in_channels: int, out_channels: int, pool_type: str
) -> nn.Sequential:
    """One block: Conv3x3 -> BatchNorm -> ReLU -> Pool2x2 (halves H and W)."""
    if pool_type == "max":
        pool = nn.MaxPool2d(kernel_size=2)
    elif pool_type == "avg":
        pool = nn.AvgPool2d(kernel_size=2)
    else:
        raise ValueError(f"pool_type must be 'max' or 'avg', got {pool_type!r}")

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        pool,
    )


class BaselineCNN(nn.Module):
    """Simple CNN: conv blocks -> global average pooling -> dropout -> linear.

    Returns raw logits of shape (batch, num_classes); softmax/sigmoid are applied outside.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        channels: tuple = (
            32,
            64,
            128,
            256,
        ),  # didn't used the value in the 'DEFAULT_BASELINE_CONFIG' in here.
        pool_type: str = "max",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels must contain at least one entry")

        blocks = []
        previous_block_channels = in_channels
        for current_block_channels in channels:
            blocks.append(
                _make_conv_block(
                    previous_block_channels, current_block_channels, pool_type
                )
            )
            previous_block_channels = current_block_channels
        self.features = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(previous_block_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # (N, C_last, H/2^k, W/2^k)
        x = self.global_pool(x)  # (N, C_last, 1, 1)
        return self.classifier(x)  # (N, num_classes)


def build_model(config: dict) -> nn.Module:
    """Build a model from a config dict (also used to rebuild from a checkpoint)."""
    name = config["model_name"]
    if name == "baseline_cnn":
        return BaselineCNN(
            num_classes=config["num_classes"],
            in_channels=config["in_channels"],
            channels=tuple(config["channels"]),  # JSON stores tuples as lists
            pool_type=config["pool_type"],
            dropout_p=config["dropout_p"],
        )
    raise ValueError(f"Unknown model_name: {name!r}")


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_device() -> torch.device:
    """Use the GPU if CUDA is available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def same_device(actual: torch.device, expected: torch.device) -> bool:
    """Return True when two torch devices refer to the same device."""
    if actual.type != expected.type:
        return False

    if actual.type == "cuda":
        actual_index = (
            torch.cuda.current_device() if actual.index is None else actual.index
        )
        expected_index = (
            torch.cuda.current_device() if expected.index is None else expected.index
        )
        return actual_index == expected_index

    return True


def print_layer_shapes(
    model: nn.Module, input_shape: tuple, save_path: Path | None = None
) -> list:
    """Print (and optionally save) a Markdown table of every layer's output shape.

    `input_shape` is (C, H, W) without the batch dimension.
    """
    rows = [("input", "-", tuple(input_shape), 0)]
    hooks = []

    def make_hook(name):
        def hook(module: nn.Module, inputs, output):
            n_params = sum(p.numel() for p in module.parameters(recurse=False))
            rows.append(
                (name, type(module).__name__, tuple(output.shape[1:]), n_params)
            )

        return hook

    # Attach a hook to every leaf module (a module with no children).
    for name, module in model.named_modules():
        if not list(module.children()):
            hooks.append(module.register_forward_hook(make_hook(name)))

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape, device=device))
    model.train(was_training)
    for hook in hooks:
        hook.remove()

    lines = [
        "| Layer | Type | Output shape (C×H×W) | Params |",
        "|---|---|---|---:|",
    ]
    for name, kind, shape, n_params in rows:
        shape_text = "×".join(str(s) for s in shape)
        lines.append(f"| {name} | {kind} | {shape_text} | {n_params:,} |")

    trainable, total = count_parameters(model)
    lines.append("")
    lines.append(f"Trainable parameters: {trainable:,} | Total parameters: {total:,}")

    print_subsection("Layer shapes")
    print("\n".join(lines))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text("\n".join(lines), encoding="utf-8")
    return rows


# ----------------------------------------------------------------------
# test_forward_pass
# ----------------------------------------------------------------------


def test_forward_pass(
    model: nn.Module,
    device: torch.device,
    train_loader,
    num_classes: int,
) -> None:
    """Test forward pass with a synthetic batch and a real training batch.

    The test checks:
    - output shape
    - output dtype
    - absence of NaN/Inf
    - output device
    - real-batch label range

    Raises:
        RuntimeError: If any check fails.
    """
    print_subsection("Forward pass test")

    # 1. Prepare model
    # --------------------------------------------------------------
    model = model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 2. Get one real batch from the DataLoader
    # --------------------------------------------------------------
    try:
        images, labels = next(iter(train_loader))
    except StopIteration as exc:
        raise RuntimeError("Forward pass test failed: train_loader is empty.") from exc

    images = images.to(device)
    labels = labels.to(device)

    batch_size, channels, height, width = images.shape
    input_shape = (batch_size, channels, height, width)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    print(f"Real batch input shape: {tuple(images.shape)}")
    print(f"Real batch labels shape: {tuple(labels.shape)}")

    # 3. Test 1: synthetic batch
    # --------------------------------------------------------------
    print_subsection("Test 1: Synthetic batch")

    synthetic_images = torch.randn(
        input_shape,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        synthetic_outputs = model(synthetic_images)

    expected_shape = (batch_size, num_classes)

    if synthetic_outputs.shape != expected_shape:
        raise RuntimeError(
            "Synthetic batch failed: "
            f"expected output shape {expected_shape}, "
            f"got {tuple(synthetic_outputs.shape)}."
        )

    print(f"✓ Output shape: {tuple(synthetic_outputs.shape)}")

    if synthetic_outputs.dtype != torch.float32:
        raise RuntimeError(
            "Synthetic batch failed: "
            f"expected dtype torch.float32, got {synthetic_outputs.dtype}."
        )

    print(f"✓ Output dtype: {synthetic_outputs.dtype}")

    if not torch.isfinite(synthetic_outputs).all():
        raise RuntimeError("Synthetic batch failed: output contains NaN or Inf values.")

    print("✓ Output contains no NaN/Inf")

    if not same_device(synthetic_outputs.device, device):
        raise RuntimeError(
            "Synthetic batch failed: "
            f"expected output device {device}, "
            f"got {synthetic_outputs.device}."
        )

    print(f"✓ Output device: {synthetic_outputs.device}")

    print("✓ Synthetic batch test passed")

    # 4. Test 2: real batch
    # --------------------------------------------------------------
    print_subsection("Test 2: Real batch")

    with torch.no_grad():
        real_outputs = model(images)

    if real_outputs.shape != expected_shape:
        raise RuntimeError(
            "Real batch failed: "
            f"expected output shape {expected_shape}, "
            f"got {tuple(real_outputs.shape)}."
        )

    print(f"✓ Output shape: {tuple(real_outputs.shape)}")

    if real_outputs.dtype != torch.float32:
        raise RuntimeError(
            "Real batch failed: "
            f"expected dtype torch.float32, got {real_outputs.dtype}."
        )

    print(f"✓ Output dtype: {real_outputs.dtype}")

    if not torch.isfinite(real_outputs).all():
        raise RuntimeError("Real batch failed: output contains NaN or Inf values.")

    print("✓ Output contains no NaN/Inf")

    if not same_device(real_outputs.device, device):
        raise RuntimeError(
            "Real batch failed: "
            f"expected output device {device}, "
            f"got {real_outputs.device}."
        )

    print(f"✓ Output device: {real_outputs.device}")

    # 5. Check real-batch labels
    # --------------------------------------------------------------
    if labels.ndim != 1:
        raise RuntimeError(
            "Real batch failed: "
            f"expected labels to have shape (batch,), "
            f"got {tuple(labels.shape)}."
        )

    if labels.shape[0] != batch_size:
        raise RuntimeError(
            f"Real batch failed: expected {batch_size} labels, got {labels.shape[0]}."
        )

    if labels.dtype not in (torch.int64, torch.long):
        raise RuntimeError(
            f"Real batch failed: expected integer labels, got {labels.dtype}."
        )

    if labels.numel() > 0:
        min_label = labels.min().item()
        max_label = labels.max().item()

        if min_label < 0 or max_label >= num_classes:
            raise RuntimeError(
                "Real batch failed: label values are outside the valid range. "
                f"Expected 0..{num_classes - 1}, "
                f"got {min_label}..{max_label}."
            )

        print(
            f"✓ Label range: {min_label}..{max_label} "
            f"(valid range: 0..{num_classes - 1})"
        )

    print("✓ Real batch test passed")

    # 6. GPU memory report
    # --------------------------------------------------------------
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"Peak GPU memory allocated: {peak_memory_mb:.2f} MB")

    print("✓ All forward-pass tests passed")


# ----------------------------------------------------------------------
# test_overfit_one_batch
# ----------------------------------------------------------------------
def test_overfit_one_batch(
    model: nn.Module,
    device: torch.device,
    train_loader,
    num_classes: int,
    max_steps: int = 500,
    learning_rate: float = 1e-3,
) -> None:
    """Try to overfit one real training batch.

    This is a training-pipeline sanity check, not a model-quality test.

    The model should be able to memorize a single batch. We use Adam and
    CrossEntropyLoss, without data augmentation, and stop early once the
    batch reaches 100% accuracy with a low loss.

    The model parameters, training/eval mode, and RNG states are restored
    after the test so this diagnostic does not affect later training.

    Raises:
        RuntimeError: If the batch cannot be overfit or an invalid numerical
            value is encountered.
    """
    print_subsection("Overfit one batch test")

    # --------------------------------------------------------------
    # 1. Get one real batch
    # --------------------------------------------------------------
    try:
        images, labels = next(iter(train_loader))
    except StopIteration as exc:
        raise RuntimeError("Overfit test failed: train_loader is empty.") from exc

    images = images.to(device)
    labels = labels.to(device)

    if images.ndim != 4:
        raise RuntimeError(
            "Overfit test failed: "
            f"expected images with shape (batch, C, H, W), "
            f"got {tuple(images.shape)}."
        )

    batch_size = images.shape[0]

    if batch_size == 0:
        raise RuntimeError("Overfit test failed: batch is empty.")

    if labels.ndim != 1 or labels.shape[0] != batch_size:
        raise RuntimeError(
            "Overfit test failed: "
            f"expected labels with shape ({batch_size},), "
            f"got {tuple(labels.shape)}."
        )

    if labels.dtype not in (torch.int64, torch.long):
        raise RuntimeError(
            f"Overfit test failed: expected integer labels, got {labels.dtype}."
        )

    if labels.numel() > 0:
        min_label = labels.min().item()
        max_label = labels.max().item()

        if min_label < 0 or max_label >= num_classes:
            raise RuntimeError(
                "Overfit test failed: label values are outside the valid range. "
                f"Expected 0..{num_classes - 1}, "
                f"got {min_label}..{max_label}."
            )

    print(f"Batch size: {batch_size}")
    print(f"Input shape: {tuple(images.shape)}")
    print(f"Labels shape: {tuple(labels.shape)}")
    print("Optimizer: Adam")
    print(f"Learning rate: {learning_rate}")
    print(f"Maximum steps: {max_steps}")

    # --------------------------------------------------------------
    # 2. Save state so the diagnostic is completely isolated
    # --------------------------------------------------------------
    original_state = deepcopy(model.state_dict())
    was_training = model.training

    cpu_rng_state = torch.get_rng_state()

    cuda_rng_state = None
    if device.type == "cuda":
        cuda_rng_state = torch.cuda.get_rng_state(device)

    # --------------------------------------------------------------
    # 3. Train repeatedly on the exact same batch
    # --------------------------------------------------------------
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    best_loss = float("inf")
    best_accuracy = 0.0
    best_step = 0

    success = False
    consecutive_successes = 0

    # We require multiple consecutive successful checks instead of
    # accepting one lucky 100% prediction.
    required_consecutive_successes = 5

    # Report frequently at first, then every 25 steps.
    report_every = 25

    try:
        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Overfit test failed: loss became NaN or Inf at step {step}."
                )

            loss.backward()

            # Check gradients before optimizer.step().
            for name, parameter in model.named_parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise RuntimeError(
                        "Overfit test failed: gradient for "
                        f"'{name}' contains NaN or Inf at step {step}."
                    )

            optimizer.step()

            # Evaluate the same batch after the optimizer update.
            with torch.no_grad():
                eval_outputs = model(images)
                eval_loss = criterion(eval_outputs, labels)

                predictions = eval_outputs.argmax(dim=1)
                correct = (predictions == labels).sum().item()
                accuracy = correct / batch_size

            if not torch.isfinite(eval_loss):
                raise RuntimeError(
                    "Overfit test failed: evaluation loss became "
                    f"NaN or Inf at step {step}."
                )

            loss_value = eval_loss.item()

            if loss_value < best_loss or accuracy > best_accuracy:
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_step = step

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_step = step

            if accuracy == 1.0 and loss_value <= 0.1:
                consecutive_successes += 1
            else:
                consecutive_successes = 0

            if (
                step == 1
                or step <= 10
                or step % report_every == 0
                or consecutive_successes == required_consecutive_successes
            ):
                print(
                    f"Step {step:>3} | "
                    f"Loss: {loss_value:.6f} | "
                    f"Accuracy: {accuracy * 100:6.2f}%"
                )

            if consecutive_successes >= required_consecutive_successes:
                success = True
                print(
                    f"✓ Batch overfit achieved at step {step}: "
                    f"100.00% accuracy, loss={loss_value:.6f}"
                )
                break

        if not success:
            print(
                f"Best result: accuracy={best_accuracy * 100:.2f}%, "
                f"loss={best_loss:.6f}, step={best_step}"
            )

            raise RuntimeError(
                "Overfit test failed: model could not reliably memorize "
                f"one training batch within {max_steps} steps. "
                f"Best accuracy={best_accuracy * 100:.2f}%, "
                f"best loss={best_loss:.6f}."
            )

    finally:
        # ----------------------------------------------------------
        # 4. Restore everything changed by this diagnostic test
        # ----------------------------------------------------------
        model.load_state_dict(original_state)
        model.zero_grad(set_to_none=True)
        model.train(was_training)

        torch.set_rng_state(cpu_rng_state)

        if device.type == "cuda" and cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)

    print("✓ Model state restored after overfit test")
    print("✓ Overfit one batch test passed")


# %%


def main() -> None:
    """Run model construction and basic forward-pass checks."""

    # -----------
    # Reproducibility, device
    # -----------
    print_section("Reproducibility, device")

    set_seed(DEFAULT_SEED)

    device = get_device()

    print_section("Model test")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # -----------
    # load data
    # -----------
    print_section("Loading data")

    class_to_idx, _idx_to_class = get_class_mapping()
    num_classes = len(class_to_idx)

    train_df = load_manifest("train")
    val_df = load_manifest("val")

    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")

    mean, std = load_or_compute_norm_stats(train_df)

    train_loader, _val_loader = get_dataloaders(
        train_df=train_df,
        val_df=val_df,
        mean=mean,
        std=std,
        batch_size=32,
        with_aug=False,
        image_size=IMAGE_SIZE,
        num_workers=4,
        seed=DEFAULT_SEED,
    )

    # -----------
    # Build model
    # -----------
    print_section("Building model...")

    model_config = {
        **DEFAULT_BASELINE_CONFIG,
        "num_classes": num_classes,
    }

    model = build_model(model_config)

    print(f"Model: {model_config['model_name']}")
    print(f"Pool type: {model_config['pool_type']}")
    print(f"Dropout: {model_config['dropout_p']}")
    print(f"Channels: {model_config['channels']}")

    # -----------
    # parameter counting
    # -----------
    print_section("counting parameters...")

    trainable_params, total_params = count_parameters(model)

    print_subsection("Parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")

    # -----------
    # Print layer shapes
    # -----------
    print_section("printing layers Shape...")

    print_layer_shapes(
        model,
        input_shape=(model_config["in_channels"], IMAGE_SIZE[0], IMAGE_SIZE[1]),
        save_path=REPORTS_DIR / "baseline_cnn_layer_shapes.md",
    )

    # -----------
    # Forward-pass tests
    # -----------
    print_section("Forward-pass tests...")

    test_forward_pass(model, device, train_loader, num_classes)

    print_subsection("Model test completed.")

    # -----------
    # Overfit one batch
    # -----------
    print_section("Overfit test...")

    test_overfit_one_batch(
        model=model,
        device=device,
        train_loader=train_loader,
        num_classes=num_classes,
        max_steps=500,
        learning_rate=1e-3,
    )

    print_subsection("Model test completed.")


if __name__ == "__main__":
    main()

"""ResNet18 transfer learning for the vehicle classifier.

One simple script that builds, trains, and saves three ResNet18 strategies.
It reuses the project's own data, loss, optimizer, scheduler, training loop,
and metric functions, and writes the same files as train.py
(config.json, history.json, best.pt, val_predictions.npz, summary.json),
so analyze.py and compare_runs work on these runs without any change.

Strategies (all start from ImageNet weights):
    frozen   : backbone frozen, only the new head is trained (feature extraction)
    finetune : starts from the best "frozen" checkpoint, then trains layer4 + head
               (layer4 uses a smaller learning rate than the head)
    full     : the whole network is trainable from the first epoch
               (the backbone uses a smaller learning rate than the head)

Run order:
    1. Cross-entropy: frozen -> finetune -> full
    2. The strategy with the best validation macro-F1 is trained once more with BCE.

Rules: only train and validation data are used; the test set is never touched.

Usage:
    python -m vehicle_classifier.resnet18
"""

import copy
import time
from typing import ClassVar

import numpy as np
import torch
from torch import nn
from torchvision import models

from vehicle_classifier.data import (
    CLASS_NAMES,
    IMAGE_SIZE,
    PROJECT_ROOT,
    get_dataloaders,
    load_manifest,
)
from vehicle_classifier.metrics import compute_confusion_matrix, print_metrics_report
from vehicle_classifier.model import count_parameters, get_device
from vehicle_classifier.train import (
    DEFAULT_CONFIG,
    build_criterion,
    build_optimizer,
    build_scheduler,
    evaluate,
    fit,
)
from vehicle_classifier.utils import (
    format_duration,
    load_json,
    print_section,
    print_subsection,
    save_json,
    set_seed,
)

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------

# Pretrained weights expect ImageNet statistics. They are also used as the
# letterbox padding color (the padding becomes ~0 after normalization).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

HEAD_LR = 1e-3  # new head
BACKBONE_LR = 1e-4  # pretrained layers that are trained (layer4, or the whole backbone)
WEIGHT_DECAY = 1e-2
DROPOUT_P = 0.3
EPOCHS = 50
OVERWRITE = False  # False: a finished run is skipped, True: it is trained again


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------


class TransferResNet(nn.Module):
    """ImageNet-pretrained ResNet18 with a new head: Dropout -> Linear(512, num_classes).

    mode:
        "frozen"   : only the head is trainable
        "finetune" : layer4 and the head are trainable
        "full"     : everything is trainable
    Returns raw logits of shape (batch, num_classes).
    """

    # Parameters whose name starts with one of these prefixes are trainable.
    TRAINABLE_PREFIXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "frozen": ("net.fc.",),
        "finetune": ("net.fc.", "net.layer4."),
        "full": ("net.",),
    }

    def __init__(self, mode, num_classes, dropout_p=0.0):
        super().__init__()
        if mode not in self.TRAINABLE_PREFIXES:
            raise ValueError(f"mode must be one of {list(self.TRAINABLE_PREFIXES)}")
        self.mode = mode

        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(net.fc.in_features, num_classes),
        )
        self.net = net

        # requires_grad is the single source of truth for what is trained
        prefixes = self.TRAINABLE_PREFIXES[mode]
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith(prefixes)

    def forward(self, x):
        return self.net(x)

    def train(self, mode=True):
        """Switch to train mode, but keep fully frozen BatchNorm layers in eval mode.

        train_one_epoch calls model.train() every epoch. BatchNorm would then update
        its running mean/var even when its weights are frozen, which silently changes
        the "frozen" backbone. Eval mode keeps the ImageNet statistics untouched.
        """
        super().train(mode)
        if mode:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d) and not any(
                    p.requires_grad for p in module.parameters()
                ):
                    module.eval()
        return self


def get_param_groups(model: nn.Module, config):
    """Split the trainable parameters into a head group and a backbone group."""
    head, backbone = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (head if name.startswith("net.fc.") else backbone).append(param)

    groups = [{"name": "head", "params": head, "lr": config["head_lr"]}]
    if backbone:
        name = "layer4" if model.mode == "finetune" else "backbone"
        groups.append({"name": name, "params": backbone, "lr": config["backbone_lr"]})
    return groups


def snapshot_frozen(model: nn.Module):
    """Copy every tensor that must not change during training (frozen weights + BatchNorm stats)."""
    prefixes = TransferResNet.TRAINABLE_PREFIXES[model.mode]
    return {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith(prefixes)
    }


# ---------------------------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------------------------


def make_config(run_name, mode, loss_type="ce", init_checkpoint=None):
    """Build the config of one run. Everything except mode/loss/init is identical across runs."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["run_name"] = run_name
    cfg["overwrite"] = OVERWRITE
    cfg["with_aug"] = True
    cfg["scheduler"] = "plateau"
    cfg["epochs"] = EPOCHS
    cfg["loss_type"] = loss_type
    cfg["lr"] = HEAD_LR
    cfg["weight_decay"] = WEIGHT_DECAY
    cfg["head_lr"] = HEAD_LR
    cfg["backbone_lr"] = BACKBONE_LR
    cfg["init_checkpoint"] = None if init_checkpoint is None else str(init_checkpoint)
    cfg["model"] = {
        "model_name": "resnet18",
        "mode": mode,
        "num_classes": len(CLASS_NAMES),
        "dropout_p": DROPOUT_P,
    }
    return cfg


def best_checkpoint_path(run_name):
    """Path of best.pt for a run name."""
    return PROJECT_ROOT / DEFAULT_CONFIG["output_root"] / run_name / "best.pt"


# ---------------------------------------------------------------------------
# 4. One complete run
# ---------------------------------------------------------------------------


def run_resnet(config):
    """Train one ResNet18 run and save the same files as train.run_experiment."""
    run_dir = PROJECT_ROOT / config["output_root"] / config["run_name"]
    if (run_dir / "summary.json").exists() and not config["overwrite"]:
        print(
            f"Skipping {config['run_name']}: finished run found (OVERWRITE = True to redo)."
        )
        return load_json(run_dir / "summary.json")
    run_dir.mkdir(parents=True, exist_ok=True)

    print_section(f"Training run: {config['run_name']}")
    set_seed(config["seed"])
    device = get_device()

    # Model first, so the random head initialization depends only on the seed
    model_cfg = config["model"]
    model = TransferResNet(
        model_cfg["mode"], model_cfg["num_classes"], model_cfg["dropout_p"]
    )
    if config["init_checkpoint"] is not None:
        checkpoint = torch.load(
            config["init_checkpoint"], map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    train_loader, val_loader = get_dataloaders(
        train_df=load_manifest("train"),
        val_df=load_manifest("val"),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        batch_size=config["batch_size"],
        with_aug=config["with_aug"],
        image_size=IMAGE_SIZE,
        num_workers=config["num_workers"],
        seed=config["seed"],
    )

    criterion = build_criterion(config["loss_type"])
    groups = get_param_groups(model, config)
    optimizer = build_optimizer(groups, config)
    scheduler = build_scheduler(optimizer, config)
    save_json(config, run_dir / "config.json")

    trainable, total = count_parameters(model)
    group_info = [
        {
            "name": g["name"],
            "num_params": sum(p.numel() for p in g["params"]),
            "lr": g["lr"],
        }
        for g in groups
    ]
    print_subsection("Setup")
    print(
        f"Mode {model_cfg['mode']} | loss {config['loss_type']} | start from: "
        f"{config['init_checkpoint'] or 'ImageNet weights'}"
    )
    print(f"Parameters: {trainable:,} trainable of {total:,}")
    for info in group_info:
        print(
            f"  group {info['name']:<9}: {info['num_params']:>10,} params | lr {info['lr']:.0e}"
        )

    # Training (validation picks the best epoch, the test set is never used)
    print_subsection("Training")
    frozen_before = snapshot_frozen(model)
    start = time.time()
    _history, best = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config,
        device,
        run_dir,
    )
    training_time = time.time() - start
    print(
        f"\nFinished in {format_duration(training_time)}. "
        f"Best epoch {best['epoch']} ({config['monitor']} = {best['score']:.4f})"
    )

    # The only essential check: frozen weights and BatchNorm statistics must be unchanged
    after = model.state_dict()
    changed = [k for k, v in frozen_before.items() if not torch.equal(v, after[k])]
    if changed:
        raise RuntimeError(
            f"Frozen tensors changed during training, e.g. {changed[:3]}"
        )

    # Final look at the best epoch (not the last epoch) on validation data
    checkpoint = torch.load(
        run_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    val = evaluate(
        model, val_loader, criterion, device, config["loss_type"], CLASS_NAMES
    )
    print_metrics_report(
        val["metrics"], title=f"Best validation result: {config['run_name']}"
    )

    # Files read by analyze.py
    cm, cm_normalized = compute_confusion_matrix(
        val["y_true"], val["y_pred"], len(CLASS_NAMES)
    )
    np.savez(
        run_dir / "val_predictions.npz",
        y_true=val["y_true"],
        y_pred=val["y_pred"],
        confidence=val["confidence"],
        probs=val["probs"],
        paths=np.array([str(p) for p in val_loader.dataset.paths]),
    )
    summary = {
        "run_name": config["run_name"],
        "best_epoch": best["epoch"],
        "monitor": config["monitor"],
        "best_score": best["score"],
        "val_loss": val["loss"],
        "val_metrics": val["metrics"],
        "confusion_matrix_counts": cm,
        "confusion_matrix_row_normalized": cm_normalized,
        "trainable_params": trainable,
        "total_params": total,
        "param_groups": group_info,
        "training_seconds": training_time,
        "config": config,
    }
    save_json(summary, run_dir / "summary.json")
    print(f"\nFiles saved in {run_dir}")
    return summary


# ---------------------------------------------------------------------------
# 5. Experiment order
# ---------------------------------------------------------------------------


def main():
    # Stage 1: cross-entropy with the three strategies
    results = {}

    results["frozen"] = run_resnet(make_config("resnet18_frozen", "frozen"))
    results["finetune"] = run_resnet(
        make_config(
            "resnet18_finetune",
            "finetune",
            init_checkpoint=best_checkpoint_path("resnet18_frozen"),
        )
    )
    results["full"] = run_resnet(make_config("resnet18_full", "full"))

    print_section("Stage 1 result (cross-entropy, validation)")
    for mode, summary in results.items():
        print(
            f"{mode:<9}: {summary['monitor']} = {summary['best_score']:.4f} "
            f"(epoch {summary['best_epoch']})"
        )

    # Stage 2: only the best strategy is trained again with BCE
    best_mode = max(results, key=lambda m: results[m]["best_score"])
    print(f"\nBest strategy: {best_mode} -> repeating it with BCE")

    init_checkpoint = None
    if best_mode == "finetune":
        # The BCE fine-tuning must start from a head that was trained with BCE too
        run_resnet(make_config("resnet18_frozen_bce", "frozen", loss_type="bce"))
        init_checkpoint = best_checkpoint_path("resnet18_frozen_bce")
    run_resnet(
        make_config(
            f"resnet18_{best_mode}_bce",
            best_mode,
            loss_type="bce",
            init_checkpoint=init_checkpoint,
        )
    )


if __name__ == "__main__":
    main()

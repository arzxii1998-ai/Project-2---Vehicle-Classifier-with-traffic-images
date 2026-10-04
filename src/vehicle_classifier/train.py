"""Training loop for the vehicle classifier project.

Contents
--------
1. Configuration       : DEFAULT_CONFIG
2. Building blocks     : build_criterion, prepare_targets, build_optimizer, build_scheduler
3. One epoch           : train_one_epoch, evaluate
4. Checkpoints         : get_normalization, save_checkpoint, load_checkpoint
5. Training loop       : fit
6. Experiment runner   : run_experiment, main

Rules this module follows
-------------------------
* Only the train and validation loaders exist here. The test set is never
  touched: every choice (best epoch, scheduler, threshold) uses validation data.
* Each run writes its files to outputs/runs/<run_name>/:
      config.json, history.json, best.pt, val_predictions.npz, summary.json
* One experiment changes one key of the config; everything else stays equal.
"""

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from vehicle_classifier.data import (
    CLASS_NAMES,
    DEFAULT_IMBALANCE,
    IMAGE_SIZE,
    PROJECT_ROOT,
    get_dataloaders,
    load_manifest,
    load_or_compute_norm_stats,
    simulate_imbalance,
)
from vehicle_classifier.metrics import (
    AverageMeter,
    compute_confusion_matrix,
    compute_metrics,
    logits_to_predictions,
    print_metrics_report,
)
from vehicle_classifier.model import DEFAULT_BASELINE_CONFIG, build_model, get_device
from vehicle_classifier.utils import (
    Colors,
    colorize,
    format_duration,
    print_section,
    print_subsection,
    save_json,
    set_seed,
)

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

# Reference baseline: max pooling, no dropout, no augmentation, no weight decay,
# fixed learning rate, cross-entropy. An ablation changes exactly one of these.
# AdamW with weight_decay=0 behaves exactly like Adam, so the weight-decay
# ablation (0 vs 1e-4) changes only that one number.
DEFAULT_CONFIG = {
    "run_name": "baseline_cnn",
    "output_root": "outputs/runs",  # relative to PROJECT_ROOT
    "overwrite": False,  # refuse to reuse the folder of an earlier run
    "seed": 42,
    # data
    "batch_size": 32,
    "num_workers": 4,
    "with_aug": False,
    # simulated class imbalance of the training split (None = use the full split);
    # otherwise {"keep_fractions": {class_name: share_kept}, "seed": int}
    "imbalance": None,
    # True = every training batch holds the same number of images of each class
    "balanced_batches": False,
    # model: the complete dict that build_model() expects
    # (model_name, in_channels, channels, pool_type, dropout_p, num_classes).
    # It is also stored in the checkpoint, so the model can be rebuilt from it.
    "model": {
        **DEFAULT_BASELINE_CONFIG,
        "num_classes": len(CLASS_NAMES),
    },
    # loss and optimization
    "loss_type": "ce",  # "ce" or "bce"
    "optimizer": "adamw",  # "adam" or "adamw"
    "lr": 1e-3,
    "weight_decay": 0.0,
    "scheduler": None,  # None, "step", or "plateau"
    "scheduler_params": {},  # overrides for SCHEDULER_DEFAULTS
    # training budget and best-checkpoint rule
    "epochs": 30,
    "monitor": "macro_f1",  # a key of compute_metrics, or "loss" (validation loss)
    "mode": "max",  # "max" for scores, "min" for loss
    "patience": None,  # stop after this many epochs without improvement; None = never
}

SCHEDULER_DEFAULTS = {
    "step": {"step_size": 10, "gamma": 0.5},
    "plateau": {"factor": 0.5, "patience": 3},
}


# ---------------------------------------------------------------------------
# 2. Building blocks
# ---------------------------------------------------------------------------


def build_criterion(loss_type):
    """Return the loss function for "ce" or "bce".

    "ce"  : CrossEntropyLoss, one mutually exclusive decision per image.
    "bce" : BCEWithLogitsLoss, eight independent yes/no decisions per image.
    The two loss values are on different scales and must not be compared
    directly; compare the metrics instead.
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"loss_type must be 'ce' or 'bce', got {loss_type!r}")


def prepare_targets(labels, loss_type, num_classes):
    """Convert integer labels [B] to the target format the loss expects.

    "ce"  : the integer labels themselves (int64, shape [B]).
    "bce" : one-hot float matrix of shape [B, num_classes].
    """
    labels = labels.long()
    if loss_type == "ce":
        return labels
    return F.one_hot(labels, num_classes).float()


def build_optimizer(params, config):
    """Return Adam or AdamW over params.

    params is an iterable of tensors, or a list of parameter-group dicts
    ({"params": ..., "lr": ...}), which the ResNet fine-tuning runs will use.
    """
    name = config["optimizer"].lower()
    if name == "adam":
        return optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])
    if name == "adamw":
        return optim.AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"])
    raise ValueError(
        f"optimizer must be 'adam' or 'adamw', got {config['optimizer']!r}"
    )


def build_scheduler(optimizer, config):
    """Return None, StepLR, or ReduceLROnPlateau (which watches validation loss)."""
    name = config["scheduler"]
    if name is None:
        return None
    if name not in SCHEDULER_DEFAULTS:
        raise ValueError(f"scheduler must be None, 'step', or 'plateau', got {name!r}")

    params = {**SCHEDULER_DEFAULTS[name], **config["scheduler_params"]}
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, **params)
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", **params)


# ---------------------------------------------------------------------------
# 3. One epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model, loader, criterion, optimizer, device, loss_type, num_classes
):
    """Run one pass over the training data and update the weights.

    Returns (mean loss, accuracy) of the epoch, both weighted by batch size.
    Accuracy here is measured while training (with augmentation and with
    dropout / BatchNorm in training mode), so it is only a rough guide.
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, prepare_targets(labels, loss_type, num_classes))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        n = labels.size(0)
        loss_meter.update(loss.item(), n)
        acc_meter.update((logits.argmax(dim=1) == labels).float().mean().item(), n)

    return loss_meter.average, acc_meter.average


@torch.no_grad()
def evaluate(model, loader, criterion, device, loss_type, class_names):
    """Evaluate the model on a loader without changing it.

    Returns a dict with:
        "loss"       : mean loss over the loader
        "metrics"    : the dict from compute_metrics
        "y_true", "y_pred", "confidence": NumPy arrays, shape [N]
        "probs"      : NumPy array, shape [N, num_classes]
    The order of samples follows the loader, so with shuffle=False index i
    matches loader.dataset.paths[i].
    """
    model.eval()
    loss_meter = AverageMeter()
    all_logits, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, prepare_targets(labels, loss_type, len(class_names)))

        loss_meter.update(loss.item(), labels.size(0))
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    y_true = torch.cat(all_labels).numpy()
    y_pred, probs, confidence = logits_to_predictions(logits, loss_type)
    y_pred = y_pred.numpy()

    return {
        "loss": loss_meter.average,
        "metrics": compute_metrics(y_true, y_pred, class_names),
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidence.numpy(),
        "probs": probs.numpy(),
    }


# ---------------------------------------------------------------------------
# 4. Checkpoints
# ---------------------------------------------------------------------------


def get_normalization(loader):
    """Read mean and std from the Normalize step of the loader's transform.

    Returns {"mean": [...], "std": [...]}, or None if the transform cannot be
    inspected. The checkpoint stores it so predict.py can rebuild the transform.
    """
    transform = getattr(loader.dataset, "transform", None)
    for step in getattr(transform, "transforms", []):
        if hasattr(step, "mean") and hasattr(step, "std"):
            return {
                "mean": [float(m) for m in step.mean],
                "std": [float(s) for s in step.std],
            }
    return None


def save_checkpoint(path, model, config, epoch, score, val_metrics, normalization):
    """Save the weights together with everything needed to rebuild and use the model."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": config["model"],  # complete dict for build_model()
        "class_names": list(CLASS_NAMES),
        "class_to_idx": {name: i for i, name in enumerate(CLASS_NAMES)},
        "image_size": list(IMAGE_SIZE),
        "normalization": normalization,
        "loss_type": config["loss_type"],
        "seed": config["seed"],
        "review_threshold": None,  # filled in after the validation analysis
        "config": config,
        "epoch": epoch,
        "monitor": config["monitor"],
        "best_score": score,
        "val_metrics": val_metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild the model from a checkpoint. Returns (model in eval mode, checkpoint dict)."""
    # weights_only=False: the checkpoint also holds plain Python objects (config,
    # metrics), and we only load files that this project wrote itself.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    assert checkpoint["class_names"] == list(CLASS_NAMES), (
        "Class names in the checkpoint differ from CLASS_NAMES in data.py"
    )
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    config,
    device,
    run_dir,
):
    """Train for config["epochs"] epochs and keep the best checkpoint.

    After every epoch it validates, steps the scheduler (using validation loss
    for "plateau"), logs one record to history.json, and saves best.pt when the
    monitored validation value improves.

    Returns (history, best), where history is a list with one dict per epoch and
    best holds the epoch, score, and full validation metrics of the best epoch.
    """
    epochs = config["epochs"]
    monitor, mode, patience = config["monitor"], config["mode"], config["patience"]
    num_classes = len(CLASS_NAMES)
    normalization = get_normalization(val_loader)

    sign = 1 if mode == "max" else -1  # compare sign * score so higher is always better
    best = {"epoch": None, "score": None, "metrics": None}
    history = []
    stale_epochs = 0
    width = len(str(epochs))

    for epoch in range(1, epochs + 1):
        start = time.time()
        lr_groups = [
            group["lr"] for group in optimizer.param_groups
        ]  # lr used in this epoch

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config["loss_type"],
            num_classes,
        )
        val = evaluate(
            model, val_loader, criterion, device, config["loss_type"], CLASS_NAMES
        )

        if scheduler is not None:
            if config["scheduler"] == "plateau":
                scheduler.step(val["loss"])
            else:
                scheduler.step()

        metrics = val["metrics"]
        score = val["loss"] if monitor == "loss" else metrics[monitor]
        improved = best["score"] is None or sign * score > sign * best["score"]
        if improved:
            best = {"epoch": epoch, "score": score, "metrics": metrics}
            save_checkpoint(
                run_dir / "best.pt", model, config, epoch, score, metrics, normalization
            )
            stale_epochs = 0
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val["loss"],
                "val_acc": metrics["accuracy"],
                "val_macro_precision": metrics["macro_precision"],
                "val_macro_recall": metrics["macro_recall"],
                "val_macro_f1": metrics["macro_f1"],
                "gap": val["loss"] - train_loss,  # validation loss minus train loss
                "lr": lr_groups[0],
                "lr_groups": lr_groups,
                "seconds": time.time() - start,
            }
        )
        save_json(
            history, run_dir / "history.json"
        )  # rewritten each epoch, safe if interrupted

        line = (
            f"Epoch {epoch:0{width}d}/{epochs} | "
            f"train loss {train_loss:.4f} {Colors.DIM}|{Colors.RESET} acc {train_acc:.4f} | "
            f"val loss {val['loss']:.4f} {Colors.DIM}|{Colors.RESET} acc {metrics['accuracy']:.4f} "
            f"{Colors.DIM}|{Colors.RESET} f1 {metrics['macro_f1']:.4f} | "
            f"lr {lr_groups[0]:.1e} | {format_duration(history[-1]['seconds'])}"
        )
        if improved:
            line += colorize(f"  <- best {monitor}", Colors.GREEN)
        print(line)

        if patience is not None and stale_epochs >= patience:
            print(f"Stopping early: no improvement for {patience} epochs")
            break

    return history, best


# ---------------------------------------------------------------------------
# 6. Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(config):
    """Run one complete experiment from a config dict and return its summary."""
    run_dir = PROJECT_ROOT / config["output_root"] / config["run_name"]
    if (run_dir / "history.json").exists() and not config["overwrite"]:
        raise FileExistsError(
            f"{run_dir} already holds a run. Change run_name or set overwrite=True."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    assert config["model"]["num_classes"] == len(CLASS_NAMES)

    print_section(f"Training run: {config['run_name']}")

    # Reproducibility, device, data, model
    set_seed(config["seed"])
    device = get_device()

    train_df = load_manifest("train")
    val_df = load_manifest("val")
    mean, std = load_or_compute_norm_stats(train_df)

    # Optional simulated imbalance. The normalization stats above come from the
    # FULL train split, so they do not change with the imbalance setting.
    if config.get("imbalance") is not None:
        train_df = simulate_imbalance(
            train_df,
            keep_fractions=config["imbalance"]["keep_fractions"],
            seed=config["imbalance"]["seed"],
            save_dir=run_dir,
        )

    train_loader, val_loader = get_dataloaders(
        train_df=train_df,
        val_df=val_df,
        mean=mean,
        std=std,
        batch_size=config["batch_size"],
        with_aug=config["with_aug"],
        image_size=IMAGE_SIZE,
        num_workers=config["num_workers"],
        seed=config["seed"],
        balanced_batches=config.get("balanced_batches", False),
    )
    model = build_model(config["model"]).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    criterion = build_criterion(config["loss_type"])
    optimizer = build_optimizer(
        [p for p in model.parameters() if p.requires_grad], config
    )
    scheduler = build_scheduler(optimizer, config)
    save_json(config, run_dir / "config.json")

    print_subsection("Setup")
    print(f"Device        : {device}")
    print(
        f"Samples       : {len(train_loader.dataset)} train, {len(val_loader.dataset)} validation"
    )
    print(f"Batches/epoch : {len(train_loader)} train, {len(val_loader)} validation")
    print(f"Parameters    : {trainable:,} trainable of {total:,}")
    print(
        f"Loss {config['loss_type']} | optimizer {config['optimizer']} "
        f"(lr {config['lr']}, weight decay {config['weight_decay']}) | "
        f"scheduler {config['scheduler']} | augmentation {config['with_aug']}"
    )
    print(
        f"Monitor       : {config['monitor']} ({config['mode']}) | epochs {config['epochs']}"
    )
    print(
        f"Batches       : "
        f"{'balanced' if config.get('balanced_batches', False) else 'standard (shuffled)'}"
        f" | imbalance {'simulated' if config.get('imbalance') else 'none'}"
    )

    # Training
    print_subsection("Training")
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

    # Final look at the best checkpoint on validation data
    best_model, _ = load_checkpoint(run_dir / "best.pt", device)
    val = evaluate(
        best_model, val_loader, criterion, device, config["loss_type"], CLASS_NAMES
    )
    print_metrics_report(
        val["metrics"], title=f"Best validation result: {config['run_name']}"
    )

    cm, cm_normalized = compute_confusion_matrix(
        val["y_true"], val["y_pred"], len(CLASS_NAMES)
    )
    arrays = {
        "y_true": val["y_true"],
        "y_pred": val["y_pred"],
        "confidence": val["confidence"],
        "probs": val["probs"],
    }
    paths = getattr(val_loader.dataset, "paths", None)
    if paths is not None:
        arrays["paths"] = np.array([str(p) for p in paths])
    np.savez(run_dir / "val_predictions.npz", **arrays)

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
        "training_seconds": training_time,
        "config": config,
    }
    save_json(summary, run_dir / "summary.json")
    print(f"\nFiles saved in {run_dir}")
    return summary


def main():
    # -- uncomment each you want ---------------------------------------------------

    # config = copy.deepcopy(DEFAULT_CONFIG)
    # config["overwrite"] = True
    # run_experiment(copy.deepcopy(config))

    # cfg = copy.deepcopy(DEFAULT_CONFIG)
    # cfg["run_name"] = "aug"
    # cfg["with_aug"] = True
    # run_experiment(cfg)

    # cfg = copy.deepcopy(DEFAULT_CONFIG)
    # cfg["run_name"] = "step_lr"
    # cfg["scheduler"] = "step"
    # run_experiment(cfg)

    # cfg = copy.deepcopy(DEFAULT_CONFIG)
    # cfg["run_name"] = "step_lr_&_aug_50epoch"
    # cfg["with_aug"] = True
    # cfg["scheduler"] = "step"
    # cfg["epochs"] = 50
    # run_experiment(cfg)

    # cfg = copy.deepcopy(DEFAULT_CONFIG)
    # cfg["run_name"] = "depth5"
    # cfg["model"]["channels"] = [32, 64, 128, 256, 256]
    # run_experiment(cfg)

    # cfg = copy.deepcopy(DEFAULT_CONFIG)
    # cfg["run_name"] = "full_ablation"
    # cfg["with_aug"] = True
    # cfg["scheduler"] = "step"
    # cfg["epochs"] = 50
    # cfg["model"]["channels"] = [32, 64, 128, 256, 256]
    # run_experiment(cfg)

    # cfg = copy.deepcopy(DEFAULT_CONFIG)
    # cfg["run_name"] = "full_ablation_DO.3"
    # cfg["with_aug"] = True
    # cfg["scheduler"] = "step"
    # cfg["epochs"] = 50
    # cfg["model"]["channels"] = [32, 64, 128, 256, 256]
    # cfg["dropout_p"] = 0.3
    # run_experiment(cfg)

    # cfg = copy.deepcopy(DEFAULT_CONFIG)
    # cfg["run_name"] = "full_ablation_DO.5"
    # cfg["with_aug"] = True
    # cfg["scheduler"] = "step"
    # cfg["epochs"] = 50
    # cfg["model"]["channels"] = [32, 64, 128, 256, 256]
    # cfg["dropout_p"] = 0.5
    # run_experiment(cfg)

    # cfg = copy.deepcopy(DEFAULT_CONFIG)
    # cfg["run_name"] = "full_ablation_bce"
    # cfg["with_aug"] = True
    # cfg["scheduler"] = "step"
    # cfg["epochs"] = 50
    # cfg["model"]["channels"] = [32, 64, 128, 256, 256]
    # cfg["loss_type"] = "bce"
    # run_experiment(cfg)

    # cfg = copy.deepcopy(DEFAULT_CONFIG)
    # cfg["overwrite"] = True
    # cfg["run_name"] = "full_ablation_avgpool"
    # cfg["with_aug"] = True
    # cfg["scheduler"] = "step"
    # cfg["epochs"] = 50
    # cfg["model"]["channels"] = [32, 64, 128, 256, 256]
    # cfg["pool_type"] = "avg"
    # run_experiment(cfg)

    # cfg = copy.deepcopy(DEFAULT_CONFIG)
    # cfg["overwrite"] = True
    # cfg["run_name"] = "full_ablation_wdecay-e-4"
    # cfg["with_aug"] = True
    # cfg["scheduler"] = "step"
    # cfg["epochs"] = 50
    # cfg["model"]["channels"] = [32, 64, 128, 256, 256]
    # cfg["weight_decay"] = 1e-4
    # run_experiment(cfg)

    # cfg = copy.deepcopy(DEFAULT_CONFIG)
    # cfg["overwrite"] = True
    # cfg["run_name"] = "full_ablation_wdecay-e-2"
    # cfg["with_aug"] = True
    # cfg["scheduler"] = "step"
    # cfg["epochs"] = 50
    # cfg["model"]["channels"] = [32, 64, 128, 256, 256]
    # cfg["weight_decay"] = 1e-2
    # run_experiment(cfg)

    # cfg = copy.deepcopy(DEFAULT_CONFIG)
    # cfg["overwrite"] = True
    # cfg["run_name"] = "full_ablation_plateau"
    # cfg["with_aug"] = True
    # cfg["scheduler"] = "plateau"
    # cfg["epochs"] = 50
    # cfg["model"]["channels"] = [32, 64, 128, 256, 256]
    # run_experiment(cfg)

    # cfg = copy.deepcopy(DEFAULT_CONFIG)
    # cfg["run_name"] = "full_ablation_bce_plateau_decoy"
    # cfg["with_aug"] = True
    # cfg["scheduler"] = "plateau"
    # cfg["epochs"] = 50
    # cfg["model"]["channels"] = [32, 64, 128, 256, 256]
    # cfg["loss_type"] = "bce"
    # cfg["weight_decay"] = 1e-2
    # run_experiment(cfg)

    """Simulated imbalance and balanced batches"""

    base = copy.deepcopy(DEFAULT_CONFIG)
    base["with_aug"] = True
    base["scheduler"] = "step"
    base["epochs"] = 50
    base["model"]["channels"] = [32, 64, 128, 256, 256]
    base["imbalance"] = copy.deepcopy(DEFAULT_IMBALANCE)

    cfg = copy.deepcopy(base)
    cfg["run_name"] = "imbalanced_standard"
    run_experiment(cfg)

    cfg = copy.deepcopy(base)
    cfg["run_name"] = "imbalanced_balanced"
    cfg["balanced_batches"] = True
    run_experiment(cfg)


if __name__ == "__main__":
    main()

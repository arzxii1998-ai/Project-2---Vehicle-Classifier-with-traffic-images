# %%
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import InterpolationMode, v2
from torchvision.utils import save_image  # noqa: F401, RUF100

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def print_section(title):
    print()
    print(f"{BLUE}{'=' * 70}{RESET}")
    print(f"{GREEN}{title.center(70)}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")


def print_subsection(title):
    print()
    print(f"{BLUE}--- {GREEN}{title}{BLUE} ---{RESET}")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_REL_PATH = "data/train_val_split_Manifest.csv"

CLASS_NAMES = [
    "ambulance",
    "autobus",
    "kamyun",
    "kamyunet",
    "minibus",
    "savari",
    "taxi",
    "vanet",
]

IMAGE_SIZE = (288, 224)  # (height, width)
DEFAULT_STATS_REL_PATH = "data/train_norm_stats.json"


def get_class_mapping():
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


def load_manifest(split: str, csv_path: str = DEFAULT_MANIFEST_REL_PATH):
    manifest_path = PROJECT_ROOT / csv_path
    df = pd.read_csv(manifest_path)
    df = df[df["split"] == split].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows found for split '{split}' in {csv_path}")

    class_to_idx, _ = get_class_mapping()
    unknown = set(df["class"]) - set(class_to_idx)
    if unknown:
        raise ValueError(f"Classes not in CLASS_NAMES: {sorted(unknown)}")

    df["label"] = df["class"].map(class_to_idx)
    df["path"] = df["path"].apply(lambda p: PROJECT_ROOT / p)
    return df


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


# ::::::::::
# Mean and STD funcs
# ::::::::::


def compute_mean_std(df, image_size=IMAGE_SIZE):
    """Per-channel mean/std over the real image content (no padding), scale 0-1."""
    height, width = image_size
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in df["path"]:
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (width, height), Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float64) / 255.0

        channel_sum += pixels.sum(axis=(0, 1))
        channel_sq_sum += (pixels**2).sum(axis=(0, 1))
        pixel_count += pixels.shape[0] * pixels.shape[1]

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_sq_sum / pixel_count - mean**2)
    return mean.tolist(), std.tolist()


def _train_fingerprint(df: pd.DataFrame):
    """Short hash identifying exactly which train images are in df."""
    keys = sorted((df["class"] + "/" + df["filename"]).tolist())
    return hashlib.md5("\n".join(keys).encode("utf-8")).hexdigest()


def load_or_compute_norm_stats(
    train_df: pd.DataFrame, image_size=IMAGE_SIZE, stats_path=DEFAULT_STATS_REL_PATH
):
    """Return (mean, std) from the JSON cache, recomputing if it is stale or missing."""

    full_path = PROJECT_ROOT / stats_path
    fingerprint = _train_fingerprint(train_df)

    if full_path.exists():
        try:
            with open(full_path) as f:
                stats = json.load(f)

            if stats["fingerprint"] == fingerprint and stats["image_size"] == list(
                image_size
            ):
                return stats["mean"], stats["std"]
        except (json.JSONDecodeError, KeyError):
            pass

    mean, std = compute_mean_std(train_df, image_size)
    stats = {
        "mean": mean,
        "std": std,
        "fingerprint": fingerprint,
        "image_size": list(image_size),
    }
    with open(full_path, "w") as f:
        json.dump(stats, f, indent=2)
    return mean, std


# ::::::::::
# letterbox resize class and funcs.
# ::::::::::


class LetterboxResize:
    def __init__(self, image_size, mean):
        self.height, self.width = image_size
        self.fill_color = tuple(round(m * 255) for m in mean)

    def __call__(self, image):
        return ImageOps.pad(
            image,
            (self.width, self.height),
            method=Image.Resampling.BICUBIC,
            color=self.fill_color,
            centering=(0.5, 0.5),
        )

    def __repr__(self):
        return f"LetterboxResize(size=({self.height}, {self.width}), fill={self.fill_color})"


# ::::::::::
# Transforms funcs.
# ::::::::::


def build_transform(mean, std, image_size=IMAGE_SIZE, with_aug=False):
    """Return train, Val transforms."""

    letterbox = LetterboxResize(image_size, mean)

    augmentations = []

    if with_aug:
        augmentations = [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.8, 1.2),
                interpolation=InterpolationMode.BILINEAR,
                fill=letterbox.fill_color,
            ),
            v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03),
        ]

    to_normalized_tensor = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]

    train_transform = v2.Compose([letterbox, *augmentations, *to_normalized_tensor])
    eval_transform = v2.Compose([letterbox, *to_normalized_tensor])
    return train_transform, eval_transform


def denormalize(tensor, mean, std):
    """Undo Normalize so a tensor can be viewed as a normal image (values in 0-1)."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return (tensor * std_t + mean_t).clamp(0, 1)


# ::::::::::
# Simulated imbalance and balanced batches
# ::::::::::

# Settings of the "standard vs balanced batches" experiment.
#   keep_fractions : share of the training images KEPT for each listed class
#                    (classes that are not listed keep 100%).
#   seed           : decides only WHICH images are kept. It is separate from the
#                    training seed, so every run sees the identical subset.
DEFAULT_IMBALANCE = {
    "keep_fractions": {"ambulance": 0.25, "kamyun": 0.25, "minibus": 0.25},
    "seed": 42,
}


def simulate_imbalance(train_df, keep_fractions, seed, save_dir=None):
    """Return a reproducible, class-imbalanced subset of the training split.

    For every class in keep_fractions, a fixed random share of its images is
    kept (chosen without replacement); all other classes are kept completely.
    Only the training split is reduced: validation and test stay untouched.

    The choice for one class depends only on (seed, class index), so editing the
    fraction of one class never changes which images are kept for another class.

    If save_dir is given, two files are written there:
        imbalanced_train_indices.csv : original_index, class, filename of every kept image
        imbalance_summary.json       : fractions, seed, class counts, subset fingerprint
    The fingerprint is identical for two runs exactly when their subsets are identical.

    Returns the subset as a new DataFrame with a fresh 0..n-1 index, so row i
    matches position i of a ManifestDataset built from it.
    """
    unknown = set(keep_fractions) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Classes not in CLASS_NAMES: {sorted(unknown)}")
    for name, fraction in keep_fractions.items():
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"keep fraction for {name!r} must be in (0, 1], got {fraction}"
            )

    kept_positions = []
    for class_index, name in enumerate(CLASS_NAMES):
        class_positions = np.flatnonzero((train_df["class"] == name).to_numpy())
        if class_positions.size == 0:
            raise ValueError(f"Class {name!r} has no training images")
        n_keep = max(1, round(class_positions.size * keep_fractions.get(name, 1.0)))
        rng = np.random.default_rng([seed, class_index])
        chosen = rng.choice(class_positions, size=n_keep, replace=False)
        kept_positions.extend(chosen.tolist())
    kept_positions.sort()  # keep the original row order

    subset = train_df.iloc[kept_positions]
    original_index = subset.index.to_numpy()
    subset = subset.reset_index(drop=True)

    before = train_df["class"].value_counts()
    after = subset["class"].value_counts()
    counts = {
        name: {"before": int(before.get(name, 0)), "after": int(after.get(name, 0))}
        for name in CLASS_NAMES
    }
    fingerprint = _train_fingerprint(subset)

    print_subsection("Simulated imbalance (training split only)")
    print(f"{'class':<12}{'before':>8}{'after':>8}")
    for name, count in counts.items():
        print(f"{name:<12}{count['before']:>8}{count['after']:>8}")
    sizes = [count["after"] for count in counts.values()]
    print(
        f"largest/smallest class: {max(sizes) / min(sizes):.1f} | "
        f"subset fingerprint: {fingerprint[:12]}"
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "original_index": original_index,
                "class": subset["class"].to_numpy(),
                "filename": subset["filename"].to_numpy(),
            }
        ).to_csv(save_dir / "imbalanced_train_indices.csv", index=False)
        with open(save_dir / "imbalance_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "keep_fractions": keep_fractions,
                    "seed": seed,
                    "counts": counts,
                    "fingerprint": fingerprint,
                },
                f,
                indent=2,
            )

    return subset


class BalancedBatchSampler(Sampler):
    """Batch sampler that puts the same number of images of every class in each batch.

    With batch_size=32 and 8 classes, every batch holds exactly 4 images per class.
    batch_size must be divisible by the number of classes.

    How the images are drawn: for each class, the sampler walks through a shuffled
    copy of that class's images, 4 at a time. When the copy is used up, the class is
    reshuffled and the walk starts again. A large class therefore shows (almost) each
    image once per pass, and a small class is cycled through several times: this is
    sampling with replacement across passes, but every image of a class is used about
    equally often. The augmentation of the Dataset makes each repeat look different.

    There is no natural epoch length (small classes never "run out"), so the epoch
    length is chosen by num_batches. Use len(standard_loader) to give the balanced
    and the standard run the same number of optimizer steps.

    Pass it to DataLoader as batch_sampler. DataLoader then refuses batch_size,
    shuffle, sampler, and drop_last, because the sampler already decides all of them.

    Reproducible: the same seed gives the same sequence of batches.
    """

    def __init__(
        self,
        labels,
        batch_size,
        num_batches,
        num_classes: int = len(CLASS_NAMES),
        seed=42,
    ):
        super().__init__()
        if batch_size % num_classes != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"the number of classes ({num_classes})"
            )
        if num_batches < 1:
            raise ValueError(f"num_batches must be at least 1, got {num_batches}")

        labels = np.asarray(labels)
        self.class_indices = [np.flatnonzero(labels == c) for c in range(num_classes)]
        missing = [c for c, idx in enumerate(self.class_indices) if idx.size == 0]
        if missing:
            raise ValueError(
                f"No training image for class indices {missing}: "
                "balanced batches need at least one image of every class"
            )

        self.num_classes = num_classes
        self.per_class = batch_size // num_classes
        self.num_batches = num_batches
        self.rng = np.random.default_rng(seed)  # keeps its state across epochs

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        orders = [self.rng.permutation(idx) for idx in self.class_indices]
        positions = [0] * self.num_classes

        for _ in range(self.num_batches):
            batch = []
            for c in range(self.num_classes):
                needed = self.per_class
                while needed > 0:
                    if positions[c] == len(orders[c]):  # class used up: reshuffle
                        orders[c] = self.rng.permutation(self.class_indices[c])
                        positions[c] = 0
                    take = min(needed, len(orders[c]) - positions[c])
                    batch.extend(orders[c][positions[c] : positions[c] + take].tolist())
                    positions[c] += take
                    needed -= take
            yield self.rng.permutation(
                batch
            ).tolist()  # mix the classes inside the batch


# ::::::::::
# Reproducibility and DataLoaders
# ::::::::::


def set_seed(seed: int = 42, deterministic: bool = True):
    """Seeding every random number generator used in the project."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed numpy and random inside each DataLoader worker from torch's worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(
    train_df,
    val_df,
    mean,
    std,
    batch_size=32,
    with_aug=False,
    image_size=IMAGE_SIZE,
    num_workers=4,
    seed=42,
    balanced_batches=False,
):
    """Build (train_loader, val_loader).

    balanced_batches=False (default): the train loader shuffles the images.
    balanced_batches=True : every train batch holds the same number of images of
        each class (see BalancedBatchSampler). The number of batches per epoch
        equals that of the standard loader, so both modes take the same number of
        optimizer steps. The validation loader is never changed.
    """

    train_tf, val_tf = build_transform(
        mean, std, image_size=image_size, with_aug=with_aug
    )

    train_ds = ManifestDataset(train_df, transform=train_tf)
    val_ds = ManifestDataset(val_df, transform=val_tf)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
    }

    if balanced_batches:
        batch_sampler = BalancedBatchSampler(
            labels=train_df["label"].to_numpy(),
            batch_size=batch_size,
            num_batches=math.ceil(len(train_ds) / batch_size),
            seed=seed,
        )
        # DataLoader forbids batch_size together with batch_sampler, so the
        # train loader gets the shared options without batch_size.
        train_kwargs = {k: v for k, v in loader_kwargs.items() if k != "batch_size"}
        train_loader = DataLoader(
            train_ds, generator=generator, batch_sampler=batch_sampler, **train_kwargs
        )
    else:
        train_loader = DataLoader(
            train_ds, generator=generator, shuffle=True, **loader_kwargs
        )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader


# ::::::::::
# Visual and speed checks
# ::::::::::


def show_samples(
    images,
    labels,
    mean,
    std,
    preds=None,
    confidences=None,
    max_images=12,
    ncols=4,
    save_path=None,
):
    """Plot normalized image tensors with true labels (and predictions, if given)."""

    _, idx_to_class = get_class_mapping()
    n = min(max_images, len(images))
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3 * ncols, 3.6 * nrows), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")

    for i in range(n):
        ax = axes.flat[i]
        ax.imshow(denormalize(images[i], mean, std).permute(1, 2, 0).numpy())

        title = f"true: {idx_to_class[int(labels[i])]}"
        color = "black"
        if preds is not None:
            title += f"\npred: {idx_to_class[int(preds[i])]}"
            if confidences is not None:
                title += f" ({float(confidences[i]):.2f})"
            color = "green" if int(preds[i]) == int(labels[i]) else "red"
        ax.set_title(title, fontsize=9, color=color)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    plt.close(fig)


def time_epochs(loader, epochs=2):
    """Return the seconds taken by each full pass over the loader (loading only)."""
    times = []
    for _ in range(epochs):
        start = time.perf_counter()
        for _images, _labels in loader:
            pass
        times.append(time.perf_counter() - start)
    return times


#  %%
if __name__ == "__main__":
    class_to_idx, _ = get_class_mapping()
    print_section("1. labeling ...")
    for name, label in class_to_idx.items():
        print(f"- {name:<10} ---> label: {label:<10}")

    # -----------
    # loading Train and Val Manifest and save in a DF
    # -----------

    print_section("2.loading Train and Val ...")
    print_subsection("1. Train:")

    train_df = load_manifest("train")

    print(train_df.head())
    print("All paths exist:", train_df["path"].map(lambda p: p.exists()).all())

    print_subsection("2. Val")

    val_df = load_manifest("val")

    print(val_df.head())
    print("All paths exist:", val_df["path"].map(lambda p: p.exists()).all())

    # -----------
    # using a Class for getting images and labels based on the DFs
    # -----------

    print_subsection("3. Dataset check (no transform)")
    train_ds = ManifestDataset(train_df)
    print("Number of samples:", len(train_ds))
    image, label = train_ds[0]
    print(
        "Type:",
        type(image),
        "| size:",
        image.size,
        "| mode:",
        image.mode,
        "| label:",
        label,
    )

    # -----------
    # computing or loading Train images STD & Mean, for normalization.
    # -----------

    print_section("4. Train mean/std (per channel) ... ")
    mean, std = load_or_compute_norm_stats(train_df)
    print("mean:", [round(m, 4) for m in mean])
    print("std :", [round(s, 4) for s in std])

    # -----------
    # Letterbox Resizing with padding in mean color (Testing)
    # -----------

    print_section("5. Letterbox Resizing testing ... ")
    letterbox = LetterboxResize(IMAGE_SIZE, mean)
    print(letterbox)

    sizes = [Image.open(p).size for p in train_df["path"]]
    ratios = [w / h for w, h in sizes]
    widest = ratios.index(max(ratios))
    tallest = ratios.index(min(ratios))
    print(f"widest path: {train_df['path'][widest]}")
    print(f"tallest path: {train_df['path'][tallest]}")

    height, width = IMAGE_SIZE
    for name, i in [("widest", widest), ("tallest", tallest)]:
        original = Image.open(train_df["path"][i]).convert("RGB")
        result = letterbox(original)
        assert result.size == (IMAGE_SIZE[1], IMAGE_SIZE[0]), (
            f"{name}: got {result.size}, expected {(width, height)} (width, height)"
        )
        print(f"{name}: original {original.size} -> {result.size}")
        # result.show()

    # -----------
    # Transform Testing
    # -----------

    print_section("6. Transform Testing ... ")

    train_tf, val_tf = build_transform(mean, std, image_size=IMAGE_SIZE, with_aug=True)

    print("train_tf:\n", train_tf, sep="")
    print("val_tf:\n", val_tf, sep="")

    sample_path = train_df.loc[train_df["class"] == "minibus", "path"].iloc[15]
    sample = Image.open(sample_path).convert("RGB")

    x = val_tf(sample)
    print("shape:", tuple(x.shape), "| dtype:", x.dtype)
    print("min/max:", round(x.min().item(), 3), round(x.max().item(), 3))

    views = [denormalize(train_tf(sample), mean, std) for _ in range(8)]
    save_path = PROJECT_ROOT / "outputs" / "debug" / "augmentation_grid.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(views, save_path, nrow=4)
    print("Saved:", save_path)

    # -----------
    # DataLoader Testing
    # -----------

    print_section("7. DataLoader testing ... ")

    def first_batch(seed):
        set_seed(seed)
        train_loader, _ = get_dataloaders(
            train_df, val_df, mean, std, batch_size=32, with_aug=True, seed=seed
        )

        return next(iter(train_loader))

    images_a, labels_a = first_batch(42)
    images_b, labels_b = first_batch(42)
    images_c, labels_c = first_batch(7)

    print("images:", tuple(images_a.shape), images_a.dtype)
    print("labels:", tuple(labels_a.shape), labels_a.dtype)
    print("class counts in batch:", torch.bincount(labels_a, minlength=8).tolist())
    print(
        "same seed  -> identical batch:",
        torch.equal(images_a, images_b) and torch.equal(labels_a, labels_b),
    )
    print("other seed -> identical labels:", torch.equal(labels_a, labels_c))

    train_loader, val_loader = get_dataloaders(train_df, val_df, mean, std)
    print("batches -> train:", len(train_loader), "| val:", len(val_loader))

    # -----------
    # Labelled samples (no augmentation)
    # -----------

    print_section("8. Labelled samples check ... ")
    plain_loader, _ = get_dataloaders(train_df, val_df, mean, std, with_aug=False)
    images, labels = next(iter(plain_loader))
    batch_save_path = PROJECT_ROOT / "outputs" / "debug" / "train_batch_samples.png"
    show_samples(images, labels, mean, std, save_path=batch_save_path)
    print("Saved:", batch_save_path)

    # -----------
    # num_workers benchmark
    # -----------

    print_section("9. num_workers benchmark ... ")
    print("CPU cores:", os.cpu_count())
    for workers in [0, 2, 4]:
        loader, _ = get_dataloaders(
            train_df, val_df, mean, std, with_aug=True, num_workers=workers
        )
        times = time_epochs(loader, epochs=2)
        print(f"num_workers={workers}: epoch times = {[round(t, 1) for t in times]} s")

    # -----------
    # Simulated imbalance and balanced batches
    # -----------

    print_section("10. Simulated imbalance and balanced batches ... ")

    keep = DEFAULT_IMBALANCE["keep_fractions"]
    imb_seed = DEFAULT_IMBALANCE["seed"]
    subset = simulate_imbalance(train_df, keep, imb_seed)
    subset_again = simulate_imbalance(train_df, keep, imb_seed)
    assert subset["path"].tolist() == subset_again["path"].tolist()
    assert set(subset["path"]) <= set(train_df["path"])
    print("same seed -> identical subset: True | subset is part of train: True")

    labels_np = subset["label"].to_numpy()
    n_batches = len(
        get_dataloaders(subset, val_df, mean, std, num_workers=0)[0]
    )  # batches of the standard loader
    sampler = BalancedBatchSampler(labels_np, batch_size=32, num_batches=n_batches)
    batches = list(sampler)
    per_class_counts = [np.bincount(labels_np[b], minlength=8) for b in batches]
    assert len(batches) == n_batches == len(sampler)
    assert all((counts == 4).all() for counts in per_class_counts)
    print(f"{n_batches} batches, every batch has exactly 4 images per class: True")

    again = list(BalancedBatchSampler(labels_np, 32, n_batches))
    assert again == batches
    print("same seed -> identical batches: True")

    bal_loader, bal_val_loader = get_dataloaders(
        subset, val_df, mean, std, with_aug=True, balanced_batches=True
    )
    images, labels = next(iter(bal_loader))
    print(
        "real balanced batch, class counts:",
        torch.bincount(labels, minlength=8).tolist(),
    )
    assert len(bal_val_loader.dataset) == len(val_df)
    print("validation loader unchanged: True")
# %%

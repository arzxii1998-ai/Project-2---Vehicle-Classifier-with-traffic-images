# %%
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, v2
from torchvision.utils import save_image

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
    Val_df = load_manifest("val")
    print(Val_df.head())
    print("All paths exist:", Val_df["path"].map(lambda p: p.exists()).all())

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
    # %%
    train_tf, val_tf = build_transform(mean, std, image_size=IMAGE_SIZE, with_aug=True)
    print("train_tf:\n", train_tf, sep="")
    print("val_tf:\n", Val_df, sep="")

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

# %%

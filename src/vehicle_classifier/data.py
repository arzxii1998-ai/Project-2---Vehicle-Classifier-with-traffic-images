# %%
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

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


#  %%
if __name__ == "__main__":
    class_to_idx, _ = get_class_mapping()
    print_section("labeling ...")
    for name, label in class_to_idx.items():
        print(f"- {name:<10} ---> label: {label:<10}")

    print_section("loding Train and Val")
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

# %%

import json
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from vehicle_classifier import duplicate_detection as dd

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

SEED = 42


def print_section(title):
    print()
    print(f"{BLUE}{'=' * 70}{RESET}")
    print(f"{GREEN}{title.center(70)}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")


def print_subsection(title):
    print()
    print(f"{BLUE}--- {GREEN}{title}{BLUE} ---{RESET}")


def save_the_df_CSV(
    the_df: DataFrame, filename: str, folder_name: str = "data"
) -> Path:
    output_path = PROJECT_ROOT / folder_name / (filename + ".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    the_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return output_path


# %%

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_POOL_DIR = PROJECT_ROOT / "dataset" / "Combined Dataset" / "train"
DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# %%


def load_image_DF(
    train_dir: Path = DEFAULT_TRAIN_POOL_DIR, extensions: set[str] = DEFAULT_EXTENSIONS
) -> pd.DataFrame:
    class_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())

    rows = []

    for class_dir in class_dirs:
        class_name = class_dir.name

        for file_path in sorted(class_dir.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in extensions:
                continue

            relative_path = file_path.resolve().relative_to(PROJECT_ROOT).as_posix()

            rows.append(
                {"path": relative_path, "class": class_name, "filename": file_path.name}
            )

    df = pd.DataFrame(rows, columns=["path", "class", "filename"])

    if df.empty:
        raise ValueError(f"No images found in '{train_dir}'. Check the path.")

    return df


# %%


def hash_image_df(
    image_df: pd.DataFrame,
    hash_method: str = dd.DEFAULT_HASH_METHOD,
    hash_size: int = dd.DEFAULT_HASH_SIZE,
) -> pd.DataFrame:
    if hash_method not in dd.HASH_METHODS:
        raise ValueError(
            f"Unknown hash_method '{hash_method}'. Choose one of: {sorted(dd.HASH_METHODS)}"
        )

    hash_fn = dd.HASH_METHODS[hash_method]
    hashes = []
    for relative_path in image_df["path"]:
        full_path = PROJECT_ROOT / relative_path
        with Image.open(full_path) as img:
            hashes.append(str(hash_fn(img, hash_size=hash_size)))

    result = image_df.copy()
    result["phash"] = hashes
    return result


# %%
similarity_threshold_DEFAULT = 0.82


def assign_group_ids(
    image_df: pd.DataFrame,
    hash_fn=hash_image_df,
    hash_method: str = dd.DEFAULT_HASH_METHOD,
    similarity_threshold: float = similarity_threshold_DEFAULT,
) -> pd.DataFrame:

    hashed_df = hash_fn(image_df, hash_method=hash_method).reset_index(drop=True)

    bits = dd._to_bit_matrix(hashed_df["phash"])
    max_distance = dd._max_distance(similarity_threshold, bits.shape[1])
    idx_a, idx_b, distances = dd._find_similar_pairs(bits, max_distance)

    clusters, root_of = dd._cluster_images(
        n_images=len(hashed_df),
        idx_a=idx_a,
        idx_b=idx_b,
        distances=distances,
        cluster_max_distance=max_distance,
    )

    group_ids = [root_of.get(i, i) for i in range(len(hashed_df))]

    result = hashed_df.drop(columns=["phash"])
    result["group_id"] = group_ids

    n_grouped = sum(len(members) for members in clusters.values())
    print(
        f"Found {len(clusters)} groups covering {n_grouped} images (similarity >= {similarity_threshold:.2f})"
    )

    return result


# %%
n_split = 5


def split_train_val(
    grouped_df: DataFrame, n_split: int = n_split, seed: int = SEED
) -> DataFrame:

    grouped_df = grouped_df.reset_index(drop=True)

    X = grouped_df["path"]
    y = grouped_df["class"]
    groups = grouped_df["group_id"]

    splitter = StratifiedGroupKFold(n_splits=n_split, shuffle=True, random_state=seed)

    _train_idx, val_idx = next(splitter.split(X, y, groups))

    result = grouped_df.copy()
    result["split"] = "train"
    result.loc[val_idx, "split"] = "val"

    print_subsection("splitting result:")
    print("value count:\n", result["split"].value_counts(), sep="")
    print()
    print(
        f"{YELLOW}stratified check:\n{RESET}",
        pd.crosstab(result["class"], result["split"], normalize="index"),
        sep="",
    )
    print()

    return result


def validate_split(df: pd.DataFrame) -> None:
    train_paths = set(df.loc[df["split"] == "train", "path"])
    val_paths = set(df.loc[df["split"] == "val", "path"])
    assert train_paths.isdisjoint(val_paths), "Some path is in both train and val!"

    train_groups = set(df.loc[df["split"] == "train", "group_id"])
    val_groups = set(df.loc[df["split"] == "val", "group_id"])
    assert train_groups.isdisjoint(val_groups), (
        "Some group_id is in both train and val!"
    )

    assert len(df) == df["path"].nunique(), "Duplicate paths found in the manifest!"

    print("All split-validity checks passed.")


# %%


def save_split_config(
    seed: int, n_splits: int, similarity_threshold: float, class_counts: dict
) -> Path:
    config = {
        "seed": seed,
        "n_splits": n_splits,
        "similarity_threshold": similarity_threshold,
        "class_counts": class_counts,
    }
    output_path = PROJECT_ROOT / "configs" / "split_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2))
    print(f"Saved config to {output_path}")
    return output_path


# %%


def main():
    train_pool_df = load_image_DF()
    print(train_pool_df["class"].value_counts())
    print(f"\nTotal: {len(train_pool_df)} images")
    print()
    print(train_pool_df.info())

    print_section("Now grouping...")

    grouped_df = assign_group_ids(
        train_pool_df, similarity_threshold=similarity_threshold_DEFAULT
    )
    print(grouped_df.head())

    print(YELLOW, "-" * 20, RESET)

    save_the_df_CSV(grouped_df, "grouped_train_pool")

    groups_NUM = grouped_df["group_id"].unique()
    multi_groups_NUM = (grouped_df["group_id"].value_counts() > 1).sum()
    print(f"\nUnique group_ids: {groups_NUM}")
    print(f"Groups with 2+ images: {multi_groups_NUM}")

    print_section("splitting...")
    split_df = split_train_val(grouped_df)
    print("head (5):\n", split_df.head())
    print()

    save_the_df_CSV(split_df, "train_val_split_Manifest")

    print_section("validating splits:...")
    validate_split(split_df)

    print_section("saving Json setting ... ")
    save_split_config(SEED, n_split, similarity_threshold_DEFAULT, class_counts=8)


if __name__ == "__main__":
    main()

# %%

"""
Near-duplicate image detection based on perceptual hashing (pHash / aHash /
dHash / wHash).

Typical workflow (called from a notebook):

    hashed = compute_phashes("../dataset/cleaned_V2", hash_method="phash")
    report = build_duplicate_report(hashed, similarity_threshold=0.9)

    display(HTML(table_to_html(report)))   # one table, links + thumbnails

    save_duplicate_report(report, "../data/duplicate_pairs_report_V2.csv")
    save_label_corrections(report, "../data/label_corrections_auto_duplicates_V2.csv")

    compare_images(
        r"dataset\cleaned_V2\train\kamyun\a.jpg",
        r"dataset\cleaned_V2\test\kamyun\b.jpg",
    )

Nothing is deleted or moved on disk. Every decision is written to a table.
"""

from __future__ import annotations

import base64
import html
import io
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_HASH_SIZE = 16  # 16 x 16 = 256-bit hash
DEFAULT_HASH_METHOD = "phash"

# Every pair at least this similar appears in the report table.
DEFAULT_SIMILARITY_THRESHOLD = 0.9
# Only pairs at least this similar can put images into the same cluster.
# 1.0 means identical hashes, which is transitive, so clusters can never chain.
DEFAULT_CLUSTER_SIMILARITY = 1.0

# Hash models available for compute_phashes() and compare_images().
HASH_METHODS = {
    "ahash": imagehash.average_hash,
    "phash": imagehash.phash,
    "dhash": imagehash.dhash,
    "whash": imagehash.whash,
}

DEFAULT_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
    ".tif",
    ".tiff",
}

# Lower number = more protected. The keeper of a duplicate cluster is the member
# from the split with the lowest number: test is frozen, train comes second,
# unclean is the most expendable. Unknown split names get the lowest protection.
SPLIT_PRIORITY = {"test": 0, "train": 1, "unclean": 2}
UNKNOWN_SPLIT_PRIORITY = 99

# This file lives in <project>/src/<package>/, so the project root is two
# levels above the package folder. Used only as a fallback base for
# project-relative paths passed to compare_images() / _project_relative().
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_REPORT_PATH = DATA_DIR / "duplicate_pairs_report.csv"
DEFAULT_LABEL_CORRECTIONS_PATH = DATA_DIR / "label_corrections_auto_duplicates.csv"

# Possible values of the `auto_decision` column (A = first image, B = second image)
DECISION_KEEP_A_EXCLUDE_B = "keep A, exclude B"
DECISION_EXCLUDE_A_KEEP_B = "exclude A, keep B"
DECISION_EXCLUDE_BOTH = "exclude A, exclude B"  # both are duplicates of a third image
DECISION_NEEDS_REVIEW = "needs review"

_PAIR_DECISIONS = {
    ("keep", "exclude"): DECISION_KEEP_A_EXCLUDE_B,
    ("exclude", "keep"): DECISION_EXCLUDE_A_KEEP_B,
    ("exclude", "exclude"): DECISION_EXCLUDE_BOTH,
}

REPORT_COLUMNS = [
    "similarity",
    "cluster_num",
    "same_class",
    "class_a",
    "class_b",
    "same_split",
    "split_a",
    "split_b",
    "filename_a",
    "path_a",
    "filename_b",
    "path_b",
    "auto_decision",
]


# ---------------------------------------------------------------------------
# 1. Hashing (now walks a dataset folder directly — no image_metadata input)
# ---------------------------------------------------------------------------


def _iter_dataset_images(dataset_dir: Path, image_extensions: set[str]):
    """
    Yields (split, class_name, filename, full_path) for every image file
    under dataset_dir, walked in a deterministic (sorted) order:
    split -> class -> filename. Mirrors the layout used by
    apply_corrections.py: <dataset_dir>/<split>/<class>/<filename>.
    """
    for split_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        split = split_dir.name
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            for file_path in sorted(class_dir.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in image_extensions:
                    continue
                yield split, cls, file_path.name, file_path


def compute_phashes(
    dataset_dir: str | Path,
    hash_method: str = DEFAULT_HASH_METHOD,
    hash_size: int = DEFAULT_HASH_SIZE,
    image_extensions: set[str] | None = None,
) -> pd.DataFrame:
    """Walk a dataset folder (split/class/filename layout) and hash every image.

    Parameters
    ----------
    dataset_dir : path to the dataset root, e.g. "../dataset/cleaned_V2".
        Expected layout: <dataset_dir>/<split>/<class>/<filename>.
    hash_method : one of "ahash", "phash", "dhash", "whash".
    hash_size : hash grid size (must be a multiple of 4 so the hex-encoded
        hash fills whole bytes — this matters because build_duplicate_report()
        unpacks the hex string back into individual bits).
    image_extensions : file extensions to treat as images. Defaults to the
        common set (.jpg, .jpeg, .png, .bmp, .webp, .gif, .tif, .tiff).

    Returns
    -------
    DataFrame with columns: path, split, class, filename, phash
    (one row per successfully hashed image; unreadable files are skipped
    and counted in the printed summary, never raise).
    """
    if hash_method not in HASH_METHODS:
        raise ValueError(
            f"Unknown hash_method '{hash_method}'. Choose one of: {sorted(HASH_METHODS)}"
        )
    if hash_size % 4 != 0:
        raise ValueError(
            "hash_size must be a multiple of 4 so the hash fills whole bytes."
        )

    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")

    extensions = image_extensions or DEFAULT_IMAGE_EXTENSIONS
    hash_fn = HASH_METHODS[hash_method]

    rows = []
    failed = 0
    for split, cls, filename, path in _iter_dataset_images(dataset_dir, extensions):
        try:
            with Image.open(path) as img:
                digest = str(hash_fn(img, hash_size=hash_size))
        except Exception:
            failed += 1
            continue
        rows.append(
            {
                "path": str(path),
                "split": split,
                "class": cls,
                "filename": filename,
                "phash": digest,
            }
        )

    hashed = pd.DataFrame(rows, columns=["path", "split", "class", "filename", "phash"])

    print(
        f"Scanned: {dataset_dir}  |  hash method: {hash_method} ({hash_size}x{hash_size})  |  "
        f"Hashed images: {len(hashed)}  |  Failed to hash: {failed}"
    )
    return hashed


def _to_bit_matrix(hex_hashes: pd.Series) -> np.ndarray:
    """Convert hex hash strings to an (n_images, n_bits) matrix of 0/1 values."""
    packed = np.array(
        [np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hex_hashes]
    )
    return np.unpackbits(packed, axis=1)


def _max_distance(similarity: float, n_bits: int) -> int:
    """Largest Hamming distance that still counts as 'similar enough'."""
    return int(np.floor((1 - similarity) * n_bits + 1e-9))


# ---------------------------------------------------------------------------
# 2. Pairwise comparison
# ---------------------------------------------------------------------------


def _find_similar_pairs(
    bits: np.ndarray,
    max_distance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare every image only with the images after it in the list.

    Returns three aligned arrays: index of image A, index of image B, and the
    Hamming distance between them (only pairs with distance <= max_distance).
    """
    idx_a: list[int] = []
    idx_b: list[int] = []
    distances: list[int] = []

    for i in range(len(bits) - 1):
        # Hamming distance between image i and every image after it
        distance_to_rest = np.count_nonzero(bits[i + 1 :] != bits[i], axis=1)

        close = np.flatnonzero(distance_to_rest <= max_distance)
        idx_a.extend([i] * len(close))
        idx_b.extend((close + i + 1).tolist())
        distances.extend(distance_to_rest[close].tolist())

    return (
        np.array(idx_a, dtype=int),
        np.array(idx_b, dtype=int),
        np.array(distances, dtype=int),
    )


# ---------------------------------------------------------------------------
# 3. Clusters and automatic decisions
# ---------------------------------------------------------------------------


def _cluster_images(
    n_images: int,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    distances: np.ndarray,
    cluster_max_distance: int,
) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Group images into clusters using only the pairs that meet the cluster criterion.

    A cluster can have any number of members (2, 3, 5, ...).

    Returns
    -------
    clusters : root image index -> list of member indexes (only clusters with 2+ images)
    root_of  : member image index -> root image index
    """
    parent = list(range(n_images))

    def find_root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    is_cluster_edge = distances <= cluster_max_distance
    edge_a = idx_a[is_cluster_edge].tolist()
    edge_b = idx_b[is_cluster_edge].tolist()

    # Union-find: merge the two images of every cluster-level pair into one cluster
    for a, b in zip(edge_a, edge_b):
        root_a, root_b = find_root(a), find_root(b)
        if root_a != root_b:
            parent[root_b] = root_a

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in sorted(set(edge_a) | set(edge_b)):
        clusters[find_root(i)].append(i)

    root_of = {i: root for root, members in clusters.items() for i in members}
    return dict(clusters), root_of


def _decide_members(
    hashed: pd.DataFrame,
    bits: np.ndarray,
    clusters: dict[int, list[int]],
    cluster_max_distance: int,
) -> dict[int, str]:
    """Decide keep / exclude / review for every image that belongs to a cluster.

    Rules
    -----
    * Cluster contains different classes           -> every member is 'review'.
    * Chain cluster (some member is not similar
      enough to the reference copy)                -> every member is 'review'.
      (Cannot happen with cluster similarity 1.0, kept as a safety net.)
    * Otherwise the member from the most protected split (test > train > unclean,
      then path order) is kept and all other members are 'exclude'.
    """

    def sort_key(i: int):
        priority = SPLIT_PRIORITY.get(hashed.at[i, "split"], UNKNOWN_SPLIT_PRIORITY)
        return (priority, hashed.at[i, "path"])

    decisions: dict[int, str] = {}
    for members in clusters.values():
        ordered = sorted(members, key=sort_key)
        keeper = ordered[0]

        has_mixed_classes = len({hashed.at[i, "class"] for i in ordered}) > 1
        is_chain = any(
            np.count_nonzero(bits[i] != bits[keeper]) > cluster_max_distance
            for i in ordered
        )

        if has_mixed_classes or is_chain:
            decisions.update({i: "review" for i in ordered})
        else:
            decisions[keeper] = "keep"
            decisions.update({i: "exclude" for i in ordered[1:]})

    return decisions


def _pair_decision(decision_a: str, decision_b: str) -> str:
    """Translate the per-image decisions of both images into one text for the pair."""
    return _PAIR_DECISIONS.get((decision_a, decision_b), DECISION_NEEDS_REVIEW)


# ---------------------------------------------------------------------------
# 4. The single report table
# ---------------------------------------------------------------------------


def build_duplicate_report(
    hashed: pd.DataFrame,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    cluster_similarity: float = DEFAULT_CLUSTER_SIMILARITY,
) -> pd.DataFrame:
    """Return one table with every pair of similar images and the automatic decision.

    Parameters
    ----------
    hashed : the DataFrame returned by compute_phashes() — columns
        path, split, class, filename, phash.
    similarity_threshold : pairs at least this similar are listed in the table.
    cluster_similarity : only pairs at least this similar can put images into the
        same cluster (default 1.0 = identical hashes). Must be >= similarity_threshold.

    `cluster_num` is the number of images in the cluster that contains both images
    of the row. It is empty for near-duplicate rows (similarity below the cluster
    criterion), because those two images do not share a cluster. Such rows always
    get 'needs review'.
    """
    if cluster_similarity < similarity_threshold:
        raise ValueError("cluster_similarity must be >= similarity_threshold.")

    if len(hashed) < 2:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    bits = _to_bit_matrix(hashed["phash"])
    n_images, n_bits = bits.shape
    max_distance = _max_distance(similarity_threshold, n_bits)
    cluster_max_distance = _max_distance(cluster_similarity, n_bits)

    idx_a, idx_b, distances = _find_similar_pairs(bits, max_distance)

    print(
        f"Compared {n_images * (n_images - 1) // 2} pairs  |  "
        f"listed: similarity >= {similarity_threshold:.2f} "
        f"(Hamming distance <= {max_distance} of {n_bits})  |  "
        f"similar pairs found: {len(idx_a)}"
    )

    if len(idx_a) == 0:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    clusters, root_of = _cluster_images(
        n_images, idx_a, idx_b, distances, cluster_max_distance
    )
    member_decisions = _decide_members(hashed, bits, clusters, cluster_max_distance)

    # One entry per report row: cluster size (or NA), decision text, and a sort key
    cluster_num: list = []
    auto_decision: list[str] = []
    group_key: list[int] = []
    for i, j in zip(idx_a.tolist(), idx_b.tolist()):
        root_i, root_j = root_of.get(i), root_of.get(j)
        if root_i is not None and root_i == root_j:
            cluster_num.append(len(clusters[root_i]))
            auto_decision.append(
                _pair_decision(member_decisions[i], member_decisions[j])
            )
            group_key.append(root_i)
        else:
            cluster_num.append(pd.NA)
            auto_decision.append(DECISION_NEEDS_REVIEW)
            group_key.append(i)

    image_a = hashed.iloc[idx_a].reset_index(drop=True)
    image_b = hashed.iloc[idx_b].reset_index(drop=True)

    report = pd.DataFrame(
        {
            "similarity": 1 - distances / n_bits,
            "cluster_num": pd.array(cluster_num, dtype="Int64"),
            "same_class": (image_a["class"] == image_b["class"]).to_numpy(),
            "class_a": image_a["class"],
            "class_b": image_b["class"],
            "same_split": (image_a["split"] == image_b["split"]).to_numpy(),
            "split_a": image_a["split"],
            "split_b": image_b["split"],
            "filename_a": image_a["filename"],
            "path_a": image_a["path"],
            "filename_b": image_b["filename"],
            "path_b": image_b["path"],
            "auto_decision": auto_decision,
            "_group": group_key,
            "_idx_a": idx_a,
            "_idx_b": idx_b,
        }
    )

    # Highest similarity first; rows of the same cluster stay next to each other
    report = report.sort_values(
        ["similarity", "_group", "_idx_a", "_idx_b"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    report = report[REPORT_COLUMNS]

    _print_summary(clusters, report)
    return report


def _print_summary(clusters: dict[int, list[int]], report: pd.DataFrame) -> None:
    """Print cluster sizes and the number of rows that need manual review."""
    if clusters:
        sizes = (
            pd.Series([len(members) for members in clusters.values()])
            .value_counts()
            .sort_index()
        )
        size_text = ", ".join(
            f"{size} images: {count}" for size, count in sizes.items()
        )
    else:
        size_text = "none"

    n_review = (report["auto_decision"] == DECISION_NEEDS_REVIEW).sum()
    print(f"Clusters (size: count) -> {size_text}")
    print(f"Rows needing manual review: {n_review}")


# ---------------------------------------------------------------------------
# 5. Comparing two images with a chosen hash model
# ---------------------------------------------------------------------------


def _resolve_project_path(path_value) -> Path:
    """Turn a project-relative path (with \\ or /) into a full path.

    Absolute paths are returned unchanged. Relative paths are resolved
    against the current working directory first (so "../dataset/..." paths
    typed from a notebook in notebooks/ work as-is); only if that does not
    exist do we fall back to resolving against PROJECT_ROOT.
    """
    path = Path(str(path_value).replace("\\", "/"))
    if path.is_absolute():
        return path
    cwd_relative = Path.cwd() / path
    if cwd_relative.exists():
        return cwd_relative
    return PROJECT_ROOT / path


def compare_images(
    path_a: str | Path,
    path_b: str | Path,
    hash_method: str = DEFAULT_HASH_METHOD,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> dict:
    """Compare two images with one hash model and return the similarity.

    Paths may be given relative to the current working directory (typical
    from a notebook, e.g. "../dataset/cleaned_V2/train/kamyun/a.jpg") or
    relative to the project root (e.g. r"dataset\\cleaned_V2\\train\\kamyun\\a.jpg").
    Use a raw string or forward slashes on Windows.

    hash_method : one of "ahash", "phash", "dhash", "whash".
    """
    if hash_method not in HASH_METHODS:
        raise ValueError(
            f"Unknown hash_method '{hash_method}'. Choose one of: {sorted(HASH_METHODS)}"
        )

    hashes = []
    for path_value in (path_a, path_b):
        full_path = _resolve_project_path(path_value)
        if not full_path.is_file():
            raise FileNotFoundError(f"Image not found: {full_path}")
        with Image.open(full_path) as img:
            hashes.append(HASH_METHODS[hash_method](img, hash_size=hash_size))

    hash_a, hash_b = hashes
    n_bits = hash_a.hash.size
    distance = int(hash_a - hash_b)

    return {
        "hash_method": hash_method,
        "hash_size": hash_size,
        "similarity": round(1 - distance / n_bits, 4),
        "hamming_distance": distance,
        "n_bits": n_bits,
    }


# ---------------------------------------------------------------------------
# 6. Notebook display
# ---------------------------------------------------------------------------


def _thumbnail_base64(path: Path, size: int) -> str | None:
    try:
        with Image.open(path) as img:
            thumb = img.convert("RGB")
        thumb.thumbnail((size, size))
        buffer = io.BytesIO()
        thumb.save(buffer, format="JPEG", quality=70)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


def _path_cell(path_value, thumbnails: bool, size: int) -> str:
    path = Path(path_value)
    cell = f'<a href="{path.resolve().as_uri()}" target="_blank">{html.escape(str(path))}</a>'
    if thumbnails:
        data = _thumbnail_base64(path, size)
        if data:
            cell += f'<br><img src="data:image/jpeg;base64,{data}" width="{size}">'
    return cell


def table_to_html(
    df: pd.DataFrame,
    path_columns: list[str] | None = None,
    thumbnails: bool = True,
    thumbnail_size: int = 140,
) -> str:
    """Render a table as HTML where path columns become links (+ thumbnails).

    Use in a notebook:  display(HTML(table_to_html(report)))
    The original DataFrame is not modified.
    """
    if path_columns is None:
        path_columns = [c for c in df.columns if c == "path" or c.startswith("path_")]

    table = df.copy()
    for column in table.columns:
        if column in path_columns:
            table[column] = table[column].map(
                lambda p: _path_cell(p, thumbnails, thumbnail_size)
            )
        elif not pd.api.types.is_numeric_dtype(table[column]):
            table[column] = table[column].map(lambda v: html.escape(str(v)))

    return table.to_html(
        escape=False, index=False, float_format="{:.3f}".format, na_rep=""
    )


# ---------------------------------------------------------------------------
# 7. CSV export
# ---------------------------------------------------------------------------


def _project_relative(path_value) -> str:
    """Return the path relative to the project root, with forward slashes."""
    resolved = Path(path_value).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path_value).as_posix()


def _non_colliding_path(output_path: Path) -> Path:
    """
    If output_path already exists, append _1, _2, ... before the extension
    until a free name is found, so a save call never overwrites a previous
    report. E.g. duplicate_pairs_report.csv -> duplicate_pairs_report_1.csv.
    """
    if not output_path.exists():
        return output_path

    stem, suffix = output_path.stem, output_path.suffix
    counter = 1
    while True:
        candidate = output_path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def save_duplicate_report(
    report: pd.DataFrame,
    output_path: str | Path | None = None,
) -> Path:
    """Save the report table as CSV (plain paths, no images).

    Default location: <project>/data/duplicate_pairs_report.csv
    If a file already exists at the target path, a numeric suffix (_1, _2, ...)
    is appended so the previous file is never overwritten.
    """
    output_path = Path(output_path) if output_path is not None else DEFAULT_REPORT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = _non_colliding_path(output_path)

    to_save = report.copy()
    for column in ("path_a", "path_b"):
        to_save[column] = to_save[column].map(_project_relative)

    to_save.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Saved {len(to_save)} rows to {output_path}")
    return output_path


def _excluded_rows(
    rows: pd.DataFrame, keeper_side: str, excluded_side: str
) -> pd.DataFrame:
    """Build label_corrections-style rows for the excluded image of each report row."""
    reasons = [
        f"duplicate of {split}/{cls}/{filename} (phash sim={similarity:.2f})"
        for split, cls, filename, similarity in zip(
            rows[f"split_{keeper_side}"],
            rows[f"class_{keeper_side}"],
            rows[f"filename_{keeper_side}"],
            rows["similarity"],
        )
    ]
    return pd.DataFrame(
        {
            "split": rows[f"split_{excluded_side}"].to_numpy(),
            "class": rows[f"class_{excluded_side}"].to_numpy(),
            "filename": rows[f"filename_{excluded_side}"].to_numpy(),
            "reason": reasons,
        }
    )


def to_label_corrections(report: pd.DataFrame) -> pd.DataFrame:
    """Return the automatically excluded images in the label_corrections.csv format.

    Columns: split, class, filename, action, new_class, reason.
    Every excluded image has a report row that pairs it with its reference copy
    ('keep A, exclude B' or 'exclude A, keep B'), which provides the reason.
    """
    keep_a = report[report["auto_decision"] == DECISION_KEEP_A_EXCLUDE_B]
    keep_b = report[report["auto_decision"] == DECISION_EXCLUDE_A_KEEP_B]

    corrections = pd.concat(
        [
            _excluded_rows(keep_a, keeper_side="a", excluded_side="b"),
            _excluded_rows(keep_b, keeper_side="b", excluded_side="a"),
        ],
        ignore_index=True,
    )
    corrections["action"] = "exclude"
    corrections["new_class"] = 0

    columns = ["split", "class", "filename", "action", "new_class", "reason"]
    return (
        corrections[columns]
        .drop_duplicates(subset=["split", "class", "filename"])
        .sort_values(["split", "class", "filename"])
        .reset_index(drop=True)
    )


def save_label_corrections(
    report: pd.DataFrame,
    output_path: str | Path | None = None,
) -> Path:
    """Save the automatic exclusions as CSV in the label_corrections.csv format.

    Default location: <project>/data/label_corrections_auto_duplicates.csv
    If a file already exists at the target path, a numeric suffix (_1, _2, ...)
    is appended so the previous file is never overwritten.
    """
    output_path = (
        Path(output_path) if output_path is not None else DEFAULT_LABEL_CORRECTIONS_PATH
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = _non_colliding_path(output_path)

    corrections = to_label_corrections(report)
    corrections.to_csv(output_path, index=False)
    print(f"Saved {len(corrections)} automatic exclusions to {output_path}")
    return output_path

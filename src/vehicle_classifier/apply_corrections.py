"""
apply_label_corrections.py

Applies manual label-correction decisions (relabel / exclude) from a CSV file
onto the raw traffic-vehicle dataset, producing a new "cleaned_V1" dataset
made of hardlinks (or symlinks as a fallback) — never copies, never touches
`raw`.

Usage
-----
    python apply_label_corrections.py \
        --raw-dir dataset/raw \
        --csv label_corrections.csv \
        --output-dir dataset/cleaned_V1

Assumptions about folder layout
--------------------------------
    <raw-dir>/<split>/<class>/<filename>

    e.g.  dataset/raw/train/vanet/215619169.jpg

CSV columns (first column is an unnamed personal bookkeeping column and is
ignored):
    <empty>, split, class, filename, action, new_split, new_class, reason

    - action        : "relabel" or "exclude" (case-insensitive)
    - new_split     : target split for a relabel. May be blank -> keep the
                       image's current split.
    - new_class     : target class for a relabel. Required for relabel rows
                       (may legitimately equal the original class -> a no-op
                       relabel, which is treated as perfectly normal and
                       simply linked back into the same class/split).

Matching logic (per image found by walking the raw dataset)
-------------------------------------------------------------
    1. Find all CSV rows whose `filename` (stripped) equals this image's
       filename (stripped). Filenames are matched case-sensitively but with
       leading/trailing whitespace ignored on both sides.
    2. Among those, keep only rows whose `class` (stripped) equals this
       image's class folder name.
    3. Among those, keep only rows whose `split` (stripped) equals this
       image's split folder name.
    4. Result:
         0 rows  -> no action for this image; link it, unchanged, into
                    cleaned_V1 at the same split/class path.
         1 row   -> apply that row's action (relabel or exclude).
         >1 rows -> conflict. Do NOT touch this image (no link is created).
                    Record it in the conflict report for manual review.

Every decision is counted and a short summary is printed at the end.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class CorrectionRow:
    split: str
    cls: str
    filename: str
    action: str
    new_split: str
    new_class: str
    reason: str
    row_number: int  # 1-based line number in the CSV, for error messages


@dataclass
class RunStats:
    total_images_scanned: int = 0
    no_action_linked: int = 0
    relabeled: int = 0
    relabeled_noop: int = (
        0  # relabel where new_class == old class and new_split == old split
    )
    excluded: int = 0
    conflicts: int = 0
    csv_rows_total: int = 0
    csv_rows_unmatched: int = (
        0  # CSV rows whose filename/class/split never matched an actual image
    )
    renamed_on_collision: int = (
        0  # relabeled images that landed on a filename already taken by a
        #    *different* image in the destination class/split, and were
        #    therefore saved under a "(n)" suffix instead of overwriting it
    )
    errors: list[str] = field(default_factory=list)
    conflict_details: list[str] = field(default_factory=list)
    unreadable_or_skipped: list[str] = field(default_factory=list)
    rename_details: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #


def load_corrections(csv_path: Path) -> list[CorrectionRow]:
    """
    Reads the corrections CSV and returns a list of CorrectionRow.
    Every string field is stripped of leading/trailing whitespace
    (the "excel trim" behaviour requested).
    """
    rows: list[CorrectionRow] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV file '{csv_path}' is empty.")

        # Expected header shape: <blank>, split, class, filename, action,
        # new_split, new_class, reason  (8 columns, first one unnamed/ignored)
        if len(header) < 8:
            raise ValueError(
                f"Expected at least 8 columns in '{csv_path}', "
                f"found {len(header)}: {header}"
            )

        for line_no, raw_row in enumerate(reader, start=2):  # header was line 1
            if not raw_row or all(cell.strip() == "" for cell in raw_row):
                continue  # skip fully blank lines

            # Pad short rows defensively so an accidental missing trailing
            # column doesn't crash the whole run.
            padded = raw_row + [""] * (8 - len(raw_row))

            _, split, cls, filename, action, new_split, new_class, reason = padded[:8]

            split = split.strip()
            cls = cls.strip()
            filename = filename.strip()
            action = action.strip().lower()
            new_split = new_split.strip()
            new_class = new_class.strip()
            reason = reason.strip()

            if not filename:
                # A row with no filename can never match anything; skip but
                # keep it silent-safe rather than crashing.
                continue

            rows.append(
                CorrectionRow(
                    split=split,
                    cls=cls,
                    filename=filename,
                    action=action,
                    new_split=new_split,
                    new_class=new_class,
                    reason=reason,
                    row_number=line_no,
                )
            )

    return rows


def index_by_filename(rows: list[CorrectionRow]) -> dict[str, list[CorrectionRow]]:
    """Groups correction rows by (stripped) filename for fast lookup."""
    index: dict[str, list[CorrectionRow]] = defaultdict(list)
    for row in rows:
        index[row.filename].append(row)
    return index


# --------------------------------------------------------------------------- #
# Linking helpers
# --------------------------------------------------------------------------- #


def make_link(source: Path, destination: Path) -> None:
    """
    Creates `destination` as a link to `source`, creating parent directories
    as needed. Tries a hardlink first (matches the "not a full copy" intent
    and is transparent to torchvision's ImageFolder/DataLoader); falls back
    to a symlink if hardlinking isn't possible (e.g. across filesystems/drives).

    Overwrite-safe *for re-runs only*: if `destination` already points at the
    same `source` (same file, already linked here on a previous run of this
    script), it is replaced so the script stays idempotent. If it exists and
    points at a *different* source, it is left alone — the caller is expected
    to have already resolved a non-colliding destination path (see
    `resolve_non_colliding_destination`) before calling this function.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        if _same_file(source, destination):
            destination.unlink()
        else:
            # Should not normally happen: callers must resolve collisions
            # via resolve_non_colliding_destination() first. Guard here so
            # we never silently clobber an unrelated file.
            raise FileExistsError(
                f"Refusing to overwrite '{destination}': it already exists "
                f"and does not point at '{source}'."
            )

    try:
        os.link(source, destination)
    except OSError:
        # Cross-device link or filesystem without hardlink support.
        os.symlink(source.resolve(), destination)


def _same_file(source: Path, destination: Path) -> bool:
    """
    True if `destination` is already a link (hard or symbolic) to `source`,
    i.e. relinking it is a safe no-op rather than a collision with a
    different image.

    NOTE: comparing resolved paths (`Path.resolve()`) is only correct for
    *symlinks* — a hardlink is just another directory entry pointing at the
    same inode, so `destination.resolve()` returns `destination`'s own path,
    not `source`'s, and would never compare equal even though the files are
    identical. `make_link` prefers hardlinks whenever possible, so we must
    compare by (device, inode) via os.stat, which is correct for both
    hardlinks and symlinks (stat follows symlinks by default).
    """
    try:
        src_stat = source.stat()
        dst_stat = destination.stat()
    except OSError:
        return False
    return (src_stat.st_dev, src_stat.st_ino) == (dst_stat.st_dev, dst_stat.st_ino)


def resolve_non_colliding_destination(dest: Path, source: Path) -> Path:
    """
    Returns a destination path guaranteed not to collide with an unrelated
    existing file.

    - If `dest` does not exist yet, or it already links to `source` (a
      harmless re-run of this script), `dest` itself is returned unchanged.
    - Otherwise `dest` is occupied by a *different* image (e.g. two source
      images that legitimately share the same filename after a relabel
      moved them into the same class/split folder). In that case a
      Windows-Explorer-style " (1)", " (2)", ... suffix is inserted before
      the file extension, incrementing until a free name is found:

          215619169.jpg -> 215619169 (1).jpg -> 215619169 (2).jpg -> ...
    """
    if not dest.exists() and not dest.is_symlink():
        return dest
    if _same_file(source, dest):
        return dest

    stem, suffix = dest.stem, dest.suffix
    counter = 1
    while True:
        candidate = dest.with_name(f"{stem} ({counter}){suffix}")
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        if _same_file(source, candidate):
            return candidate
        counter += 1


# --------------------------------------------------------------------------- #
# Core processing
# --------------------------------------------------------------------------- #


def iter_raw_images(raw_dir: Path):
    """
    Yields (split, class_name, filename, full_path) for every image file
    under raw_dir, walked in a deterministic (sorted) order:
    split -> class -> filename.
    """
    for split_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        split = split_dir.name
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            for file_path in sorted(class_dir.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                yield split, cls, file_path.name, file_path


def resolve_action(
    split: str,
    cls: str,
    filename: str,
    candidates_by_filename: dict[str, list[CorrectionRow]],
    stats: RunStats,
) -> tuple[str, CorrectionRow | None, list[CorrectionRow]]:
    """
    Implements the 3-step narrowing (filename -> class -> split) described
    in the spec and returns:
        (outcome, matched_row, rows_that_found_this_image)

    outcome is one of "no_action", "apply", "conflict".
    `rows_that_found_this_image` lists every CSV row that correctly
    identified this exact image (i.e. matched on filename+class+split),
    regardless of whether the action could ultimately be applied — this is
    used so a row involved in a genuine conflict is never also reported as
    a "typo / no matching image" row.
    """
    same_filename = candidates_by_filename.get(filename, [])
    if not same_filename:
        return "no_action", None, []

    same_class = [r for r in same_filename if r.cls == cls]
    if not same_class:
        return "no_action", None, []

    same_split = [r for r in same_class if r.split == split]

    if len(same_split) == 0:
        # Class matched for at least one row but none of those also match
        # the split -> none of the same-class rows apply to this image's
        # actual split, so this image has no action.
        return "no_action", None, []

    if len(same_split) > 1:
        stats.conflicts += 1
        row_numbers = ", ".join(str(r.row_number) for r in same_split)
        stats.conflict_details.append(
            f"{split}/{cls}/{filename}: {len(same_split)} matching CSV rows "
            f"(lines {row_numbers}) — needs manual review."
        )
        return "conflict", None, same_split

    return "apply", same_split[0], same_split


def process_dataset(raw_dir: Path, output_dir: Path, csv_path: Path) -> RunStats:
    stats = RunStats()

    corrections = load_corrections(csv_path)
    stats.csv_rows_total = len(corrections)
    candidates_by_filename = index_by_filename(corrections)

    matched_row_ids: set[int] = set()  # row_number of every CSV row actually applied

    for split, cls, filename, src_path in iter_raw_images(raw_dir):
        stats.total_images_scanned += 1

        outcome, row, found_rows = resolve_action(
            split, cls, filename, candidates_by_filename, stats
        )
        for r in found_rows:
            matched_row_ids.add(r.row_number)

        if outcome == "conflict":
            continue  # do not touch this image at all

        if outcome == "no_action":
            dest = output_dir / split / cls / filename
            dest = resolve_non_colliding_destination(dest, src_path)
            if dest.name != filename:
                stats.renamed_on_collision += 1
                stats.rename_details.append(
                    f"{split}/{cls}/{filename}: name already taken there by "
                    f"a different image (likely relabeled in from elsewhere), "
                    f"saved as '{dest.name}' instead."
                )
            make_link(src_path, dest)
            stats.no_action_linked += 1
            continue

        # outcome == "apply"
        assert row is not None

        if row.action == "exclude":
            stats.excluded += 1
            continue  # no link created

        if row.action == "relabel":
            target_split = row.new_split if row.new_split else split
            target_class = row.new_class if row.new_class else cls

            if not row.new_class:
                stats.errors.append(
                    f"Line {row.row_number}: action=relabel but new_class is "
                    f"empty for {split}/{cls}/{filename}. Skipped — image was "
                    f"NOT linked. Please fix the CSV."
                )
                continue

            dest = output_dir / target_split / target_class / filename
            dest = resolve_non_colliding_destination(dest, src_path)
            if dest.name != filename:
                stats.renamed_on_collision += 1
                stats.rename_details.append(
                    f"Line {row.row_number}: {split}/{cls}/{filename} "
                    f"relabeled to {target_split}/{target_class}/ — name "
                    f"already taken there by a different image, saved as "
                    f"'{dest.name}' instead."
                )
            make_link(src_path, dest)
            stats.relabeled += 1
            if target_class == cls and target_split == split:
                stats.relabeled_noop += 1
            continue

        # Unknown action string
        stats.errors.append(
            f"Line {row.row_number}: unrecognized action '{row.action}' for "
            f"{split}/{cls}/{filename}. Skipped — image was NOT linked."
        )

    # Report CSV rows that never matched any real image on disk at all
    # (not even as part of a conflict) — these are very likely typos in
    # split/class/filename inside the CSV itself. Rows that DID match an
    # image but were skipped due to a conflict are excluded from this list
    # (they are already reported under CONFLICTS above).
    for row in corrections:
        if row.row_number not in matched_row_ids:
            stats.csv_rows_unmatched += 1
            stats.unreadable_or_skipped.append(
                f"Line {row.row_number}: split='{row.split}', class='{row.cls}', "
                f"filename='{row.filename}' never matched an image under "
                f"'{raw_dir}/{row.split}/{row.cls}/{row.filename}'."
            )

    return stats


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_report(stats: RunStats) -> None:
    print("=" * 70)
    print("LABEL CORRECTION REPORT")
    print("=" * 70)
    print(f"Images scanned in raw dataset : {stats.total_images_scanned}")
    print(f"  -> linked unchanged (no action found)   : {stats.no_action_linked}")
    print(
        f"  -> relabeled                            : {stats.relabeled}"
        f"  (of which no-op, same class/split: {stats.relabeled_noop})"
    )
    print(f"  -> excluded                             : {stats.excluded}")
    print(f"  -> conflicts (multiple actions, skipped): {stats.conflicts}")
    print(
        f"  -> relabeled but renamed to avoid overwrite: {stats.renamed_on_collision}"
    )

    total_linked = stats.no_action_linked + stats.relabeled
    print(f"\nTotal files created in cleaned dataset: {total_linked}")

    print(f"\nCSV rows total       : {stats.csv_rows_total}")
    print(f"CSV rows never matched to a real image on disk: {stats.csv_rows_unmatched}")

    if stats.conflict_details:
        print("\n--- CONFLICTS (no action taken, review manually) ---")
        for line in stats.conflict_details:
            print(f"  - {line}")

    if stats.unreadable_or_skipped:
        print("\n--- CSV ROWS WITH NO MATCHING IMAGE (possible typos) ---")
        for line in stats.unreadable_or_skipped:
            print(f"  - {line}")

    if stats.rename_details:
        print("\n--- RELABELS RENAMED TO AVOID OVERWRITE ---")
        for line in stats.rename_details:
            print(f"  - {line}")

    if stats.errors:
        print("\n--- ERRORS (image skipped, needs manual fix) ---")
        for line in stats.errors:
            print(f"  - {line}")

    print("=" * 70)


# --------------------------------------------------------------------------- #
# Notebook-friendly entry point
# --------------------------------------------------------------------------- #


def run_corrections(
    raw_dir: str | Path,
    csv_path: str | Path,
    output_dir: str | Path = "dataset/cleaned_V1",
    verbose: bool = True,
) -> RunStats:
    """
    Callable entry point for notebooks / other scripts:

        from vehicle_classifier.apply_corrections import run_corrections

        stats = run_corrections(
            raw_dir="dataset/raw",
            csv_path="label_corrections.csv",
            output_dir="dataset/cleaned_V1",
        )

    Returns the RunStats object (so you can inspect stats.conflict_details,
    stats.relabeled, etc. programmatically) and, by default, also prints the
    same human-readable report as the CLI.
    """
    raw_dir = Path(raw_dir)
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw dataset directory not found: {raw_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    stats = process_dataset(raw_dir, output_dir, csv_path)
    if verbose:
        print_report(stats)
    return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Path to dataset/raw (contains train/test/unclean).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to the label_corrections CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/cleaned_V1"),
        help="Path to create cleaned_V1 dataset (default: dataset/cleaned_V1).",
    )
    args = parser.parse_args()

    try:
        run_corrections(args.raw_dir, args.csv, args.output_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

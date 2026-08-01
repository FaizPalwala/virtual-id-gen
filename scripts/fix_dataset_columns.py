#!/usr/bin/env python3
"""fix_dataset_columns.py — Rename columns in released dataset artifacts in place.

Applies the standardized column schema to dataset artifacts produced by an
older build (before the `clusterid` -> `identity_id` rename).  Intended for
one-off migration of already-released files on HPC so a full pipeline re-run
is unnecessary.

Handles both the balanced (`dataset.csv` / `dataset.parquet`) and imbalanced
(`dataset_imbalanced.csv` / `dataset_imbalanced.parquet`) artifacts.  Only
columns present in the file are renamed; missing ones are skipped, so the
script is idempotent and safe to re-run.

Usage:
    python scripts/fix_dataset_columns.py --dir /scratch/$USER/datagen/data/dataset
    python scripts/fix_dataset_columns.py --dir ... --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Standardized output schema (must match OUTPUT_COLUMNS in src/build_dataset.py).
# Key: old column name -> new standardized name.
RENAME_MAP = {
    "clusterid": "identity_id",
}

ARTIFACTS = (
    "dataset.csv",
    "dataset.parquet",
    "dataset_imbalanced.csv",
    "dataset_imbalanced.parquet",
)


def fix_one(path: Path, dry_run: bool) -> int:
    """Rename columns in a single CSV/Parquet artifact.  Returns rows changed."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported artifact type: {path}")

    present = {old: new for old, new in RENAME_MAP.items() if old in df.columns}
    if not present:
        return 0

    df.rename(columns=present, inplace=True)
    if not dry_run:
        if suffix == ".csv":
            df.to_csv(path, index=False)
        else:
            df.to_parquet(path, index=False)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing dataset artifacts (dataset.csv/.parquet, ...)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing files",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")

    total_files = 0
    total_rows = 0
    for name in ARTIFACTS:
        path = root / name
        if not path.exists():
            continue
        rows = fix_one(path, args.dry_run)
        action = "would rename" if args.dry_run else "renamed"
        print(f"[{action}] {name}: {rows:,} rows ({path.stat().st_size:,} bytes)")
        total_files += 1
        total_rows += rows

    print(f"\nTotal: {total_files} artifacts, {total_rows:,} rows affected.")
    if args.dry_run:
        print("Dry run — no files were written.  Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()

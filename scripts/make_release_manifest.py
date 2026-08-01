#!/usr/bin/env python3
"""Rewrite dataset CSV paths from absolute/HPC paths to release-relative paths.

Reads a dataset.csv (or dataset_imbalanced.csv, or identitymanifest.csv)
produced by the HPC pipeline, rewrites every ``image_path`` to a
release-relative path of the form ``images/identity_NNN/accepted_XXX.jpg``,
and writes the result back.

Also validates that no stale ``/tmp/``, ``/home/``, ``seeds/``, or seed
filenames remain in the published metadata — exits non-zero if any are found.

Example:
    python scripts/make_release_manifest.py \\
        --dataset-csv release_v1/metadata/dataset.csv \\
        --image-root release_v1/images
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def normalise_path(path_str: str) -> str:
    """Convert an absolute HPC path to a release-relative image path.

    Examples
    --------
    /scratch/kxvs0578/datagen/data/merged/processed/images/identity_037/accepted_012.jpg
        → images/identity_037/accepted_012.jpg
    /tmp/job.6848258/shard_0/data_shard_0/identities/candidates/identity_001/candidate_038.png
        → images/identity_001/candidate_038.png  (raw candidates never released)
    """
    p = Path(path_str)
    parts = p.parts

    # Strategy: walk backwards to find identity_NNN, take it + filename.
    idx = None
    for i in range(len(parts) - 1, -1, -1):
        if re.match(r"^identity_\d{3,}$", parts[i]):
            idx = i
            break
    if idx is None:
        raise ValueError(f"Cannot extract identity from path: {path_str}")

    identity_dir = parts[idx]
    filename = parts[-1]
    return f"images/{identity_dir}/{filename}"


FORBIDDEN_PATTERNS = [
    (r"^/tmp/", "TMPDIR path"),
    (r"^/scratch/", "absolute scratch path"),
    (r"/home/", "home directory path"),
    (r"/seeds/", "local seed image directory"),
    (r"SFHQ_pt\d_", "SFHQ seed filename"),
    (r"^candidate_", "raw candidate filename (not release-relative)"),
    (r"^\d+\.(jpg|png)$", "bare filename (not release-relative)"),
]


def validate_no_leaked_paths(df: pd.DataFrame) -> None:
    """Raise RuntimeError if any 'image_path' leaks internal paths."""
    leaked: list[tuple[str, str, str]] = []
    for _, row in df.iterrows():
        path = row["image_path"]
        for pattern, label in FORBIDDEN_PATTERNS:
            if re.search(pattern, path):
                leaked.append((path, label, pattern))
    if leaked:
        for path, label, pattern in leaked[:10]:
            print(f"  LEAK: {label}  pattern={pattern!r}  path={path!r}",
                  file=sys.stderr)
        raise RuntimeError(
            f"{len(leaked)} leaked paths found in metadata — "
            f"do not publish without cleaning"
        )


def make_manifest(
    dataset_csv: str,
    image_root: str,
    output_csv: str | None = None,
) -> None:
    """Normalise paths and validate the dataset CSV."""
    df = pd.read_csv(dataset_csv)
    original_count = len(df)

    # Accept both legacy 'imagepath' and standardised 'image_path'
    col = "image_path" if "image_path" in df.columns else "imagepath"
    if col not in df.columns:
        raise KeyError(
            f"'image_path' column missing. Columns: {list(df.columns)}"
        )

    df["image_path"] = df[col].apply(normalise_path)
    if col != "image_path":
        df = df.drop(columns=[col])

    # Verify every normalised path exists on disk
    root = Path(image_root)
    missing: list[str] = []
    for path in df["image_path"]:
        if not (root / path).is_file():
            missing.append(path)
    if missing:
        raise RuntimeError(
            f"{len(missing)} normalised paths do not resolve. "
            f"First 10: {missing[:10]}"
        )

    validate_no_leaked_paths(df)

    output = output_csv or dataset_csv
    df.to_csv(output, index=False)
    print(f"Normalised {original_count} paths → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite dataset CSV paths for release"
    )
    parser.add_argument(
        "--dataset-csv",
        required=True,
        help="Path to dataset.csv / dataset_imbalanced.csv (read and overwritten)",
    )
    parser.add_argument(
        "--image-root",
        required=True,
        help="Root directory containing images/ (for path verification)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (default: overwrite input)",
    )
    args = parser.parse_args()
    make_manifest(args.dataset_csv, args.image_root, args.output_csv)


if __name__ == "__main__":
    main()

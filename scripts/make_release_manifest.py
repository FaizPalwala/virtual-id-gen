#!/usr/bin/env python3
"""Rewrite dataset CSV paths from absolute/HPC paths to release-relative paths.

Reads a dataset CSV (dataset.csv, dataset_imbalanced.csv, dataset_candidates.csv,
or dataset_candidates_imbalanced.csv) produced by the HPC pipeline, rewrites
every ``image_path`` to a release-relative path of the form
``images/identity_NNN/accepted_XXX.jpg`` (Bench) or
``images/identity_NNN/candidate_YYY.png`` (Full), and writes the result back.

Also validates that no stale ``/tmp/``, ``/home/``, ``seeds/``, or seed
filenames remain in the published metadata — exits non-zero if any are found.

Pruning (--prune-images --source-root): copies only the images referenced by
the CSV from the source tree into the release image root, so orphan
crops/candidates never ship.  Run once per CSV with the same release root and
the union of referenced images is materialised.

Example:
    python scripts/make_release_manifest.py \\
        --dataset-csv release_v1/metadata/dataset.csv \\
        --image-root release_v1 \\
        --source-root /scratch/$USER/datagen/data \\
        --release-type bench \\
        --prune-images
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


def normalise_path(path_str: str, release_type: str = "bench") -> str:
    """Convert an absolute HPC path to a release-relative image path.

    Examples
    --------
    /scratch/kxvs0578/datagen/data/processed/images/identity_037/accepted_012.jpg
        → images/identity_037/accepted_012.jpg   (bench)
    /scratch/kxvs0578/datagen/data/identities/candidates/identity_037/candidate_012.png
        → images/identity_037/candidate_012.png  (full)
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
    # The release tree stores images under images/, regardless of whether the
    # source lived under processed/images/ (bench) or identities/candidates/ (full).
    return f"images/{identity_dir}/{filename}"


FORBIDDEN_PATTERNS = [
    (r"^/tmp/", "TMPDIR path"),
    (r"^/scratch/", "absolute scratch path"),
    (r"/home/", "home directory path"),
    (r"/seeds/", "local seed image directory"),
    (r"SFHQ_pt\d_", "SFHQ seed filename"),
    (r"^\d+\.(jpg|png)$", "bare filename (not release-relative)"),
]
# candidate_* filenames are legitimate in the Full release, forbidden in Bench.
BENCH_ONLY_PATTERNS = [(r"(^|/)candidate_\d{3}\.png$", "raw candidate filename in Bench release")]


def validate_no_leaked_paths(df: pd.DataFrame, release_type: str = "bench") -> None:
    """Raise RuntimeError if any 'image_path' leaks internal paths."""
    patterns = FORBIDDEN_PATTERNS + (BENCH_ONLY_PATTERNS if release_type == "bench" else [])
    leaked: list[tuple[str, str, str]] = []
    for _, row in df.iterrows():
        path = row["image_path"]
        for pattern, label in patterns:
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
    release_type: str = "bench",
    prune_images: bool = False,
    source_root: str | None = None,
) -> None:
    """Normalise paths, prune (optionally), and validate the dataset CSV."""
    df = pd.read_csv(dataset_csv)
    original_count = len(df)

    # Accept both legacy 'imagepath' and standardised 'image_path'
    col = "image_path" if "image_path" in df.columns else "imagepath"
    if col not in df.columns:
        raise KeyError(
            f"'image_path' column missing. Columns: {list(df.columns)}"
        )

    df["image_path"] = df[col].apply(lambda p: normalise_path(p, release_type))
    if col != "image_path":
        df = df.drop(columns=[col])

    # Leak check FIRST: a path is a leak (e.g. candidate_* in a bench release)
    # regardless of whether the source file happens to resolve on disk.
    validate_no_leaked_paths(df, release_type)

    # Prune: copy only CSV-referenced images from the SOURCE tree into the
    # release image root.  Without --source-root, the release root is assumed
    # to already contain images/ (verification only).
    #
    # The release tree flattens to images/identity_NNN/file, but the source
    # tree keeps its pipeline prefix: processed/images/identity_NNN/ (bench)
    # or identities/candidates/identity_NNN/ (full).  Reconstruct the source
    # path from the release type so --source-root points at the data root.
    source_prefix = {
        "bench": "processed/images",
        "full": "identities/candidates",
    }[release_type]

    root = Path(image_root)
    src_base = Path(source_root) if source_root else root
    missing: list[str] = []
    for path in df["image_path"]:
        # path is images/identity_NNN/file.ext — map back to source layout
        identity_dir, filename = path.split("/", 2)[1], path.rsplit("/", 1)[1]
        src = src_base / source_prefix / identity_dir / filename
        if not src.is_file():
            missing.append(path)
            continue
        if prune_images:
            dst = root / path
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    if missing:
        raise RuntimeError(
            f"{len(missing)} normalised paths do not resolve. "
            f"First 10: {missing[:10]}"
        )

    output = output_csv or dataset_csv
    df.to_csv(output, index=False)
    action = "Pruned + normalised" if prune_images else "Normalised"
    print(f"{action} {original_count} paths → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite dataset CSV paths for release"
    )
    parser.add_argument(
        "--dataset-csv",
        required=True,
        help="Path to dataset CSV (read and overwritten)",
    )
    parser.add_argument(
        "--image-root",
        required=True,
        help="Release root (the dir that CONTAINS images/)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (default: overwrite input)",
    )
    parser.add_argument(
        "--release-type",
        default="bench",
        choices=["bench", "full"],
        help="bench (224 crops) or full (1024 candidates) — controls path checks",
    )
    parser.add_argument(
        "--prune-images",
        action="store_true",
        default=False,
        help="Copy only CSV-referenced images into the release image root",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Source tree containing the images referenced by the CSV "
             "(e.g. the HPC data root).  Required when --prune-images is set; "
             "defaults to --image-root for verification-only runs.",
    )
    args = parser.parse_args()
    if args.prune_images and not args.source_root:
        parser.error("--source-root is required when --prune-images is set")
    make_manifest(
        args.dataset_csv,
        args.image_root,
        args.output_csv,
        release_type=args.release_type,
        prune_images=args.prune_images,
        source_root=args.source_root,
    )


if __name__ == "__main__":
    main()

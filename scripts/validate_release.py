#!/usr/bin/env python3
"""Release gate — run before every publication.

Validates the entire release directory against the criteria in Strat.md §4.5:
every path exists, correct resolution/RGB, no duplicates, no split leaks,
no internal paths in metadata, and checksums match.  The column schema is
loaded from scripts/schema.json so validation tracks the documented schema.

Exit 0 = clean pass; exit 1 = gate failure (do not publish).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


VALID_SPLITS = {"retain", "test", "forget"}
VALID_AGEGROUPS = {0, 1, 2, 3}
DEFAULT_EXPECTED_SIZE = (224, 224)
FORBIDDEN_PATH_PATTERNS: list[tuple[str, str]] = [
    (r"^/tmp/", "TMPDIR path"),
    (r"^/scratch/", "absolute scratch path"),
    (r"/home/", "home directory path"),
    (r"/seeds/", "local seed image directory"),
    (r"SFHQ_pt\d_", "SFHQ seed filename in path"),
    (r"^candidate_", "raw candidate filename (not release-relative)"),
]


def _load_schema(schema_path: Path) -> dict:
    """Load the JSON schema and extract required column metadata."""
    with open(schema_path) as fh:
        schema = json.load(fh)
    props = schema["definitions"]["base_columns"]["properties"]
    required = set(schema["definitions"]["base_columns"]["required"])
    return {"properties": props, "required": required}


def _sha256(path: Path) -> str:
    """Return hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8192):
            h.update(chunk)
    return h.hexdigest()


def validate(
    release_dir: str,
    metadata_csv: str,
    schema_path: str | None = None,
    expected_width: int = 224,
    expected_height: int = 224,
    require_relative_paths: bool = True,
    check_cluster_split_isolation: bool = True,
    checksums_file: str | None = None,
) -> bool:
    root = Path(release_dir).resolve()
    csv_path = root / metadata_csv
    if not csv_path.is_file():
        print(f"FAIL: metadata CSV not found: {csv_path}", file=sys.stderr)
        return False

    # Schema-driven column requirements
    schema = None
    if schema_path:
        schema = _load_schema(Path(schema_path))

    df = pd.read_csv(csv_path)
    failures: list[str] = []
    expected_size = (expected_width, expected_height)

    # ---- 1. Column presence (schema-driven when available) ----
    if schema is not None:
        missing_cols = schema["required"] - set(df.columns)
        if missing_cols:
            failures.append(f"Missing columns: {missing_cols}")
    else:
        missing_cols = {"image_path", "identity_id", "age_group", "split", "forget_step"} - set(df.columns)
        if missing_cols:
            failures.append(f"Missing base columns: {missing_cols}")

    # ---- 2. Duplicate paths ----
    dupes = df[df.duplicated(subset="image_path", keep=False)]
    if len(dupes) > 0:
        failures.append(
            f"{len(dupes)} duplicate image paths: "
            f"{dupes['image_path'].iloc[:5].tolist()}"
        )

    # ---- 3. Valid split values ----
    invalid_splits = set(df["split"]) - VALID_SPLITS
    if invalid_splits:
        failures.append(f"Invalid split values: {invalid_splits}")

    # ---- 4. Valid agegroup values ----
    if "age_group" in df.columns:
        invalid_ages = set(df["age_group"]) - VALID_AGEGROUPS
        if invalid_ages:
            failures.append(f"Invalid age_group values: {invalid_ages}")

    # ---- 5. Cluster split isolation ----
    if check_cluster_split_isolation:
        multi_split = (
            df.groupby("identity_id")["split"]
            .nunique()
            .loc[lambda x: x > 1]
        )
        if len(multi_split) > 0:
            failures.append(
                f"{len(multi_split)} clusters appear in multiple splits: "
                f"{multi_split.index.tolist()[:10]}"
            )

    # ---- 6. Forget step validity ----
    forget_df = df[df["split"] == "forget"]
    if len(forget_df) > 0:
        invalid_steps = set(forget_df["forget_step"]) - set(range(-1, 100))
        if invalid_steps:
            failures.append(f"Invalid forget_step values: {invalid_steps}")
        # Every forget cluster should have exactly one forget_step
        step_per_cluster = forget_df.groupby("identity_id")["forget_step"].nunique()
        inconsistent = step_per_cluster[step_per_cluster > 1]
        if len(inconsistent) > 0:
            failures.append(
                f"{len(inconsistent)} forget clusters have inconsistent forget_step"
            )

    # ---- 7. Image file validation ----
    missing_files: list[str] = []
    bad_size: list[tuple[str, tuple[int, int]]] = []
    bad_mode: list[tuple[str, str]] = []
    leaked_paths: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        path = root / row["image_path"]
        if not path.is_file():
            missing_files.append(row["image_path"])
            continue
        try:
            with Image.open(path) as img:
                if img.size != expected_size:
                    bad_size.append((row["image_path"], img.size))
                if img.mode != "RGB":
                    bad_mode.append((row["image_path"], img.mode))
        except Exception as exc:
            failures.append(f"Cannot decode {row['image_path']}: {exc}")

    if missing_files:
        failures.append(
            f"{len(missing_files)} images missing. "
            f"First 10: {missing_files[:10]}"
        )
    if bad_size:
        failures.append(
            f"{len(bad_size)} images have wrong dimensions. "
            f"First 5: {bad_size[:5]}"
        )
    if bad_mode:
        failures.append(
            f"{len(bad_mode)} images are not RGB. "
            f"First 5: {bad_mode[:5]}"
        )

    # ---- 8. No internal paths in metadata ----
    if require_relative_paths:
        for _, row in df.iterrows():
            path_str = row["image_path"]
            for pattern, label in FORBIDDEN_PATH_PATTERNS:
                if re.search(pattern, path_str):
                    leaked_paths.append((path_str, label))
        if leaked_paths:
            failures.append(
                f"{len(leaked_paths)} leaked internal paths in metadata. "
                f"First 5: {leaked_paths[:5]}"
            )

    # ---- 9. Checksums ----
    if checksums_file:
        csum_path = root / checksums_file
        if not csum_path.is_file():
            failures.append(f"Checksums file not found: {csum_path}")
        else:
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with open(csum_path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        expected_hash, _, file_path = line.partition("  ")
                        file_path = file_path.lstrip("*")
                        actual = Path(file_path)
                        if not actual.is_file():
                            failures.append(
                                f"Checksum file missing: {file_path}"
                            )
                        elif _sha256(actual) != expected_hash:
                            failures.append(
                                f"Checksum mismatch: {file_path}"
                            )
            finally:
                os.chdir(cwd)

    # ---- Report ----
    if failures:
        print("\n".join(f"FAIL: {f}" for f in failures), file=sys.stderr)
        print(f"\n{len(failures)} gate failure(s).  Do not publish.", file=sys.stderr)
        return False

    summary = (
        df.groupby("split")
        .agg(images=("image_path", "size"), identities=("identity_id", "nunique"))
    )
    print(f"Images:       {len(df)}")
    print(f"Identities:   {df['identity_id'].nunique()}")
    print(summary.to_string())
    print("Release QA: PASS")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a dataset release directory before publication"
    )
    parser.add_argument(
        "--release-dir",
        required=True,
        help="Root of the release directory (contains images/ and metadata/)",
    )
    parser.add_argument(
        "--metadata",
        default="metadata/dataset.csv",
        help="Path to the dataset CSV relative to --release-dir",
    )
    parser.add_argument(
        "--schema",
        default="scripts/schema.json",
        help="Path to JSON schema (relative to release-dir or absolute)",
    )
    parser.add_argument(
        "--expected-width",
        type=int,
        default=224,
        help="Expected image width (default: 224)",
    )
    parser.add_argument(
        "--expected-height",
        type=int,
        default=224,
        help="Expected image height (default: 224)",
    )
    parser.add_argument(
        "--require-relative-paths",
        action="store_true",
        default=True,
        help="Reject metadata containing /tmp/, /home/, seeds/, etc.",
    )
    parser.add_argument(
        "--no-require-relative-paths",
        action="store_false",
        dest="require_relative_paths",
        help="Skip internal-path leak check",
    )
    parser.add_argument(
        "--check-cluster-split-isolation",
        action="store_true",
        default=True,
        help="Verify every cluster is in exactly one split",
    )
    parser.add_argument(
        "--no-check-cluster-split-isolation",
        action="store_false",
        dest="check_cluster_split_isolation",
        help="Skip split isolation check",
    )
    parser.add_argument(
        "--checksums",
        default=None,
        help="Path to checksums file relative to --release-dir",
    )
    args = parser.parse_args()

    ok = validate(
        release_dir=args.release_dir,
        metadata_csv=args.metadata,
        schema_path=args.schema,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
        require_relative_paths=args.require_relative_paths,
        check_cluster_split_isolation=args.check_cluster_split_isolation,
        checksums_file=args.checksums,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

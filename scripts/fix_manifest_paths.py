#!/usr/bin/env python3
"""
fix_manifest_paths.py
Rewrite absolute manifest paths to the portable relative contract.

Pre-fix ``merge_shards`` wrote ``raw_candidatepath`` and ``seedpath`` as
absolute strings (e.g. ``/scratch/.../identities/candidates/identity_000/
candidate_000.png``), which breaks builds on any machine where the data
root differs (local mirrors, CI).  This script rewrites them in place:

  raw_candidatepath  → relative to the identities dir
                       (candidates/identity_000/candidate_000.png)
  seedpath           → relative to the data root
                       (seeds/seed_0536.jpg)

The transformation is prefix-independent: it locates the ``candidates`` /
``seeds`` path segment and keeps everything from it onward, so it works on
any machine regardless of the original mount (``/scratch/...`` on HPC,
``~/Projects/data`` locally).

Idempotent: already-relative rows pass through untouched.  Intended for
one-shot migration of existing manifests; the fixed merge_shards writes
relative paths natively.

Usage:
    python fix_manifest_paths.py --manifest ../data/identities/raw_candidate_manifest.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to raw_candidate_manifest.csv (the merged manifest).",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    df = pd.read_csv(manifest_path)
    changes = 0

    def _strip_to_segment(value: str, anchor: str) -> str:
        """Return the path suffix starting at *anchor* (prefix-independent).

        Legacy manifests carry absolute paths with machine-specific mounts
        (/scratch/... on HPC, ~/Projects/data locally).  relative_to() cannot
        handle a foreign mount, so instead we locate the anchor segment and
        keep everything from it onward — the portable relative form.
        """
        path = Path(value)
        if not path.is_absolute():
            return value  # already relative — leave alone
        parts = path.parts
        for i, part in enumerate(parts):
            if part == anchor:
                return str(Path(*parts[i:]))
        return value  # anchor not found — leave unchanged

    for column, anchor in (
        ("raw_candidatepath", "candidates"),
        ("seedpath", "seeds"),
    ):
        if column not in df.columns:
            continue
        rewritten = df[column].apply(lambda v: _strip_to_segment(v, anchor))
        col_changes = int((rewritten != df[column]).sum())
        df[column] = rewritten
        print(f"[OK] {column}: {col_changes}/{len(df)} rows relativised")
        changes += col_changes

    if changes:
        df.to_csv(manifest_path, index=False)
        print(f"[OK] Rewrote {manifest_path} ({changes} path fixes)")
    else:
        print("[OK] Nothing to fix — manifest already relative")


if __name__ == "__main__":
    main()

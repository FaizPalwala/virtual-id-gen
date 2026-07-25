#!/usr/bin/env python3
"""merge_shards.py — Combine 4-way Slurm job-array manifests into one.

Reads shard_N/ directories (each a copy of a generate step's identities/ output),
remaps identity/cluster IDs to be globally unique, copies candidate images into
a unified candidates/ tree, and writes a single merged manifest + skipped-seed log.

The merged output is written directly into *mergeddir*:
    mergeddir/raw_candidate_manifest.csv
    mergeddir/candidates/identity_XXX/candidate_YYY.png
    mergeddir/skipped_seed_manifest.csv

After the merge, point Hydra's ``dataset.dataroot`` to the *parent* of mergeddir
so that ``identities/`` resolves to mergeddir — e.g. if mergeddir is
``/data/merged/identities``, set ``dataset.dataroot=/data/merged``.

Usage:
    python scripts/merge_shards.py --outputdir identities/ \\
                                   --mergeddir identities/final/
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

IDENTITIES_PER_SHARD = 100


def merge_shards(outputdir: str, mergeddir: str, shard_count: int = 4) -> str:
    root = Path(outputdir)
    merged = Path(mergeddir)
    merged.mkdir(parents=True, exist_ok=True)

    merged_candidates = merged / "candidates"
    merged_candidates.mkdir(exist_ok=True)

    all_records: list[pd.DataFrame] = []
    all_skipped: list[pd.DataFrame] = []

    for shard_id in range(shard_count):
        shard_dir = root / f"shard_{shard_id}"
        manifest = shard_dir / "raw_candidate_manifest.csv"
        skipped = shard_dir / "skipped_seed_manifest.csv"

        if not manifest.exists():
            print(f"[WARN] shard {shard_id}: manifest missing — skipping")
            continue

        df = pd.read_csv(manifest)
        base_id = shard_id * IDENTITIES_PER_SHARD
        df["identityid"] = df["identityid"].astype(int) + base_id
        df["clusterid"] = df["clusterid"].astype(int) + base_id

        # Copy candidate images using shard-relative paths (not the stale
        # absolute TMPDIR paths stored in the manifest).
        shard_candidates = shard_dir / "candidates"
        for _, row in df.iterrows():
            cid = int(row["clusterid"])
            trial = int(row["trial"])
            src = (
                shard_candidates
                / f"identity_{cid - base_id:03d}"
                / f"candidate_{trial:03d}.png"
            )
            if not src.is_file():
                continue
            dst_cluster = merged_candidates / f"identity_{cid:03d}"
            dst_cluster.mkdir(exist_ok=True)
            dst = dst_cluster / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            df.at[_, "raw_candidatepath"] = str(dst)

        all_records.append(df)
        if skipped.exists():
            all_skipped.append(pd.read_csv(skipped))

    if not all_records:
        raise FileNotFoundError("No shard manifests found under " + str(root))

    combined = pd.concat(all_records, ignore_index=True)
    combined.sort_values(["clusterid", "trial"], inplace=True)
    combined.to_csv(merged / "raw_candidate_manifest.csv", index=False)

    if all_skipped:
        combined_skipped = pd.concat(all_skipped, ignore_index=True)
        combined_skipped.to_csv(merged / "skipped_seed_manifest.csv", index=False)

    n_candidates = len(combined)
    n_identities = combined["clusterid"].nunique()
    print(
        f"Merged {len(all_records)}/{shard_count} shards → "
        f"{n_candidates} candidates across {n_identities} identities (at {merged})"
    )
    return str(merged)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outputdir", required=True, help="Root directory containing shard_*/ subdirs"
    )
    p.add_argument(
        "--mergeddir", required=True, help="Where to write the unified identities/ output"
    )
    p.add_argument("--shardcount", type=int, default=4)
    merge_shards(**vars(p.parse_args()))

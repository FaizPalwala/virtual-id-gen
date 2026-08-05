"""
build_dataset.py

Constructs the final Phase 2 dataset mapping by merging generated identity 
manifests with extracted embedding attributes. Applies the train/test/forget splits.
Build the backward-compatible final ``dataset.csv`` file.

Example:
    python main.py --config-name step5_build \\
        dataset.dataroot=../data
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

OUTPUT_COLUMNS = [
    "image_path", "identity_id", "age_group", "age", "gender",
    "split", "forget_step", "forget_variant", "image_subset",
    "arcface_similarity", "laplacian_variance", "detection_confidence",
    "pose", "expression", "lighting", "setting", "camera",
]


def _load_attributes(embeddingsdir: str) -> pd.DataFrame:
    """Load image-level attributes saved by extract_embeddings.py."""
    embeddings = Path(embeddingsdir)
    df = pd.DataFrame(
        {
            "imagepath": np.load(embeddings / "imagepaths.npy").astype(str),
            "agegroup": np.load(embeddings / "agegroups.npy"),
            "age": np.load(embeddings / "ages.npy"),
            "gender": np.load(embeddings / "genders.npy"),
            "embedding": list(np.load(embeddings / "embeddings.npy")),
        }
    )
    id_path = embeddings / "identity_ids.npy"
    if id_path.exists():
        df["identity_id"] = np.load(id_path)
    # Derive trial from candidate filename for per-crop embedding matching.
    # Format: candidates/identity_NNN/candidate_YYY.png → trial=YYY.
    # identitymanifest.csv carries the same trial — the join key that links
    # each 224 crop to the 1024 candidate it was cropped from.
    t = df["imagepath"].str.extract(r"candidate_(\d+)")[0]
    df["trial"] = pd.to_numeric(t, errors="coerce").astype("Int64")
    return df


def _add_arcface_similarity(final: pd.DataFrame) -> pd.DataFrame:
    """Add per-image cosine similarity to the identity's mean embedding.

    Confound control: an image far from its identity's canonical face
    (unusual angle/expression) is memorised differently from a canonical
    portrait.  Without this control, per-identity MIA AUC is confounded
    by within-identity variance.
    """
    from common import normalised_cosine_similarity

    # Mean embedding per cluster (embeddings are already L2-normalised).
    means = final.groupby("identity_id")["embedding"].mean()
    final["arcface_similarity"] = [
        normalised_cosine_similarity(row.embedding, means[row.identity_id])
        for row in final.itertuples(index=False)
    ]
    return final


def _load_candidate_metadata(identitydir: str) -> pd.DataFrame:
    """Load per-candidate metadata from the generate step's manifest.

    Returns a DataFrame indexed by ``identity_id`` with a single row per
    identity (the metadata is identical for all candidates of an identity,
    since it describes the generation prompt attributes).
    """
    raw = Path(identitydir) / "identities" / "raw_candidate_manifest.csv"
    if not raw.exists():
        raw = Path(identitydir).parent / "identities" / "raw_candidate_manifest.csv"
    df = pd.read_csv(raw)
    # The raw manifest may still use 'clusterid' if merged by an older script.
    id_col = "identity_id" if "identity_id" in df.columns else "clusterid"
    metadata_cols = [
        "pose", "expression", "lighting", "setting", "camera",
    ]
    available = [c for c in metadata_cols if c in df.columns]
    if not available:
        # Older manifests without metadata columns — return empty.
        return pd.DataFrame(index=pd.Index([], name="identity_id"))
    return df[[id_col] + available].rename(columns={id_col: "identity_id"})\
        .drop_duplicates("identity_id").set_index("identity_id")[available]


def _distribute_forget(
    forget_ids: list[int], nforget: int, forget_steps: int, rng: np.random.RandomState,
) -> dict[int, tuple[int, int]]:
    """Assign forget identities to steps with uniform+remainder distribution.

    Returns ``{identity_id: (step, variant)}`` where *step* is 0-based and
    *variant* is the index within the step.
    """
    base = nforget // forget_steps
    remainder = nforget % forget_steps

    mapping: dict[int, tuple[int, int]] = {}
    pos = 0
    for step in range(forget_steps):
        count = base + (1 if step < remainder else 0)
        for variant in range(count):
            mapping[forget_ids[pos]] = (step, variant)
            pos += 1
    return mapping


def _to_dataroot_relative(p: str, base_dir: Path, dataroot: Path) -> str:
    """Convert a manifest image path to a dataroot-relative release path.

    Handles two conventions:
    - absolute paths (legacy manifests written before the portability fix)
    - relative-to-*base_dir* paths (current portable manifest contract;
      base_dir is the identities dir for candidates, processed for crops).

    This keeps release CSVs portable: every ``image_path`` resolves against
    the dataset root on any machine, regardless of where the pipeline ran.
    """
    path = Path(p)
    if path.is_absolute():
        try:
            return str(path.relative_to(dataroot))
        except ValueError:
            pass
        for anchor in ("candidates", "images"):
            parts = path.parts
            for i, part in enumerate(parts):
                if part == anchor:
                    return str(Path(*parts[i:]))
        return str(path)
    return str((base_dir / path).relative_to(dataroot))


def _compute_holdout_size(
    total_per_id: int, holdout_frac: float, min_holdout: int
) -> int:
    """Return the number of holdout images per identity.

    MUFAC evaluation requires held-out probe images for every identity.
    The holdout is split off BEFORE any cardinality-based trimming so it
    is independent of the train set size.
    """
    return max(min_holdout, round(total_per_id * holdout_frac))


def build_dataset(
    identitydir: str,
    embeddingsdir: str,
    outputdir: str,
    nforget: int = 40,
    forget_steps: int = 15,
    randomstate: int = 42,
    imagesperidentity: int = 75,
    holdout_frac: float = 0.20,
    min_holdout: int = 5,
) -> str:
    """Build the balanced 224x224 crop dataset with MUFAC-aligned splits.

    Every identity contributes to both training and evaluation via the
    ``image_subset`` column, replacing the old identity-disjoint ``test`` split.
    """
    manifest = pd.read_csv(Path(identitydir) / "identitymanifest.csv")
    attributes = _load_attributes(embeddingsdir)

    # Match each 224 crop to its parent 1024 candidate embedding via the
    # (identity_id, trial) join key.  identitymanifest.csv carries trial
    # from preprocess; _load_attributes derives trial from the candidate
    # filename.  This gives per-crop embeddings → arcface_similarity is
    # a real within-identity spread, not trivially 1.0.
    #
    # NOTE — per-identity age variance is a FEATURE, not noise to suppress:
    # each identity's images carry their own InsightFace gender/age estimate
    # (taken on the parent 1024×1024 candidate), so one identity can span
    # multiple age_group values (e.g. identity 558 ranges 23–65).  The
    # synthetic identity stays recognisably the same person while appearing
    # at different ages — a realism property that also prevents the age
    # classifier from shortcutting to identity (identity_id → constant age).
    # Do not collapse to a per-identity constant (first/mode); keep the
    # per-image proxy labels.
    attrs_merge = attributes[
        ["identity_id", "trial", "agegroup", "age", "gender", "embedding"]
    ].copy()
    final = manifest[
        ["imagepath", "identity_id", "trial", "detection_confidence", "laplacian_variance"]
    ].merge(attrs_merge, on=["identity_id", "trial"], how="left")
    if len(final) != len(manifest):
        raise RuntimeError(
            f"Trial-based join lost rows: {len(manifest)} → {len(final)}"
        )

    # Join generation metadata (pose, expression, lighting, ...) from the
    # raw candidate manifest.  Metadata is per-identity (same prompt grid for
    # all candidates), so the join is identity-level.
    metadata = _load_candidate_metadata(identitydir)
    if not metadata.empty:
        final = final.merge(
            metadata, on="identity_id", how="left", validate="many_to_one"
        )

    # Per-image cosine similarity to the identity's mean embedding.
    # Computed BEFORE trimming so the reference mean spans ALL quality-passing
    # crops from preprocess — the identity's canonical face.  The column is
    # therefore dataset-invariant: a given image has the same arcface_similarity
    # in the balanced and imbalanced variants, making it a valid confound
    # control across both.  Trimming below only removes rows; it never changes
    # the values.
    final = _add_arcface_similarity(final)

    # Trim each identity to imagesperidentity (preprocess may have saved
    # more than that).  The manifest is quality-ranked, so the first N
    # rows per cluster are the best crops.
    final = (
        final.sort_values(["identity_id", "detection_confidence"], ascending=[True, False])
        .groupby("identity_id")
        .head(imagesperidentity)
        .reset_index(drop=True)
    )

    identities = sorted(final["identity_id"].unique())
    if nforget >= len(identities):
        raise ValueError("forget count must leave at least one retain identity.")
    rng = np.random.RandomState(randomstate)
    rng.shuffle(identities)
    forget_ids = identities[:nforget]
    split_map = (
        {identity: "forget" for identity in forget_ids}
        | {identity: "retain" for identity in identities[nforget:]}
    )
    final["split"] = final["identity_id"].map(split_map)

    # MUFAC-aligned per-identity holdout.
    holdout_n = _compute_holdout_size(imagesperidentity, holdout_frac, min_holdout)
    rng_holdout = np.random.RandomState(randomstate + 7777)
    for cid in identities:
        rows = final[final["identity_id"] == cid].index
        shuffled_idx = list(rows)
        rng_holdout.shuffle(shuffled_idx)
        for j, idx in enumerate(shuffled_idx):
            value = "holdout" if j < holdout_n else "train"
            final.at[idx, "image_subset"] = value
    final["image_subset"] = final["image_subset"].fillna("train")

    # Distribute forget identities across steps with uniform base + remainder.
    step_map = _distribute_forget(forget_ids, nforget, forget_steps, rng)
    final["forgetstep"] = (
        final["identity_id"].map(
            {cid: step for cid, (step, _) in step_map.items()}
        )
        .fillna(-1)
        .astype(int)
    )
    final["forgetvariant"] = (
        final["identity_id"].map(
            {cid: variant for cid, (_, variant) in step_map.items()}
        )
        .fillna(-1)
        .astype(int)
    )
    final = final[final.agegroup != -1].copy()

    # Cluster samples: asymmetric matplotlib grid per identity.
    samples_root = Path(outputdir) / "cluster_samples"
    dataroot = Path(outputdir).parent
    if samples_root.exists():
        import shutil

        shutil.rmtree(samples_root)
    samples_root.mkdir(parents=True)

    raw_manifest = Path(identitydir).parent / "identities" / "raw_candidate_manifest.csv"
    if not raw_manifest.exists():
        raw_manifest = Path(identitydir) / "identities" / "raw_candidate_manifest.csv"
    if raw_manifest.exists():
        seeds = (
            pd.read_csv(raw_manifest)[["clusterid", "seedpath"]]
            .rename(columns={"clusterid": "identity_id"})
            .drop_duplicates("identity_id")
            .set_index("identity_id")
            .seedpath
        )
    else:
        seeds = pd.Series(dtype=str)

    for cid in sorted(final["identity_id"].unique()):
        split = split_map[cid]
        (samples_root / split).mkdir(parents=True, exist_ok=True)

        # Collect up to 4 images: seed + 3 examples
        image_paths = []
        seed_path = seeds.get(cid, None)
        if seed_path:
            resolved_seed = Path(seed_path)
            if not resolved_seed.is_absolute():
                # Manifest seedpath is relative to the data root.
                resolved_seed = dataroot / resolved_seed
            if resolved_seed.exists():
                image_paths.append(str(resolved_seed))
        examples = final[final["identity_id"] == cid].imagepath.head(3)
        identity_root = Path(identitydir)
        for i, img_path in enumerate(examples, 1):
            resolved = identity_root / img_path
            if resolved.exists():
                image_paths.append(str(resolved))

        if not image_paths:
            continue

        # Asymmetric grid: seed spans left column (3 rows), variations stack right.
        img_list = [Image.open(p).convert("RGB") for p in image_paths]
        cell_w, cell_h = img_list[0].size

        fig = plt.figure(figsize=(10, 7.5), facecolor="#f8f9fa")
        gs = fig.add_gridspec(
            3, 2, width_ratios=[2.2, 1], hspace=0.25, wspace=0.15,
        )
        ax_seed = fig.add_subplot(gs[:, 0])  # spans all left cells
        ax_vars = [fig.add_subplot(gs[i, 1]) for i in range(3)]

        ax_seed.imshow(img_list[0])
        ax_seed.set_title(
            "SEED", fontsize=9, fontweight="bold", color="#c0392b", pad=4,
        )
        ax_seed.axis("off")
        for spine in ax_seed.spines.values():
            spine.set_visible(True)
            spine.set_color("#c0392b")
            spine.set_linewidth(3)

        for i in range(1, min(4, len(img_list))):
            ax = ax_vars[i - 1]
            ax.imshow(img_list[i])
            ax.set_title(f"Example {i}", fontsize=8, color="#555", pad=3)
            ax.axis("off")
            rect = mpatches.Rectangle(
                (0, 0), cell_w - 1, cell_h - 1,
                linewidth=1, edgecolor="#ccc", facecolor="none",
            )
            ax.add_patch(rect)

        for j in range(len(img_list) - 1, 3):
            ax_vars[j].axis("off")

        fig.suptitle(
            f"Identity {int(cid):03d} — {split.upper()}",
            fontsize=13, fontweight="bold", y=0.97,
        )
        out = samples_root / split / f"identity_{int(cid):03d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)
    dataroot = output.parent
    output_df = final.rename(
        columns={
            "imagepath": "image_path",
            "agegroup": "age_group",
            "forgetstep": "forget_step",
            "forgetvariant": "forget_variant",
        }
    )
    # Fill missing metadata columns with defaults for older manifests.
    for col in OUTPUT_COLUMNS:
        if col not in output_df.columns:
            output_df[col] = "" if col in ("pose", "expression", "lighting", "setting", "camera") else 0.0
    output_df = output_df[OUTPUT_COLUMNS]
    output_df["image_path"] = (
        output_df["image_path"]
        .apply(lambda p: _to_dataroot_relative(p, Path(identitydir), dataroot))
    )
    csv_path = output / "dataset.csv"
    parquet_path = output / "dataset.parquet"
    output_df.to_csv(csv_path, index=False)
    output_df.to_parquet(parquet_path, index=False)
    (output / "datasetsummary.json").write_text(
        json.dumps(
            {
                "total_images": len(final),
                "nidentities": len(identities),
                "images_per_identity": imagesperidentity,
                "nforget_ids": nforget,
                "forget_steps": forget_steps,
                "forget_pct": nforget / len(identities),
                "distribution": {str(s): cnt for s, cnt in sorted(
                    Counter(step_map[cid][0] for cid in step_map).items()
                )},
                "split_sizes": {
                    key: int(value)
                    for key, value in final.groupby("split").size().items()
                },
                "holdout_per_id": holdout_n,
                "columns": OUTPUT_COLUMNS,
            },
            indent=2,
        )
    )
    return str(csv_path)


def build_imbalanced_dataset(
    identitydir: str,
    embeddingsdir: str,
    outputdir: str,
    nforget: int = 40,
    forget_steps: int = 15,
    randomstate: int = 42,
    holdout_frac: float = 0.20,
    min_holdout: int = 5,
    # Imbalance: train counts as ratios of the max train pool.
    # holdout is constant per identity regardless of bin.
    imagesperidentity: int = 75,
    candidatesperidentity: int = 85,
    high_bin_pct: float = 0.10,
    low_bin_pct: float = 0.60,
    train_gradient_ratio: dict[str, float] | None = None,
) -> str:
    """Build the imbalanced 224x224 dataset with MUFAC-aligned splits.

    Same ``image_subset`` design as balanced: every identity contributes
    train + holdout.  The popularity gradient applies only to the train
    subset -- holdout is constant per identity regardless of bin.

    Holdout per identity: ``max(min_holdout, round(imagesperidentity *
    holdout_frac))`` — the SAME reserve as the balanced release, so probe
    stability is uniform across all popularity tiers.

    Train gradient: ratios applied to ``max_train = candidatesperidentity
    - holdout``.  Defaults give 70:40:20 (85 - 15 = 70; {1.0, 0.57, 0.29}).
    The computation scales with the candidate pool: if a future release
    generates 550 candidates/id targeting 500 images, holdout = 100 and
    max_train = 450, so the gradient re-derives train counts accordingly.
    """
    # ── 1. Read and merge manifests (same as build_dataset) ──
    manifest = pd.read_csv(Path(identitydir) / "identitymanifest.csv")
    attributes = _load_attributes(embeddingsdir)

    attrs_merge = attributes[
        ["identity_id", "trial", "agegroup", "age", "gender", "embedding"]
    ].copy()
    final = manifest[
        ["imagepath", "identity_id", "trial", "detection_confidence", "laplacian_variance"]
    ].merge(attrs_merge, on=["identity_id", "trial"], how="left")
    if len(final) != len(manifest):
        raise RuntimeError(
            f"Trial-based join lost rows: {len(manifest)} -> {len(final)}"
        )

    metadata = _load_candidate_metadata(identitydir)
    if not metadata.empty:
        final = final.merge(
            metadata, on="identity_id", how="left", validate="many_to_one"
        )

    # Per-image cosine similarity to the identity's mean embedding.
    # Computed BEFORE pruning so the reference mean spans ALL quality-passing
    # crops from preprocess — the identity's canonical face.  The column is
    # therefore dataset-invariant: a given image has the same arcface_similarity
    # in the balanced and imbalanced variants, making it a valid confound
    # control across both.  Pruning below only removes rows; it never changes
    # the values.
    final = _add_arcface_similarity(final)

    identities = sorted(final["identity_id"].unique())

    # ── 2. Assign popularity bins ──
    # Design choice: seeded random shuffle ensures reproducibility and avoids
    # any correlation with the identity's original identity ID (which may be
    # correlated with seed quality or prompt order).  Bin boundaries are
    # based on identity count, not image count.
    rng = np.random.RandomState(randomstate + 9999)  # different seed from balanced
    shuffled_ids = identities.copy()
    rng.shuffle(shuffled_ids)

    n_high = max(1, int(len(shuffled_ids) * high_bin_pct))
    n_low = max(1, int(len(shuffled_ids) * low_bin_pct))
    n_medium = len(shuffled_ids) - n_high - n_low

    high_ids = set(shuffled_ids[:n_high])
    low_ids = set(shuffled_ids[-n_low:])
    # medium_ids = set(shuffled_ids[n_high : n_high + n_medium])  # implicit

    bin_map = {}
    for cid in shuffled_ids:
        if cid in high_ids:
            bin_map[cid] = "high"
        elif cid in low_ids:
            bin_map[cid] = "low"
        else:
            bin_map[cid] = "medium"

    # ── 3. Down-sample low- and medium-bin identities ──
    # Design choice: train counts are set as ratios of max_train_per_id
    # (candidate ceiling − per-identity holdout).  Higher bins get more
    # train images; holdout is constant per identity.  Produces a 70:40:20
    # train gradient with defaults (85 − 15 = 70; {1.0, 0.57, 0.29}).
    # High-bin identities keep all available crops (up to the ceiling).
    # Per-identity random draws are seeded for reproducibility.
    rng_prune = np.random.RandomState(randomstate + 9998)
    if train_gradient_ratio is None:
        train_gradient_ratio = {"high": 1.0, "medium": 0.57, "low": 0.29}
    # Holdout reserve — SAME formula as the balanced release, applied per
    # identity: max(min_holdout, round(imagesperidentity × holdout_frac)).
    # Defaults: max(5, round(75 × 0.20)) = 15 — probe stability in every tier.
    holdout_n = _compute_holdout_size(imagesperidentity, holdout_frac, min_holdout)
    # Train pool = candidate ceiling minus the per-identity holdout reserve.
    # Defaults: 85 − 15 = 70 → {1.0, 0.57, 0.29} × 70 → 70:40:20 train.
    max_train = max(1, candidatesperidentity - holdout_n)
    # KEPT pool per identity = train + holdout (gradient on train only;
    # holdout carved afterwards).  Defaults: 85 / 55 / 35 kept.
    per_bin_images = {
        "high": round(max_train * train_gradient_ratio["high"]) + holdout_n,
        "medium": round(max_train * train_gradient_ratio["medium"]) + holdout_n,
        "low": round(max_train * train_gradient_ratio["low"]) + holdout_n,
    }
    drop_mask = pd.Series(False, index=final.index)
    for cid in low_ids | (set(bin_map.keys()) - high_ids - low_ids):  # low + medium
        target = per_bin_images[bin_map[cid]]
        rows = final[final["identity_id"] == cid]
        n_keep = min(target, len(rows))
        keep_idx = set(rng_prune.choice(rows.index, n_keep, replace=False))
        drop_idx = set(rows.index) - keep_idx
        drop_mask.loc[list(drop_idx)] = True

    imbalanced = final[~drop_mask].copy()

    new_sizes = imbalanced.groupby("identity_id").size()
    for cid in set(bin_map.keys()) - high_ids:
        expected = per_bin_images[bin_map[cid]]
        actual = new_sizes.get(cid, 0)
        if actual < expected * 0.5:
            raise RuntimeError(
                f"Identity {cid} ({bin_map[cid]} bin): got {actual} < {expected * 0.5}"
            )

    imbalanced_ids = sorted(imbalanced["identity_id"].unique())
    # Same identity-level split as balanced.
    rng_split = np.random.RandomState(randomstate)
    rng_split.shuffle(imbalanced_ids)
    forget_ids = imbalanced_ids[:nforget]
    split_map = (
        {cid: "forget" for cid in forget_ids}
        | {cid: "retain" for cid in imbalanced_ids[nforget:]}
    )
    imbalanced["split"] = imbalanced["identity_id"].map(split_map)

    # MUFAC holdout: carve the SAME per-identity reserve (holdout_n, computed
    # above from imagesperidentity × holdout_frac) out of each identity's
    # kept pool — 15/id with defaults, identical across all popularity bins.
    rng_h = np.random.RandomState(randomstate + 7777)
    for cid in imbalanced_ids:
        rows = imbalanced[imbalanced["identity_id"] == cid].index
        shuffled = list(rows)
        rng_h.shuffle(shuffled)
        n_hold = min(holdout_n, len(shuffled) - 1)
        for j, idx in enumerate(shuffled):
            imbalanced.at[idx, "image_subset"] = "holdout" if j < n_hold else "train"
    imbalanced["image_subset"] = imbalanced["image_subset"].fillna("train")

    # Forget step distribution (same algorithm as balanced).
    step_map = _distribute_forget(forget_ids, nforget, forget_steps, rng_split)
    imbalanced["forgetstep"] = (
        imbalanced["identity_id"].map(
            {cid: step for cid, (step, _) in step_map.items()}
        )
        .fillna(-1)
        .astype(int)
    )
    imbalanced["forgetvariant"] = (
        imbalanced["identity_id"].map(
            {cid: variant for cid, (_, variant) in step_map.items()}
        )
        .fillna(-1)
        .astype(int)
    )
    imbalanced = imbalanced[imbalanced.agegroup != -1].copy()

    # ── 6. Add popularity annotation columns ──
    # Design choice: adding both a categorical label (popularity_bin) and a
    # cardinality value (images_per_identity) lets downstream evaluation decide
    # which axis to test — discrete bins or continuous count.
    imbalanced["popularity_bin"] = imbalanced["identity_id"].map(bin_map)
    id_counts = imbalanced.groupby("identity_id").size()
    imbalanced["images_per_identity"] = imbalanced["identity_id"].map(id_counts)

    # ── 7. Write output ──
    # Design choice: imbalanced variant is a separate file paired alongside
    # dataset.csv.  Users compare forget efficacy across popularity_bin values
    # without modifying the balanced baseline.
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)
    dataroot = output.parent

    output_columns = OUTPUT_COLUMNS + ["popularity_bin", "images_per_identity"]
    output_df = imbalanced.rename(
        columns={
            "imagepath": "image_path",
            "agegroup": "age_group",
            "forgetstep": "forget_step",
            "forgetvariant": "forget_variant",
        }
    )
    for col in OUTPUT_COLUMNS:
        if col not in output_df.columns:
            output_df[col] = "" if col in ("pose", "expression", "lighting", "setting", "camera") else 0.0
    output_df = output_df[output_columns]
    output_df["image_path"] = (
        output_df["image_path"]
        .apply(lambda p: _to_dataroot_relative(p, Path(identitydir), dataroot))
    )

    csv_path = output / "dataset_imbalanced.csv"
    parquet_path = output / "dataset_imbalanced.parquet"
    output_df.to_csv(csv_path, index=False)
    output_df.to_parquet(parquet_path, index=False)

    # Summary with per-bin breakdown.
    bin_stats = imbalanced.groupby("popularity_bin").agg(
        nidentities=("identity_id", "nunique"),
        nimages=("identity_id", "count"),
    ).to_dict("index")

    (output / "datasetsummary_imbalanced.json").write_text(
        json.dumps(
            {
                "total_images": len(imbalanced),
                "nidentities": len(imbalanced_ids),
                "forget_pct": nforget / len(imbalanced_ids),
                "holdout_per_id": holdout_n,
                "gradient_ratios": train_gradient_ratio,
                "images_per_bin": {
                    "high": per_bin_images["high"],
                    "medium": per_bin_images["medium"],
                    "low": per_bin_images["low"],
                },
                "bin_sizes": {
                    "high_pct": high_bin_pct,
                    "medium_pct": 1.0 - high_bin_pct - low_bin_pct,
                    "low_pct": low_bin_pct,
                    "n_high": n_high,
                    "n_medium": n_medium,
                    "n_low": n_low,
                },
                "per_bin": {
                    bin: {
                        "identities": stats["nidentities"],
                        "images": stats["nimages"],
                        "images_per_id": round(stats["nimages"] / stats["nidentities"], 1),
                    }
                    for bin, stats in bin_stats.items()
                },
                "forget_steps": forget_steps,
                "split_sizes": {
                    key: int(value)
                    for key, value in imbalanced.groupby("split").size().items()
                },
                "columns": OUTPUT_COLUMNS + ["popularity_bin", "images_per_identity"],
            },
            indent=2,
        )
    )

    print(f"[OK] Imbalanced dataset: {len(imbalanced)} images across "
          f"{len(imbalanced_ids)} identities — "
          f"{n_high} high, {n_medium} medium, {n_low} low.")
    return str(csv_path)


# ── Candidate (full-resolution) dataset columns ──
# Same core as OUTPUT_COLUMNS but excludes crop-level quality metrics
# (laplacian_variance, detection_confidence) — meaningless on 1024
# candidates before alignment.
RAW_OUTPUT_COLUMNS = [
    "image_path", "identity_id", "age_group", "age", "gender",
    "arcface_similarity", "pose", "expression", "lighting",
    "setting", "camera",
]


def build_raw_dataset(
    identitydir: str,
    embeddingsdir: str,
    outputdir: str,
    randomstate: int = 42,
) -> str:
    """Build the full-resolution 1024x1024 raw dataset.

    Max-size reference dataset -- all 85 portraits per identity, no
    splits.  General-purpose release for identity recognition, face
    generation evaluation, and demographic bias studies.  Researchers
    can construct balanced subsets of up to 85 images/identity with
    custom train/holdout splits.

    Columns: identity_id, age, gender, arcface_similarity, pose,
    expression, lighting, setting, camera.
    """
    manifest = pd.read_csv(Path(identitydir) / "identities" / "raw_candidate_manifest.csv")
    id_col = "identity_id" if "identity_id" in manifest.columns else "clusterid"
    manifest.rename(columns={id_col: "identity_id", "raw_candidatepath": "imagepath"}, inplace=True)

    candidates = manifest[["imagepath", "identity_id", "trial"]].copy()
    candidates.drop_duplicates(inplace=True)

    attributes = _load_attributes(embeddingsdir)
    attrs_merge = attributes[
        ["identity_id", "trial", "agegroup", "age", "gender", "embedding"]
    ].copy()
    final = candidates.merge(attrs_merge, on=["identity_id", "trial"], how="left")

    metadata = _load_candidate_metadata(identitydir)
    if not metadata.empty:
        final = final.merge(metadata, on="identity_id", how="left", validate="many_to_one")

    final = _add_arcface_similarity(final)
    identities = sorted(final["identity_id"].unique())

    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)
    dataroot = output.parent
    output_df = final.rename(
        columns={"imagepath": "image_path", "agegroup": "age_group"}
    )
    for col in RAW_OUTPUT_COLUMNS:
        if col not in output_df.columns:
            output_df[col] = "" if col in ("pose", "expression", "lighting", "setting", "camera") else 0.0
    output_df = output_df[RAW_OUTPUT_COLUMNS]
    output_df["image_path"] = output_df["image_path"].apply(
        lambda p: _to_dataroot_relative(p, Path(identitydir) / "identities", dataroot)
    )

    csv_path = output / "dataset_raw.csv"
    parquet_path = output / "dataset_raw.parquet"
    output_df.to_csv(csv_path, index=False)
    output_df.to_parquet(parquet_path, index=False)

    max_balanced = int(output_df.groupby("identity_id").size().min())
    max_per_id = int(output_df.groupby("identity_id").size().max())
    (output / "datasetsummary_raw.json").write_text(
        json.dumps({
            "total_images": len(final),
            "nidentities": len(identities),
            "max_images_per_id": max_per_id,
            "max_balanced_subset": max_balanced,
            "columns": RAW_OUTPUT_COLUMNS,
        }, indent=2)
    )

    print(f"[OK] Raw dataset: {len(final)} images across {len(identities)} identities.")
    return str(csv_path)

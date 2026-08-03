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
    "split", "forget_step", "forget_variant",
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
        return str(path.relative_to(dataroot))
    return str((base_dir / path).relative_to(dataroot))


def build_dataset(
    identitydir: str,
    embeddingsdir: str,
    outputdir: str,
    nforget: int = 40,
    ntest: int = 60,
    forget_steps: int = 15,
    randomstate: int = 42,
    imagesperidentity: int = 75,
) -> str:
    """Merge final manifest and attributes, trim to imagesperidentity per cluster."""
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
    if nforget + ntest >= len(identities):
        raise ValueError("Split sizes must leave at least one retain identity.")
    rng = np.random.RandomState(randomstate)
    rng.shuffle(identities)
    forget_ids = identities[:nforget]
    test = identities[nforget : nforget + ntest]
    split_map = (
        {identity: "forget" for identity in forget_ids}
        | {identity: "test" for identity in test}
        | {identity: "retain" for identity in identities[nforget + ntest :]}
    )
    final["split"] = final["identity_id"].map(split_map)

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
    ntest: int = 60,
    forget_steps: int = 15,
    randomstate: int = 42,
    # Imbalance distribution parameters.
    # Design choice: we down-sample low-popularity identities rather than
    # up-sample high-popularity ones, because no extra candidate images are
    # available (preprocess selects exactly 75/identity).  Down-sampling
    # creates a realistic long-tail distribution without regeneration.
    high_bin_pct: float = 0.10,    # top 10% of identities keep all 75 images
    low_bin_pct: float = 0.60,     # bottom 60% keep only low_bin_images
    low_bin_images: int = 20,      # prune to 20 randomly selected images
    medium_bin_images: int = 40,   # middle 30% keep 40 images (gradient 85:40:20)
) -> str:
    """Build an imbalanced variant of the dataset for unlearning stress-testing.

    Follows the same merge/split logic as :func:`build_dataset`, then
    applies a popularity-based down-sampling to create a **4.25:2:1
    gradient** across three bins:

    * High-popularity (top 10%): up to 85 images/identity (the max
      available from generation) — over-represented, hardest to forget.
    * Medium (next 30%): **{medium_bin_images}** images/identity — moderately
      represented.
    * Low-popularity (bottom 60%): {low_bin_images} images/identity — under-
      represented, easiest to forget.

    The 85 → 40 → 20 gradient mirrors real-world long-tail distributions
    where a minority of identities (celebrities) have many images while
    most have very few.  Downstream evaluation can plot unlearning efficacy
    against *images_per_identity* and test for a monotonic relationship.

    Output columns include ``popularity_bin`` and ``images_per_identity``
    for per-bin analysis.
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
    # Design choice: low-bin identities retain low_bin_images (20), medium-bin
    # identities retain medium_bin_images (40), producing the 85:40:20
    # gradient.  High-bin identities keep all available crops (up to 85).
    # Per-identity random draws are seeded for reproducibility.
    rng_prune = np.random.RandomState(randomstate + 9998)
    per_bin_images = {
        "low": low_bin_images,
        "medium": medium_bin_images,
        "high": 85,  # max available from generate step
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
    for cid, expected in [(cid, low_bin_images) for cid in low_ids] + [
        (cid, medium_bin_images)
        for cid in set(bin_map.keys()) - high_ids - low_ids
    ]:
        actual = new_sizes.get(cid, 0)
        if actual < expected * 0.5:
            raise RuntimeError(
                f"Identity {cid} ({bin_map[cid]} bin) has {actual} images "
                f"but expected ~{expected}"
            )

    imbalanced_ids = sorted(imbalanced["identity_id"].unique())
    # Design choice: split assignment uses the same identity pool as balanced,
    # only the per-ID image count differs.  This ensures the same identities
    # are forget/test/retain in both variants, making cross-variant comparison
    # valid.
    rng_split = np.random.RandomState(randomstate)
    rng_split.shuffle(imbalanced_ids)
    forget_ids = imbalanced_ids[:nforget]
    test_ids = imbalanced_ids[nforget : nforget + ntest]
    split_map = (
        {cid: "forget" for cid in forget_ids}
        | {cid: "test" for cid in test_ids}
        | {cid: "retain" for cid in imbalanced_ids[nforget + ntest :]}
    )
    imbalanced["split"] = imbalanced["identity_id"].map(split_map)

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
        nimages=("imagepath", "count"),
    ).to_dict("index")

    (output / "datasetsummary_imbalanced.json").write_text(
        json.dumps(
            {
                "total_images": len(imbalanced),
                "nidentities": len(imbalanced_ids),
                "forget_pct": nforget / len(imbalanced_ids),
                "design": (
                    "Down-sampled to a 4.25:2:1 gradient: high up to 85 "
                    f"images, medium {medium_bin_images} images, "
                    f"low {low_bin_images} images."
                ),
                "images_per_bin": {
                    "high": 85,
                    "medium": medium_bin_images,
                    "low": low_bin_images,
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
          f"{n_high} high (up to 85 img), {n_medium} medium ({medium_bin_images} img), "
          f"{n_low} low ({low_bin_images} img).")
    return str(csv_path)


# ── Candidate (full-resolution) dataset columns ──
# Same core as OUTPUT_COLUMNS but excludes crop-level quality metrics
# (laplacian_variance, detection_confidence) — meaningless on 1024
# candidates before alignment.
CANDIDATE_OUTPUT_COLUMNS = [
    "image_path", "identity_id", "age_group", "age", "gender",
    "split", "forget_step", "forget_variant",
    "arcface_similarity", "pose", "expression", "lighting",
    "setting", "camera",
]


def build_candidates_dataset(
    identitydir: str,
    embeddingsdir: str,
    outputdir: str,
    nforget: int = 40,
    ntest: int = 60,
    forget_steps: int = 15,
    randomstate: int = 42,
) -> str:
    """Build the full-resolution 1024×1024 candidate dataset.

    Reads the raw candidate manifest directly (not the processed identity
    manifest), merges with 1024-based embeddings (identity-level join),
    applies the same split/forget assignments as the balanced 224 dataset,
    and outputs ``dataset_candidates.csv`` / ``.parquet``.

    Differs from :func:`build_dataset` in that:
    - Images are 1024×1024 candidates, not 224×224 aligned crops
    - No ``laplacian_variance`` or ``detection_confidence`` (crop metrics)
    - All 85 candidates per identity are included (no quality trim)
    - 15 output columns (vs. 17 for the standard dataset)
    """
    manifest = pd.read_csv(Path(identitydir) / "identities" / "raw_candidate_manifest.csv")
    # The manifest may use 'clusterid' if merged by an older script.
    id_col = "identity_id" if "identity_id" in manifest.columns else "clusterid"
    manifest.rename(columns={id_col: "identity_id", "raw_candidatepath": "imagepath"}, inplace=True)

    candidates = manifest[["imagepath", "identity_id", "trial"]].copy()
    candidates.drop_duplicates(inplace=True)

    attributes = _load_attributes(embeddingsdir)
    attrs_merge = attributes[
        ["identity_id", "trial", "agegroup", "age", "gender", "embedding"]
    ].copy()

    final = candidates.merge(attrs_merge, on=["identity_id", "trial"], how="left")

    # Metadata columns from the manifest.
    metadata = _load_candidate_metadata(identitydir)
    if not metadata.empty:
        final = final.merge(metadata, on="identity_id", how="left", validate="many_to_one")

    # ArcFace similarity for each candidate to its identity's candidate mean.
    final = _add_arcface_similarity(final)

    identities = sorted(final["identity_id"].unique())
    if nforget + ntest >= len(identities):
        raise ValueError("Split sizes must leave at least one retain identity.")

    rng = np.random.RandomState(randomstate)
    rng.shuffle(identities)
    forget_ids = identities[:nforget]
    test = identities[nforget : nforget + ntest]
    split_map = (
        {i: "forget" for i in forget_ids}
        | {i: "test" for i in test}
        | {i: "retain" for i in identities[nforget + ntest :]}
    )
    final["split"] = final["identity_id"].map(split_map)

    step_map = _distribute_forget(forget_ids, nforget, forget_steps, rng)
    final["forgetstep"] = (
        final["identity_id"].map({cid: step for cid, (step, _) in step_map.items()})
        .fillna(-1).astype(int)
    )
    final["forgetvariant"] = (
        final["identity_id"].map({cid: variant for cid, (_, variant) in step_map.items()})
        .fillna(-1).astype(int)
    )

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
    for col in CANDIDATE_OUTPUT_COLUMNS:
        if col not in output_df.columns:
            output_df[col] = "" if col in ("pose", "expression", "lighting", "setting", "camera") else 0.0
    output_df = output_df[CANDIDATE_OUTPUT_COLUMNS]
    # Candidates manifest paths are relative to the identities dir, so base
    # the resolver at identitydir/identities (identitydir == dataroot here).
    output_df["image_path"] = output_df["image_path"].apply(
        lambda p: _to_dataroot_relative(p, Path(identitydir) / "identities", dataroot)
    )

    csv_path = output / "dataset_candidates.csv"
    parquet_path = output / "dataset_candidates.parquet"
    output_df.to_csv(csv_path, index=False)
    output_df.to_parquet(parquet_path, index=False)

    (output / "datasetsummary_candidates.json").write_text(
        json.dumps({
            "total_images": len(final),
            "nidentities": len(identities),
            "nforget_ids": nforget,
            "forget_steps": forget_steps,
            "forget_pct": nforget / len(identities),
            "split_sizes": {k: int(v) for k, v in final.groupby("split").size().items()},
            "columns": CANDIDATE_OUTPUT_COLUMNS,
        }, indent=2)
    )

    print(f"[OK] Candidates dataset: {len(final)} images across {len(identities)} identities.")
    return str(csv_path)


def build_candidates_imbalanced(
    identitydir: str,
    embeddingsdir: str,
    outputdir: str,
    nforget: int = 40,
    ntest: int = 60,
    forget_steps: int = 15,
    randomstate: int = 42,
    high_bin_pct: float = 0.10,
    low_bin_pct: float = 0.60,
    low_bin_images: int = 20,
    medium_bin_images: int = 40,
) -> str:
    """Imbalanced variant of the full-resolution candidate dataset.

    Same 85:40:20 gradient as the 224 imbalanced, applied to the candidate
    pool.  Shares ``identity_id``, ``split``, and forget assignments with
    the balanced candidates dataset.
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

    # Popularity bins (same algorithm as imbalanced crops).
    rng = np.random.RandomState(randomstate + 9999)
    shuffled = identities.copy()
    rng.shuffle(shuffled)
    n_high = max(1, int(len(shuffled) * high_bin_pct))
    n_low = max(1, int(len(shuffled) * low_bin_pct))
    n_medium = len(shuffled) - n_high - n_low
    high_ids = set(shuffled[:n_high])
    low_ids = set(shuffled[-n_low:])
    bin_map = {}
    for cid in shuffled:
        bin_map[cid] = "high" if cid in high_ids else ("low" if cid in low_ids else "medium")

    rng_prune = np.random.RandomState(randomstate + 9998)
    per_bin_images = {"low": low_bin_images, "medium": medium_bin_images, "high": 85}
    drop_mask = pd.Series(False, index=final.index)
    for cid in low_ids | (set(bin_map.keys()) - high_ids - low_ids):
        target = per_bin_images[bin_map[cid]]
        rows = final[final["identity_id"] == cid]
        n_keep = min(target, len(rows))
        keep_idx = set(rng_prune.choice(rows.index, n_keep, replace=False))
        drop_mask.loc[list(set(rows.index) - keep_idx)] = True
    imbalanced = final[~drop_mask].copy()

    # Add popularity annotation columns (same as build_imbalanced_dataset).
    imbalanced["popularity_bin"] = imbalanced["identity_id"].map(bin_map)
    id_counts = imbalanced.groupby("identity_id").size()
    imbalanced["images_per_identity"] = imbalanced["identity_id"].map(id_counts)

    # ── Split + forget assignment (MUST match the other three artifacts) ──
    # Design choice: identical algorithm and seed to build_dataset /
    # build_candidates_dataset / build_imbalanced_dataset — same identity
    # pool, same randomstate — so the same identities are forget/test/retain
    # in all four artifacts.  Cross-variant comparison (balanced vs
    # imbalanced, 224 vs 1024) is only valid if split membership is
    # invariant; this block guarantees it.
    imbalanced_ids = sorted(imbalanced["identity_id"].unique())
    if nforget + ntest >= len(imbalanced_ids):
        raise ValueError("Split sizes must leave at least one retain identity.")
    rng_split = np.random.RandomState(randomstate)
    rng_split.shuffle(imbalanced_ids)
    forget_ids = imbalanced_ids[:nforget]
    test_ids = imbalanced_ids[nforget : nforget + ntest]
    split_map = (
        {cid: "forget" for cid in forget_ids}
        | {cid: "test" for cid in test_ids}
        | {cid: "retain" for cid in imbalanced_ids[nforget + ntest :]}
    )
    imbalanced["split"] = imbalanced["identity_id"].map(split_map)

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

    output_columns = CANDIDATE_OUTPUT_COLUMNS + ["popularity_bin", "images_per_identity"]
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)
    dataroot = output.parent
    output_df = imbalanced.rename(
        columns={
            "imagepath": "image_path",
            "agegroup": "age_group",
            "forgetstep": "forget_step",
            "forgetvariant": "forget_variant",
        }
    )
    for col in CANDIDATE_OUTPUT_COLUMNS:
        if col not in output_df.columns:
            output_df[col] = "" if col in ("pose", "expression", "lighting", "setting", "camera") else 0.0
    output_df = output_df[output_columns]
    output_df["image_path"] = output_df["image_path"].apply(
        lambda p: _to_dataroot_relative(p, Path(identitydir) / "identities", dataroot)
    )

    csv_path = output / "dataset_candidates_imbalanced.csv"
    parquet_path = output / "dataset_candidates_imbalanced.parquet"
    output_df.to_csv(csv_path, index=False)
    output_df.to_parquet(parquet_path, index=False)

    bin_stats = imbalanced.groupby("popularity_bin").agg(
        nidentities=("identity_id", "nunique"),
        nimages=("imagepath", "count"),
    ).to_dict("index")

    (output / "datasetsummary_candidates_imbalanced.json").write_text(
        json.dumps({
            "total_images": len(imbalanced),
            "nidentities": len(identities),
            "forget_pct": nforget / len(identities),
            "images_per_bin": {"high": 85, "medium": medium_bin_images, "low": low_bin_images},
            "bin_sizes": {"high_pct": high_bin_pct, "low_pct": low_bin_pct,
                          "n_high": n_high, "n_medium": n_medium, "n_low": n_low},
            "per_bin": {bin: {"identities": s["nidentities"], "images": s["nimages"]}
                        for bin, s in bin_stats.items()},
            "columns": CANDIDATE_OUTPUT_COLUMNS + ["popularity_bin", "images_per_identity"],
        }, indent=2)
    )

    print(f"[OK] Candidates imbalanced: {len(imbalanced)} images across "
          f"{len(identities)} identities — "
          f"{n_high} high, {n_medium} medium, {n_low} low.")
    return str(csv_path)

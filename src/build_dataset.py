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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

OUTPUT_COLUMNS = [
    "image_path", "clusterid", "age_group", "age", "gender",
    "split", "forget_step", "forget_variant",
    "arcface_similarity", "laplacian_variance", "detection_confidence",
]


def _load_attributes(embeddingsdir: str) -> pd.DataFrame:
    """Load image-level attributes saved by extract_embeddings.py."""
    embeddings = Path(embeddingsdir)
    return pd.DataFrame(
        {
            "imagepath": np.load(embeddings / "imagepaths.npy").astype(str),
            "agegroup": np.load(embeddings / "agegroups.npy"),
            "age": np.load(embeddings / "ages.npy"),
            "gender": np.load(embeddings / "genders.npy"),
            "embedding": list(np.load(embeddings / "embeddings.npy")),
        }
    )


def _add_arcface_similarity(final: pd.DataFrame) -> pd.DataFrame:
    """Add per-image cosine similarity to the identity's mean embedding.

    Confound control: an image far from its identity's canonical face
    (unusual angle/expression) is memorised differently from a canonical
    portrait.  Without this control, per-identity MIA AUC is confounded
    by within-identity variance.
    """
    from common import normalised_cosine_similarity

    # Mean embedding per cluster (embeddings are already L2-normalised).
    means = final.groupby("clusterid")["embedding"].mean()
    final["arcface_similarity"] = [
        normalised_cosine_similarity(row.embedding, means[row.clusterid])
        for row in final.itertuples(index=False)
    ]
    return final


def _distribute_forget(
    forget_ids: list[int], nforget: int, forget_steps: int, rng: np.random.RandomState,
) -> dict[int, tuple[int, int]]:
    """Assign forget identities to steps with uniform+remainder distribution.

    Returns ``{clusterid: (step, variant)}`` where *step* is 0-based and
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
    final = manifest[["imagepath", "clusterid", "detection_confidence", "laplacian_variance"]].merge(
        attributes, on="imagepath", how="inner", validate="one_to_one"
    )
    if len(final) != len(manifest):
        print(f"Manifest: {len(manifest)} rows, final: {len(final)} rows")
        print(f"  Manifest sample: {manifest['imagepath'].iloc[0]}")
        print(f"  Attributes sample: {attributes['imagepath'].iloc[0]}")
        only_manifest = set(manifest["imagepath"]) - set(attributes["imagepath"])
        only_attrs = set(attributes["imagepath"]) - set(manifest["imagepath"])
        if only_manifest:
            print(f"  In manifest only ({len(only_manifest)}): {sorted(only_manifest)[:3]}")
        if only_attrs:
            print(f"  In attributes only ({len(only_attrs)}): {sorted(only_attrs)[:3]}")
        raise RuntimeError("Attribute extraction is missing final images.")

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
        final.sort_values(["clusterid", "detection_confidence"], ascending=[True, False])
        .groupby("clusterid")
        .head(imagesperidentity)
        .reset_index(drop=True)
    )

    identities = sorted(final.clusterid.unique())
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
    final["split"] = final.clusterid.map(split_map)

    # Distribute forget identities across steps with uniform base + remainder.
    step_map = _distribute_forget(forget_ids, nforget, forget_steps, rng)
    final["forgetstep"] = (
        final.clusterid.map(
            {cid: step for cid, (step, _) in step_map.items()}
        )
        .fillna(-1)
        .astype(int)
    )
    final["forgetvariant"] = (
        final.clusterid.map(
            {cid: variant for cid, (_, variant) in step_map.items()}
        )
        .fillna(-1)
        .astype(int)
    )
    final = final[final.agegroup != -1].copy()

    # Cluster samples: asymmetric matplotlib grid per identity.
    samples_root = Path(outputdir) / "cluster_samples"
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
            .drop_duplicates("clusterid")
            .set_index("clusterid")
            .seedpath
        )
    else:
        seeds = pd.Series(dtype=str)

    for cid in sorted(final.clusterid.unique()):
        split = split_map[cid]
        (samples_root / split).mkdir(parents=True, exist_ok=True)

        # Collect up to 4 images: seed + 3 examples
        image_paths = []
        seed_path = seeds.get(cid, None)
        if seed_path and Path(seed_path).exists():
            image_paths.append(seed_path)
        examples = final[final.clusterid == cid].imagepath.head(3)
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
    )[OUTPUT_COLUMNS]
    output_df["image_path"] = (
        output_df["image_path"]
        .apply(lambda p: str((Path(identitydir) / p).relative_to(dataroot)))
    )
    csv_path = output / "dataset.csv"
    parquet_path = output / "dataset.parquet"
    output_df.to_csv(csv_path, index=False)
    output_df.to_parquet(parquet_path, index=False)
    (output / "datasetsummary.json").write_text(
        json.dumps(
            {
                "total_images": len(final),
                "nclusters": len(identities),
                "nforget_ids": nforget,
                "forget_steps": forget_steps,
                "distribution": {
                    "base_per_step": nforget // forget_steps,
                    "remainder_steps": nforget % forget_steps,
                },
                "split_sizes": {
                    key: int(value)
                    for key, value in final.groupby("split").size().items()
                },
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
    final = manifest[["imagepath", "clusterid", "detection_confidence", "laplacian_variance"]].merge(
        attributes, on="imagepath", how="inner", validate="one_to_one"
    )
    if len(final) != len(manifest):
        raise RuntimeError("Attribute extraction is missing final images.")

    # Per-image cosine similarity to the identity's mean embedding.
    # Computed BEFORE pruning so the reference mean spans ALL quality-passing
    # crops from preprocess — the identity's canonical face.  The column is
    # therefore dataset-invariant: a given image has the same arcface_similarity
    # in the balanced and imbalanced variants, making it a valid confound
    # control across both.  Pruning below only removes rows; it never changes
    # the values.
    final = _add_arcface_similarity(final)

    identities = sorted(final.clusterid.unique())

    # ── 2. Assign popularity bins ──
    # Design choice: seeded random shuffle ensures reproducibility and avoids
    # any correlation with the identity's original cluster ID (which may be
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
        rows = final[final.clusterid == cid]
        n_keep = min(target, len(rows))
        keep_idx = set(rng_prune.choice(rows.index, n_keep, replace=False))
        drop_idx = set(rows.index) - keep_idx
        drop_mask.loc[list(drop_idx)] = True

    imbalanced = final[~drop_mask].copy()

    # ── 4. Validate post-pruning structure ──
    new_sizes = imbalanced.groupby("clusterid").size()
    for cid, expected in [(cid, low_bin_images) for cid in low_ids] + [
        (cid, medium_bin_images) for cid in set(bin_map.keys()) - high_ids - low_ids
    ]:
        actual = new_sizes.get(cid, 0)
        if actual != expected:
            raise RuntimeError(
                f"Identity {cid} ({bin_map[cid]}): "
                f"expected {expected} images, got {actual}"
            )

    # ── 5. Assign splits on the imbalanced set ──
    # Design choice: split assignment uses the same identity pool as balanced,
    # only the per-ID image count differs.  This ensures the same identities
    # are forget/test/retain in both variants, making cross-variant comparison
    # valid.
    imbalanced_ids = sorted(imbalanced.clusterid.unique())
    rng_split = np.random.RandomState(randomstate)
    rng_split.shuffle(imbalanced_ids)
    forget_ids = imbalanced_ids[:nforget]
    test_ids = imbalanced_ids[nforget : nforget + ntest]
    split_map = (
        {cid: "forget" for cid in forget_ids}
        | {cid: "test" for cid in test_ids}
        | {cid: "retain" for cid in imbalanced_ids[nforget + ntest :]}
    )
    imbalanced["split"] = imbalanced.clusterid.map(split_map)

    # Forget step distribution (same algorithm as balanced).
    step_map = _distribute_forget(forget_ids, nforget, forget_steps, rng_split)
    imbalanced["forgetstep"] = (
        imbalanced.clusterid.map(
            {cid: step for cid, (step, _) in step_map.items()}
        )
        .fillna(-1)
        .astype(int)
    )
    imbalanced["forgetvariant"] = (
        imbalanced.clusterid.map(
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
    imbalanced["popularity_bin"] = imbalanced.clusterid.map(bin_map)
    id_counts = imbalanced.groupby("clusterid").size()
    imbalanced["images_per_identity"] = imbalanced.clusterid.map(id_counts)

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
    )[output_columns]
    output_df["image_path"] = (
        output_df["image_path"]
        .apply(lambda p: str((Path(identitydir) / p).relative_to(dataroot)))
    )

    csv_path = output / "dataset_imbalanced.csv"
    parquet_path = output / "dataset_imbalanced.parquet"
    output_df.to_csv(csv_path, index=False)
    output_df.to_parquet(parquet_path, index=False)

    # Summary with per-bin breakdown.
    bin_stats = imbalanced.groupby("popularity_bin").agg(
        nidentities=("clusterid", "nunique"),
        nimages=("image_path", "count"),
    ).to_dict("index")

    (output / "datasetsummary_imbalanced.json").write_text(
        json.dumps(
            {
                "total_images": len(imbalanced),
                "nclusters": len(imbalanced_ids),
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
            },
            indent=2,
        )
    )

    print(f"[OK] Imbalanced dataset: {len(imbalanced)} images across "
          f"{len(imbalanced_ids)} identities — "
          f"{n_high} high (up to 85 img), {n_medium} medium ({medium_bin_images} img), "
          f"{n_low} low ({low_bin_images} img).")
    return str(csv_path)

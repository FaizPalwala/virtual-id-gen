"""
build_dataset.py

Constructs the final Phase 2 dataset mapping by merging generated identity 
manifests with extracted embedding attributes. Applies the train/test/forget splits.
Build the backward-compatible final ``dataset.csv`` file.
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
    "split", "forget_step",
]


def _normalise_image_path(path_str: str) -> str:
    """Strip machine-specific prefix, keeping the stable relative suffix.

    Absolute paths vary by machine (macOS vs HPC), but the portion from
    ``images/`` onward is identical.  This lets us merge manifests and
    attributes generated on different hosts.
    """
    parts = Path(path_str).parts
    try:
        idx = parts.index("images")
        return str(Path(*parts[idx:]))
    except ValueError:
        return path_str


def build_dataset(
    identitydir: str,
    embeddingsdir: str,
    outputdir: str,
    nforget: int = 40,
    ntest: int = 60,
    randomstate: int = 42,
) -> str:
    """Merge final manifest and attributes without changing CSV output schema."""
    manifest = pd.read_csv(Path(identitydir) / "identitymanifest.csv")
    embeddings = Path(embeddingsdir)
    attributes = pd.DataFrame(
        {
            "imagepath": np.load(embeddings / "imagepaths.npy").astype(str),
            "agegroup": np.load(embeddings / "agegroups.npy"),
            "age": np.load(embeddings / "ages.npy"),
            "gender": np.load(embeddings / "genders.npy"),
        }
    )
    # Normalise to machine-agnostic relative paths for cross-host merge.
    manifest["_rel"] = manifest["imagepath"].apply(_normalise_image_path)
    attributes["_rel"] = attributes["imagepath"].apply(_normalise_image_path)
    final = manifest[["_rel", "imagepath", "clusterid"]].merge(
        attributes[["_rel", "agegroup", "age", "gender"]],
        on="_rel", how="inner", validate="one_to_one",
    ).drop(columns="_rel")
    if len(final) != len(manifest):
        print(f"Manifest: {len(manifest)} rows, final: {len(final)} rows")
        print(f"  Manifest sample: {manifest['imagepath'].iloc[0]}")
        print(f"  Attributes sample: {attributes['imagepath'].iloc[0]}")
        manifest_rels = set(manifest["imagepath"].apply(_normalise_image_path))
        attr_rels = set(attributes["imagepath"].apply(_normalise_image_path))
        only_manifest = manifest_rels - attr_rels
        only_attrs = attr_rels - manifest_rels
        if only_manifest:
            print(f"  In manifest only ({len(only_manifest)}): {sorted(only_manifest)[:3]}")
        if only_attrs:
            print(f"  In attributes only ({len(only_attrs)}): {sorted(only_attrs)[:3]}")
        raise RuntimeError("Attribute extraction is missing final images.")
    sizes = final.groupby("clusterid").size()
    if sizes.nunique() != 1:
        raise RuntimeError(f"Uneven identity clusters: {sizes.to_dict()}")
    identities = sorted(final.clusterid.unique())
    if nforget + ntest >= len(identities):
        raise ValueError("Split sizes must leave at least one retain identity.")
    rng = np.random.RandomState(randomstate)
    rng.shuffle(identities)
    forget, test = identities[:nforget], identities[nforget : nforget + ntest]
    split_map = (
        {identity: "forget" for identity in forget}
        | {identity: "test" for identity in test}
        | {identity: "retain" for identity in identities[nforget + ntest :]}
    )
    final["split"] = final.clusterid.map(split_map)
    final["forgetstep"] = (
        final.clusterid.map(
            {identity: step // 2 for step, identity in enumerate(forget)}
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

    raw_manifest = Path(identitydir).parent / "merged" / "identities" / "raw_candidate_manifest.csv"
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
        for i, img_path in enumerate(examples, 1):
            if Path(img_path).exists():
                image_paths.append(img_path)

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
        }
    )[OUTPUT_COLUMNS]
    output_df["image_path"] = (
        output_df["image_path"]
        .apply(lambda p: str(Path(p).relative_to(dataroot)))
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
                "nforgetsteps": nforget,
                "split_sizes": {
                    key: int(value)
                    for key, value in final.groupby("split").size().items()
                },
            },
            indent=2,
        )
    )
    return str(csv_path)


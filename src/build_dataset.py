"""
build_dataset.py

Constructs the final Phase 2 dataset mapping by merging generated identity 
manifests with extracted embedding attributes. Applies the train/test/forget splits.
Build the backward-compatible final ``dataset.csv`` file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUTPUT_COLUMNS = ["imagepath", "clusterid", "agegroup", "split", "forgetstep"]


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
        }
    )
    final = manifest[["imagepath", "clusterid"]].merge(
        attributes, on="imagepath", how="inner", validate="one_to_one"
    )
    if len(final) != len(manifest):
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
        final.clusterid.map({identity: step for step, identity in enumerate(forget)})
        .fillna(-1)
        .astype(int)
    )
    final = final[final.agegroup != -1].copy()

    # Cluster samples: 2×2 grid (seed + 3 examples) per identity.
    CELL = 256
    samples_root = Path(outputdir) / "cluster_samples"
    if samples_root.exists():
        import shutil

        shutil.rmtree(samples_root)
    samples_root.mkdir(parents=True)

    raw_manifest = Path(identitydir) / "raw_candidate_manifest.csv"
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
        labels = []
        seed_path = seeds.get(cid, None)
        if seed_path and Path(seed_path).exists():
            image_paths.append(seed_path)
            labels.append("seed")
        examples = final[final.clusterid == cid].imagepath.head(3)
        for i, img_path in enumerate(examples, 1):
            if Path(img_path).exists():
                image_paths.append(img_path)
                labels.append(f"ex.{i}")

        if not image_paths:
            continue

        grid = Image.new("RGB", (CELL * 2, CELL * 2), (255, 255, 255))
        positions = [(0, 0), (CELL, 0), (0, CELL), (CELL, CELL)]
        for pos, img_path, label in zip(positions, image_paths, labels):
            img = Image.open(img_path).convert("RGB").resize((CELL, CELL))
            grid.paste(img, pos)

            draw = ImageDraw.Draw(grid)
            # Drop-shadow for readability
            for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                draw.text((pos[0] + 5 + dx, pos[1] + 5 + dy), label, fill=(0, 0, 0))
            draw.text((pos[0] + 5, pos[1] + 5), label, fill=(255, 255, 255))

        out = samples_root / split / f"identity_{int(cid):03d}.png"
        grid.save(out)

    # ------------------------------------------------------------------
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)
    output_df = final[OUTPUT_COLUMNS]
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


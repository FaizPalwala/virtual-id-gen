"""
build_dataset.py

Constructs the final Phase 2 dataset mapping by merging generated identity 
manifests with extracted embedding attributes. Applies the train/test/forget splits.
Build the backward-compatible final ``sfhqdataset.csv`` file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "sfhqdataset.csv"
    final[OUTPUT_COLUMNS].to_csv(csv_path, index=False)
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


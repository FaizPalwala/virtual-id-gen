"""Shared fixtures for the SFHQ-VirtualID test suite.

build_dataset imports matplotlib.pyplot at module level; force the Agg
backend before any test module imports it so the suite runs headless on
CI and macOS (no display server needed).
"""
import matplotlib
import numpy as np
import pandas as pd
import pytest


matplotlib.use("Agg", force=True)


@pytest.fixture
def dataset_inputs(tmp_path):
    """Build a synthetic pipeline output tree matching production layout.

    Mirrors the HPC layout::

        <root>/processed/identitymanifest.csv   (224 crops, per (id, trial))
        <root>/identities/raw_candidate_manifest.csv  (1024 candidates)
        <root>/embeddings/*.npy                 (attributes per candidate)
        <root>/dataset/                         (build output)

    Every identity has ``candidates_per_id`` candidates; the 224 crop
    manifest is one row per candidate (preprocess keeps quality-passing
    crops).  The raw manifest carries metadata columns (pose, expression,
    lighting, setting, camera) plus ``seedpath`` for cluster samples.
    """
    n_ids = 10
    candidates_per_id = 85
    root = tmp_path
    processed = root / "processed"
    identities = root / "identities"
    embeddings = root / "embeddings"
    dataset = root / "dataset"
    processed.mkdir()
    identities.mkdir()
    embeddings.mkdir()

    crop_rows = []
    cand_rows = []
    for cid in range(n_ids):
        for trial in range(candidates_per_id):
            crop_rows.append({
                "imagepath": f"images/identity_{cid:03d}/accepted_{trial:03d}.jpg",
                "identity_id": cid,
                "trial": trial,
                "detection_confidence": 0.9,
                "laplacian_variance": 100.0 + trial,
            })
            cand_rows.append({
                "identityid": cid,
                "clusterid": cid,
                "trial": trial,
                "raw_candidatepath": f"candidates/identity_{cid:03d}/candidate_{trial:03d}.png",
                "seedpath": f"seeds/seed_{cid:03d}.png",
                "pose": "frontal", "expression": "neutral",
                "lighting": "studio", "setting": "backdrop", "camera": "85mm",
            })
    pd.DataFrame(crop_rows).to_csv(processed / "identitymanifest.csv", index=False)
    pd.DataFrame(cand_rows).to_csv(identities / "raw_candidate_manifest.csv", index=False)

    # Embeddings: one-hot-ish per identity so the mean embedding is stable;
    # trial derived from candidate_YYY in the path (extract contract).
    n = len(cand_rows)
    rng = np.random.RandomState(42)
    emb = rng.randn(n, 8).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    emb += np.eye(8)[np.array([r % 8 for r in range(n)])].astype(np.float32) * 0.5
    np.save(embeddings / "imagepaths.npy",
            np.array([r["raw_candidatepath"] for r in cand_rows]))
    np.save(embeddings / "agegroups.npy", np.ones(n, dtype=np.int8))
    np.save(embeddings / "ages.npy", np.full(n, 30, dtype=np.int8))
    np.save(embeddings / "genders.npy", np.zeros(n, dtype=np.int8))
    np.save(embeddings / "embeddings.npy", emb)
    np.save(embeddings / "identity_ids.npy",
            np.array([r["clusterid"] for r in cand_rows]))

    return {
        "root": str(root),
        "processed": str(processed),
        "identities": str(identities),
        "embeddings": str(embeddings),
        "dataset": str(dataset),
        "n_ids": n_ids,
        "candidates_per_id": candidates_per_id,
    }

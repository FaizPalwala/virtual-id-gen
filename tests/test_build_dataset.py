import numpy as np
import pandas as pd
from build_dataset import OUTPUT_COLUMNS, build_dataset


def test_build_preserves_legacy_csv_schema(tmp_path):
    identities, embeddings, output = tmp_path / "identities", tmp_path / "emb", tmp_path / "out"
    identities.mkdir(); embeddings.mkdir()

    # Manifest contract: imagepath (crop path), identity_id, trial — the
    # (identity_id, trial) join key links each 224 crop to its parent 1024
    # candidate embedding.
    rows = [
        {"imagepath": f"images/identity_{cluster:03d}/accepted_{number:03d}.jpg",
         "identity_id": cluster, "trial": number,
         "detection_confidence": 0.9, "laplacian_variance": 100.0}
        for cluster in range(3) for number in range(2)
    ]
    pd.DataFrame(rows).to_csv(identities / "identitymanifest.csv", index=False)

    # Embeddings contract: imagepaths (candidate paths, trial from
    # candidate_YYY), agegroup (must be != -1), age, gender, embedding,
    # identity_ids.
    n = len(rows)
    np.save(embeddings / "imagepaths.npy",
            np.array([f"candidates/identity_{c:03d}/candidate_{t:03d}.png"
                      for c in range(3) for t in range(2)]))
    np.save(embeddings / "agegroups.npy", np.ones(n, dtype=np.int8))
    np.save(embeddings / "ages.npy", np.full(n, 30, dtype=np.int8))
    np.save(embeddings / "genders.npy", np.zeros(n, dtype=np.int8))
    np.save(embeddings / "embeddings.npy", np.eye(3)[np.array([0, 0, 1, 1, 2, 2])].astype(np.float32))
    np.save(embeddings / "identity_ids.npy", np.array([0, 0, 1, 1, 2, 2]))

    # Raw candidate manifest (metadata loader falls back to
    # identitydir/raw_candidate_manifest.csv when the identities/ sibling
    # is absent in a tmp test tree).
    pd.DataFrame(
        {"clusterid": [0, 1, 2], "seedpath": ["s0.png", "s1.png", "s2.png"]}
    ).to_csv(identities / "raw_candidate_manifest.csv", index=False)

    result = pd.read_csv(
        build_dataset(str(identities), str(embeddings), str(output),
                      nforget=1, forget_steps=15, randomstate=42,
                      imagesperidentity=2, holdout_frac=0.20, min_holdout=1)
    )
    assert list(result.columns) == OUTPUT_COLUMNS
    assert len(result) == 6
    # MUFAC invariant: every identity contributes BOTH train and holdout.
    subsets = result.groupby("identity_id")["image_subset"].apply(set)
    assert all(s == {"train", "holdout"} for s in subsets), subsets.to_dict()
    # Identity-level split isolation: exactly one split per identity.
    assert result.groupby("identity_id")["split"].nunique().eq(1).all()
    # Bench strips prompt metadata (no pose/expression/lighting/setting/camera).
    for col in ("pose", "expression", "lighting", "setting", "camera"):
        assert col not in result.columns, f"{col} should not exist in bench"

    # ── BOTH forget schedules ship as columns ──
    # uniform `forget_step`: equal ids per step (1 forget id over 15 steps
    # with nforget=1 → step 0 carries it; -1 elsewhere is retain).
    forget = result[result["split"] == "forget"]
    assert len(forget) == 2  # 1 forget identity × 2 images
    steps = forget["forget_step"].unique()
    assert len(steps) == 1 and 0 <= steps[0] < 15
    # poisson `forget_step_poisson`: same forget set, valid step range.
    psteps = forget["forget_step_poisson"].unique()
    assert len(psteps) == 1 and 0 <= psteps[0] < 15
    # Retain rows carry -1 in both schedule columns.
    retain = result[result["split"] == "retain"]
    assert (retain["forget_step"] == -1).all()
    assert (retain["forget_step_poisson"] == -1).all()

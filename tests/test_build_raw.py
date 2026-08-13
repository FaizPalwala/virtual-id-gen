"""Contract tests for build_raw_dataset — the max-size Raw release.

The Raw release is the general-purpose 1024×1024 reference: all
candidates per identity, NO split protocol (no split / forget_step /
forget_step_poisson / image_subset columns).
"""
import pandas as pd

from build_dataset import RAW_OUTPUT_COLUMNS, build_raw_dataset


def test_raw_is_max_size_without_splits(dataset_inputs):
    result_path = build_raw_dataset(
        dataset_inputs["root"],
        dataset_inputs["embeddings"],
        dataset_inputs["dataset"] + "_raw",
        randomstate=42,
        jpeg_expected_size=(64, 64),  # fixture candidates are 64x64
    )
    df = pd.read_csv(result_path)

    # In-place JPEG conversion is UNCONDITIONAL: shipped paths are .jpg
    assert df["image_path"].str.endswith(".jpg").all()

    # All candidates shipped, no quality trim.
    assert len(df) == dataset_inputs["n_ids"] * dataset_inputs["candidates_per_id"]
    assert df.groupby("identity_id").size().nunique() == 1  # uniform 100/id
    assert df["identity_id"].nunique() == dataset_inputs["n_ids"]

    # Exactly the documented columns — nothing else.
    assert list(df.columns) == RAW_OUTPUT_COLUMNS

    # No split protocol columns exist at all.
    for col in ("split", "forget_step", "forget_step_poisson", "image_subset",
                "laplacian_variance", "detection_confidence", "popularity_bin"):
        assert col not in df.columns, f"{col} should not exist in Raw"

    # Metadata joined PER CANDIDATE (20×5 prompt grid — each portrait has
    # its own pose/expression/lighting).  This is the contract the D1 fix
    # restores: the old per-identity collapse reported one value per
    # identity, which the build's variance guard now rejects.
    for col in ("pose", "expression", "lighting"):
        per_id_var = df.groupby("identity_id")[col].nunique()
        assert (per_id_var > 1).all(), \
            f"{col} is constant per identity — per-candidate metadata lost"
    # fixture: 4 poses / 3 expressions / 5 lightings per identity
    assert df["pose"].nunique() == 4
    assert df["expression"].nunique() == 3
    assert df["lighting"].nunique() == 5

    # Every row has real demographics + similarity.  Cosine similarity to
    # the identity mean spans (−1, 1]; the frozen release is −0.08…0.97.
    assert df["age_group"].notna().all()
    assert df["gender"].notna().all()
    assert df["arcface_similarity"].between(-1, 1).all()

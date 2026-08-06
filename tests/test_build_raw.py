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
    )
    df = pd.read_csv(result_path)

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

    # Metadata joined per identity (same prompt grid for all candidates).
    for col in ("pose", "expression", "lighting", "setting", "camera"):
        assert (df[col] == "frontal").all() or (df[col] == "studio").all() or \
            (df[col] == "backdrop").all() or (df[col] == "85mm").all() or \
            (df[col] == "neutral").all(), f"{col} not populated from manifest"

    # Every row has real demographics + similarity.  Cosine similarity to
    # the identity mean spans (−1, 1]; the frozen release is −0.08…0.97.
    assert df["age_group"].notna().all()
    assert df["gender"].notna().all()
    assert df["arcface_similarity"].between(-1, 1).all()

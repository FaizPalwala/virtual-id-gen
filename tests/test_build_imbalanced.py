"""Contract tests for build_imbalanced_dataset — the MUFAC invariants.

These lock the data-freeze properties that the dissertation depends on:
  - holdout is the same per-identity reserve as balanced (18/id with
    defaults), uniform across ALL popularity bins
  - train gradient is ratio-based 5:1 (82:41:16; max_train = candidates - holdout)
  - identity-level split assignment is IDENTICAL to the balanced release
  - every identity contributes both image_subset values
  - NO forget-schedule columns (schedule axis lives in balanced)
"""
import numpy as np
import pandas as pd

from build_dataset import (
    _compute_holdout_size,
    _distribute_forget,
    _distribute_forget_poisson,
    build_dataset,
    build_imbalanced_dataset,
)


def _read(csv_path):
    return pd.read_csv(csv_path)


def test_holdout_formula_matches_balanced_reserve():
    # Defaults: imagesperidentity=90, holdout_frac=0.20, min_holdout=5
    assert _compute_holdout_size(90, 0.20, 5) == 18
    # Scales: 500 images @ 20% → 100 holdout (guidance §2.2 example)
    assert _compute_holdout_size(500, 0.20, 5) == 100
    # Floor: thin datasets never get probe-breaking holdouts
    assert _compute_holdout_size(10, 0.20, 5) == 5


def test_distribute_forget_uniform_plus_remainder():
    ids = list(range(60))
    rng = np.random.RandomState(42)
    mapping = _distribute_forget(ids, 60, 15, rng)
    steps = [s for s, _ in mapping.values()]
    counts = pd.Series(steps).value_counts().sort_index()
    # 60 ids over 15 steps → 4 per step, uniform
    assert set(counts) == {4}
    assert all(0 <= s < 15 for s in steps)
    # Variants are 0-based within each step
    for step, variant in mapping.values():
        assert variant == sum(1 for s, v in mapping.values() if s == step and v < variant)


def test_distribute_forget_poisson_totals_and_shape():
    """Seeded-Poisson forget-variant contract.

    Same 15 steps and same total forget set as uniform, but per-step counts
    follow the deletion-request arrival distribution (arXiv:2507.15280,
    arXiv:2012.01668).  The schedule must be reproducible from its seed and
    MUST NOT be uniform (that's the whole point of the variant).
    """
    ids = list(range(75))
    rng = np.random.RandomState(42)
    mapping = _distribute_forget_poisson(ids, 75, 15, rng)
    steps = [s for s, _ in mapping.values()]

    # Total forget set preserved: every id assigned, exactly once.
    assert len(mapping) == 75
    assert sorted(mapping.keys()) == ids

    # Exactly 15 steps, all within range.
    assert len(set(steps)) == 15
    assert set(steps) == set(range(15))

    # Per-step counts sum to nforget.
    counts = pd.Series(steps).value_counts().sort_index()
    assert counts.sum() == 75

    # The variant is NOT uniform — steps carry different batch sizes.
    assert counts.nunique() > 1, f"expected non-uniform schedule, got {counts.to_dict()}"

    # Reproducible: same seed → identical schedule.
    rng2 = np.random.RandomState(42)
    mapping2 = _distribute_forget_poisson(list(range(75)), 75, 15, rng2)
    assert mapping == mapping2

    # Different seed → (overwhelmingly likely) different schedule.
    rng3 = np.random.RandomState(7)
    mapping3 = _distribute_forget_poisson(list(range(75)), 75, 15, rng3)
    assert mapping != mapping3


def test_imbalanced_holdout_and_gradient(dataset_inputs):
    root = dataset_inputs["root"]
    result_path = build_imbalanced_dataset(
        dataset_inputs["processed"],
        dataset_inputs["embeddings"],
        dataset_inputs["dataset"],
        nforget=2, randomstate=42,
        holdout_frac=0.20, min_holdout=5,
        imagesperidentity=90, candidatesperidentity=100,
    )
    df = _read(result_path)

    # Column contract: exactly the 12 imbalanced columns, no forget-schedule
    # columns (the schedule axis lives in balanced), no prompt metadata.
    assert list(df.columns) == [
        "image_path", "identity_id", "age_group", "age", "gender", "split",
        "image_subset", "arcface_similarity", "laplacian_variance",
        "detection_confidence", "popularity_bin", "images_per_identity",
    ], list(df.columns)
    for col in ("forget_step", "forget_step_poisson", "pose", "expression",
                "lighting", "setting", "camera"):
        assert col not in df.columns, f"{col} should not exist in imbalanced"

    # Every identity contributes BOTH subsets.
    subsets = df.groupby("identity_id")["image_subset"].apply(set)
    assert all(s == {"train", "holdout"} for s in subsets), subsets.to_dict()

    # Holdout is uniform: 18/id across ALL bins (probe stability).
    holdout = df[df.image_subset == "holdout"].groupby("identity_id").size()
    assert holdout.nunique() == 1 and holdout.iloc[0] == 18, \
        holdout.value_counts().to_dict()

    # Train gradient: max_train = 100 - 18 = 82 → ratios {1.0, 0.50, 0.20}
    # → 82:41:16 train per id (kept pools 100/59/34).
    train = df[df.image_subset == "train"].groupby("identity_id").size()
    bins = df.drop_duplicates("identity_id").set_index("identity_id")["popularity_bin"]
    per_bin = pd.DataFrame({"train": train, "bin": bins}).groupby("bin")["train"].mean()
    assert per_bin["high"] == 82, per_bin.to_dict()
    assert per_bin["medium"] == 41, per_bin.to_dict()
    assert per_bin["low"] == 16, per_bin.to_dict()

    # Kept pools: high keeps all 100, medium 59, low 34.
    kept = df.groupby("identity_id").size()
    kept_per_bin = pd.DataFrame({"kept": kept, "bin": bins}).groupby("bin")["kept"].mean()
    assert kept_per_bin["high"] == 100
    assert kept_per_bin["medium"] == 59
    assert kept_per_bin["low"] == 34

    # Split isolation: exactly one identity-level split each; forget=2.
    assert df.groupby("identity_id")["split"].nunique().eq(1).all()
    assert df.drop_duplicates("identity_id")["split"].value_counts().to_dict() == \
        {"retain": 8, "forget": 2}


def test_imbalanced_split_matches_balanced(dataset_inputs):
    """The imbalanced variant MUST share the balanced identity→split map."""
    balanced_path = build_dataset(
        dataset_inputs["processed"],
        dataset_inputs["embeddings"],
        dataset_inputs["dataset"] + "_bal",
        nforget=2, forget_steps=15, randomstate=42,
        imagesperidentity=90, holdout_frac=0.20, min_holdout=5,
    )
    imbalanced_path = build_imbalanced_dataset(
        dataset_inputs["processed"],
        dataset_inputs["embeddings"],
        dataset_inputs["dataset"] + "_imb",
        nforget=2, randomstate=42,
        holdout_frac=0.20, min_holdout=5,
        imagesperidentity=90, candidatesperidentity=100,
    )
    bal = _read(balanced_path).drop_duplicates("identity_id").set_index("identity_id")["split"]
    imb = _read(imbalanced_path).drop_duplicates("identity_id").set_index("identity_id")["split"]
    assert (bal != imb).sum() == 0
    assert bal.value_counts().to_dict() == {"retain": 8, "forget": 2}


def test_balanced_ships_both_forget_schedules(dataset_inputs):
    """The balanced artifact carries BOTH schedules as columns.

    `forget_step` = uniform (equal ids/step); `forget_step_poisson` =
    seeded arrival-model schedule.  Same forget set, different step
    assignment — and per-step counts differ (that's the stress test).
    """
    result_path = build_dataset(
        dataset_inputs["processed"],
        dataset_inputs["embeddings"],
        dataset_inputs["dataset"] + "_both",
        nforget=3, forget_steps=3, randomstate=42,
        imagesperidentity=90, holdout_frac=0.20, min_holdout=5,
    )
    df = _read(result_path)

    # Column contract: 12 balanced columns incl. both schedules.
    assert "forget_step" in df.columns and "forget_step_poisson" in df.columns
    assert "forget_variant" not in df.columns
    for col in ("pose", "expression", "lighting", "setting", "camera"):
        assert col not in df.columns, f"{col} should not exist in bench"

    forget = df[df["split"] == "forget"]
    n_ids = forget["identity_id"].nunique()
    assert n_ids == 3

    # Uniform: 3 ids / 3 steps → 1 per step, step index == cumulative.
    unif = forget.groupby("identity_id")["forget_step"].first()
    assert set(unif) == {0, 1, 2}

    # Poisson: same 3 ids, valid step range, all assigned.
    pois = forget.groupby("identity_id")["forget_step_poisson"].first()
    assert set(pois) <= {0, 1, 2}
    assert pois.notna().all()

    # Retain rows: -1 in both schedule columns.
    retain = df[df["split"] == "retain"]
    assert (retain["forget_step"] == -1).all()
    assert (retain["forget_step_poisson"] == -1).all()

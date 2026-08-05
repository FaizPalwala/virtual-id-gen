"""Contract tests for build_imbalanced_dataset — the MUFAC invariants.

These lock the data-freeze properties that the dissertation depends on:
  - holdout is the same per-identity reserve as balanced (15/id with
    defaults), uniform across ALL popularity bins
  - train gradient is ratio-based 70:40:20 (max_train = candidates - holdout)
  - identity-level split assignment is IDENTICAL to the balanced release
  - every identity contributes both image_subset values
"""
import numpy as np
import pandas as pd

from build_dataset import (
    _compute_holdout_size,
    _distribute_forget,
    build_dataset,
    build_imbalanced_dataset,
)


def _read(csv_path):
    return pd.read_csv(csv_path)


def test_holdout_formula_matches_balanced_reserve():
    # Defaults: imagesperidentity=75, holdout_frac=0.20, min_holdout=5
    assert _compute_holdout_size(75, 0.20, 5) == 15
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


def test_imbalanced_holdout_and_gradient(dataset_inputs):
    root = dataset_inputs["root"]
    result_path = build_imbalanced_dataset(
        dataset_inputs["processed"],
        dataset_inputs["embeddings"],
        dataset_inputs["dataset"],
        nforget=2, forget_steps=15, randomstate=42,
        holdout_frac=0.20, min_holdout=5,
        imagesperidentity=75, candidatesperidentity=85,
    )
    df = _read(result_path)

    # Every identity contributes BOTH subsets.
    subsets = df.groupby("identity_id")["image_subset"].apply(set)
    assert all(s == {"train", "holdout"} for s in subsets), subsets.to_dict()

    # Holdout is uniform: 15/id across ALL bins (probe stability).
    holdout = df[df.image_subset == "holdout"].groupby("identity_id").size()
    assert holdout.nunique() == 1 and holdout.iloc[0] == 15, \
        holdout.value_counts().to_dict()

    # Train gradient: max_train = 85 - 15 = 70 → ratios {1.0, 0.57, 0.29}
    # → 70:40:20 train per id (kept pools 85/55/35).
    train = df[df.image_subset == "train"].groupby("identity_id").size()
    bins = df.drop_duplicates("identity_id").set_index("identity_id")["popularity_bin"]
    per_bin = pd.DataFrame({"train": train, "bin": bins}).groupby("bin")["train"].mean()
    assert per_bin["high"] == 70, per_bin.to_dict()
    assert per_bin["medium"] == 40, per_bin.to_dict()
    assert per_bin["low"] == 20, per_bin.to_dict()

    # Kept pools: high keeps all 85, medium 55, low 35.
    kept = df.groupby("identity_id").size()
    kept_per_bin = pd.DataFrame({"kept": kept, "bin": bins}).groupby("bin")["kept"].mean()
    assert kept_per_bin["high"] == 85
    assert kept_per_bin["medium"] == 55
    assert kept_per_bin["low"] == 35

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
        imagesperidentity=75, holdout_frac=0.20, min_holdout=5,
    )
    imbalanced_path = build_imbalanced_dataset(
        dataset_inputs["processed"],
        dataset_inputs["embeddings"],
        dataset_inputs["dataset"] + "_imb",
        nforget=2, forget_steps=15, randomstate=42,
        holdout_frac=0.20, min_holdout=5,
        imagesperidentity=75, candidatesperidentity=85,
    )
    bal = _read(balanced_path).drop_duplicates("identity_id").set_index("identity_id")["split"]
    imb = _read(imbalanced_path).drop_duplicates("identity_id").set_index("identity_id")["split"]
    assert (bal != imb).sum() == 0
    assert bal.value_counts().to_dict() == {"retain": 8, "forget": 2}

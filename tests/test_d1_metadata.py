"""Regression tests for the Phase D1 prompt-metadata fixes + in-place JPEG.

Covers:
- per-candidate metadata join in build_raw_dataset (the D1 bug: a
  per-identity collapse silently reported one pose/expression/lighting per
  identity; the build now raises)
- validate_release.py prompt-metadata variance gate (release-side catch)
- validate_prompt_plan.py pre-flight (generation-side catch)
- unconditional in-place PNG→JPEG q95 conversion in build_raw_dataset
  (verify-then-delete contract)
"""
import json
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from build_dataset import build_raw_dataset  # noqa: E402

# The conftest fixture materialises 64x64 candidate PNGs; the raw build's
# verify step must be told the expected size.
EXPECTED = (64, 64)


def _build_raw_with_metadata(dataset_inputs):
    """Run build_raw_dataset and return the emitted CSV path."""
    return build_raw_dataset(
        dataset_inputs["root"],
        dataset_inputs["embeddings"],
        dataset_inputs["dataset"] + "_raw_d1",
        randomstate=42,
        jpeg_expected_size=EXPECTED,
    )


def test_d1_build_raises_on_per_identity_collapse(dataset_inputs, monkeypatch):
    """The D1 bug pattern (per-identity metadata) must fail the build guard."""
    # Simulate the old collapse faithfully: the manifest KEEPS all its
    # columns (raw_candidatepath, trial, ...) but the prompt-metadata
    # columns carry one value per identity (the drop_duplicates bug).
    identities = pathlib.Path(dataset_inputs["identities"])
    man = pd.read_csv(identities / "raw_candidate_manifest.csv")
    for col in ("pose", "expression", "lighting"):
        man[col] = man.groupby("identity_id")[col].transform("first")
    man.to_csv(identities / "raw_candidate_manifest.csv", index=False)

    with pytest.raises(RuntimeError, match="Prompt-metadata collapse"):
        _build_raw_with_metadata(dataset_inputs)


def test_d1_per_candidate_metadata_survives(dataset_inputs):
    """Per-candidate metadata must reach the shipped CSV (the fix)."""
    result = _build_raw_with_metadata(dataset_inputs)
    df = pd.read_csv(result)
    for col in ("pose", "expression", "lighting"):
        per_id = df.groupby("identity_id")[col].nunique()
        assert (per_id > 1).all(), f"{col} collapsed per identity"
    # fixture supplies 4 poses / 3 expressions / 5 lightings per identity
    assert df["pose"].nunique() == 4
    assert df["expression"].nunique() == 3
    assert df["lighting"].nunique() == 5


def test_validate_release_metadata_gate_catches_collapse(dataset_inputs, tmp_path):
    """validate_release.py's variance gate must fail on collapsed metadata."""
    # Build a release tree from the FIXTURE (per-candidate metadata) then
    # collapse the CSV's prompt columns to simulate D1.
    result = _build_raw_with_metadata(dataset_inputs)
    release = tmp_path / "release"
    images = release / "images"
    images.mkdir(parents=True)
    df = pd.read_csv(result)
    # fabricate a collapsed CSV: constant per identity
    df["pose"] = df.groupby("identity_id")["pose"].transform("first")
    df["expression"] = df.groupby("identity_id")["expression"].transform("first")
    df["lighting"] = df.groupby("identity_id")["lighting"].transform("first")
    csv = release / "dataset_raw.csv"
    df.to_csv(csv, index=False)
    for cid in df["identity_id"].unique():
        (images / f"identity_{cid:03d}").mkdir(parents=True, exist_ok=True)
    (release / "schema.json").write_text(
        json.dumps({"required": ["image_path", "identity_id", "pose"]})
    )

    import validate_release as vr

    ok = vr.validate(
        str(release),
        metadata_csv="dataset_raw.csv",
        schema_path=None,
        schema_def="raw",
        require_relative_paths=False,
        check_cluster_split_isolation=False,
    )
    assert ok is False, "collapsed metadata must fail the release gate"


def test_validate_prompt_plan_go():
    """The pre-flight prompt validator must pass on the real 20×5 plan."""
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve().parent.parent
                              / "scripts" / "validate_prompt_plan.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "GO" in proc.stdout


def _manifest_rows(dataset_inputs):
    """Return (identities root, manifest) from the production-shaped fixture."""
    identities = pathlib.Path(dataset_inputs["identities"])
    man = pd.read_csv(identities / "raw_candidate_manifest.csv")
    return identities, man


def test_inplace_jpeg_verify_then_delete(dataset_inputs):
    """The raw build converts PNGs in place, verifies, then deletes them."""
    identities, man = _manifest_rows(dataset_inputs)
    result = build_raw_dataset(
        dataset_inputs["root"],
        dataset_inputs["embeddings"],
        dataset_inputs["dataset"] + "_raw_inplace",
        randomstate=42,
        jpeg_expected_size=EXPECTED,
    )
    df = pd.read_csv(result)

    # CSV image_path rewritten .png → .jpg (unconditional — no toggle)
    assert df["image_path"].str.endswith(".jpg").all()

    # Every original PNG deleted; every JPEG present + decodes RGB 64x64
    from PIL import Image

    checked = 0
    for _, row in man.iterrows():
        png = identities / row["raw_candidatepath"]
        jpg = png.with_suffix(".jpg")
        assert not png.exists(), f"PNG not deleted: {png}"
        assert jpg.is_file(), f"JPEG missing: {jpg}"
        with Image.open(jpg) as im:
            assert im.mode == "RGB" and im.size == EXPECTED
        checked += 1
    assert checked == len(man)


def test_inplace_jpeg_aborts_on_missing_png_keeps_everything(dataset_inputs):
    """A missing source PNG aborts conversion; no PNG may be deleted."""
    identities, man = _manifest_rows(dataset_inputs)
    # Delete one PNG to simulate a corrupt/missing source.
    victim = identities / man.iloc[0]["raw_candidatepath"]
    victim.unlink()

    with pytest.raises(RuntimeError, match="missing"):
        build_raw_dataset(
            dataset_inputs["root"],
            dataset_inputs["embeddings"],
            dataset_inputs["dataset"] + "_raw_inplace_fail",
            randomstate=42,
            jpeg_expected_size=EXPECTED,
        )

    # Phase 1 checks missing BEFORE converting, so no JPEG was created and
    # no surviving PNG was deleted.
    for _, row in man.iloc[1:10].iterrows():
        jpg = identities / row["raw_candidatepath"].replace(".png", ".jpg")
        assert not jpg.exists(), f"JPEG created despite missing source: {jpg}"
        png = identities / row["raw_candidatepath"]
        assert png.exists(), f"PNG deleted despite abort: {png}"

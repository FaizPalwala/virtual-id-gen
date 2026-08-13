"""
build_dataset.py

Constructs the final Phase 2 dataset mapping by merging generated identity 
manifests with extracted embedding attributes. Applies the train/test/forget splits.
Build the backward-compatible final ``dataset.csv`` file.

Example:
    python main.py --config-name step5_build \\
        dataset.dataroot=../data
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Bench output schema (224×224 crops, MUFAC-aligned).
# Design: prompt metadata (pose/expression/lighting/setting/camera) is NOT
# shipped with Bench — it is unlearning-irrelevant noise there (the Raw
# release carries it for the general-purpose use case).  The balanced
# artifact carries BOTH forget schedules: `forget_step` (uniform baseline,
# 5 ids/step) and `forget_step_poisson` (seeded Poisson arrival-model
# stress test; evaluate by cumulative count — see _distribute_forget_poisson).
OUTPUT_COLUMNS = [
    "image_path", "identity_id", "age_group", "age", "gender",
    "split", "forget_step", "forget_step_poisson", "image_subset",
    "arcface_similarity", "laplacian_variance", "detection_confidence",
]

# Imbalanced Bench schema: same core as balanced but NO forget-schedule
# columns — the popularity gradient (train-count 82:41:16) is the only
# experimental axis; the schedule axis lives in balanced.
IMBALANCED_OUTPUT_COLUMNS = [
    "image_path", "identity_id", "age_group", "age", "gender",
    "split", "image_subset",
    "arcface_similarity", "laplacian_variance", "detection_confidence",
    "popularity_bin", "images_per_identity",
]

def _load_attributes(embeddingsdir: str) -> pd.DataFrame:
    """Load image-level attributes saved by extract_embeddings.py."""
    embeddings = Path(embeddingsdir)
    df = pd.DataFrame(
        {
            "imagepath": np.load(embeddings / "imagepaths.npy").astype(str),
            "agegroup": np.load(embeddings / "agegroups.npy"),
            "age": np.load(embeddings / "ages.npy"),
            "gender": np.load(embeddings / "genders.npy"),
            "embedding": list(np.load(embeddings / "embeddings.npy")),
        }
    )
    id_path = embeddings / "identity_ids.npy"
    if id_path.exists():
        df["identity_id"] = np.load(id_path)
    # Derive trial from candidate filename for per-crop embedding matching.
    # Format: candidates/identity_NNN/candidate_YYY.png → trial=YYY.
    # identitymanifest.csv carries the same trial — the join key that links
    # each 224 crop to the 1024 candidate it was cropped from.
    t = df["imagepath"].str.extract(r"candidate_(\d+)")[0]
    df["trial"] = pd.to_numeric(t, errors="coerce").astype("Int64")
    return df

def _add_arcface_similarity(final: pd.DataFrame) -> pd.DataFrame:
    """Add per-image cosine similarity to the identity's mean embedding.

    Confound control: an image far from its identity's canonical face
    (unusual angle/expression) is memorised differently from a canonical
    portrait.  Without this control, per-identity MIA AUC is confounded
    by within-identity variance.
    """
    from common import normalised_cosine_similarity

    # Guard: rows without an embedding (extract skips faces it cannot
    # detect) produce NaN, which np.dot propagates into a length-N NaN
    # array and float() crashes with a cryptic TypeError.  Fail loudly
    # with the offending rows instead.
    missing = final["embedding"].isna()
    if missing.any():
        bad = final.loc[missing, ["identity_id", "trial"]].drop_duplicates()
        raise ValueError(
            f"{missing.sum()} rows have no embedding (extract skipped their "
            f"candidates — no detectable face).  First offenders: "
            f"{bad.head(5).to_dict('records')}.  Drop these rows before "
            f"computing similarity, or re-run extraction."
        )

    # Mean embedding per cluster (embeddings are already L2-normalised).
    means = final.groupby("identity_id")["embedding"].mean()
    final["arcface_similarity"] = [
        normalised_cosine_similarity(row.embedding, means[row.identity_id])
        for row in final.itertuples(index=False)
    ]
    return final

def _convert_one_inplace(args: tuple[Path, Path], quality: int = 95) -> None:
    """Module-level worker (picklable under macOS spawn): PNG→JPEG in place."""
    _png_to_jpeg(*args, quality)

def _convert_candidates_inplace(
    df: pd.DataFrame,
    dataroot: Path,
    quality: int = 95,
    workers: int = 4,
    expected_size: tuple[int, int] = (1024, 1024),
) -> None:
    """Convert raw candidates PNG→JPEG q95 (4:4:4) IN PLACE, verify-then-delete.

    Three phases (RELEASE_TODO Phase D2 P1-4, user-approved design):
      1. convert every shipped candidate ``candidate_YYY.png`` →
         ``candidate_YYY.jpg`` in the SAME directory (atomic temp+replace),
         collecting failures WITHOUT deleting anything
      2. verify every JPEG decodes at the expected size, RGB (decode is the
         whole point of the "verify before delete" contract)
      3. ONLY then delete the source PNGs — if any conversion or
         verification failed, nothing is deleted and the build raises

    Runs inside the build (Aire-side), AFTER extract/preprocess have read
    the PNGs, so the off-node transfer bundle is ~22 GB of JPEG instead of
    ~100 GB of PNG.  The CSV image_path columns are rewritten .png→.jpg by
    the caller so shipped metadata matches the surviving files.
    """
    from concurrent.futures import ProcessPoolExecutor

    jobs: list[tuple[Path, Path]] = []
    for _, row in df.iterrows():
        rel = row["image_path"]  # dataroot-relative: identities/candidates/...
        src = dataroot / rel
        dst = src.with_suffix(".jpg")
        jobs.append((src, dst))

    # ---- Phase 1: convert (never delete on failure) ----
    missing = [str(s) for s, _ in jobs if not s.is_file()]
    if missing:
        raise RuntimeError(
            f"{len(missing)} raw candidates missing for JPEG conversion: "
            f"{missing[:5]}"
        )
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_convert_one_inplace, j, quality) for j in jobs]
        failures: list[str] = []
        for fut in futures:
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 — collect all
                failures.append(str(exc))
            done += 1
            if done % 5000 == 0:
                print(f"  [raw-jpeg] converted {done}/{len(jobs)}")
    if failures:
        raise RuntimeError(
            f"{len(failures)}/{len(jobs)} raw JPEG conversions failed — "
            f"PNGs kept, nothing deleted.  First 5: {failures[:5]}"
        )

    # ---- Phase 2: verify every JPEG decodes at the expected size, RGB ----
    from PIL import Image

    bad: list[str] = []
    for i, (src, dst) in enumerate(jobs):
        try:
            with Image.open(dst) as im:
                im.load()
                if im.mode != "RGB" or im.size != expected_size:
                    bad.append(f"{dst} → {im.mode} {im.size}")
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{dst} → {exc}")
        if (i + 1) % 5000 == 0:
            print(f"  [raw-jpeg] verified {i + 1}/{len(jobs)}")
    if bad:
        raise RuntimeError(
            f"{len(bad)}/{len(jobs)} JPEGs failed verification — PNGs kept. "
            f"First 5: {bad[:5]}"
        )

    # ---- Phase 3: only now delete the sources ----
    for src, _ in jobs:
        src.unlink(missing_ok=True)
    print(f"[OK] In-place raw JPEG: {len(jobs)} portraits converted, "
          f"PNGs deleted, tree is now ~22 GB")

def _png_to_jpeg(src: Path, dst: Path, quality: int = 95) -> None:
    """Re-encode a PNG portrait as JPEG q95 with 4:4:4 chroma (atomic)."""
    from PIL import Image

    with Image.open(src) as im:
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        icc = im.info.get("icc_profile")
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        save_kwargs = {"quality": quality, "subsampling": 0}
        if icc:
            save_kwargs["icc_profile"] = icc
        im.save(tmp, "JPEG", **save_kwargs)
        tmp.replace(dst)

def _load_candidate_metadata(identitydir: str) -> pd.DataFrame:
    """Load per-candidate prompt metadata from the generate step's manifest.

    Returns a DataFrame keyed by ``(identity_id, trial)`` with one row per
    candidate: each generated portrait carries its own variation attributes
    (the 20×5 prompt grid gives every candidate a unique rendering
    instruction).  A per-identity collapse here would silently report the
    first candidate's attributes for the whole identity, contradicting the
    prompt-diversity design — see RELEASE_TODO Phase D1.
    """
    raw = Path(identitydir) / "identities" / "raw_candidate_manifest.csv"
    if not raw.exists():
        raw = Path(identitydir).parent / "identities" / "raw_candidate_manifest.csv"
    df = pd.read_csv(raw)
    # The raw manifest may still use 'clusterid' if merged by an older script.
    id_col = "identity_id" if "identity_id" in df.columns else "clusterid"
    metadata_cols = [
        "pose", "expression", "lighting", "setting", "camera",
    ]
    available = [c for c in metadata_cols if c in df.columns]
    if not available:
        # Older manifests without metadata columns — return empty.
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["identity_id", "trial"])
        )
    key_cols = [id_col, "trial"]
    out = df[key_cols + available].rename(columns={id_col: "identity_id"})
    # Guard: the manifest must be per-candidate (one row per (id, trial)).
    dup = out.duplicated(subset=["identity_id", "trial"]).sum()
    if dup:
        raise RuntimeError(
            f"raw_candidate_manifest has {dup} duplicate (identity_id, trial) "
            f"rows — cannot build per-candidate prompt metadata"
        )
    return out.set_index(["identity_id", "trial"])[available]

def _distribute_forget(
    forget_ids: list[int], nforget: int, forget_steps: int, rng: np.random.RandomState,
) -> dict[int, tuple[int, int]]:
    """Assign forget identities to steps with uniform+remainder distribution.

    Returns ``{identity_id: (step, variant)}`` where *step* is 0-based and
    *variant* is the index within the step.

    Uniform is the MUFAC baseline: every step deletes the same number of
    identities, so per-step evaluation curves are directly comparable and
    step index == cumulative forgotten count (no batch-size confound).
    """
    base = nforget // forget_steps
    remainder = nforget % forget_steps

    mapping: dict[int, tuple[int, int]] = {}
    pos = 0
    for step in range(forget_steps):
        count = base + (1 if step < remainder else 0)
        for variant in range(count):
            mapping[forget_ids[pos]] = (step, variant)
            pos += 1
    return mapping

def _distribute_forget_poisson(
    forget_ids: list[int], nforget: int, forget_steps: int, rng: np.random.RandomState,
) -> dict[int, tuple[int, int]]:
    """Assign forget identities to steps with a seeded Poisson batch schedule.

    Models real-world data-deletion request streams: erasure requests under
    GDPR-style regimes arrive as an (approximately) independent arrival
    process, so batching them into fixed windows yields Poisson-distributed
    batch sizes — occasional near-empty steps and occasional bursts
    (Ginart et al., 2019 "Making AI Forget You"; the online-forgetting
    framing of arXiv:2012.01668).

    This is the "forget-variant" stress test: same number of steps, same
    total forget set, but per-step counts follow the arrival distribution.
    Downstream evaluation MUST compare schedules by cumulative forgotten
    count, not step index, because step index no longer equals cumulative
    count (Shen et al., 2025 "Machine Unlearning for Streaming Forgetting",
    arXiv:2507.15280 — the total variation between consecutive forget sets,
    V_T, is what drives difficulty).

    Implementation: draw forget_steps Poisson(lambda=nforget/forget_steps)
    counts, then rebalance by adding/removing one identity at a time from
    randomly chosen steps until the total is exactly nforget.  Seeded via
    *rng*, so the schedule is a fixed dataset property (recorded in
    RELEASE_MANIFEST.json), not a runtime variable.

    Returns ``{identity_id: (step, variant)}`` as in :func:`_distribute_forget`.
    """
    lam = nforget / forget_steps
    counts = [int(rng.poisson(lam)) for _ in range(forget_steps)]

    # Rebalance to exactly nforget while preserving the Poisson shape.
    diff = nforget - sum(counts)
    while diff != 0:
        step = rng.randint(forget_steps)
        if diff > 0:
            counts[step] += 1
            diff -= 1
        elif counts[step] > 0:
            counts[step] -= 1
            diff += 1

    mapping: dict[int, tuple[int, int]] = {}
    pos = 0
    for step, count in enumerate(counts):
        for variant in range(count):
            mapping[forget_ids[pos]] = (step, variant)
            pos += 1
    assert pos == nforget, f"Poisson schedule misplaced {nforget - pos} identities"
    return mapping

def _to_dataroot_relative(p: str, base_dir: Path, dataroot: Path) -> str:
    """Convert a manifest image path to a dataroot-relative release path.

    Handles two conventions:
    - absolute paths (legacy manifests written before the portability fix)
    - relative-to-*base_dir* paths (current portable manifest contract;
      base_dir is the identities dir for candidates, processed for crops).

    This keeps release CSVs portable: every ``image_path`` resolves against
    the dataset root on any machine, regardless of where the pipeline ran.
    """
    path = Path(p)
    if path.is_absolute():
        try:
            return str(path.relative_to(dataroot))
        except ValueError:
            pass
        for anchor in ("candidates", "images"):
            parts = path.parts
            for i, part in enumerate(parts):
                if part == anchor:
                    return str(Path(*parts[i:]))
        return str(path)
    return str((base_dir / path).relative_to(dataroot))

def _compute_holdout_size(
    total_per_id: int, holdout_frac: float, min_holdout: int
) -> int:
    """Return the number of holdout images per identity.

    MUFAC evaluation requires held-out probe images for every identity.
    The holdout is split off BEFORE any cardinality-based trimming so it
    is independent of the train set size.
    """
    return max(min_holdout, round(total_per_id * holdout_frac))

def build_dataset(
    identitydir: str,
    embeddingsdir: str,
    outputdir: str,
    nforget: int = 40,
    forget_steps: int = 15,
    randomstate: int = 42,
    imagesperidentity: int = 90,
    holdout_frac: float = 0.20,
    min_holdout: int = 5,
) -> str:
    """Build the balanced 224x224 crop dataset with MUFAC-aligned splits.

    Every identity contributes to both training and evaluation via the
    ``image_subset`` column, replacing the old identity-disjoint ``test`` split.

    Ships BOTH forget schedules as columns:
    - ``forget_step`` — uniform baseline (equal ids/step; step index ==
      cumulative forgotten count, so per-step curves are directly comparable)
    - ``forget_step_poisson`` — seeded Poisson batch sizes modelling
      GDPR-style deletion request streams (see
      :func:`_distribute_forget_poisson`); evaluate by cumulative count.
    """
    manifest = pd.read_csv(Path(identitydir) / "identitymanifest.csv")
    attributes = _load_attributes(embeddingsdir)

    # Match each 224 crop to its parent 1024 candidate embedding via the
    # (identity_id, trial) join key.  identitymanifest.csv carries trial
    # from preprocess; _load_attributes derives trial from the candidate
    # filename.  This gives per-crop embeddings → arcface_similarity is
    # a real within-identity spread, not trivially 1.0.
    #
    # NOTE — per-identity age variance is a FEATURE, not noise to suppress:
    # each identity's images carry their own InsightFace gender/age estimate
    # (taken on the parent 1024×1024 candidate), so one identity can span
    # multiple age_group values (e.g. identity 558 ranges 23–65).  The
    # synthetic identity stays recognisably the same person while appearing
    # at different ages — a realism property that also prevents the age
    # classifier from shortcutting to identity (identity_id → constant age).
    # Do not collapse to a per-identity constant (first/mode); keep the
    # per-image proxy labels.
    attrs_merge = attributes[
        ["identity_id", "trial", "agegroup", "age", "gender", "embedding"]
    ].copy()
    final = manifest[
        ["imagepath", "identity_id", "trial", "detection_confidence", "laplacian_variance"]
    ].merge(attrs_merge, on=["identity_id", "trial"], how="left")
    if len(final) != len(manifest):
        raise RuntimeError(
            f"Trial-based join lost rows: {len(manifest)} → {len(final)}"
        )

    # NOTE — prompt metadata join (pose/expression/lighting/setting/camera)
    # happens only in build_raw_dataset; bench OUTPUT_COLUMNS deliberately
    # strips it (unlearning-irrelevant noise), so merging it here would be
    # dead work.

    # Per-image cosine similarity to the identity's mean embedding.
    # Computed BEFORE trimming so the reference mean spans ALL quality-passing
    # crops from preprocess — the identity's canonical face.  The column is
    # therefore dataset-invariant: a given image has the same arcface_similarity
    # in the balanced and imbalanced variants, making it a valid confound
    # control across both.  Trimming below only removes rows; it never changes
    # the values.
    final = _add_arcface_similarity(final)

    # Trim each identity to imagesperidentity (preprocess may have saved
    # more than that).  The manifest is quality-ranked, so the first N
    # rows per cluster are the best crops.
    final = (
        final.sort_values(["identity_id", "detection_confidence"], ascending=[True, False])
        .groupby("identity_id")
        .head(imagesperidentity)
        .reset_index(drop=True)
    )

    identities = sorted(final["identity_id"].unique())
    if nforget >= len(identities):
        raise ValueError("forget count must leave at least one retain identity.")
    rng = np.random.RandomState(randomstate)
    rng.shuffle(identities)
    forget_ids = identities[:nforget]
    split_map = (
        {identity: "forget" for identity in forget_ids}
        | {identity: "retain" for identity in identities[nforget:]}
    )
    final["split"] = final["identity_id"].map(split_map)

    # MUFAC-aligned per-identity holdout.
    holdout_n = _compute_holdout_size(imagesperidentity, holdout_frac, min_holdout)
    rng_holdout = np.random.RandomState(randomstate + 7777)
    for cid in identities:
        rows = final[final["identity_id"] == cid].index
        shuffled_idx = list(rows)
        rng_holdout.shuffle(shuffled_idx)
        for j, idx in enumerate(shuffled_idx):
            value = "holdout" if j < holdout_n else "train"
            final.at[idx, "image_subset"] = value
    final["image_subset"] = final["image_subset"].fillna("train")

    # ── Forget schedules (BOTH shipped as columns) ──
    # Uniform: deterministic base+remainder (step index == cumulative count).
    # Poisson: seeded arrival-model schedule — dedicated RNG stream so the
    # two schedules are independent of each other and of the split shuffle.
    rng_poisson = np.random.RandomState(randomstate + 4242)
    step_map = _distribute_forget(forget_ids, nforget, forget_steps, rng)
    poisson_map = _distribute_forget_poisson(forget_ids, nforget, forget_steps, rng_poisson)
    final["forgetstep"] = (
        final["identity_id"].map(
            {cid: step for cid, (step, _) in step_map.items()}
        )
        .fillna(-1)
        .astype(int)
    )
    final["forgetstep_poisson"] = (
        final["identity_id"].map(
            {cid: step for cid, (step, _) in poisson_map.items()}
        )
        .fillna(-1)
        .astype(int)
    )
    final = final[final.agegroup != -1].copy()

    # ------------------------------------------------------------------
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)
    dataroot = output.parent
    output_df = final.rename(
        columns={
            "imagepath": "image_path",
            "agegroup": "age_group",
            "forgetstep": "forget_step",
            "forgetstep_poisson": "forget_step_poisson",
        }
    )
    # Fill missing columns with defaults for older manifests.
    for col in OUTPUT_COLUMNS:
        if col not in output_df.columns:
            output_df[col] = ""
    output_df = output_df[OUTPUT_COLUMNS]
    output_df["image_path"] = (
        output_df["image_path"]
        .apply(lambda p: _to_dataroot_relative(p, Path(identitydir), dataroot))
    )
    csv_path = output / "dataset.csv"
    parquet_path = output / "dataset.parquet"
    output_df.to_csv(csv_path, index=False)
    output_df.to_parquet(parquet_path, index=False)
    (output / "datasetsummary.json").write_text(
        json.dumps(
            {
                "total_images": len(final),
                "nidentities": len(identities),
                "images_per_identity": imagesperidentity,
                "nforget_ids": nforget,
                "forget_steps": forget_steps,
                "forget_pct": nforget / len(identities),
                "distribution": {str(s): cnt for s, cnt in sorted(
                    Counter(step_map[cid][0] for cid in step_map).items()
                )},
                "split_sizes": {
                    key: int(value)
                    for key, value in final.groupby("split").size().items()
                },
                "holdout_per_id": holdout_n,
                "columns": OUTPUT_COLUMNS,
            },
            indent=2,
        )
    )
    return str(csv_path)

def build_imbalanced_dataset(
    identitydir: str,
    embeddingsdir: str,
    outputdir: str,
    nforget: int = 40,
    randomstate: int = 42,
    holdout_frac: float = 0.20,
    min_holdout: int = 5,
    # Imbalance: train counts as ratios of the max train pool.
    # holdout is constant per identity regardless of bin.
    imagesperidentity: int = 90,
    candidatesperidentity: int = 100,
    high_bin_pct: float = 0.10,
    low_bin_pct: float = 0.60,
    train_gradient_ratio: dict[str, float] | None = None,
) -> str:
    """Build the imbalanced 224x224 dataset with MUFAC-aligned splits.

    Same ``image_subset`` design as balanced: every identity contributes
    train + holdout.  The popularity gradient applies only to the train
    subset -- holdout is constant per identity regardless of bin.

    Holdout per identity: ``max(min_holdout, round(imagesperidentity *
    holdout_frac))`` — the SAME reserve as the balanced release, so probe
    stability is uniform across all popularity tiers.

    Train gradient: ratios applied to ``max_train = candidatesperidentity
    - holdout``.  Defaults give 82:41:16 (100 - 18 = 82; {1.0, 0.50, 0.20}).
    This 5:1 ratio is within the VGG-Face2 range (Cao et al., 2018;
    ~10:1 max-min, 5:1 inter-quartile) and matches the real-world
    distribution curated face training sets exhibit (Wang et al., 2019
    "Deep Face Recognition: A Survey"; Liu et al., 2019 "Large-Scale
    Long-Tailed Recognition in an Open World", CVPR).
    The computation scales with the candidate pool: if a future release
    generates 550 candidates/id targeting 500 images, holdout = 100 and
    max_train = 450, so the gradient re-derives train counts accordingly.
    """
    # ── 1. Read and merge manifests (same as build_dataset) ──
    manifest = pd.read_csv(Path(identitydir) / "identitymanifest.csv")
    attributes = _load_attributes(embeddingsdir)

    attrs_merge = attributes[
        ["identity_id", "trial", "agegroup", "age", "gender", "embedding"]
    ].copy()
    final = manifest[
        ["imagepath", "identity_id", "trial", "detection_confidence", "laplacian_variance"]
    ].merge(attrs_merge, on=["identity_id", "trial"], how="left")
    if len(final) != len(manifest):
        raise RuntimeError(
            f"Trial-based join lost rows: {len(manifest)} -> {len(final)}"
        )

    # NOTE — prompt metadata join (pose/expression/lighting/setting/camera)
    # happens only in build_raw_dataset; imbalanced bench OUTPUT_COLUMNS
    # deliberately strips it, so merging it here would be dead work.

    # Per-image cosine similarity to the identity's mean embedding.
    # Computed BEFORE pruning so the reference mean spans ALL quality-passing
    # crops from preprocess — the identity's canonical face.  The column is
    # therefore dataset-invariant: a given image has the same arcface_similarity
    # in the balanced and imbalanced variants, making it a valid confound
    # control across both.  Pruning below only removes rows; it never changes
    # the values.
    final = _add_arcface_similarity(final)

    identities = sorted(final["identity_id"].unique())

    # ── 2. Assign popularity bins ──
    # Design choice: seeded random shuffle ensures reproducibility and avoids
    # any correlation with the identity's original identity ID (which may be
    # correlated with seed quality or prompt order).  Bin boundaries are
    # based on identity count, not image count.
    rng = np.random.RandomState(randomstate + 9999)  # different seed from balanced
    shuffled_ids = identities.copy()
    rng.shuffle(shuffled_ids)

    n_high = max(1, int(len(shuffled_ids) * high_bin_pct))
    n_low = max(1, int(len(shuffled_ids) * low_bin_pct))
    n_medium = len(shuffled_ids) - n_high - n_low

    high_ids = set(shuffled_ids[:n_high])
    low_ids = set(shuffled_ids[-n_low:])
    # medium_ids = set(shuffled_ids[n_high : n_high + n_medium])  # implicit

    bin_map = {}
    for cid in shuffled_ids:
        if cid in high_ids:
            bin_map[cid] = "high"
        elif cid in low_ids:
            bin_map[cid] = "low"
        else:
            bin_map[cid] = "medium"

    # ── 3. Down-sample low- and medium-bin identities ──
    # Design choice: train counts are set as ratios of max_train_per_id
    # (candidate ceiling − per-identity holdout).  Higher bins get more
    # train images; holdout is constant per identity.  Produces an 82:41:16
    # train gradient with defaults (100 − 18 = 82; {1.0, 0.50, 0.20}).
    # A 5:1 ratio matches the real-world imbalance in curated face
    # training sets (VGG-Face2, Cao et al. 2018; ~10:1 max-min,
    # 5:1 inter-quartile; Wang et al. 2019; Liu et al. 2019, CVPR).
    # High-bin identities keep all available crops (up to the ceiling).
    # Per-identity random draws are seeded for reproducibility.
    rng_prune = np.random.RandomState(randomstate + 9998)
    if train_gradient_ratio is None:
        train_gradient_ratio = {"high": 1.0, "medium": 0.50, "low": 0.20}
    # Holdout reserve — SAME formula as the balanced release, applied per
    # identity: max(min_holdout, round(imagesperidentity × holdout_frac)).
    # Defaults: max(5, round(90 × 0.20)) = 18 — probe stability in every tier.
    holdout_n = _compute_holdout_size(imagesperidentity, holdout_frac, min_holdout)
    # higher imagesperidentity, which in turn scales train-gradients
    # gracefully.  Defaults: 100−18=82 → {1.0, 0.50, 0.20}×82 → 82:41:16.
    max_train = max(1, candidatesperidentity - holdout_n)
    # KEPT pool per identity = train + holdout (gradient on train only;
    # holdout carved afterwards).  Defaults: 100 / 59 / 34 kept.
    per_bin_images = {
        "high": round(max_train * train_gradient_ratio["high"]) + holdout_n,
        "medium": round(max_train * train_gradient_ratio["medium"]) + holdout_n,
        "low": round(max_train * train_gradient_ratio["low"]) + holdout_n,
    }
    drop_mask = pd.Series(False, index=final.index)
    for cid in low_ids | (set(bin_map.keys()) - high_ids - low_ids):  # low + medium
        target = per_bin_images[bin_map[cid]]
        rows = final[final["identity_id"] == cid]
        n_keep = min(target, len(rows))
        keep_idx = set(rng_prune.choice(rows.index, n_keep, replace=False))
        drop_idx = set(rows.index) - keep_idx
        drop_mask.loc[list(drop_idx)] = True

    imbalanced = final[~drop_mask].copy()

    new_sizes = imbalanced.groupby("identity_id").size()
    for cid in set(bin_map.keys()) - high_ids:
        expected = per_bin_images[bin_map[cid]]
        actual = new_sizes.get(cid, 0)
        if actual < expected * 0.5:
            raise RuntimeError(
                f"Identity {cid} ({bin_map[cid]} bin): got {actual} < {expected * 0.5}"
            )

    imbalanced_ids = sorted(imbalanced["identity_id"].unique())
    # Same identity-level split as balanced.
    rng_split = np.random.RandomState(randomstate)
    rng_split.shuffle(imbalanced_ids)
    forget_ids = imbalanced_ids[:nforget]
    split_map = (
        {cid: "forget" for cid in forget_ids}
        | {cid: "retain" for cid in imbalanced_ids[nforget:]}
    )
    imbalanced["split"] = imbalanced["identity_id"].map(split_map)

    # MUFAC holdout: carve the SAME per-identity reserve (holdout_n, computed
    # above from imagesperidentity × holdout_frac) out of each identity's
    # kept pool — 15/id with defaults, identical across all popularity bins.
    rng_h = np.random.RandomState(randomstate + 7777)
    for cid in imbalanced_ids:
        rows = imbalanced[imbalanced["identity_id"] == cid].index
        shuffled = list(rows)
        rng_h.shuffle(shuffled)
        n_hold = min(holdout_n, len(shuffled) - 1)
        for j, idx in enumerate(shuffled):
            imbalanced.at[idx, "image_subset"] = "holdout" if j < n_hold else "train"
    imbalanced["image_subset"] = imbalanced["image_subset"].fillna("train")

    imbalanced = imbalanced[imbalanced.agegroup != -1].copy()

    # ── 6. Add popularity annotation columns ──
    # Design choice: adding both a categorical label (popularity_bin) and a
    # cardinality value (images_per_identity) lets downstream evaluation decide
    # which axis to test — discrete bins or continuous count.
    imbalanced["popularity_bin"] = imbalanced["identity_id"].map(bin_map)
    id_counts = imbalanced.groupby("identity_id").size()
    imbalanced["images_per_identity"] = imbalanced["identity_id"].map(id_counts)

    # ── 7. Write output ──
    # Design choice: imbalanced variant is a separate file paired alongside
    # dataset.csv.  Users compare forget efficacy across popularity_bin values
    # without modifying the balanced baseline.
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)
    dataroot = output.parent

    output_columns = IMBALANCED_OUTPUT_COLUMNS
    output_df = imbalanced.rename(
        columns={
            "imagepath": "image_path",
            "agegroup": "age_group",
        }
    )
    for col in IMBALANCED_OUTPUT_COLUMNS:
        if col not in output_df.columns:
            output_df[col] = ""
    output_df = output_df[output_columns]
    output_df["image_path"] = (
        output_df["image_path"]
        .apply(lambda p: _to_dataroot_relative(p, Path(identitydir), dataroot))
    )

    csv_path = output / "dataset_imbalanced.csv"
    parquet_path = output / "dataset_imbalanced.parquet"
    output_df.to_csv(csv_path, index=False)
    output_df.to_parquet(parquet_path, index=False)

    # Summary with per-bin breakdown.
    bin_stats = imbalanced.groupby("popularity_bin").agg(
        nidentities=("identity_id", "nunique"),
        nimages=("identity_id", "count"),
    ).to_dict("index")

    (output / "datasetsummary_imbalanced.json").write_text(
        json.dumps(
            {
                "total_images": len(imbalanced),
                "nidentities": len(imbalanced_ids),
                "forget_pct": nforget / len(imbalanced_ids),
                "holdout_per_id": holdout_n,
                "gradient_ratios": train_gradient_ratio,
                "images_per_bin": {
                    "high": per_bin_images["high"],
                    "medium": per_bin_images["medium"],
                    "low": per_bin_images["low"],
                },
                "bin_sizes": {
                    "high_pct": high_bin_pct,
                    "medium_pct": 1.0 - high_bin_pct - low_bin_pct,
                    "low_pct": low_bin_pct,
                    "n_high": n_high,
                    "n_medium": n_medium,
                    "n_low": n_low,
                },
                "per_bin": {
                    bin: {
                        "identities": stats["nidentities"],
                        "images": stats["nimages"],
                        "images_per_id": round(stats["nimages"] / stats["nidentities"], 1),
                    }
                    for bin, stats in bin_stats.items()
                },
                "forget_schedules": "none (schedule axis lives in balanced)",
                "split_sizes": {
                    key: int(value)
                    for key, value in imbalanced.groupby("split").size().items()
                },
                "columns": IMBALANCED_OUTPUT_COLUMNS,
            },
            indent=2,
        )
    )

    print(f"[OK] Imbalanced dataset: {len(imbalanced)} images across "
          f"{len(imbalanced_ids)} identities — "
          f"{n_high} high, {n_medium} medium, {n_low} low.")
    return str(csv_path)

# ── Candidate (full-resolution) dataset columns ──
# Same core as OUTPUT_COLUMNS but excludes crop-level quality metrics
# (laplacian_variance, detection_confidence) — meaningless on 1024
# candidates before alignment.
RAW_OUTPUT_COLUMNS = [
    "image_path", "identity_id", "age_group", "age", "gender",
    "arcface_similarity", "pose", "expression", "lighting",
    "setting", "camera",
]

def build_raw_dataset(
    identitydir: str,
    embeddingsdir: str,
    outputdir: str,
    randomstate: int = 42,
    jpeg_expected_size: tuple[int, int] = (1024, 1024),
) -> str:
    """Build the full-resolution 1024x1024 raw dataset.

    Max-size reference dataset -- all 100 portraits per identity, no
    splits.  General-purpose release for identity recognition, face
    generation evaluation, and demographic bias studies.  Researchers
    can construct balanced subsets of up to 100 images/identity with
    custom train/holdout splits.

    Columns: identity_id, age, gender, arcface_similarity, pose,
    expression, lighting, setting, camera.

    JPEG is the shipped format, decided HERE: every shipped candidate is
    converted PNG→JPEG q95 (4:4:4) IN PLACE with verify-then-delete (see
    ``_convert_candidates_inplace``) and the CSV image_path is rewritten
    .png→.jpg.  Unconditional — no toggle, no Mac-side conversion.  Runs
    after extract/preprocess (they read the PNGs first).
    """
    manifest = pd.read_csv(Path(identitydir) / "identities" / "raw_candidate_manifest.csv")
    id_col = "identity_id" if "identity_id" in manifest.columns else "clusterid"
    manifest.rename(columns={id_col: "identity_id", "raw_candidatepath": "imagepath"}, inplace=True)

    candidates = manifest[["imagepath", "identity_id", "trial"]].copy()
    candidates.drop_duplicates(inplace=True)

    attributes = _load_attributes(embeddingsdir)
    attrs_merge = attributes[
        ["identity_id", "trial", "agegroup", "age", "gender", "embedding"]
    ].copy()
    final = candidates.merge(attrs_merge, on=["identity_id", "trial"], how="left")

    metadata = _load_candidate_metadata(identitydir)
    if not metadata.empty:
        final = final.merge(
            metadata, on=["identity_id", "trial"], how="left", validate="one_to_one"
        )

    # Drop candidates extract could not embed (no detectable face) BEFORE
    # similarity: they cannot ship (arcface_similarity is a required column)
    # and NaN embeddings crash _add_arcface_similarity with a cryptic
    # TypeError.  The raw release is "max-size among embeddable portraits" —
    # the summary records the drop so the manifest stays honest.
    n_pre = len(final)
    final = final[final["embedding"].notna()].copy()
    n_dropped = n_pre - len(final)
    if n_dropped:
        print(f"[build_raw_dataset] dropped {n_dropped}/{n_pre} candidates "
              f"without embeddings (extract skipped undetectable faces)")

    final = _add_arcface_similarity(final)
    identities = sorted(final["identity_id"].unique())

    # Airtight guard (RELEASE_TODO Phase D1): the 20×5 prompt grid gives
    # every candidate a unique rendering instruction, so every identity's
    # shipped portraits MUST span >1 pose/expression/lighting (the
    # variation dimensions).  setting/camera can legitimately repeat (the
    # grid has only 3 lenses); a metadata join collapse (per-identity
    # instead of per-candidate) fails here at build time rather than
    # surfacing at release QA.
    variation_cols = [c for c in ("pose", "expression", "lighting")
                      if c in final.columns]
    if variation_cols:
        min_var = final.groupby("identity_id")[variation_cols].nunique().min(axis=1)
        collapsed = min_var.loc[lambda x: x <= 1]
        if len(collapsed) > 0:
            raise RuntimeError(
                f"Prompt-metadata collapse at build: {len(collapsed)} identities "
                f"have <=1 unique value across {variation_cols} (per-candidate "
                f"variation missing).  First 10: {collapsed.index.tolist()[:10]}"
            )
        print(f"[build_raw_dataset] prompt-metadata variance OK "
              f"(all {len(identities)} identities >1 unique pose/expr/lighting)")

    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)
    dataroot = output.parent
    output_df = final.rename(
        columns={"imagepath": "image_path", "agegroup": "age_group"}
    )
    for col in RAW_OUTPUT_COLUMNS:
        if col not in output_df.columns:
            output_df[col] = "" if col in ("pose", "expression", "lighting", "setting", "camera") else 0.0
    output_df = output_df[RAW_OUTPUT_COLUMNS]
    output_df["image_path"] = output_df["image_path"].apply(
        lambda p: _to_dataroot_relative(p, Path(identitydir) / "identities", dataroot)
    )

    # In-place raw JPEG conversion — unconditional (RELEASE_TODO Phase D2
    # P1-4): the raw release ships as JPEG q95 (4:4:4), decided here in the
    # build.  Every shipped candidate is converted PNG→JPEG with
    # verify-then-delete, so the off-node transfer bundle is ~22 GB instead
    # of ~100 GB and the shipped CSV image_path is .jpg.  Must run BEFORE
    # the CSV write.  Runs after extract/preprocess have read the PNGs.
    _convert_candidates_inplace(output_df, dataroot, expected_size=jpeg_expected_size)
    output_df["image_path"] = output_df["image_path"].str.replace(
        r"\.png$", ".jpg", regex=True
    )

    csv_path = output / "dataset_raw.csv"
    parquet_path = output / "dataset_raw.parquet"
    output_df.to_csv(csv_path, index=False)
    output_df.to_parquet(parquet_path, index=False)

    max_balanced = int(output_df.groupby("identity_id").size().min())
    max_per_id = int(output_df.groupby("identity_id").size().max())
    (output / "datasetsummary_raw.json").write_text(
        json.dumps({
            "total_images": len(final),
            "nidentities": len(identities),
            "max_images_per_id": max_per_id,
            "max_balanced_subset": max_balanced,
            "columns": RAW_OUTPUT_COLUMNS,
        }, indent=2)
    )

    print(f"[OK] Raw dataset: {len(final)} images across {len(identities)} identities.")
    return str(csv_path)

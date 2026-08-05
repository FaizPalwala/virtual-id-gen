# Changelog

All notable changes to **SFHQ-VirtualID** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — dev → main (PR #1)

> Initial release of the full pipeline + dataset.  `main` becomes the
> release branch with this merge; everything below is what ships in v1.0.0.

### Data freeze status

- **Frozen artifacts** (verified, release-ready, staged in `release_v1/`):
  - `SFHQ-VirtualID-Bench` — 224×224 aligned crops
    - `dataset.csv` — balanced, **45,000** images (75/id; 60 train + 15 holdout)
    - `dataset_imbalanced.csv` — ratio-based **70:40:20** train gradient,
      **27,593** images (kept 85/55/35 per bin; 15 holdout/id)
  - `SFHQ-VirtualID-Raw` — 1024×1024 portraits, **51,000** images (85/id,
    max-size, no splits)
- All 3 release gates pass (`validate_release.py`: schema conformance,
  split isolation, `image_subset` invariant, union orphan check, checksums).
- Holdout is the **same per-identity reserve as the balanced release**
  (`max(min_holdout, round(imagesperidentity × holdout_frac))` = 15/id),
  uniform across all popularity bins — probe stability in every tier.
- Imbalanced gradient scales with the candidate pool:
  `max_train = candidatesperidentity − holdout` (defaults 85−15=70 → ratios
  {1.0, 0.57, 0.29} → 70:40:20 train).

### Added

- **MUFAC-aligned split protocol**: every identity contributes both
  `image_subset` values (`train` + `holdout`); identity-level `split` is
  `retain` (540) or `forget` (60); no identity-disjoint `test` split
  (unseen-identity accuracy was structurally 0).
- **Sequential forgetting protocol**: 15 steps × 4 identities, uniform +
  remainder; `forget_step` / `forget_variant` columns.
- **Three dataset artifacts** across two releases:
  `dataset.csv` / `dataset_imbalanced.csv` (Bench 224) and
  `dataset_raw.csv` (Raw 1024).
- **Proxy labels**: per-image `age_group`/`age`/`gender` inherited from the
  parent 1024×1024 candidate (per-image variance is a deliberate feature).
- **Confound controls**: `arcface_similarity` (per-candidate embedding,
  real within-identity spread), `laplacian_variance`, `detection_confidence`.
- **Prompt system**: 100 unique deterministic variations (20 compositions ×
  5 lighting), shuffled per identity, sliced to 85 candidates/id.
- **Identity-conditioned generation**: InstantID + Juggernaut-XL-v9 +
  ControlNet (Canny SDXL), guidance 3.0 / ip-adapter 0.85 / 30 steps.
- **CLIP + KMeans seed selection**: diverse 600-identity pool from ~90k
  SFHQ CC0 synthetic portraits (no real-person photographs).
- **HPC pipeline scripts** (Slurm, Aire cluster, L40S GPUs):
  `hpc_download.sh`, `hpc_generate.sh` (12-shard array × 50 ids),
  `hpc_merge.sh`, `hpc_extract.sh`, `hpc_preprocess.sh`, `hpc_build.sh`,
  `hpc_full_pipeline.sh` (chained with `--dependency=afterok`),
  `hpc_smoke_test.sh`, `hpc_juggernaut_sweep.sh`, `hpc_preprocess_smoke.sh`,
  `gpu_preflight.sh`.
- **Release tooling**: `make_release_manifest.py` (path normalisation +
  prune-to-union), `validate_release.py` (schema + orphan + checksum gates),
  `schema.json` (3 defs), `RELEASE_MANIFEST.json` (provenance pin).
- **Tests**: 32 unit tests covering build, common utilities, embedding
  extraction, InstantID adapter, prompt variations, seed pool.
- **Documentation**: README (two-release, at-a-glance, pipeline, schema),
  `DATASET_CARD.md`, `CITATION.cff`, `THIRD_PARTY_NOTICES.md`,
  `HPC CookBook.md`, `licence_evidence/` (CC0 + OpenRAIL-M).

### Changed

- **Full/candidates → Raw**: the 1024×1024 release is now
  `SFHQ-VirtualID-Raw` (`dataset_raw.csv`, `portrait_*.png`); the imbalanced
  variant is exclusive to Bench.  `raw_arcface_similarity` dropped (never
  populated).
- **Imbalance gradient**: absolute 85:40:20 counts → **ratio-based** on the
  train pool (defaults 70:40:20); holdout carved per identity (15/id).
- **Image naming at release**: `accepted_*` → `crop_*` (Bench),
  `candidate_*` → `portrait_*` (Raw) — consumer-facing names at release time
  only; pipeline keeps internal names.
- **fp16 runtime + perf**: ~40–55% faster generation, ~50% less VRAM
  (xFormers/SDPA, VAE slicing, DPM++, text-embedding cache, CUDA auto-tune).
- **Extract runs on 1024 candidates** before preprocessing (siblings, not
  chained) — demographics reliable, embeddings real.
- **Relative paths** throughout manifests + release CSVs (legacy
  absolute-path manifests migrated in-place).

### Fixed

- Imbalanced holdout computed from the median per-identity count (5/id) →
  per-identity reserve formula (15/id) — matches the approved MUFAC design.
- `dataset_candidates_imbalanced` split/forget columns were unassigned
  (`KeyError` / all-zero) — fixed and verified.
- Trial-based `(identity_id, trial)` embedding join — `arcface_similarity`
  is real per-candidate similarity (std 0.034–0.043), not 1.0.
- `torch.autocast` fp16 wrapper removed — VAE decode in fp16 produced black
  images.
- GPU OOM on shard 8 — expandable segments, cache clear, `--exclusive` on
  generate only.
- `--partition` removed from CPU-only Slurm scripts (Aire `gpu`-only
  partition + QOSMinGRES).
- DATA_DIR moved from home quota (65 GB) to Lustre scratch.
- Release tooling: union-based orphan check, anchored leak patterns,
  schema defs for 3 artifacts.

### Security / hygiene

- `.gitignore` now covers local-only tracking docs (`Strat.md`,
  `RELEASE_TODO.md`), Hermes desktop app artifacts (`.hermes/`), and
  `.DS_Store` — never committed or pushed.
- History rewritten to remove `.hermes/`, `Strat.md`, `RELEASE_TODO.md`,
  `.DS_Store` from a single refactor commit before this PR.
- No weights, no source-to-output linkage, no embeddings, no seeds shipped.

### Known limitations (see `DATASET_CARD.md`)

- Synthetic, not anonymous — residual likeness / memorisation possible.
- Age labels are model-estimated proxy labels.
- No fairness balancing applied to the SFHQ source distribution.

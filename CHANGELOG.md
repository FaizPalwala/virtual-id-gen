# Changelog

All notable changes to **SFHQ-VirtualID** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — dev → main (PR #1)

> Initial release of the full pipeline + dataset.  `main` becomes the
> release branch with this merge; everything below is what ships in v1.0.0.

### Data freeze status (v1.0.0 — SHIPPED)

750 identities × 100 candidates, 5:1 train gradient.  Frozen + verified:
release QA 3/3 PASS, both HF repos live (private), Zenodo DOIs reserved.

- **Shipped artifacts**:
  - `SFHQ-VirtualID-Bench` — 224×224 aligned crops (JPEG)
    - `dataset.csv` — balanced, **67,500** images (90/id; 72 train + 18 holdout)
    - `dataset_imbalanced.csv` — ratio-based **5:1 (82:41:16)** train gradient,
      **36,064** images (kept 100/59/34 per bin; 18 holdout/id; 11 high-bin
      rows absent — 122 preprocess rejects capped 10 pools at 98–99)
  - `SFHQ-VirtualID-Raw` — 1024×1024 portraits, **74,999** images (100/id,
    max-size, no splits; JPEG q95 4:4:4; 1 corrupt candidate excluded)
- Holdout uniform at 18/id across all bins:
  `max(min_holdout, round(imagesperidentity × holdout_frac))` = 18/id.
- Train gradient = 5:1 (1.0:0.50:0.20), calibrated to the VGG-Face2 range
  (Cao et al., 2018; Wang et al., 2019; Liu et al., 2019, CVPR).
  `max_train = candidatesperidentity − holdout` (100−18=82 → ratios → 82:41:16 train).
- Forget: 75 identities (10% of 750), 15 steps × 5 ids.
- Shards: 15 (5 concurrent × 3 waves), 50 ids per shard.

### Added

- **MUFAC-aligned split protocol**: every identity contributes both
  `image_subset` values (`train` + `holdout`); identity-level `split` is
  `retain` (675) or `forget` (75); no identity-disjoint `test` split
  (unseen-identity accuracy was structurally 0).
- **Sequential forgetting protocol**: 15 steps × 5 identities, uniform +
  remainder baseline (`forget_step` column); seeded-Poisson stress-test
  schedule (`forget_step_poisson` column) with per-step counts drawn from
  λ=5 Poisson rebalanced to total 75 — models GDPR-style deletion
  request arrival (Ginart et al., 2019; arXiv:2012.01668;
  Shen et al., 2025, arXiv:2507.15280); evaluate by cumulative count.
  Both schedules ship as columns in the balanced artifact; the imbalanced
  artifact carries no schedule columns (its axis is the popularity gradient).
- **Three dataset artifacts** across two releases:
  `dataset.csv` / `dataset_imbalanced.csv` (Bench 224) and
  `dataset_raw.csv` (Raw 1024).
- **Proxy labels**: per-image `age_group`/`age`/`gender` inherited from the
  parent 1024×1024 candidate (per-image variance is a deliberate feature).
- **Confound controls**: `arcface_similarity` (per-candidate embedding,
  real within-identity spread), `laplacian_variance`, `detection_confidence`.
- **Prompt system**: 100 unique deterministic variations (20 compositions ×
  5 lighting), shuffled per identity, sliced to 100 candidates/id.
- **Identity-conditioned generation**: InstantID + Juggernaut-XL-v9 +
  ControlNet (Canny SDXL), guidance 3.0 / ip-adapter 0.85 / 30 steps.
- **CLIP + KMeans seed selection**: diverse 750-identity pool from ~90k
  SFHQ CC0 synthetic portraits (no real-person photographs).
- **HPC pipeline scripts** (Slurm, Aire cluster, L40S GPUs):
  `hpc_download.sh`, `hpc_generate.sh` (15-shard array × 50 ids),
  `hpc_merge.sh`, `hpc_extract.sh`, `hpc_preprocess.sh`, `hpc_build.sh`,
  `hpc_full_pipeline.sh` (chained with `--dependency=afterok`),
  `hpc_smoke_test.sh`, `hpc_juggernaut_sweep.sh`, `hpc_preprocess_smoke.sh`,
  `gpu_preflight.sh`.
- **Release tooling**: `make_release_manifest.py` (path normalisation +
  prune-to-union), `validate_release.py` (schema + orphan + checksum gates),
  `schema.json` (3 defs), `RELEASE_MANIFEST.json` (provenance pin).
- **Tests**: 39 unit tests covering build, common utilities, embedding
  extraction, InstantID adapter, prompt variations, seed pool, release
  validation.
- **Documentation**: README (two-release, at-a-glance, pipeline, schema),
  `DATASET_CARD.md`, `CITATION.cff`, `THIRD_PARTY_NOTICES.md`,
  `HPC CookBook.md`, `licence_evidence/` (CC0 + OpenRAIL-M).

### Changed

- **Full/candidates → Raw**: the 1024×1024 release is now
  `SFHQ-VirtualID-Raw` (`dataset_raw.csv`, `portrait_*.jpg` — JPEG q95,
  4:4:4 chroma, re-encoded from PNG at release build time);
  the imbalanced variant is exclusive to Bench.  `raw_arcface_similarity`
  dropped (never populated).
- **Imbalance gradient**: absolute 100:59:34 counts → **ratio-based** on the
  train pool (defaults 82:41:16); holdout carved per identity (18/id).
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
  per-identity reserve formula (18/id) — matches the approved MUFAC design.
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

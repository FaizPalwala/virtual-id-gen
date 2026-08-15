# SFHQ-VirtualID-Bench Dataset Card

## Summary

SFHQ-VirtualID-Bench is a **synthetic, identity-conditioned face dataset** of
750 synthetic identities, built as a benchmark for **machine unlearning**.
Each identity is a deletion unit: all 90 aligned 224×224 crops of one
identity share a single identity-level `split` (`retain` or `forget`), and 75
identities follow a sequential 15-step forgetting protocol. Every identity
contributes both per-image `image_subset` values (`train` + `holdout`), so
retention and forgetting-generalisation can be measured on genuinely held-out
images (MUFAC-aligned evaluation).

The release ships two artifacts:

| Artifact | Images | Per-identity | Purpose |
|---|---|---|---|
| `dataset.csv` (balanced) | 67,500 | 90 (72 train + 18 holdout) | Primary unlearning benchmark; ships BOTH forget schedules as columns |
| `dataset_imbalanced.csv` (imbalanced) | 36,064 | 34–100 (ratio-based) | Long-tail stress test — 5:1 popularity gradient (82:41:16 train) |

**This dataset is synthetic.  Synthetic does not mean risk-free — residual
likeness, demographic bias, and training-data memorisation are possible and
must be acknowledged.**

---

## Composition

| Property | Value |
|---|---|
| **Identities** | 750 |
| **Resolution** | 224×224 aligned crops |
| **Format** | JPEG (uint8 BGR storage) |
| **Images (balanced)** | 67,500 (90/id) |
| **Images (imbalanced)** | 36,064 (34–100/id, ratio-based) |
| **Splits** | Retain: 675 identities · Forget: 75 identities |
| **Forget protocol** | 15 steps, 75 identities — uniform 5/step (baseline) or seeded-Poisson batches (variant) |
| **Holdout** | 18 per identity (`max(min_holdout, round(imagesperidentity × holdout_frac))`) |
| **Imbalance gradient** | 5:1 — high 75 ids / medium 225 / low 450; train ratios {1.0, 0.50, 0.20} on an 82-train pool |

The imbalanced artifact ships 36,064 rows (11 fewer than 75 × 100): 123
candidates failed the preprocess quality gate (blur/unreadable), capping
10 high-bin identity pools at 98–99 crops.

## Labels

| Column | Type | Values |
|---|---|---|
| `identity_id` | int | 0–749 |
| `age_group` | int | 0=Young, 1=Adult, 2=Middle-Aged, 3=Senior |
| `age` | int | Raw InsightFace age estimate |
| `gender` | int | 0/1 (InsightFace classifier) |
| `split` | string | `retain`, `forget` (identity-level) |
| `image_subset` | string | `train`, `holdout` (per-image; MUFAC-aligned) |
| `forget_step` | int | 0–14 (forget only, **uniform** schedule), −1 (retain) |
| `forget_step_poisson` | int | 0–14 (forget only, **seeded-Poisson** schedule), −1 (retain) |
| `arcface_similarity` | float | Cosine similarity to identity's mean embedding (confound control) |
| `laplacian_variance` | float | Sharpness score (quality confound control) |
| `detection_confidence` | float | Face detector confidence (alignment control) |
| `popularity_bin` | str | `"high"` / `"medium"` / `"low"` (imbalanced only) |
| `images_per_identity` | int | Actual per-identity count (imbalanced only) |

- **Schedule columns are balanced-only.** The balanced artifact ships both
  `forget_step` (uniform baseline — step index == cumulative forgotten count)
  and `forget_step_poisson` (seeded arrival-model stress test, λ=5 rebalanced
  to 75; evaluate by **cumulative count**, not step index).  The imbalanced
  artifact deliberately carries no schedule columns — its only experimental
  axis is the popularity gradient.
- **Prompt metadata is stripped from Bench** (`pose`/`expression`/`lighting`/
  `setting`/`camera` ship only in the Raw release).
- **Age labels are per-image, not per-identity.**  Each crop inherits the
  gender/age estimate taken on its parent 1024×1024 candidate, so one identity
  can span several age groups.  This is intentional — do not collapse to a
  per-identity constant.
- **Age-group labels are proxy estimates** from the InsightFace classifier —
  not verified demographic attributes.  The confound columns are provided so
  downstream MIA/fairness analyses can control for per-identity variation.

## Split isolation invariant

Every `identity_id` maps to exactly one `split` (`retain` or `forget`), and
every identity contributes **both** `image_subset` values (`train` +
`holdout`).  No identity's images appear in multiple splits.  This invariant
is enforced by the build step and validated by `validate_release.py` before
publication.

## Intended use

The 224×224 Bench is the **primary machine-unlearning benchmark**: aligned
crops match the ImageNet training regime of downstream ResNet-18 classifiers,
so pretrained features activate at full fidelity from epoch 1.  The imbalanced
variant adds a 5:1 popularity gradient for long-tail unlearning stress-testing,
with a uniform 18-image holdout per identity so probe stability is comparable
across all popularity tiers.  Typical tasks: unlearning, membership-inference
attacks, forgetting-generalisation, retention.

## Method

Synthetic identities were generated with InstantID + Juggernaut-XL-v9 +
ControlNet from CC0 synthetic SFHQ seed images (CLIP+KMeans-diverse seed
selection); faces were detected, aligned, and cropped to 224×224, then
quality-gated (sharpness + decodability) before split assignment.  The
`arcface_similarity` column is a confound control — the similarity floor was
not applied as a filter in this release.  Exact model
revisions, generation configuration, and licence URLs are recorded in
`RELEASE_MANIFEST.json` and `THIRD_PARTY_NOTICES.md`.  No model weights are
distributed.

## Limitations

- **Synthetic, not anonymous.**  Generated faces may retain unintended
  resemblance to real persons through the training data of the generators.
- **Proxy labels.**  Age-group fields are model-estimated and carry classifier
  bias; they are not ground-truth demographics.
- **Demographic bias.**  The SFHQ source may over-/under-represent certain
  appearances; no fairness correction has been applied.
- **Artifacts.**  Diffusion artifacts (blur, distorted features) are most
  visible in 224×224 downscaled crops.
- **Imbalance is synthetic.**  The popularity gradient is a deliberate
  down-sampling, not real-world distribution.

## Privacy and release policy

The release does **not** include: raw SFHQ source images, seed-to-output
linkage, rejected candidates, ArcFace embeddings or biometric templates, or
model weights.  Users must comply with the upstream licence terms of the
generation models (see `THIRD_PARTY_NOTICES.md`).  A takedown process for
credible likeness complaints is available via the repository.

## Licence and citation

- **Code:** MIT
- **Images and metadata:** Non-commercial research use only, with a
  prohibited-use clause (no biometric identification, surveillance,
  authentication, impersonation, or high-impact decisions about people).  See
  [`LICENSE`](LICENSE).

```bibtex
@dataset{sfhq_virtualid_bench,
  title     = {{SFHQ-VirtualID-Bench}: Synthetic Identity-Conditioned Aligned
               Face Crops for Machine Unlearning},
  author    = {Faiz Palwala},
  year      = {2026},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.21877893},
  url       = {https://github.com/FaizPalwala/virtual-id-gen},
}
```

See [`CITATION.cff`](CITATION.cff) for the complete metadata file.

## Acknowledgements

This work was undertaken on the Aire HPC system at the University of Leeds, UK.

# SFHQ-VirtualID-Bench — Dataset Card

**Synthetic identity-conditioned aligned face crops (224×224) for machine-unlearning benchmark evaluation.**

| | |
|---|---|
| **Identity-conditioned** | 750 synthetic identities, each a deletion unit |
| **Images** | 67,500 (balanced) + 36,064 (imbalanced) |
| **Splits** | Identity-level `retain` (675) / `forget` (75); per-image train + holdout |
| **Forget protocol** | 15 steps × 75 identities — uniform 5/step, or seeded-Poisson variant |
| **Resolution / format** | 224×224, JPEG |
| **License** | Images/metadata: non-commercial research (see `LICENSE`) |

---

## Summary

SFHQ-VirtualID-Bench is a benchmark for evaluating **machine unlearning in
image classification**. Each of the 750 synthetic identities is a deletion
unit: all images of one identity share a single identity-level `split`
(`retain` or `forget`), so no identity's images straddle the boundary.
75 identities follow a sequential 15-step forgetting protocol, shipped in
two schedules — uniform 5/step (baseline) and a seeded-Poisson arrival
model (stress test; evaluate by **cumulative forgotten count**, not step
index — Ginart et al. 2019; arXiv:2012.01668; Shen et al. 2025,
arXiv:2507.15280).

**This dataset is synthetic. Synthetic does not mean risk-free — residual
likeness, demographic bias, and training-data memorisation are possible and
must be acknowledged.**

---

## Composition

| Property | Value |
|---|---|
| **Images (balanced)** | 67,500 |
| **Images (imbalanced)** | 36,064 |
| **Identities** | 750 |
| **Images per identity (balanced)** | 90 (72 train + 18 holdout) |
| **Imbalance gradient (imbalanced)** | Ratio-based 5:1 — high 75 ids / medium 225 / low 450; train pools {82, 41, 16} |
| **Holdout** | 18 per identity in both variants |
| **Resolution / format** | 224×224, JPEG (uint8 BGR storage) |
| **Splits** | Retain 675 identities / Forget 75 identities; per-image `image_subset` = train + holdout within each |

The imbalanced total is 36,064 (not 75 × 100): 122 candidates failed the
preprocess quality gate (121 blur, 1 unreadable), capping 10 high-bin
identity pools at 98–99 images.

---

## Labels

| Column | Type | Values |
|---|---|---|
| `identity_id` | int | 0–749 |
| `age_group` | int | 0=Young, 1=Adult, 2=Middle-Aged, 3=Senior |
| `age` | int | Raw InsightFace age estimate (per-image) |
| `gender` | int | 0/1 (InsightFace classifier) |
| `split` | string | `retain`, `forget` (identity-level) |
| `image_subset` | string | `train`, `holdout` (per-image; MUFAC-aligned) |
| `forget_step` | int | 0–14 (forget only, uniform schedule), -1 (retain) |
| `forget_step_poisson` | int | 0–14 (forget only, seeded-Poisson schedule), -1 (retain) |
| `arcface_similarity` | float | Cosine similarity to identity's mean embedding (confound control) |
| `laplacian_variance` | float | Sharpness score (quality confound control) |
| `detection_confidence` | float | Face detector confidence (alignment control) |
| `popularity_bin` | str | `"high"` / `"medium"` / `"low"` (imbalanced only) |
| `images_per_identity` | int | Actual per-identity count (imbalanced only) |

**Schedule columns are balanced-only.** The balanced artifact ships both
`forget_step` (uniform baseline — step index == cumulative forgotten count)
and `forget_step_poisson` (seeded arrival-model stress test, λ=5 rebalanced
to 75; evaluate by cumulative count). The imbalanced artifact carries **no
schedule columns** — its only experimental axis is the popularity gradient.
Prompt metadata (pose/expression/lighting/setting/camera) is stripped from
Bench; it ships only in the Raw release.

**Age-group labels are proxy estimates** from the InsightFace
gender/age classifier — not verified demographic attributes. **Age labels
are per-image, not per-identity**: one identity can span several age groups
(e.g. identity 558 ranges 23–65) while remaining recognisably the same
person; do not collapse to a per-identity constant.

---

## Split isolation invariant

Every `identity_id` maps to exactly one `split` (`retain` or `forget`), and
every identity contributes **both** `image_subset` values (`train` +
`holdout`). No identity's images appear in multiple splits. This invariant
is enforced by the build step and validated by `validate_release.py` before
publication.

---

## Intended use

The 224×224 aligned crops match the ImageNet training regime of downstream
ResNet-18 classifiers (`CROP_SIZE`, `IMAGENET_MEAN`, `IMAGENET_STD` in
`src/common.py`), so pretrained features activate at full fidelity from
epoch 1. The imbalanced variant adds a ratio-based 5:1 (82:41:16) popularity
gradient for long-tail unlearning stress-testing, with a uniform 18-image
holdout per identity so probe stability is comparable across all popularity
tiers.

**Experimental protocol for the long-tail hypothesis test** (H1: high-bin
identities are over-learned → hardest to forget; low-bin under-learned →
easiest to scrub):

| Step | Protocol | What it isolates |
|---|---|---|
| A. Baseline gate | Per-bin MIA AUC + per-identity train accuracy **before** unlearning; low-bin ≈ 0.5 is floor-bound — report, never average away | "Never learned" vs "scrubbed" |
| B. Distance-to-oracle | Per-bin retrain oracle; per-bin distance after a fixed unlearning budget | Difficulty, budget-fairly |
| C. Budget sweep | Sweep budget per bin; budget at which each bin reaches its oracle | Cleanest causal test of H1 |
| D. Balanced cross-check | Same 75 forget identities through the balanced artifact (90/id) | Count-gradient vs identity attribution |
| E. Intensity-modulation | Fixed-intensity baseline; expect heterogeneous/skewed deviation (FaLW, arXiv:2601.18650) | Whether per-sample reweighting is needed |

**Analysis rule:** all cross-schedule and cross-bin comparisons use
**cumulative forgotten count**, never raw step index. The retrain-oracle
definition follows Bourtoule et al. 2021 (IEEE S&P).

---

## Limitations

- **Synthetic, not anonymous.** Generated faces may retain unintended
  resemblance to real persons via the training data of the generator stack.
  Synthetic is not a privacy guarantee.
- **Proxy labels.** Age-group fields are model-estimated and carry InsightFace
  classifier bias.
- **Reconstruction risk.** The identity adapter encodes each seed into an
  identity embedding; the method inherently preserves similarity to the seed.
  Seed-to-output linkage and embeddings are **not** released.
- **Demographic bias.** The SFHQ source may over-/under-represent certain
  appearances; no fairness correction has been applied.
- **Artifacts.** Diffusion artifacts (blur, distorted features) are most
  visible in the 224×224 downscaled crops.
- **Imbalance is synthetic.** The popularity gradient is a deliberate
  down-sampling, not a real-world distribution.

---

## Privacy and release policy

The release does **not** include: raw SFHQ source images, seed-to-output
linkage, rejected candidates, ArcFace embeddings, or model weights.

Users must comply with upstream licence terms of the generation models. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

## Licence and citation

- **Code:** MIT
- **Images and metadata:** Non-commercial research use only, with
  prohibited-use clause (no biometric identification, surveillance,
  authentication, impersonation, or high-impact decisions about people).
  See [`LICENSE`](LICENSE).
- **Third-party models:** not included — obtain weights from upstream sources.

```bibtex
@dataset{sfhq_virtualid_bench,
  title     = {{SFHQ-VirtualID-Bench}: Synthetic Identity-Conditioned Aligned
               Face Crops for Machine Unlearning},
  author    = {TODO},
  year      = {2026},
  version   = {1.0.0},
  doi       = {TODO},
  url       = {https://github.com/FaizPalwala/virtual-id-gen},
}
```

See [`CITATION.cff`](CITATION.cff) for the complete metadata file.

---

## Acknowledgements

This work was undertaken on the Aire HPC system at the University of Leeds, UK.

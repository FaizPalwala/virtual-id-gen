---
pretty_name: SFHQ-VirtualID
license: other
task_categories:
  - image-classification
  - other
tags:
  - synthetic
  - face
  - machine-unlearning
  - identity-clustering
  - instantid
  - juggernaut
  - sdxl
  - imbalanced
  - long-tail
size_categories:
  - 10K<n<100K
---
# SFHQ-VirtualID Dataset Card

## Summary

SFHQ-VirtualID is a family of **synthetic, identity-conditioned face
datasets** built from CC0 synthetic SFHQ seed images via an InstantID +
Juggernaut-XL-v9 + ControlNet pipeline.  Each of the 600 synthetic identities
is a deletion unit — all images of a single identity share one split
(`retain`, `test`, or `forget`), and 60 identities are assigned to a
sequential 15-step forgetting protocol (4 identities per step).

The project publishes **two complementary releases**:

| Release | Resolution | Contents | Role |
|---|---|---|---|
| **SFHQ-VirtualID-Bench** | 224×224 aligned crops | Balanced + imbalanced | Machine-unlearning benchmark (primary) |
| **SFHQ-VirtualID-Full** | 1024×1024 candidates | Balanced + imbalanced | General-purpose identity-conditioned faces |

Both releases share identical `identity_id`, `split`, `forget_step`, and
`forget_variant` assignments, so results transfer between them.  Demographics
(age, gender) are derived from face detection on the 1024×1024 candidates —
where InsightFace reliably works — and inherited by the aligned 224×224 crops
via identity join.

**This dataset is synthetic and identity-conditioned.  Synthetic does not mean
risk-free — residual likeness, demographic bias, and training-data memorisation
are possible and must be acknowledged.**

---

## SFHQ-VirtualID-Bench (224×224 aligned crops)

### Composition

| Property | Value |
|---|---|
| **Images (balanced)** | 45,000 |
| **Identities** | 600 |
| **Images per identity (balanced)** | 75 |
| **Images (imbalanced)** | ~19,500 |
| **Imbalance gradient** | 85:40:20 (high 60 ids / medium 180 / low 360) |
| **Resolution** | 224×224 |
| **Format** | JPEG (uint8 BGR storage) |
| **Splits (both variants)** | Retain: 450 identities<br>Test: 90 identities<br>Forget: 60 identities |
| **Forget protocol** | 15 steps, 4 identities per step (uniform + remainder) |

### Labels

| Column | Type | Values |
|---|---|---|
| `identity_id` | int | 0–599 |
| `age_group` | int | 0=Young, 1=Adult, 2=Middle-Aged, 3=Senior |
| `age` | int | Raw InsightFace age estimate |
| `gender` | int | 0/1 (InsightFace classifier) |
| `split` | string | `retain`, `test`, `forget` |
| `forget_step` | int | 0–14 (forget only), -1 (retain/test) |
| `forget_variant` | int | Index within forget step; -1 for non-forget |
| `arcface_similarity` | float | Cosine similarity to identity's mean embedding (confound control) |
| `laplacian_variance` | float | Sharpness score (quality confound control) |
| `detection_confidence` | float | Face detector confidence (alignment control) |
| `popularity_bin` | str | `"high"` / `"medium"` / `"low"` (imbalanced only) |
| `images_per_identity` | int | Actual per-identity count (imbalanced only) |

**Age-group labels are proxy estimates** from the InsightFace gender/age
classifier.  They are not verified demographic attributes and should not be
interpreted as such.  The confound columns (`arcface_similarity`,
`laplacian_variance`, `detection_confidence`, `gender`) are included so
downstream MIA/fairness analyses can control for per-identity variation.

### Split isolation invariant

Every `identity_id` maps to exactly one `split`.  No identity's images appear
in multiple splits.  This invariant is enforced by the build step
(`build_dataset.py`) and validated by `validate_release.py` before publication.

### Intended use

The 224×224 Bench is the **primary machine-unlearning benchmark**: aligned
crops match the ImageNet training regime of downstream ResNet-18 classifiers
(`CROP_SIZE`, `IMAGENET_MEAN`, `IMAGENET_STD` in `src/common.py`), so
pretrained features activate at full fidelity from epoch 1.  The imbalanced
variant adds the 85:40:20 popularity gradient for long-tail unlearning
stress-testing.

---

## SFHQ-VirtualID-Full (1024×1024 candidates)

### Composition

| Property | Value |
|---|---|
| **Images (balanced)** | 51,000 |
| **Identities** | 600 |
| **Images per identity** | 85 (all candidates, no quality trim) |
| **Images (imbalanced)** | ~TBD |
| **Resolution** | 1024×1024 |
| **Format** | PNG |
| **Splits (both variants)** | Retain: 450 identities<br>Test: 90 identities<br>Forget: 60 identities |
| **Forget protocol** | 15 steps, 4 identities per step (uniform + remainder) |

### Labels

| Column | Type | Description |
|---|---|---|
| `image_path` | string | `identities/candidates/identity_NNN/candidate_YYY.png` |
| `identity_id` | int | 0–599 |
| `age_group` / `age` / `gender` | int | Proxy demographics from 1024 detection |
| `split` | string | `retain`, `test`, `forget` |
| `forget_step` / `forget_variant` | int | Unlearning step / variant |
| `arcface_similarity` | float | Cosine similarity to identity's mean candidate embedding |
| `pose` | string | Head/body position (from prompt) |
| `expression` | string | Facial expression (from prompt) |
| `lighting` | string | Lighting condition (from prompt) |
| `setting` | string | Background/scene (from prompt) |
| `camera` | string | Camera angle (from prompt) |
| `raw_arcface_similarity` | float | Similarity of full-res candidate to source seed (TBD) |
| `popularity_bin` / `images_per_identity` | — | Imbalanced variant only |

No `laplacian_variance` or `detection_confidence` — these are crop-level
quality metrics and are meaningless on raw candidates.

### Intended use

The 1024×1024 Full release is for **general-purpose identity research**:
identity recognition, face generation evaluation, demographic bias studies,
and **erasure-transfer testing** (does forgetting the 224 crop also hide
identity in the full context?).  It is not trimmed by quality gates — all 85
generated candidates per identity are included.

---

## Collection and generation

### Source data

The seed images are drawn from [SFHQ](https://github.com/SelfishGene/SFHQ-
dataset) part 1 (~90,000 images).  SFHQ is a CC0 / Public Domain synthetic
portrait dataset generated via StyleGAN2 and diffusion models.  No real-person
photographs are used at any stage of the pipeline.

**Seed selection:** CLIP feature embeddings of SFHQ images are clustered with
KMeans, and diverse representatives are selected so the 600 identities span
the source appearance space rather than clumping on similar faces.

### Generation pipeline

| Stage | Method | Purpose |
|---|---|---|
| Seed selection | CLIP + KMeans | Diverse 600-seed pool from ~90k SFHQ images |
| Candidate generation | InstantID + Juggernaut-XL-v9 + ControlNet | 85 identity-conditioned candidates per identity |
| Prompt variation | 100-prompt pool (20 compositions × 5 lighting) | Controlled non-identity variation; unique prompt per candidate |
| **Embedding extraction** | ArcFace on 1024×1024 candidates | Identity embeddings + demographics (detection works on full portraits) |
| Face alignment | MTCNN (`facenet-pytorch`) | Detect, align, crop faces to 224×224 |
| Quality filtering | Laplacian variance | Discard blurry crops (< 80) |
| Identity gating | ArcFace cosine similarity | Keep images ≥ 0.45 similarity to seed identity |
| Split assignment | Seeded shuffle (seed 42) | Identity-level retain/test/forget |
| Imbalance pruning | Seeded down-sample (seeds 42+9998/9999) | 85:40:20 popularity gradient |

### Model versions

Exact model revisions, commit hashes, and licence URLs are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`RELEASE_MANIFEST.json`](RELEASE_MANIFEST.json).  Users must obtain all model
weights from upstream sources under the upstream licence terms — no weights are
distributed with this dataset.

### Generation configuration

| Parameter | Value |
|---|---|
| Base model | `RunDiffusion/Juggernaut-XL-v9` |
| Inference steps | 30 |
| Guidance scale | 3.0 |
| IP adapter scale | 0.85 |
| ControlNet scale | 0.80 |
| Resolution (generation) | 1024×1024 |
| Resolution (Bench crop) | 224×224 |
| Random seed | 42 |
| Shard seeds | 42, 44, …, 64 (12 shards × 50 identities) |
| Candidates per identity | 85 |
| Prompt pool | 100 (20 compositions × 5 lighting) |

## Limitations

- **Synthetic, not anonymous.**  Generated faces may retain unintended
  resemblance to real persons through the training data of Juggernaut-XL-v9 /
  SDXL or the SFHQ generator.  Synthetic is not a privacy guarantee.
- **Proxy labels.**  Age-group fields are model-estimated.  They carry the
  biases of the InsightFace classifier and must not be treated as ground-truth
  demographics.
- **Reconstruction risk.**  The InstantID adapter encodes the seed image into
  an identity embedding.  We do not release seed-to-output linkage or
  embeddings, but the method inherently preserves similarity to each single
  seed — this is by design and is what makes the dataset useful.
- **Demographic bias.**  The SFHQ source dataset may over- or under-represent
  certain appearance characteristics.  No balancing or fairness correction has
  been applied.
- **Artifacts.**  Generated images may contain diffusion artifacts (blur,
  distorted facial features, inconsistent accessories), most visible in the
  224×224 downscaled Bench crops.
- **Imbalance is synthetic.**  The popularity gradient is a deliberate
  down-sampling, not a reflection of real-world celebrity distributions.

## Privacy and release policy

The release does **not** include:
- Raw SFHQ source images
- Seed-to-output linkage (which seed produced which identity)
- Rejected or low-quality candidates
- ArcFace embedding vectors or biometric templates
- Model weights of any kind

A reporting contact and takedown process for credible likeness or policy
complaints will be published with the release.

Users of this dataset must comply with the upstream licence terms of
Juggernaut-XL-v9, InstantID, ControlNet, and InsightFace models.  See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Licence and citation

- **Code:** MIT
- **Images and metadata:** Non-commercial research use only, with prohibited-
  use clause (no biometric identification, surveillance, authentication,
  impersonation, or high-impact decisions about people).  See [`LICENSE`](LICENSE).
- **Third-party models:** Not included — users obtain weights from upstream sources.

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

@dataset{sfhq_virtualid_full,
  title     = {{SFHQ-VirtualID-Full}: Full-Resolution Synthetic
               Identity-Conditioned Face Candidates},
  author    = {TODO},
  year      = {2026},
  version   = {1.0.0},
  doi       = {TODO},
  url       = {https://github.com/FaizPalwala/virtual-id-gen},
}
```

See [`CITATION.cff`](CITATION.cff) for the complete metadata file.

## Acknowledgements

This work was undertaken on the Aire HPC system at the University of Leeds, UK.

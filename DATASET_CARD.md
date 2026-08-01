---
pretty_name: SFHQ-InstantID
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
  - sdxl
size_categories:
  - 10K<n<100K
---
# SFHQ-InstantID Dataset Card

## Summary

SFHQ-InstantID is a **synthetic, identity-conditioned face dataset** designed
as a benchmark for **machine unlearning**.  Each of the 400 synthetic
identities is a deletion unit — all 40 images of a single identity share one
split (`retain`, `test`, or `forget`), and 40 identities are assigned to a
sequential 40-step forgetting protocol.

The dataset is constructed by a fully reproducible pipeline: CC0 synthetic SFHQ
seed images drive an InstantID + SDXL + ControlNet pipeline that introduces
controlled identity-preserving variation (pose, expression, lighting,
accessories).  Generated candidates pass through MTCNN alignment, sharpness
filtering, and ArcFace similarity gating.  The final release contains 16,000
aligned 128×128 face crops with proxy age-group labels and identity-level split
assignments.

**This dataset is synthetic and identity-conditioned.  Synthetic does not mean
risk-free — residual likeness, demographic bias, and training-data memorisation
are possible and must be acknowledged.**

## Composition

| Property | Value |
|---|---|
| **Images** | 16,000 |
| **Identities** | 400 |
| **Images per identity** | 40 |
| **Resolution** | 128×128 RGB |
| **Format** | PNG |
| **Splits** | Retain: 300 identities (12,000 images)<br>Test: 60 identities (2,400 images)<br>Forget: 40 identities (1,600 images) |
| **Forget protocol** | 40 steps, 1 identity deleted per step |

### Labels

| Column | Type | Values |
|---|---|---|
| `clusterid` | int | 0–399 |
| `agegroup` | int | 0=Young, 1=Adult, 2=Middle-Aged, 3=Senior |
| `split` | string | `retain`, `test`, `forget` |
| `forgetstep` | int | 0–39 (forget only), -1 (retain/test) |

**Age-group labels are proxy estimates** from the InsightFace gender/age
classifier.  They are not verified demographic attributes and should not be
interpreted as such.

### Split isolation invariant

Every `clusterid` maps to exactly one `split`.  No identity's images appear in
multiple splits.  This invariant is enforced by the build step
(`build_dataset.py`) and validated by `validate_release.py` before publication.

## Collection and generation

### Source data

The seed images are drawn from [SFHQ](https://github.com/SelfishGene/SFHQ-
dataset) part 1 (~10,000 images).  SFHQ is a CC0 / Public Domain synthetic
portrait dataset generated via StyleGAN2 and diffusion models.  No real-person
photographs are used at any stage of the pipeline.

### Generation pipeline

| Stage | Method | Purpose |
|---|---|---|
| Identity generation | InstantID adapter + SDXL + ControlNet | 70 identity-conditioned candidates per seed |
| Face alignment | MTCNN (`facenet-pytorch`) | Detect, align, crop faces |
| Quality filtering | Laplacian variance | Discard blurry crops (< 80) |
| Identity gating | ArcFace cosine similarity | Keep images ≥ 0.45 similarity to seed |
| Final selection | Top-N by similarity + diversity | Select 40 best per identity |
| Embedding extraction | ArcFace (InsightFace) | Identity embeddings (not released) |
| Age labelling | InsightFace gender/age classifier | Proxy age-group labels |
| Split assignment | Seeded shuffle | Identity-level retain/test/forget |

### Model versions

Exact model revisions, commit hashes, and licence URLs are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`RELEASE_MANIFEST.json`](RELEASE_MANIFEST.json).  Users must obtain all model
weights from upstream sources under the upstream licence terms — no weights are
distributed with this dataset.

### Generation configuration

| Parameter | Value |
|---|---|
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` |
| Inference steps | 25 (DPM++) |
| Guidance scale | 5.5 |
| IP adapter scale | 0.90 |
| ControlNet scale | 0.80 |
| Resolution (generation) | 1024×1024 |
| Resolution (final crop) | 128×128 |
| Random seed | 42 |
| Shard seeds | 42, 44, 46, 48 |

## Intended use

This dataset is designed for **research on machine unlearning** — specifically,
evaluating whether a model can genuinely delete knowledge of a synthetic
identity after sequential forgetting requests.  It is also suitable for:

- Controlled studies of identity-level data deletion
- Evaluation of unlearning verification methods
- Benchmarking differential privacy and federated unlearning

It is **not** intended for training production face recognition systems,
biometric authentication, or any application that makes decisions about people.

## Limitations

- **Synthetic, not anonymous.**  Generated faces may retain unintended
  resemblance to real persons through the training data of SDXL or the SFHQ
  generator.  Synthetic is not a privacy guarantee.
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
  distorted facial features, inconsistent accessories) especially in the 128×
  128 downscaled crops.

## Privacy and release policy

The release does **not** include:
- Raw SFHQ source images
- Seed-to-output linkage (which seed produced which identity)
- Rejected or low-quality candidates
- ArcFace embedding vectors or biometric templates
- Model weights of any kind

A reporting contact and takedown process for credible likeness or policy
complaints will be published with the release.

Users of this dataset must comply with the upstream licence terms of SDXL,
InstantID, ControlNet, and InsightFace models.  See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Licence and citation

- **Code:** MIT
- **Images and metadata:** Non-commercial research use only, with prohibited-
  use clause (no biometric identification, surveillance, authentication,
  impersonation, or high-impact decisions about people).  See [`LICENSE`](LICENSE).
- **Third-party models:** Not included — users obtain weights from upstream sources.

```bibtex
@dataset{sfhq_instantid_v1,
  title     = {{SFHQ-InstantID}: A Synthetic Identity-Conditioned Face Dataset
               for Machine Unlearning},
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

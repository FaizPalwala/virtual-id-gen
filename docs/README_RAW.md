---
pretty_name: SFHQ-VirtualID-Raw (Synthetic Face HQ — Virtual Identities)
license: other
license_detail: "Non-Commercial Research Licence — images and metadata (see LICENSE); MIT for code only"
task_categories:
  - image-classification
  - image-to-image
tags:
  - machine-unlearning
  - synthetic-faces
  - identity-conditioned
  - face-generation
  - portraits
size_categories:
  - 10K<n<100K
language:
  - en
---
# SFHQ-VirtualID-Raw Dataset Card

## Summary

SFHQ-VirtualID-Raw is a **synthetic, identity-conditioned portrait dataset**
of 750 synthetic identities — 75,000 full-resolution 1024×1024 portraits
(100 per identity).  It is the max-size reference companion to the
SFHQ-VirtualID-Bench benchmark: no splits, no quality trim, no forget
protocol.  Consumers construct their own train/holdout partitions.

The dataset is generated with InstantID + Juggernaut-XL-v9 + ControlNet from
CC0 synthetic SFHQ seed images.  **Synthetic does not mean risk-free —
residual likeness, demographic bias, and training-data memorisation are
possible and must be acknowledged.**

---

## Composition

| Property | Value |
|---|---|
| **Images** | 75,000 |
| **Identities** | 750 |
| **Images per identity** | 100 (all portraits, no quality trim) |
| **Resolution** | 1024×1024 |
| **Format** | JPEG q95, 4:4:4 chroma (converted in-build from PNG, verify-then-delete) |
| **Splits** | None (max-size reference) |
| **Forget protocol** | None |

All 75,000 generated candidates shipped (no exclusions in this release); the
`arcface_similarity` column is computed on the shipped JPEGs.

## Labels

| Column | Type | Description |
|---|---|---|
| `image_path` | string | `images/identity_NNN/portrait_YYY.jpg` |
| `identity_id` | int | 0–749 |
| `age_group` / `age` / `gender` | int | Proxy demographics from 1024 detection |
| `arcface_similarity` | float | Cosine similarity to identity's mean portrait embedding |
| `pose` | string | Head/body position (from prompt) |
| `expression` | string | Facial expression (from prompt) |
| `lighting` | string | Lighting condition (from prompt) |
| `setting` | string | Background/scene (from prompt) |
| `camera` | string | Camera angle (from prompt) |

No `laplacian_variance`, `detection_confidence`, `split`, or `forget_*`
columns — these are crop-level quality metrics / split-protocol fields and
are meaningless on raw portraits.

## Intended use

The 1024×1024 Raw release is for **general-purpose identity research**:
identity recognition, face generation evaluation, demographic bias studies,
and **erasure-transfer testing** (does forgetting the 224 crop also hide
identity in the full context?).  It is not trimmed by quality gates — all
generated portraits per identity are included.

## Method

Synthetic identities were generated with InstantID + Juggernaut-XL-v9 +
ControlNet from CC0 synthetic SFHQ seed images (CLIP+KMeans-diverse seed
selection); portraits were kept at full 1024×1024 resolution.  Exact model
revisions, generation configuration, and licence URLs are recorded in
`RELEASE_MANIFEST.json` and `THIRD_PARTY_NOTICES.md`.  No model weights are
distributed.

## Limitations

- **Synthetic, not anonymous.**  Generated faces may retain unintended
  resemblance to real persons through the training data of the generators.
- **Proxy labels.**  Age-group fields are model-estimated and carry classifier
  bias; they are not ground-truth demographics.
- **Reconstruction risk.**  The InstantID adapter encodes the seed image into
  an identity embedding.  Seed-to-output linkage and embeddings are not
  released, but each portrait inherently resembles its seed identity — this
  is by design.
- **Demographic bias.**  The SFHQ source may over-/under-represent certain
  appearances; no fairness correction has been applied.

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
@dataset{sfhq_virtualid_raw,
  title     = {{SFHQ-VirtualID-Raw}: Full-Resolution Synthetic
               Identity-Conditioned Face Portraits},
  author    = {Faiz Palwala},
  year      = {2026},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.21879130},
  url       = {https://github.com/FaizPalwala/virtual-id-gen},
}
```

See [`CITATION.cff`](CITATION.cff) for the complete metadata file.

## Acknowledgements

This work was undertaken on the Aire HPC system at the University of Leeds, UK.

# SFHQ-VirtualID-Raw — Dataset Card

**Synthetic identity-conditioned full-resolution face portraits (1024×1024) for general-purpose identity research.**

| | |
|---|---|
| **Identity-conditioned** | 750 synthetic identities |
| **Images** | 74,999 (all portraits, no quality trim) |
| **Splits** | None (max-size reference; consumers construct their own) |
| **Resolution / format** | 1024×1024, JPEG q95 (4:4:4 chroma) |
| **License** | Images/metadata: non-commercial research (see `LICENSE`) |

---

## Summary

SFHQ-VirtualID-Raw is the full-resolution companion release to
[SFHQ-VirtualID-Bench](https://huggingface.co/datasets/FaizPalwala/SFHQ-VirtualID-Bench):
every 224×224 aligned crop in the Bench is a downscale of one of these
1024×1024 portraits. All 100 generated portraits per identity are included —
no quality gates, no splits, no forget protocol. Demographics (age, gender)
and `arcface_similarity` are computed on these full-resolution images and
inherited by the Bench crops via identity join.

**This dataset is synthetic. Synthetic does not mean risk-free — residual
likeness, demographic bias, and training-data memorisation are possible and
must be acknowledged.**

---

## Composition

| Property | Value |
|---|---|
| **Images** | 74,999 |
| **Identities** | 750 |
| **Images per identity** | 100 (all portraits, no quality trim) |
| **Resolution / format** | 1024×1024, JPEG q95 (4:4:4 chroma; re-encoded from PNG at release time) |
| **Splits** | None (max-size reference; consumers construct their own train/holdout) |
| **Forget protocol** | None |

The total is 74,999 (not 75 × 100): one candidate
(`identity_501/trial_012`) was an unreadable/corrupt generation and was
excluded; every other generated portrait ships.

---

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

---

## Intended use

The 1024×1024 Raw release is for **general-purpose identity research**:
identity recognition, face generation evaluation, demographic bias studies,
and **erasure-transfer testing** (does forgetting the 224 crop also hide
identity in the full context?). It is not trimmed by quality gates — all
100 generated portraits per identity are included.

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
- **Artifacts.** Generated images may contain diffusion artifacts (blur,
  distorted facial features, inconsistent accessories).

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
@dataset{sfhq_virtualid_raw,
  title     = {{SFHQ-VirtualID-Raw}: Full-Resolution Synthetic
               Identity-Conditioned Face Portraits},
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

# Third-Party Notices — SFHQ-InstantID

This dataset was generated using the following third-party components.  The
components themselves (model weights, source code, pretrained checkpoints) are
**not distributed** with this dataset.  Each user must obtain them from the
upstream sources listed below under each project's own licence terms.

## Source data

| Component | Source | Licence | Notes |
|---|---|---|---|
| SFHQ | [SelfishGene/SFHQ-dataset](https://github.com/SelfishGene/SFHQ-dataset) | CC0 / Public Domain (Kaggle listing) | Synthetic portraits from StyleGAN2 + diffusion models.  Part 1 (~10k images) used. |

## Generative models

| Component | Repository | Weights | Licence | Output-use terms |
|---|---|---|---|---|
| SDXL 1.0 | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | Hugging Face | Stability AI Open RAIL++-M | Permits output distribution with attribution; prohibits harmful use |
| InstantID | [InstantX/InstantID](https://github.com/InstantX/InstantID) | Hugging Face | Apache 2.0 (code) | Model weights licence may differ — check upstream |
| ControlNet (SDXL) | [diffusers/controlnet-canny-sdxl-1.0](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0) | Hugging Face | OpenRAIL++-M | Same output-use terms as SDXL base |

## Face analysis models

| Component | Repository | Weights | Licence | Notes |
|---|---|---|---|---|
| InsightFace (ArcFace) | [deepinsight/insightface](https://github.com/deepinsight/insightface) | `buffalo_l` model pack | MIT (code); model terms vary | `w600k_r50.onnx` for ArcFace embeddings; `1k3d68.onnx`, `2d106det.onnx`, `det_10g.onnx`, `genderage.onnx` for detection and age estimation |
| MTCNN | [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch) | Bundled in package | MIT | Face detection and alignment |

## Diffusion and ML framework

| Component | Version | Licence |
|---|---|---|
| Diffusers | 0.39.0 | Apache 2.0 |
| Transformers | ≥4.40 | Apache 2.0 |
| PyTorch | 2.6.0+cu124 | BSD |
| ONNX Runtime | GPU build | MIT |

## Notes

- **`DIAMONIK7777/antelopev2`** (InsightFace model mirror):  This project does
  **not** use antelopev2.  All InsightFace models are loaded from the official
  `buffalo_l` model pack via the `insightface` Python package.
- **Exact revisions**:  Commit hashes and download dates for each model are
  recorded in [`RELEASE_MANIFEST.json`](RELEASE_MANIFEST.json).
- **Upstream terms**:  Users generating their own datasets with this pipeline
  must independently verify the current licence terms of each component at the
  time of generation, as upstream licences may change.  In particular, SDXL and
  ControlNet output-use terms apply to any images generated with those models,
  regardless of whether the weights are distributed.

---

*This file is part of the SFHQ-InstantID dataset release and should be
preserved with all copies of the dataset.*

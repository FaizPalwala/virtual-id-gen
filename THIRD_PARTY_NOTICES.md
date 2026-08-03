# Third-Party Notices — SFHQ-InstantID

This dataset was generated using the following third-party components.  The
components themselves (model weights, source code, pretrained checkpoints) are
**not distributed** with this dataset.  Each user must obtain them from the
upstream sources listed below under each project's own licence terms.

## Source data

| Component | Source | Licence | Notes |
|---|---|---|---|
| SFHQ | [SelfishGene/SFHQ-dataset](https://github.com/SelfishGene/SFHQ-dataset) | CC0 / Public Domain (Kaggle listing) | Synthetic portraits from StyleGAN2 + diffusion models.  Part 1 (~90k images) used; CLIP + KMeans diversity selection picks the 600 seeds. |

## Generative models

| Component | Repository | Weights | Licence | Output-use terms |
|---|---|---|---|---|
| Juggernaut-XL-v9 | [RunDiffusion/Juggernaut-XL-v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) | Hugging Face (`cf41923…`) | **CreativeML Open RAIL-M** (verified 2026-08-03) | Outputs: "Licensor claims no rights in the Output You generate" — research output distribution permitted; restricted uses (harm, discrimination, etc.) prohibited. Paid-API serving requires separate licence (not applicable to dataset generation) |
| InstantID | [InstantX/InstantID](https://huggingface.co/InstantX/InstantID) | Hugging Face (`57b32df…`) | **Apache 2.0** (weights + code, verified 2026-08-03) | Cleanest component: permissive licence, non-gated weights (`ip-adapter.bin` + ControlNetModel) |
| ControlNet (SDXL) | [diffusers/controlnet-canny-sdxl-1.0](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0) | Hugging Face (`eb115a1…`) | **OpenRAIL++** (verified 2026-08-03) | Same output terms as SDXL: no rights claimed in generated outputs; Attachment A restricted uses apply |
| SDXL 1.0 (Juggernaut foundation) | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | Hugging Face | Stability AI Open RAIL++-M | Juggernaut-XL-v9 is a fine-tune of SDXL; underlying output-use terms apply |

## Face analysis models

| Component | Repository | Weights | Licence | Notes |
|---|---|---|---|---|
| InsightFace (ArcFace) | [deepinsight/insightface](https://github.com/deepinsight/insightface) | `buffalo_l` model pack | MIT (code); model terms vary | `w600k_r50.onnx` for ArcFace embeddings; `1k3d68.onnx`, `2d106det.onnx`, `det_10g.onnx`, `genderage.onnx` for detection and age/gender estimation |
| MTCNN | [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch) | Bundled in package | MIT | Face detection and alignment |

## Diffusion and ML framework

| Component | Version | Licence |
|---|---|---|
| Diffusers | 0.39.0 | Apache 2.0 |
| Transformers | ≥4.40 | Apache 2.0 |
| PyTorch | 2.6.0+cu124 | BSD |
| ONNX Runtime | GPU build | MIT |
| CLIP (seed selection) | via `transformers` / `open_clip` | MIT |

## Notes

- **`DIAMONIK7777/antelopev2`** (InsightFace model mirror):  This project does
  **not** use antelopev2.  All InsightFace models are loaded from the official
  `buffalo_l` model pack via the `insightface` Python package.
- **Exact revisions**:  Commit hashes and download dates for each model are
  recorded in [`RELEASE_MANIFEST.json`](RELEASE_MANIFEST.json).
- **Upstream terms**:  Users generating their own datasets with this pipeline
  must independently verify the current licence terms of each component at the
  time of generation, as upstream licences may change.  In particular,
  Juggernaut-XL-v9 and SDXL/ControlNet output-use terms apply to any images
  generated with those models, regardless of whether the weights are
  distributed.

---

*This file is part of the SFHQ-VirtualID dataset release and should be
preserved with all copies of the dataset.*

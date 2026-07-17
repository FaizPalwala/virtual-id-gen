# Phase 1: SFHQ Dataset Acquisition, Preprocessing & Virtual Identity Generation

Part of the **Iterative Machine Unlearning for Personal Data Deletion** project.

This package replaces the legacy K-Means identity clustering approach with an explicit **DCFace identity-conditioned generation** pipeline. The final outputs remain strictly Phase-2 compatible, converting raw synthetic faces into a structured facial identity dataset ready for model unlearning experiments.

## Architecture Overview

```text
Raw SFHQ images (Kaggle)
        ↓  Step 1: Download
        ↓  Step 2: Preprocess (MTCNN detection, alignment, quality filter)
        ↓  Step 3: DCFace Virtual Identity Synthesis (400 unique identities)
        ↓  Step 4: ArcFace embedding extraction & Proxy label generation
        ↓  Step 5: Dataset construction (forget/retain/test splits)
        ↓
data/dataset/sfhq_dataset.csv  ←  training input
```

## Output Dataset Structure

`data/dataset/sfhq_dataset.csv` has 5 columns:

| Column        | Description                                              |
|---------------|----------------------------------------------------------|
| `image_path`  | Path to preprocessed 128×128 face crop                  |
| `cluster_id`  | Synthetic identity ID (0–199)                            |
| `age_group`   | Proxy label: 0=Young, 1=Adult, 2=Middle-Aged, 3=Senior  |
| `split`       | `retain` / `forget` / `test`                            |
| `forget_step` | 0–19 (which unlearning iteration deletes this identity); -1 for non-forget |

---

## Dataset Design

```
400 synthetic identities (~40 images each)
├── Retain  (300 identities, ~12,000 imgs) — training set
├── Test    ( 60 identities, ~2,400  imgs) — held-out evaluation
└── Forget  ( 40 identities, ~1,600  imgs) — sequential deletion
                                             (2 identity per unlearning step)
```



---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Kaggle API credentials
- Go to https://www.kaggle.com/account → Create New Token
- Save `kaggle.json` to `~/.kaggle/kaggle.json`


### Full Pipeline Run

To run the entire pipeline sequentially (ensure you are on a node with GPU access):

```bash
cd src
python main.py pipeline.dcface_root=/absolute/path/to/dcface

```

### Modular Execution (HPC / Job Scheduling)

You can submit discrete jobs by passing the `--config-name` argument to Hydra. This allows you to allocate hardware resources efficiently (e.g., running data prep on cheaper CPU nodes and synthesis on expensive GPU nodes).

**CPU Node (Download & Preprocess):**

```bash
python main.py --config-name step1_download
python main.py --config-name step2_preprocess

```

**GPU Node (Generation & Extraction):**

```bash
# Ensure you pass the required dcface_root
python main.py --config-name step3_generate pipeline.dcface_root=/absolute/path/to/dcface
python main.py --config-name step4_extract

```

**CPU Node (Final Dataset Build):**

```bash
python main.py --config-name step5_build

```

---

## Design Decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| **Dataset** | SFHQ (AI-generated) | Zero real individuals; bypasses privacy and ethical issues entirely. |
| **Face Detector** | MTCNN (`facenet-pytorch`) | Fast, well-maintained, and yields >98% accuracy on SFHQ images. |
| **Identity Synthesis** | DCFace Adapter | Explicit identity-conditioned generation replaces legacy K-Means clustering, guaranteeing strict, provable virtual identity boundaries. |
| **Validation Metrics** | ArcFace (Similarity) & Laplacian (Blur) | Enforces high generation quality by discarding blurry crops or images that heavily deviate from the seed identity (Similarity < 0.45, Blur < 80). |
| **Proxy Task** | Age Group Classification (4-way) | Mirrors the MUFAC benchmark, creating a realistic privacy-centric evaluation scenario. |
| **Image Size** | 128x128 | GPU-efficient for downstream ResNet-18 training while preserving necessary facial details. |
| **Forget Protocol** | 20 steps (2 identities per step) | Models real-world sequential GDPR/CCPA deletion requests. |

---

## Notes

- **No real faces.** SFHQ is entirely AI-generated via StyleGAN2 and diffusion models.
  No ethical review is required for this dataset.
- **Reproducible.** All random operations use `random_state=42`.
- **Extensible.** Adding SFHQ parts 2–4 requires only changing `--part` in Step 1.
- **Phase 2 input**: `data/dataset/sfhq_dataset.csv` is directly consumed by
  the Phase 2 PyTorch Dataset class.

---

## Citation

If you use SFHQ in your research:
```
SelfishGene (2022). SFHQ Dataset: Synthetic Faces High Quality.
GitHub: https://github.com/SelfishGene/SFHQ-dataset
```

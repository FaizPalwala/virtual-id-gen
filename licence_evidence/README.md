# Licence Evidence — SFHQ-VirtualID

Pinned evidence for the third-party licence position (Strat.md §2).  These
files are the exact captures taken at the verification date; keep them with
the release so the licence position is defensible if upstream pages change.

## Files

| File | Source | What it proves | Captured |
|---|---|---|---|
| `kaggle_sfhq_part1.html` | https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-1 | SFHQ Part 1 Kaggle listing licence = **CC0: Public Domain** (schema.org `license` field, dataset id 2437902, creator David Beniaguev). Raw capture — renders blank in a browser (JS SPA); see `kaggle_sfhq_part1.EXTRACT.md` for the human-readable record | 2026-08-03 |
| `kaggle_sfhq_part1.EXTRACT.md` | derived from the raw capture | Human-readable extract: verbatim JSON-LD licence block, dataset metadata, sha256, verification commands | 2026-08-03 |
| `openrail-m_license.txt` | https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/LICENSE.md | Full CreativeML Open RAIL-M text: "Licensor claims no rights in the Output You generate" + Attachment A restricted uses | 2026-08-03 |

## Licence summary (verified 2026-08-03)

| Component | Licence | Output-distribution for research |
|---|---|---|
| SFHQ (source) | CC0 / Public Domain (Kaggle listing) | ✅ (also synthetic, no consent chain) |
| Juggernaut-XL-v9 (weights) | CreativeML Open RAIL-M | ✅ "no rights in the Output" — restricted uses don't apply |
| InstantID (weights + code) | Apache 2.0 | ✅ |
| ControlNet SDXL (weights) | OpenRAIL++ | ✅ same output clause |
| InsightFace / MTCNN | MIT (code) + model terms | ✅ (not redistributed) |

Exact pinned revisions: `scripts/RELEASE_MANIFEST.json` → `model_revisions`.

## Position

The model stack permits generating and publicly distributing the outputs for
research, subject to the Attachment A restricted-use prohibitions (no illegal
use, no harming individuals, no discrimination, no PII-based harm).  Our own
non-commercial / prohibited-use licence (`LICENSE`) mirrors these constraints
and must ship with the data.

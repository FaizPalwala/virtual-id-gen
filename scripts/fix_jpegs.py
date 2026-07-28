"""Fix pre-saved JPEGs: apply missing padding + BGR conversion.

Existing images were saved as 128×128 RGB (missing 33 % reflective padding
and BGR conversion).  Re-pad and re-save so extract_embeddings can find faces.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def fix_images(imagedir: str) -> None:
    root = Path(imagedir)
    paths = sorted(root.rglob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No JPEG files found under {root}")

    for path in tqdm(paths, desc="Fixing JPEGs"):
        # Read → PIL saved BGR-as-RGB, so cv2 reads BGR with R↔B swapped.
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            continue

        # Undo the PIL swap: recover the original RGB crop
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply 33 % reflective padding (same as preprocess)
        pad = img_rgb.shape[0] // 3
        padded = cv2.copyMakeBorder(img_rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

        # Convert to BGR for cv2.imwrite
        crop_bgr = cv2.cvtColor(padded, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), crop_bgr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagedir", required=True)
    args = parser.parse_args()
    fix_images(args.imagedir)

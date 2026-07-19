"""
common.py
Shared image, quality, and manifest utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_image_paths(directory: str | Path, recursive: bool = True) -> list[Path]:
    """Return deterministic image paths below *directory*."""
    root = Path(directory)
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        path for path in iterator if path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def laplacian_variance(rgb_image: np.ndarray) -> float:
    """Return a simple sharpness score; larger values indicate sharper images."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalised_cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate cosine similarity safely for feature vectors."""
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("Cannot compare zero-norm embeddings.")
    return float(np.dot(first, second) / (first_norm * second_norm))

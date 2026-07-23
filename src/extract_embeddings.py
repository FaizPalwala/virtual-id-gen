"""
extract_embeddings.py
Extract ArcFace features and demographic proxy attributes from final crops.

Only run this module after ``preprocess_identity_candidates`` has created the
final ``identities/images`` directory.  Paths written here are used verbatim by
``builddataset.py``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import get_image_paths

AGE_GROUP_NAMES = {
    0: "Young 0-24",
    1: "Adult 25-44",
    2: "Middle-Aged 45-64",
    3: "Senior 65+",
    -1: "Unknown",
}


def load_arcface_model(ctx_id: int = 0):
    """Create InsightFace detection, recognition, and age/gender models."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError as error:
        raise ImportError(
            "Install insightface and an ONNX Runtime provider."
        ) from error
    app = FaceAnalysis(allowed_modules=["detection", "recognition", "genderage"])
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def get_embedding_and_attributes(app, image_bgr: np.ndarray):
    """Return normalised embedding, estimated age and gender, or Nones."""
    faces = app.get(image_bgr)
    if not faces:
        return None, None, None
    face = max(faces, key=lambda item: float(item.det_score))
    return (
        face.normed_embedding,
        int(getattr(face, "age", -1)),
        int(getattr(face, "gender", -1)),
    )

def get_embedding_and_attributes_robust(app, image_bgr: np.ndarray, ctxid: int):
    """Return largest-face attributes from an unaligned/raw image.

    The standard extractor intentionally uses a 128-pixel detector for aligned
    final crops.  Generation seeds are full-resolution portraits, so this
    function retries larger detectors and reflected context before declaring a
    seed unusable.  It does not mutate InsightFace face objects.
    """
    if image_bgr is None:
        return None, None, None
    height, width = image_bgr.shape[:2]
    for ratio in (0.0, 0.25, 0.50):
        pad_y, pad_x = int(height * ratio), int(width * ratio)
        candidate = image_bgr if ratio == 0.0 else cv2.copyMakeBorder(
            image_bgr, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT_101
        )
        for det_size in ((640, 640), (512, 512), (320, 320)):
            app.prepare(ctx_id=ctxid, det_size=det_size)
            faces = app.get(candidate)
            if not faces:
                continue
            face = max(faces, key=lambda item: float(
                (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
            ))
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            return embedding, int(getattr(face, "age", -1)), int(getattr(face, "gender", -1))
    return None, None, None

def age_to_group(age: int) -> int:
    """Map InsightFace's age estimate to the stable four-class proxy label."""
    if age < 0:
        return -1
    if age < 25:
        return 0
    if age < 45:
        return 1
    if age < 65:
        return 2
    return 3


def extract_embeddings(inputdir: str, outputdir: str, ctxid: int = 0) -> str:
    """Extract arrays for all final crops, retaining only detectable faces."""
    input_path, output_path = Path(inputdir), Path(outputdir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_paths = get_image_paths(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No final images found under {input_path}")
    app = load_arcface_model(ctxid)
    embeddings, paths, ages, genders, groups = [], [], [], [], []
    for image_path in tqdm(image_paths, desc="Extracting ArcFace features"):
        image = cv2.imread(str(image_path))
        embedding, age, gender = (
            get_embedding_and_attributes(app, image)
            if image is not None
            else (None, None, None)
        )
        if embedding is None:
            continue
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        paths.append(str(image_path))
        ages.append(age)
        genders.append(gender)
        groups.append(age_to_group(age))
    if not embeddings:
        raise RuntimeError("No final images contained a detectable face.")
    np.save(output_path / "embeddings.npy", np.stack(embeddings))
    np.save(output_path / "imagepaths.npy", np.asarray(paths))
    np.save(output_path / "ages.npy", np.asarray(ages, dtype=np.int16))
    np.save(output_path / "genders.npy", np.asarray(genders, dtype=np.int8))
    np.save(output_path / "agegroups.npy", np.asarray(groups, dtype=np.int8))
    summary = {
        "total_input": len(image_paths),
        "total_extracted": len(paths),
        "failed": len(image_paths) - len(paths),
        "embedding_dim": 512,
        "age_group_distribution": dict(Counter(groups)),
    }
    (output_path / "embeddingsummary.json").write_text(json.dumps(summary, indent=2))
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputdir", required=True)
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--ctxid", type=int, default=0)
    arguments = parser.parse_args()
    extract_embeddings(arguments.inputdir, arguments.outputdir, arguments.ctxid)

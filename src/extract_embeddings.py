"""
extract_embeddings.py
Extract ArcFace features and demographic proxy attributes from final crops.

Only run this module after ``preprocess_identity_candidates`` has created the
final ``identities/images`` directory.  Paths written here are used verbatim by
``builddataset.py``.

Example:
    python extract_embeddings.py \\
        --inputdir ../data/processed/images \\
        --outputdir ../data/embeddings \\
        --ctxid 0
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import get_image_paths


def load_arcface_model(ctx_id: int = 0):
    """Create InsightFace detection, recognition, and age/gender models."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError as error:
        raise ImportError(
            "Install insightface and an ONNX Runtime provider."
        ) from error
    app = FaceAnalysis(
        allowed_modules=["detection", "recognition", "genderage"],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
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


def get_embedding_cpu(app, crop_bgr: np.ndarray):
    """Crops are already MTCNN-aligned — use standard 5‑point landmarks.

    The InsightFace detector fails on tight aligned crops (face fills the
    frame), so we bypass detection entirely.  Recognition and genderage both
    use the same standard landmarks via a real ``Face`` object that supports
    both attribute access (``face.kps`` for recognition) and dict-style
    writes (``face['gender']`` for genderage).

    Returns real demographics — not hardcoded constants.
    """
    from insightface.app.common import Face

    h, w = crop_bgr.shape[:2]
    kps_112 = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.6963],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.3655],
        ],
        dtype=np.float32,
    )
    kps = kps_112 * (w / 112.0)
    face = Face(bbox=np.array([0, 0, w, h], dtype=np.float32), kps=kps)

    embedding = app.models["recognition"].get(crop_bgr, face)
    if embedding is None:
        return None, None, None
    app.models["genderage"].get(crop_bgr, face)  # sets face.age, face.gender
    return (
        np.asarray(embedding, dtype=np.float32),
        int(getattr(face, "age", 25)),
        int(getattr(face, "gender", 0)),
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


def extract_embeddings(
    inputdir: str,
    outputdir: str,
    ctxid: int = 0,
    candidate_manifest: str | None = None,
) -> str:
    """Extract arrays for all images, retaining only detectable faces.

    When *candidate_manifest* is provided (raw_candidate_manifest.csv from the
    merge step), identity IDs are joined through the manifest and saved as
    ``identity_ids.npy``.  This decouples embedding storage from image paths,
    allowing the build step to join through identity rather than file name.
    """
    input_path, output_path = Path(inputdir), Path(outputdir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_paths = get_image_paths(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No final images found under {input_path}")

    # Build identity lookup from the candidate manifest when available.
    # The manifest lives next to the candidates dir by pipeline convention
    # (dataroot/identities/raw_candidate_manifest.csv); infer it rather than
    # require a config override.
    identity_map: dict[str, int] = {}
    if candidate_manifest is None:
        inferred = input_path.parent / "raw_candidate_manifest.csv"
        if inferred.is_file():
            candidate_manifest = str(inferred)
    if candidate_manifest:
        manifest = pd.read_csv(candidate_manifest)
        # Identity column in the raw manifest (pre-sweep: identity_id).
        id_col = "identity_id" if "identity_id" in manifest.columns else "clusterid"
        for _, row in manifest.iterrows():
            # Normalize: merge_shards may write absolute paths (HPC scratch
            # mount), but the lookup below produces paths relative to
            # input_path.parent (identities/).  Derive the relative form so
            # the key space matches regardless of how the manifest was written.
            raw = Path(row["raw_candidatepath"])
            try:
                lookup_key = str(raw.relative_to(input_path.parent))
            except ValueError:
                lookup_key = str(raw)
            identity_map[lookup_key] = int(row[id_col])

    app = load_arcface_model(ctxid)
    embeddings, paths, ages, genders, groups, identity_ids = [], [], [], [], [], []
    for image_path in tqdm(image_paths, desc="Extracting ArcFace features"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        embedding, age, gender = get_embedding_and_attributes(app, image)
        if embedding is None:
            embedding, age, gender = get_embedding_cpu(app, image)
        if embedding is None:
            continue
        relative = str(image_path.relative_to(input_path.parent))
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        paths.append(relative)
        ages.append(age)
        genders.append(gender)
        groups.append(age_to_group(age))
        if identity_map:
            identity_ids.append(identity_map.get(relative, -1))
    if not embeddings:
        raise RuntimeError("No images contained a detectable face.")
    np.save(output_path / "embeddings.npy", np.stack(embeddings))
    np.save(output_path / "imagepaths.npy", np.asarray(paths))
    np.save(output_path / "ages.npy", np.asarray(ages, dtype=np.int16))
    np.save(output_path / "genders.npy", np.asarray(genders, dtype=np.int8))
    np.save(output_path / "agegroups.npy", np.asarray(groups, dtype=np.int8))
    if identity_map:
        np.save(output_path / "identity_ids.npy", np.asarray(identity_ids, dtype=np.int16))
    summary = {
        "total_input": len(image_paths),
        "total_extracted": len(paths),
        "failed": len(image_paths) - len(paths),
        "embedding_dim": 512,
        "age_group_distribution": dict(Counter(groups)),
    }
    if identity_map:
        summary["candidate_manifest"] = candidate_manifest
        summary["matched_identity_ids"] = len([i for i in identity_ids if i >= 0])
    (output_path / "embeddingsummary.json").write_text(json.dumps(summary, indent=2))
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputdir", required=True)
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--ctxid", type=int, default=0)
    arguments = parser.parse_args()
    extract_embeddings(arguments.inputdir, arguments.outputdir, arguments.ctxid)

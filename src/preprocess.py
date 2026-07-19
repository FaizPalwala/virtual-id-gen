"""Align, quality-filter, and select final generated identity images.

``preprocess_identity_candidates`` consumes the raw candidate manifest produced
by ``generateidentities.py``.  It writes all downstream artifacts beneath ``processeddir``: final
``identitymanifest.csv`` and cluster-preserving image folders. This isolates
dataset construction from the candidate-generation ``identities`` directory.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import laplacian_variance, normalised_cosine_similarity
from extract_embeddings import get_embedding_and_attributes, load_arcface_model


def _make_mtcnn(imgsize: int, confthreshold: float, device: str):
    """Build MTCNN lazily so modules can be imported without torch installed."""
    import torch
    from facenet_pytorch import MTCNN

    resolved = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    return MTCNN(
        image_size=imgsize,
        margin=20,
        keep_all=False,
        min_face_size=40,
        thresholds=[0.6, 0.7, confthreshold],
        device=resolved,
    )


def preprocess_identity_candidates(
    identitydir: str,
    processeddir: str,
    imagesperidentity: int,
    imgsize: int = 128,
    blurthreshold: float = 80.0,
    confthreshold: float = 0.85,
    min_similarity_final: float = 0.45,
    ctxid: int = 0,
    device: str = "auto",
) -> str:
    """Create exactly ``imagesperidentity`` aligned final samples per cluster.

    Candidates are ranked by final-crop ArcFace similarity, detection confidence,
    and sharpness after thresholding.  The process aborts if any cluster cannot
    satisfy the requested cardinality, preventing uneven identity clusters.
    """
    root = Path(identitydir)
    processed_root = Path(processeddir)
    manifest_path = root / "raw_candidate_manifest.csv"
    candidates = pd.read_csv(manifest_path)
    candidates = candidates[candidates.raw_status == "accepted_raw"].copy()
    output_root = processed_root / "images"
    rejected_root = processed_root / "rejected"
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    rejected_root.mkdir(parents=True, exist_ok=True)
    mtcnn = _make_mtcnn(imgsize, confthreshold, device)
    app = load_arcface_model(ctxid)
    seed_embeddings = {}
    accepted = []
    rejected = []
    for row in tqdm(
        candidates.itertuples(index=False),
        total=len(candidates),
        desc="Preprocessing candidates",
    ):
        source = Path(row.raw_candidatepath)
        reason = None
        confidence = None
        sharpness = None
        similarity = None
        try:
            image = Image.open(source).convert("RGB")
            face_tensor, probabilities = mtcnn(image, return_prob=True)
            if face_tensor is None or probabilities is None:
                reason = "no_face_after_generation"
            else:
                confidence = float(
                    probabilities if np.isscalar(probabilities) else probabilities[0]
                )
                if confidence < confthreshold:
                    reason = "low_detection_confidence"
            if reason is None:
                crop = (
                    ((face_tensor.permute(1, 2, 0).numpy() + 1.0) * 127.5)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                sharpness = laplacian_variance(crop)
                if sharpness < blurthreshold:
                    reason = "blur"
            if reason is None:
                seed_path = str(row.seedpath)
                if seed_path not in seed_embeddings:
                    seed, _, _ = get_embedding_and_attributes(
                        app, cv2.imread(seed_path)
                    )
                    if seed is None:
                        raise RuntimeError(f"No detectable seed face: {seed_path}")
                    seed_embeddings[seed_path] = seed
                final_embedding, _, _ = get_embedding_and_attributes(
                    app, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                )
                if final_embedding is None:
                    reason = "no_face_in_final_crop"
                else:
                    similarity = normalised_cosine_similarity(
                        seed_embeddings[seed_path], final_embedding
                    )
                    if similarity < min_similarity_final:
                        reason = "low_final_similarity"
            record = row._asdict() | {
                "detection_confidence": confidence,
                "laplacian_variance": sharpness,
                "arcface_similarity": similarity,
                "rejection_reason": reason,
            }
            if reason is None:
                record["_crop"] = crop
                accepted.append(record)
            else:
                rejected.append(record)
        except Exception as error:
            rejected.append(
                row._asdict()
                | {
                    "detection_confidence": confidence,
                    "laplacian_variance": sharpness,
                    "arcface_similarity": similarity,
                    "rejection_reason": f"processing_error:{type(error).__name__}",
                }
            )
    final_rows = []
    for cluster_id, group in pd.DataFrame(accepted).groupby("clusterid", sort=True):
        ranked = group.sort_values(
            ["arcface_similarity", "detection_confidence", "laplacian_variance"],
            ascending=False,
        )
        if len(ranked) < imagesperidentity:
            reasons = Counter(
                item["rejection_reason"]
                for item in rejected
                if item["clusterid"] == cluster_id
            )
            raise RuntimeError(
                f"Identity {cluster_id} has {len(ranked)}/{imagesperidentity} final samples; rejections={dict(reasons)}"
            )
        destination = output_root / f"identity_{int(cluster_id):03d}"
        destination.mkdir()
        for index, record in enumerate(
            ranked.head(imagesperidentity).to_dict("records")
        ):
            final_path = destination / f"accepted_{index:03d}.jpg"
            Image.fromarray(record.pop("_crop")).save(final_path, quality=95)
            record["imagepath"] = str(final_path)
            final_rows.append(record)
    final = pd.DataFrame(final_rows).sort_values(["clusterid", "imagepath"])
    final.to_csv(processed_root / "identitymanifest.csv", index=False)
    pd.DataFrame(rejected).drop(columns=["_crop"], errors="ignore").to_csv(
        processed_root / "preprocessing_rejection_manifest.csv", index=False
    )
    (processed_root / "preprocessing_metadata.json").write_text(
        json.dumps(
            {
                "images_per_identity": imagesperidentity,
                "final_images": len(final),
                "rejected": len(rejected),
                "imgsize": imgsize,
                "min_similarity_final": min_similarity_final,
                "candidate_manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return str(processed_root / "identitymanifest.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identitydir", required=True)
    parser.add_argument("--processeddir", required=True)
    parser.add_argument("--imagesperidentity", type=int, required=True)
    parser.add_argument("--imgsize", type=int, default=128)
    parser.add_argument("--blurthreshold", type=float, default=80.0)
    parser.add_argument("--confthreshold", type=float, default=0.85)
    parser.add_argument("--minsimilarityfinal", type=float, default=0.45)
    parser.add_argument("--ctxid", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    preprocess_identity_candidates(
        args.identitydir,
        args.processeddir,
        args.imagesperidentity,
        args.imgsize,
        args.blurthreshold,
        args.confthreshold,
        args.minsimilarityfinal,
        args.ctxid,
        args.device,
    )

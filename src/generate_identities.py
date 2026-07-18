"""
generate_identities.py

Selects SFHQ seed faces, calls an external DCFace adapter to generate synthetic
identities, and validates outputs using ArcFace similarity and blur thresholds.
"""
from __future__ import annotations

from collections import Counter
import json
import logging
import random
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess import laplacian_variance
from extract_embeddings import load_arcface_model, get_embedding_and_attributes

logger = logging.getLogger(__name__)
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

def get_image_paths(directory: str | Path) -> List[Path]:
    """Recursively retrieves all valid image paths from a directory."""
    return sorted(p for p in Path(directory).rglob('*') if p.suffix.lower() in VALID_EXTENSIONS)

def invoke_generator(command_template: str, seed_path: Path, style_path: Path, output_path: Path, generation_seed: int) -> None:
    """
    Invokes the external generator command to create a synthetic image.
    
    Args:
        command_template (str): The CLI command template.
        seed_path (Path): Path to the identity seed image.
        style_path (Path): Path to the style image.
        output_path (Path): Path where the generated image should be saved.
        generation_seed (int): The seed for random number generation.
    """
    command = command_template.format(
        id_image=seed_path.resolve(),
        style_image=style_path.resolve(),
        output_path=output_path.resolve(),
        generation_seed=generation_seed,
    )

    result = subprocess.run(shlex.split(command), capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Generator error: %s", result.stderr[-2000:])
        raise RuntimeError("External generator failed")
    if not output_path.exists():
        raise RuntimeError(f"Generator did not create expected output at {output_path}")

def _validate(app, bgr: np.ndarray, seed_embedding: np.ndarray,
              min_similarity: float, blur_threshold: float) -> tuple[bool, dict]:
    """Return validation result and always return all diagnostics."""
    embedding, age, gender = get_embedding_and_attributes(app, bgr)
    blur = float(laplacian_variance(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    similarity = None
    if embedding is not None:
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            similarity = float(np.dot(seed_embedding, np.asarray(embedding, np.float32) / norm))

    reasons = []
    if embedding is None:
        reasons.append("no_face")
    elif similarity is None:
        reasons.append("invalid_embedding")
    elif similarity < min_similarity:
        reasons.append("low_similarity")
    if blur < blur_threshold:
        reasons.append("blur")

    return not reasons, {
        "arcface_similarity": similarity,
        "laplacian_variance": blur,
        "detected_age": None if age is None else int(age),
        "detected_gender": None if gender is None else int(gender),
        "rejection_reason": "+".join(reasons) if reasons else "accepted",
    }

def generate_identities(
    processed_dir: str,
    output_dir: str,
    n_identities: int = 400,
    images_per_identity: int = 40,
    candidates_per_identity: int = 70,
    generator_cmd: str = "",
    ctx_id: int = 0,
    random_state: int = 42,
    min_similarity: float = 0.45,
    blur_threshold: float = 80.0,
    visualise: bool = True,
    keep_rejected: bool = True,
) -> str:
    """
    Generates verified synthetic identities using an external adapter.
    """
    required = {"{id_image}", "{style_image}", "{output_path}"}
    present = set(re.findall(r"\{[^}]+\}", generator_cmd))
    if not required.issubset(present):
        raise ValueError(f"generator_cmd must contain {required}")
    if candidates_per_identity < images_per_identity:
        raise ValueError("candidates_per_identity must be at least images_per_identity")

    source_images = get_image_paths(processed_dir)
    if len(source_images) < 2 * n_identities:
        raise ValueError(f"Need {2 * n_identities} source crops; found {len(source_images)}")

    rng = random.Random(random_state)
    rng.shuffle(source_images)
    seeds, styles = source_images[:n_identities], source_images[n_identities:2 * n_identities]

    out = Path(output_dir)
    images_root, candidates_root, rejected_root = out / "images", out / "_candidates", out / "rejected"
    for directory in (images_root, candidates_root, rejected_root):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    logger.info("Loading ArcFace model for seed embedding extraction...")
    app = load_arcface_model(ctx_id)
    seed_embeddings = []
    for seed in tqdm(seeds, desc="Embedding seeds"):
        embedding, _, _ = get_embedding_and_attributes(app, cv2.imread(str(seed)))
        if embedding is None or np.linalg.norm(embedding) == 0:
            raise RuntimeError(f"Undetectable face in seed: {seed}")
        seed_embeddings.append(np.asarray(embedding, np.float32) / np.linalg.norm(embedding))

    manifest_rows, rejection_rows = [], []
    logger.info("Generating %d identities...", n_identities)
    for ident_idx, (seed_path, seed_emb) in enumerate(tqdm(zip(seeds, seed_embeddings), total=n_identities, desc="Generating")):
        destination = images_root / f"identity_{ident_idx:03d}"
        destination.mkdir()
        accepted = 0

        for trial in range(candidates_per_identity):
            if accepted >= images_per_identity:
                break
            style_path = styles[(ident_idx * candidates_per_identity + trial) % len(styles)]
            candidate = candidates_root / f"{ident_idx:03d}_{trial:03d}.png"
            generation_seed = random_state + ident_idx * candidates_per_identity + trial
            invoke_generator(generator_cmd, seed_path, style_path, candidate, generation_seed)

            bgr = cv2.imread(str(candidate))
            if bgr is None:
                metrics = {"arcface_similarity": None, "laplacian_variance": None,
                           "detected_age": None, "detected_gender": None,
                           "rejection_reason": "unreadable_image"}
                valid = False
            else:
                valid, metrics = _validate(app, bgr, seed_emb, min_similarity, blur_threshold)

            record = {"identity_id": ident_idx, "trial": trial, "seed_path": str(seed_path),
                      "style_path": str(style_path), "generation_seed": generation_seed,
                      "candidate_path": str(candidate), **metrics}
            if not valid:
                if keep_rejected and candidate.exists():
                    rejected = rejected_root / f"identity_{ident_idx:03d}_trial_{trial:03d}.png"
                    shutil.move(candidate, rejected)
                    record["candidate_path"] = str(rejected)
                else:
                    candidate.unlink(missing_ok=True)
                rejection_rows.append(record)
                logger.info("Rejected id=%03d trial=%03d: %s; similarity=%s; blur=%s",
                            ident_idx, trial, metrics["rejection_reason"],
                            metrics["arcface_similarity"], metrics["laplacian_variance"])
                continue

            final_path = destination / f"{accepted:03d}.png"
            shutil.move(candidate, final_path)
            manifest_rows.append({**record, "image_path": str(final_path), "cluster_id": ident_idx})
            accepted += 1

        if accepted < images_per_identity:
            pd.DataFrame(rejection_rows).to_csv(out / "rejection_manifest.csv", index=False)
            counts = Counter(row["rejection_reason"] for row in rejection_rows if row["identity_id"] == ident_idx)
            raise RuntimeError(
                f"Identity {ident_idx} accepted {accepted}/{images_per_identity}. "
                f"Rejection counts: {dict(counts)}. Inspect {rejected_root} and rejection_manifest.csv "
                "before changing thresholds or candidate count."
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out / "identity_manifest.csv", index=False)
    pd.DataFrame(rejection_rows).to_csv(out / "rejection_manifest.csv", index=False)
    summary = {"n_identities": n_identities, "images_per_identity": images_per_identity,
               "accepted_images": len(manifest), "rejected_candidates": len(rejection_rows),
               "min_identity_similarity": min_similarity, "blur_threshold": blur_threshold,
               "random_state": random_state}
    (out / "identity_summary.json").write_text(json.dumps(summary, indent=2))
    if visualise:
        _generate_visualisations(manifest, out, n_identities, rng)
    return str(out)


def _generate_visualisations(df: pd.DataFrame, out_path: Path, n_identities: int, rng: random.Random) -> None:
    """Generates 3x3 grids for a random subset of identities."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    vis_dir = out_path / 'identity_samples'
    vis_dir.mkdir(exist_ok=True)
    
    sample_ids = rng.sample(range(n_identities), min(6, n_identities))
    
    for ident_id in sample_ids:
        paths = df[df.cluster_id == ident_id].image_path.tolist()[:9]
        fig, axes = plt.subplots(3, 3, figsize=(6, 6))
        
        for ax, img_path in zip(axes.flat, paths):
            ax.imshow(Image.open(img_path))
            ax.axis('off')
            
        for ax in axes.flat[len(paths):]:
            ax.axis('off')
            
        fig.suptitle(f'Virtual Identity {ident_id:03d}')
        fig.savefig(vis_dir / f'identity_{ident_id:03d}.png')
        plt.close(fig)
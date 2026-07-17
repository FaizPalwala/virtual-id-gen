"""
(Legacy File)
Step 3: ArcFace Embedding Extraction
- Loads processed 128x128 face crops
- Extracts 512-dim ArcFace embeddings using insightface
- Also extracts age/gender estimates for proxy label construction
- Saves embeddings and metadata to disk

Usage:
    python extract_embeddings.py [--input_dir ../data/processed]
                                 [--output_dir ../data/embeddings]
"""

import argparse
import gc
import math
import json
import subprocess
import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm


def load_arcface_model(ctx_id: int = 0):
    """Load insightface FaceAnalysis model (detection + recognition + age/gender)."""
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError(
            "insightface not installed.\n"
            "Install with: pip install insightface onnxruntime-gpu\n"
            "  (or onnxruntime for CPU-only)"
        )

    app = FaceAnalysis(allowed_modules=["detection", "recognition", "genderage"])
    app.prepare(ctx_id=ctx_id, det_size=(128, 128))
    return app


def get_embedding_and_attributes(app, img_bgr: np.ndarray):
    """
    Extract ArcFace embedding + age + gender from a face image.
    Returns (embedding, age, gender) or (None, None, None) if no face found.
    Embedding is L2-normalised 512-dim vector.
    """
    faces = app.get(img_bgr)
    if not faces:
        return None, None, None

    face = faces[0]
    emb = face.normed_embedding  # shape: (512,), L2-normalised
    age = int(getattr(face, "age", -1))
    gender = int(getattr(face, "gender", -1))  # 0=female, 1=male
    return emb, age, gender


def age_to_group(age: int) -> int:
    """Map age estimate to 4-way classification label."""
    if age < 0:   return -1   # unknown
    if age < 25:  return 0    # Young (0-24)
    if age < 45:  return 1    # Adult (25-44)
    if age < 65:  return 2    # Middle-Aged (45-64)
    return 3                  # Senior (65+)


AGE_GROUP_NAMES = {0: "Young (0-24)", 1: "Adult (25-44)",
                   2: "Middle-Aged (45-64)", 3: "Senior (65+)", -1: "Unknown"}


def extract_embeddings(
    input_dir: str,
    output_dir: str,
    ctx_id: int = 0,
    chunk_size: int = 10000,
    batch_log_every: int = 500
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Gather image paths
    img_paths = sorted(
        list(input_path.glob("*.jpg")) +
        list(input_path.glob("*.png"))
    )
    # Exclude metadata files
    img_paths = [p for p in img_paths if "metadata" not in p.name]
    print(f"[INFO] Found {len(img_paths)} processed face images")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    print(f"[INFO] Processing in chunks of {chunk_size}")

    total_chunks = math.ceil(len(img_paths) / chunk_size) if img_paths else 0
    for chunk_idx, start in enumerate(range(0, len(img_paths), chunk_size), start=1):
        end = min(start + chunk_size, len(img_paths))
        print(f"[INFO] Starting chunk {chunk_idx}/{total_chunks} ({end - start} images)")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--input_dir", str(input_path),
            "--output_dir", str(output_path),
            "--ctx_id", str(ctx_id),
            "--chunk_size", str(chunk_size),
            "--chunk_start", str(start),
            "--chunk_end", str(end),
            "--batch_log_every", str(batch_log_every),
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Chunk {chunk_idx}/{total_chunks} failed with exit code {result.returncode}. "
                "The extractor now runs each chunk in a separate process to avoid native memory growth."
            )

    chunk_dir = output_path / "chunks"
    chunk_files = sorted(chunk_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files were produced in {chunk_dir}")

    embeddings_parts = []
    image_path_parts = []
    age_parts = []
    gender_parts = []
    age_group_parts = []

    for chunk_file in chunk_files:
        with np.load(chunk_file) as chunk:
            embeddings_parts.append(chunk["embeddings"])
            image_path_parts.append(chunk["image_paths"])
            age_parts.append(chunk["ages"])
            gender_parts.append(chunk["genders"])
            age_group_parts.append(chunk["age_groups"])

    embeddings_np = np.concatenate(embeddings_parts, axis=0)
    paths_np = np.concatenate(image_path_parts, axis=0)
    ages_np = np.concatenate(age_parts, axis=0)
    genders_np = np.concatenate(gender_parts, axis=0)
    age_groups_np = np.concatenate(age_group_parts, axis=0)

    np.save(output_path / "embeddings.npy", embeddings_np)
    np.save(output_path / "image_paths.npy", paths_np)
    np.save(output_path / "ages.npy", ages_np)
    np.save(output_path / "genders.npy", genders_np)
    np.save(output_path / "age_groups.npy", age_groups_np)

    age_group_counts = Counter(age_groups_np.tolist())
    summary = {
        "total_input": int(len(img_paths)),
        "total_extracted": int(embeddings_np.shape[0]),
        "failed": int(len(img_paths) - embeddings_np.shape[0]),
        "embedding_dim": int(embeddings_np.shape[1]) if len(embeddings_np) else 512,
        "chunk_size": int(chunk_size),
        "chunks": int(len(chunk_files)),
        "age_group_distribution": {
            AGE_GROUP_NAMES[k]: v for k, v in sorted(age_group_counts.items())
        },
        "gender_distribution": {
            "female (0)": int((genders_np == 0).sum()),
            "male (1)": int((genders_np == 1).sum())
        }
    }
    with open(output_path / "embedding_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    summary_path = output_path / "embedding_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Expected summary file {summary_path} was not created by chunk runs"
        )

    with open(summary_path, "r") as f:
        summary = json.load(f)

    print("\n[RESULTS]")
    print(f"  Embeddings extracted : {summary.get('total_extracted', 0)}")
    print(f"  Failed               : {summary.get('failed', 0)}")
    print(f"  Chunks saved         : {summary.get('chunks', 0)}")
    print(f"  Chunk size           : {chunk_size}")
    print(f"  Saved to: {output_path}")
    return str(output_path)


def process_chunk(
    input_dir: str,
    output_dir: str,
    ctx_id: int,
    chunk_size: int,
    chunk_start: int,
    chunk_end: int,
    batch_log_every: int,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        list(input_path.glob("*.jpg")) +
        list(input_path.glob("*.png"))
    )
    img_paths = [p for p in img_paths if "metadata" not in p.name]
    chunk_paths = img_paths[chunk_start:chunk_end]
    chunk_idx = chunk_start // chunk_size + 1
    total_chunks = math.ceil(len(img_paths) / chunk_size) if img_paths else 0
    print(f"[INFO] Chunk process {chunk_idx}/{total_chunks}: {len(chunk_paths)} images")

    chunk_dir = output_path / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    app = load_arcface_model(ctx_id=ctx_id)
    print(f"[INFO] ArcFace model loaded for chunk {chunk_idx}/{total_chunks} (ctx_id={ctx_id})")

    embeddings = []
    valid_paths = []
    ages = []
    genders = []
    age_groups = []
    failed = 0

    for local_idx, img_path in enumerate(tqdm(chunk_paths, desc=f"Chunk {chunk_idx}/{total_chunks}")):
        global_idx = chunk_start + local_idx
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            failed += 1
            continue

        emb, age, gender = get_embedding_and_attributes(app, img_bgr)
        del img_bgr
        if emb is None:
            failed += 1
            continue

        embeddings.append(np.asarray(emb, dtype=np.float32))
        valid_paths.append(str(img_path))
        ages.append(age)
        genders.append(gender)
        age_groups.append(age_to_group(age))

        if (global_idx + 1) % batch_log_every == 0:
            print(f"  [{global_idx+1}/{len(img_paths)}] processed, {failed} failed so far")

    chunk_file = chunk_dir / f"chunk_{chunk_idx:05d}.npz"
    np.savez_compressed(
        chunk_file,
        embeddings=np.stack(embeddings, axis=0) if embeddings else np.empty((0, 512), dtype=np.float32),
        image_paths=np.asarray(valid_paths),
        ages=np.asarray(ages, dtype=np.int16),
        genders=np.asarray(genders, dtype=np.int8),
        age_groups=np.asarray(age_groups, dtype=np.int8)
    )

    manifest_entry = {
        "file": chunk_file.name,
        "source_images": len(chunk_paths),
        "saved_embeddings": len(embeddings),
        "failed": failed,
        "chunk_start": int(chunk_start),
        "chunk_end": int(chunk_end)
    }
    manifest_path = output_path / "chunk_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = []
    manifest = [entry for entry in manifest if entry.get("file") != chunk_file.name]
    manifest.append(manifest_entry)
    manifest.sort(key=lambda item: item.get("chunk_start", 0))
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    summary_path = output_path / "embedding_summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        summary = {
            "total_input": 0,
            "total_extracted": 0,
            "failed": 0,
            "embedding_dim": 512,
            "chunk_size": int(chunk_size),
            "chunks": 0,
            "age_group_distribution": {label: 0 for label in AGE_GROUP_NAMES.values()},
            "gender_distribution": {"female (0)": 0, "male (1)": 0},
            "chunks_manifest": []
        }

    summary["total_input"] = max(summary.get("total_input", 0), len(img_paths))
    summary["total_extracted"] = int(summary.get("total_extracted", 0) + len(embeddings))
    summary["failed"] = int(summary.get("failed", 0) + failed)
    summary["embedding_dim"] = 512
    summary["chunk_size"] = int(chunk_size)
    summary["chunks"] = int(len(manifest))

    age_group_distribution = summary.get("age_group_distribution", {})
    for label in AGE_GROUP_NAMES.values():
        age_group_distribution.setdefault(label, 0)
    for age_group in age_groups:
        age_group_distribution[AGE_GROUP_NAMES[age_group]] += 1
    summary["age_group_distribution"] = age_group_distribution

    gender_distribution = summary.get("gender_distribution", {"female (0)": 0, "male (1)": 0})
    gender_distribution["female (0)"] = int(gender_distribution.get("female (0)", 0) + sum(1 for g in genders if g == 0))
    gender_distribution["male (1)"] = int(gender_distribution.get("male (1)", 0) + sum(1 for g in genders if g == 1))
    summary["gender_distribution"] = gender_distribution
    summary["chunks_manifest"] = manifest

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Saved chunk {chunk_idx}/{total_chunks}: {len(embeddings)} embeddings, {failed} failed")
    del app, embeddings, valid_paths, ages, genders, age_groups
    gc.collect()
    return str(chunk_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ArcFace embeddings from face images")
    parser.add_argument("--input_dir", type=str, default="../data/processed")
    parser.add_argument("--output_dir", type=str, default="../data/embeddings")
    parser.add_argument("--ctx_id", type=int, default=0,
                        help="GPU context ID (-1 for CPU)")
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="Number of images to process per chunk")
    parser.add_argument("--chunk_start", type=int, default=None,
                        help="Start index for a single chunk run")
    parser.add_argument("--chunk_end", type=int, default=None,
                        help="End index for a single chunk run")
    parser.add_argument("--batch_log_every", type=int, default=10000,
                        help="Progress log interval within a chunk")
    args = parser.parse_args()

    if args.chunk_start is not None or args.chunk_end is not None:
        if args.chunk_start is None or args.chunk_end is None:
            raise ValueError("Both --chunk_start and --chunk_end must be provided together")
        process_chunk(
            args.input_dir,
            args.output_dir,
            args.ctx_id,
            args.chunk_size,
            args.chunk_start,
            args.chunk_end,
            args.batch_log_every,
        )
    else:
        extract_embeddings(args.input_dir, args.output_dir, args.ctx_id, args.chunk_size, args.batch_log_every)

"""
Step 1: Diverse SFHQ Seed Extraction via Precomputed CLIP Features

Downloads the SFHQ Part 1 dataset from Kaggle (one-time, ~15 min), then
loads precomputed CLIP ViT-L/14 embeddings from the included pickle files
to bypass local inference.  Clusters embeddings with KMeans and saves the
most representative image from each cluster as a seed.

Requires: kaggle CLI credentials, scikit-learn, numpy.

Example:
    python download.py --num_images 400 --output_dir ../data/seeds
"""
from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm

# Which CLIP variant to use from the precomputed pickle files.
# Options: CLIP_ViTL_14@336 (768d), CLIP_ResNet50x64 (1024d),
#          CLIP_ViTL_14 (768d), ConvNext_XL_Imagenet21k (2048d)
FEATURE_KEY = "CLIP_ViTL_14"


def download_sfhq(
    part: int = 1,
    output_dir: str = "../data/seeds",
    num_images: int = 400,
    skip_download: bool = False,
) -> str:
    """Extract *num_images* diverse seeds from SFHQ Part *part*.

    Downloads the full dataset on first run (or when *skip_download* is
    False).  On subsequent runs, pass ``skip_download=True`` to reuse the
    already-downloaded data.  The full dataset directory is cleaned up
    after seed extraction.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_dir = output_path / f"sfhq_part_{part}_full"
    dataset_name = f"selfishgene/synthetic-faces-high-quality-sfhq-part-{part}"

    # ── 1. Download (unless skip_download and data exists) ──
    if skip_download and dataset_dir.exists():
        print(f"[INFO] Skipping download — {dataset_dir} already exists.")
    else:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise ImportError(
                "Install the Kaggle CLI: pip install kaggle, then configure "
                "~/.kaggle/kaggle.json credentials."
            )

        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Authenticating Kaggle API for {dataset_name} ...")
        api = KaggleApi()
        api.authenticate()

        print(f"[INFO] Downloading and unzipping to {dataset_dir} ...")
        api.dataset_download_files(
            dataset_name, path=str(dataset_dir), unzip=True, quiet=False,
        )

    # ── 2. Locate images ──
    images_dir = dataset_dir / "images" / "images"
    if not images_dir.exists():
        images_dir = dataset_dir / "images"
        if not images_dir.exists():
            images_dir = dataset_dir

    print(f"[INFO] Indexing images in {images_dir} ...")
    image_paths = sorted(list(images_dir.glob("*.jpg")))
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}.")
    actual_pool_size = len(image_paths)
    print(f"[INFO] Successfully loaded ALL {actual_pool_size} images.")

    if actual_pool_size < num_images:
        print(f"\n[WARN] Only {actual_pool_size} images; reducing target seeds.")
        num_images = actual_pool_size

    # ── 3. Load precomputed CLIP features from pickle files ──
    features_dir = dataset_dir / "pretrained_features" / "pretrained_features"
    if not features_dir.exists():
        raise RuntimeError(
            f"Precomputed features not found at {features_dir}. "
            "Download may be incomplete."
        )

    print(f"\n[INFO] Loading precomputed CLIP embeddings ({FEATURE_KEY}) ...")
    pickle_files = sorted(features_dir.glob("*.pickle"))
    if len(pickle_files) != actual_pool_size:
        print(
            f"[WARN] Pickle count ({len(pickle_files)}) != image count "
            f"({actual_pool_size}); some images may be skipped."
        )

    # Map pickle filename → image path for matching
    image_map = {p.stem: p for p in image_paths}

    embeddings: list[np.ndarray] = []
    valid_paths: list[str] = []

    for pkl_path in tqdm(pickle_files, desc="Loading features"):
        stem = pkl_path.stem  # e.g. SFHQ_pt1_00000001
        img_path = image_map.get(stem)
        if img_path is None:
            continue
        try:
            data = pickle.load(open(pkl_path, "rb"))
            vec = np.asarray(data[FEATURE_KEY], dtype=np.float32).ravel()
            # Normalise to unit length for cosine-similarity-based clustering
            vec /= max(np.linalg.norm(vec), 1e-12)
            embeddings.append(vec)
            valid_paths.append(str(img_path))
        except Exception as exc:
            continue

    if len(valid_paths) < num_images:
        print(f"[WARN] Only {len(valid_paths)} valid features; reducing seeds.")
        num_images = len(valid_paths)

    embeddings_arr = np.array(embeddings)
    print(f"[OK] Loaded {embeddings_arr.shape[0]} vectors, dim={embeddings_arr.shape[1]}")

    # ── 4. KMeans clustering → most-representative per cluster ──
    print(f"\n[INFO] Clustering {embeddings_arr.shape[0]} vectors into {num_images} groups ...")
    kmeans = KMeans(n_clusters=num_images, random_state=42, n_init="auto")
    kmeans.fit(embeddings_arr)
    closest, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, embeddings_arr
    )

    # ── 4b. Hash-dedupe the selected representatives ──
    # Two clusters can land on the SAME source image (SFHQ's pool contains
    # byte-identical duplicates).  The v1.0.0 seed pool shipped 2 duplicate
    # pairs (600 seeds); at 750 clusters it produces 5 — those would become
    # near-duplicate identities (the A3 seed-reuse bug).  Dedupe by sha1 and
    # refill each duplicate slot from the next-closest image to that
    # cluster's center, so every seed is a distinct face.
    def file_sha1(path: str) -> str:
        import hashlib
        return hashlib.sha1(open(path, "rb").read()).hexdigest()

    seen_hashes: set[str] = set()
    picks: dict[int, str] = {}  # cluster_id -> src path
    for cluster_id, src_idx in enumerate(closest):
        src = valid_paths[src_idx]
        h = file_sha1(src)
        if h in seen_hashes:
            continue  # refilled below
        seen_hashes.add(h)
        picks[cluster_id] = src

    if len(picks) < num_images:
        print(f"[INFO] {num_images - len(picks)} duplicate picks found — "
              f"refilling from next-closest unique images ...")
        # Cosine similarity to each cluster centre (embeddings normalised).
        for cluster_id in range(num_images):
            if cluster_id in picks:
                continue
            centre = kmeans.cluster_centers_[cluster_id]
            sims = embeddings_arr @ centre
            for order in np.argsort(-sims):
                cand = valid_paths[order]
                h = file_sha1(cand)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    picks[cluster_id] = cand
                    break

    if len(picks) < num_images:
        print(f"[WARN] Only {len(picks)} unique seeds available "
              f"(requested {num_images}); continuing with what we have.")
    else:
        print(f"[OK] {len(picks)} unique (hash-distinct) seeds selected.")

    # ── 5. Save seeds sequentially ──
    print(f"\n[INFO] Saving {len(picks)} diverse seeds to {output_path} ...")
    for cluster_id, src in sorted(picks.items()):
        dest = output_path / f"seed_{cluster_id:04d}.jpg"
        shutil.copy2(src, dest)

    # ── 6. Clean up full dataset ──
    print("[INFO] Cleaning up full dataset download ...")
    shutil.rmtree(dataset_dir, ignore_errors=True)
    # Also remove any stray zip files left by the Kaggle download
    for z in output_path.glob("*.zip"):
        z.unlink()

    print(f"[OK] {num_images} diverse seeds saved to {output_path.resolve()}")
    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download SFHQ and extract diverse seeds."
    )
    parser.add_argument(
        "--num_images", type=int, default=400, help="Number of diverse seeds"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data/seeds", help="Output directory"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse already-downloaded data instead of downloading again",
    )
    args = parser.parse_args()

    download_sfhq(
        num_images=args.num_images,
        output_dir=args.output_dir,
        skip_download=args.skip_download,
    )

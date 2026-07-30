"""
Step 1: Diverse SFHQ Seed Extraction via CLIP + KMeans (Bulk Download)

Downloads the ENTIRE SFHQ Part 1 database locally as a single bulk operation.
Since we now have the full dataset locally, we target the `images/images`
directory directly for our seed pool. We also attempt to load precomputed
embeddings provided with the dataset to completely bypass the slow local 
CLIP inference step. If precomputed embeddings aren't found or don't match,
it gracefully falls back to local PyTorch inference.

Requires: kaggle CLI credentials, torch, transformers, scikit-learn.

Example:
    python download.py --num_images 400 --output_dir ../data/seeds
"""
from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm


@contextmanager
def suppress_stdout():
    """Context manager to completely silence noisy API prints."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def download_sfhq(
    part: int = 1,
    output_dir: str = "../data/seeds",
    num_images: int = 400,
) -> str:
    """Download *num_images* diverse seed images from SFHQ Part *part*.

    Downloads the entire database as the evaluation pool, attempts to load
    precomputed embeddings (falling back to OpenAI CLIP if missing), clusters
    into *num_images* groups, and keeps the centroid-closest image per cluster.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Install the Kaggle CLI: pip install kaggle, then configure "
            "~/.kaggle/kaggle.json credentials."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Full dataset downloaded here, then cleaned up after seed extraction
    dataset_dir = output_path / f"sfhq_part_{part}_full"
    dataset_dir.mkdir(exist_ok=True)

    dataset_name = f"selfishgene/synthetic-faces-high-quality-sfhq-part-{part}"

    # ── 1. Authenticate & Download ENTIRE Dataset ──
    print(f"[INFO] Authenticating Kaggle API for {dataset_name} ...")
    api = KaggleApi()
    api.authenticate()

    print(f"[INFO] Downloading and unzipping the ENTIRE database to {dataset_dir} ...")
    print(f"[INFO] (This may take a while depending on your network connection)")
    # Using the Kaggle API to bulk download and extract
    api.dataset_download_files(dataset_name, path=str(dataset_dir), unzip=True, quiet=False)

    # ── 2. Locate Images ──
    images_dir = dataset_dir / "images" / "images"
    if not images_dir.exists():
        # Fallback if structural path differs slightly
        images_dir = dataset_dir / "images"
        if not images_dir.exists():
             images_dir = dataset_dir
             
    print(f"[INFO] Indexing images in {images_dir} ...")
    image_paths = sorted(list(images_dir.glob("*.jpg")))
    
    actual_pool_size = len(image_paths)
    if not image_paths:
        raise RuntimeError(f"Failed to find any images in {images_dir}. Check dataset extraction.")
    
    print(f"[INFO] Successfully loaded ALL {actual_pool_size} images for the evaluation pool.")
    
    if actual_pool_size < num_images:
        print(f"\n[WARN] Only found {actual_pool_size} images; reducing target seeds.")
        num_images = actual_pool_size

    # ── 3. Attempt to Load Precomputed Embeddings ──
    embeddings_arr = None
    valid_paths = [str(p) for p in image_paths]
    
    print("\n[INFO] Searching for precomputed CLIP embeddings in the dataset...")
    # Typically saved as .npy (e.g., clip_features.npy, embeddings.npy)
    npy_files = list(dataset_dir.rglob("*.npy"))
    
    for npy_file in npy_files:
        if "clip" in npy_file.name.lower() or "embed" in npy_file.name.lower() or "feature" in npy_file.name.lower():
            print(f"[INFO] Found potential precomputed embeddings: {npy_file}")
            try:
                loaded_arr = np.load(npy_file)
                # Ensure the loaded array matches our image count exactly
                if loaded_arr.shape[0] == actual_pool_size:
                    print(f"[SUCCESS] Loaded precomputed embeddings perfectly matching image count ({loaded_arr.shape[0]}).")
                    embeddings_arr = loaded_arr
                    break
                else:
                    print(f"[WARN] Size mismatch: Embeddings {loaded_arr.shape[0]} vs Images {actual_pool_size}.")
            except Exception as e:
                print(f"[WARN] Failed to load {npy_file}: {e}")

    # ── 4. Fallback: Compute CLIP embeddings locally ──
    if embeddings_arr is None:
        print("\n[INFO] No valid precomputed embeddings found. Falling back to local CLIP inference...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "openai/clip-vit-base-patch32"
        print(f"[INFO] Loading CLIP model ({model_id}) on {device} ...")
        
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)

        embeddings: list[np.ndarray] = []
        valid_paths = []

        print(f"[INFO] Extracting CLIP semantic features for {len(image_paths)} images ...")
        for img_path in tqdm(image_paths, desc="Embedding images"):
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().numpy().flatten())
                valid_paths.append(str(img_path))
            except Exception:
                continue
                
        embeddings_arr = np.array(embeddings)
        
    if len(valid_paths) < num_images:
        num_images = len(valid_paths)

    # ── 5. KMeans clustering → most-representative per cluster ──
    print(f"\n[INFO] Clustering {len(embeddings_arr)} vectors into {num_images} groups ...")
    kmeans = KMeans(n_clusters=num_images, random_state=42, n_init="auto")
    kmeans.fit(embeddings_arr)
    closest, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, embeddings_arr
    )

    # ── 6. Save seeds directly to output_dir ──
    print(f"\n[INFO] Saving {num_images} diverse seeds to {output_path} ...")
    for idx in tqdm(closest, desc="Saving seeds"):
        src = valid_paths[idx]
        dest = output_path / f"seed_{idx:04d}.jpg"
        shutil.copy2(src, dest)

    # Clean up full dataset — only the seeds remain in output_dir
    print("[INFO] Cleaning up full dataset download ...")
    shutil.rmtree(dataset_dir, ignore_errors=True)

    print(f"[OK] {num_images} diverse seeds successfully saved to {output_path.resolve()}")
    
    return str(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download SFHQ and extract diverse seeds.")
    parser.add_argument("--num_images", type=int, default=400, help="Number of diverse seeds to extract")
    parser.add_argument("--output_dir", type=str, default="../data/seeds", help="Output directory")
    args = parser.parse_args()
    
    download_sfhq(num_images=args.num_images, output_dir=args.output_dir)
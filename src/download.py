"""
Step 1: Diverse SFHQ Seed Download via CLIP + KMeans

Downloads a pool of SFHQ images from Kaggle, embeds them with CLIP to
capture natural demographic diversity (age, gender, ethnicity), clusters
embeddings with KMeans, and saves the single most representative image
from each cluster as a seed.  This replaces bulk download + age-stratified
selection with a single, mathematically diverse pass.

Requires: kaggle CLI credentials, torch, transformers, scikit-learn.

Example:
    python download.py --num_images 400 --pool_size 4000 --output_dir ../data/raw
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm


def download_sfhq(
    part: int = 1,
    output_dir: str = "../data/raw",
    num_images: int = 400,
    pool_size: int = 4000,
) -> str:
    """Download *num_images* diverse seed images from SFHQ Part *part*.

    Downloads *pool_size* images as an evaluation pool, embeds them with
    OpenAI CLIP, clusters into *num_images* groups, and keeps the centroid-
    closest image per cluster.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Install the Kaggle CLI: pip install kaggle, then configure "
            "~/.kaggle/kaggle.json credentials."
        )

    from transformers import CLIPModel, CLIPProcessor

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path / "_temp_pool"
    temp_dir.mkdir(exist_ok=True)

    dataset_name = f"selfishgene/synthetic-faces-high-quality-sfhq-part-{part}"

    # ── 1. Authenticate ──
    print(f"[INFO] Authenticating Kaggle API for {dataset_name} ...")
    api = KaggleApi()
    api.authenticate()

    import concurrent.futures
    import threading

    # ── 2. Direct Streaming (Bypassing API Truncation via Multithreading) ──
    print(f"[INFO] Bypassing Kaggle API truncation. Streaming {pool_size} images concurrently...")
    
    image_paths: list[str] = []
    
    # Thread-safe lock for appending to our results list
    lock = threading.Lock()
    
    # We will generate a broad index range. We generate 2x the pool size to account 
    # for missing indices (404 errors) in the dataset sequence.
    index_range = range(0, pool_size * 2)
    
    # Helper function to run inside each thread
    def download_single_image(idx: int) -> bool:
        # If we've already hit our target pool size, abort early
        with lock:
            if len(image_paths) >= pool_size:
                return False
                
        filename = f"SFHQ_pt1_{idx:08d}.jpg"
        
        # Test valid Kaggle target paths
        candidates = [
            f"images/images/{filename}",
            f"images/{filename}",
            filename
        ]
        
        for target_path in candidates:
            try:
                # Kaggle API is thread-safe for reading/downloading
                api.dataset_download_file(dataset_name, target_path, path=str(temp_dir))
                
                # Check for and extract zip wrappers concurrently
                zip_candidate = temp_dir / f"{filename}.zip"
                if zip_candidate.exists():
                    with zipfile.ZipFile(zip_candidate) as zf:
                        zf.extractall(temp_dir)
                    zip_candidate.unlink()
                    
                # Use glob to find the extracted file, as Kaggle might nest it
                downloaded = list(temp_dir.rglob(filename))
                
                if downloaded:
                    with lock:
                        # Double-check pool size inside lock before appending
                        if len(image_paths) < pool_size:
                            image_paths.append(str(downloaded[0]))
                            return True
            except Exception:
                continue
                
        return False

    # Execute downloads concurrently using 20 threads
    max_threads = 20
    print(f"[INFO] Launching {max_threads} download threads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_threads=max_threads) as executor:
        # Submit tasks and wrap in tqdm for a progress bar
        futures = {executor.submit(download_single_image, i): i for i in index_range}
        
        with tqdm(total=pool_size, desc="Downloading pool images") as pbar:
            for future in concurrent.futures.as_completed(futures):
                success = future.result()
                if success:
                    pbar.update(1)
                    
                # Abort remaining futures if we hit our target size
                with lock:
                    if len(image_paths) >= pool_size:
                        # Cancel pending futures in the queue (Python 3.9+)
                        for f in futures:
                            f.cancel()
                        break

    actual_pool_size = len(image_paths)
    if not image_paths:
        raise RuntimeError("Failed to download any images. The dataset naming convention may have changed.")
    elif actual_pool_size < num_images:
        print(f"\n[WARN] Only downloaded {actual_pool_size} valid images; reducing target.")
        num_images = actual_pool_size

    pool_files = image_files[:pool_size]
    actual_pool_size = len(pool_files)
    if actual_pool_size < num_images:
        print(
            f"[WARN] Only {actual_pool_size} images available; "
            f"reducing num_images from {num_images} to {actual_pool_size}."
        )
        num_images = actual_pool_size

    # ── 3. CLIP embedding ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    print(f"[INFO] Loading CLIP model ({model_id}) on {device} ...")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    embeddings: list[np.ndarray] = []
    valid_paths: list[str] = []

    print("[INFO] Extracting CLIP semantic features ...")
    for img_path in tqdm(image_paths, desc="Embedding images"):
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy().flatten())
            valid_paths.append(img_path)
        except Exception:
            continue

    if len(valid_paths) < num_images:
        num_images = len(valid_paths)

    embeddings_arr = np.array(embeddings)

    # ── 4. KMeans clustering → most-representative per cluster ──
    print(f"[INFO] Clustering {len(embeddings_arr)} vectors into {num_images} groups ...")
    kmeans = KMeans(n_clusters=num_images, random_state=42, n_init="auto")
    kmeans.fit(embeddings_arr)
    closest, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, embeddings_arr
    )

    # ── 5. Save seeds ──
    print(f"[INFO] Saving {num_images} diverse seeds to {output_path} ...")
    for idx in tqdm(closest, desc="Saving seeds"):
        src = valid_paths[idx]
        dest = output_path / f"seed_{idx:04d}.jpg"
        shutil.copy2(src, dest)

    # ── 6. Cleanup ──
    print("[INFO] Cleaning up temporary pool ...")
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"[OK] {num_images} diverse seeds saved to {output_path.resolve()}")
    return str(output_path)

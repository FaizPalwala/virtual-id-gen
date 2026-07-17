"""
(Legacy File)
Step 2: Preprocessing
- Face detection and alignment using MTCNN (facenet-pytorch)
- Resize to 128x128
- Quality filtering (blur check, detection confidence)
- Saves processed crops to output_dir

Usage:
    python preprocess.py [--input_dir ../data/raw] [--output_dir ../data/processed]
                               [--img_size 128] [--max_images 30000]
"""

import argparse
import os
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def laplacian_variance(img_array: np.ndarray) -> float:
    """Measure image sharpness. Low value = blurry."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    img_size: int = 128,
    max_images: int = 30000,
    blur_threshold: float = 80.0,
    conf_threshold: float = 0.85,
    device: str = "auto"
):
    try:
        import torch
        from facenet_pytorch import MTCNN
    except ImportError:
        raise ImportError(
            "facenet-pytorch not installed.\n"
            "Install with: pip install facenet-pytorch"
        )

    if device == "auto":
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device

    print(f"[INFO] Using device: {device_str}")

    # MTCNN: outputs aligned face crops at img_size x img_size
    mtcnn = MTCNN(
        image_size=img_size,
        margin=20,
        keep_all=False,
        min_face_size=40,
        thresholds=[0.6, 0.7, conf_threshold],
        device=device_str
    )

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_images = sorted(
        list(input_path.rglob("*.jpg")) +
        list(input_path.rglob("*.png")) +
        list(input_path.rglob("*.jpeg"))
    )

    if max_images and len(all_images) > max_images:
        import random
        random.seed(42)
        all_images = random.sample(all_images, max_images)
        print(f"[INFO] Randomly sampled {max_images} images from {len(all_images)} total")

    print(f"[INFO] Processing {len(all_images)} images → {output_path}")

    stats = {"total": len(all_images), "success": 0, "no_face": 0,
             "blurry": 0, "low_conf": 0}
    metadata = []

    for img_path in tqdm(all_images, desc="Preprocessing"):
        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(img_pil)

            # Detect and align face
            face_tensor, probs = mtcnn(img_pil, return_prob=True)

            if face_tensor is None or probs is None:
                stats["no_face"] += 1
                continue

            conf = float(probs) if not hasattr(probs, '__len__') else float(probs[0])
            if conf < conf_threshold:
                stats["low_conf"] += 1
                continue

            # Convert tensor to numpy for blur check (tensor: C,H,W float [-1,1])
            face_np = face_tensor.permute(1, 2, 0).numpy()
            face_np = ((face_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

            blur_score = laplacian_variance(face_np)
            if blur_score < blur_threshold:
                stats["blurry"] += 1
                continue

            # Save processed image
            out_filename = f"{img_path.stem}_face.jpg"
            out_path = output_path / out_filename
            face_pil = Image.fromarray(face_np)
            face_pil.save(out_path, quality=92)

            metadata.append({
                "original_path": str(img_path),
                "processed_path": str(out_path),
                "filename": out_filename,
                "confidence": round(conf, 4),
                "blur_score": round(blur_score, 2)
            })
            stats["success"] += 1

        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}") 
            continue

    # Save metadata
    meta_path = output_path / "preprocessing_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"stats": stats, "images": metadata}, f, indent=2)

    print("\n[RESULTS]")
    print(f"  Total processed : {stats['total']}")
    print(f"  Saved (success) : {stats['success']}")
    print(f"  No face detected: {stats['no_face']}")
    print(f"  Low confidence  : {stats['low_conf']}")
    print(f"  Blurry          : {stats['blurry']}")
    print(f"  Yield rate      : {stats['success']/stats['total']*100:.1f}%")
    print(f"  Metadata saved  : {meta_path}")
    return str(output_path), metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SFHQ face images")
    parser.add_argument("--input_dir", type=str, default="../data/raw")
    parser.add_argument("--output_dir", type=str, default="../data/processed")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=30000)
    parser.add_argument("--blur_threshold", type=float, default=80.0)
    parser.add_argument("--conf_threshold", type=float, default=0.85)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', or 'cpu'")
    args = parser.parse_args()
    preprocess_dataset(
        args.input_dir, args.output_dir, args.img_size,
        args.max_images, args.blur_threshold, args.conf_threshold, args.device
    )

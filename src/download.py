"""
(Legacy File)
Step 1: SFHQ Dataset Acquisition
Downloads SFHQ Part 1 from Kaggle. Requires kaggle API credentials in ~/.kaggle/kaggle.json
Usage:
    python download.py [--part 1] [--output_dir ./data/raw]
"""

import argparse
import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def download_sfhq(part: int = 1, output_dir: str = "./data/raw"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_slug = f"selfishgene/synthetic-faces-high-quality-sfhq-part-{part}"
    
    print(f"[INFO] Initialising Kaggle API for SFHQ Part {part}...")

    # 1. Validate Credentials
    kaggle_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_cfg.exists() and 'KAGGLE_KEY' not in os.environ:
        raise FileNotFoundError(
            f"Kaggle credentials not found at {kaggle_cfg} or in ENV vars."
        )

    # 2. Fix permissions (Required for Linux/macOS to avoid errors)
    if kaggle_cfg.exists() and os.name != 'nt':
        os.chmod(kaggle_cfg, 0o600)

    try:
        # 3. Use Native API instead of subprocess
        api = KaggleApi()
        api.authenticate()
        
        print(f"[INFO] Downloading: {dataset_slug}")
        # unzip=True handles the extraction automatically
        api.dataset_download_files(dataset_slug, path=str(output_path), unzip=True, quiet=False)
        
        print(f"[OK] Download and extraction complete.")

    except Exception as e:
        print(f"[ERROR] Kaggle download failed: {e}")
        raise RuntimeError(f"Download failed: {e}")

    # 4. Count images
    img_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    img_files = [f for f in output_path.rglob("*") if f.suffix.lower() in img_extensions]
    
    print(f"[INFO] Found {len(img_files)} image files in {output_path.resolve()}")
    return str(output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SFHQ dataset from Kaggle")
    parser.add_argument("--part", type=int, default=1, choices=[1, 2, 3, 4],
                        help="SFHQ part number to download (default: 1)")
    parser.add_argument("--output_dir", type=str, default="../data/raw",
                        help="Directory to save downloaded data")
    args = parser.parse_args()
    download_sfhq(args.part, args.output_dir)
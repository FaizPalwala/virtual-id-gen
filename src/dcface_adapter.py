"""
dcface_adapter.py

External adapter for the official DCFace synthesis interface.
Accepts an identity image and a style image, invokes DCFace, and extracts
the newest generated image, renaming it to the requested output path.
"""

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("DCFaceAdapter")

def inject_pkg_resources_polyfill(target_dir: Path) -> None:
    """
    Injects a modern importlib polyfill to satisfy legacy torchmetrics imports.
    This prevents crashes on modern setuptools versions without breaking dependencies.
    """
    polyfill_path = target_dir / "pkg_resources.py"
    if not polyfill_path.exists():
        polyfill_code = (
            "from importlib.metadata import version, PackageNotFoundError\n\n"
            "class DistributionNotFound(Exception):\n"
            "    pass\n\n"
            "class Distribution:\n"
            "    def __init__(self, version):\n"
            "        self.version = version\n\n"
            "def get_distribution(pkg_name):\n"
            "    try:\n"
            "        return Distribution(version(pkg_name))\n"
            "    except PackageNotFoundError:\n"
            "        raise DistributionNotFound(pkg_name)\n"
        )
        polyfill_path.write_text(polyfill_code)
        logger.info(f"Injected legacy pkg_resources polyfill at {polyfill_path}")

def find_newest_image(directory: Path, before_set: Set[Path]) -> Path:
    """Identifies the newest image generated in the directory."""
    candidates = [
        p for p in directory.rglob("*") 
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p not in before_set
    ]
    if not candidates:
        raise RuntimeError("DCFace executed but produced no new output images.")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def main() -> None:
    parser = argparse.ArgumentParser(description="DCFace Generation Adapter")
    parser.add_argument("--dcface_root", required=True, help="Root path of the DCFace repository")
    parser.add_argument("--id_image", required=True, help="Path to seed identity image")
    parser.add_argument("--style_image", required=True, help="Path to style image")
    parser.add_argument("--output_path", required=True, help="Path to save the generated face")
    args = parser.parse_args()

    root = Path(args.dcface_root).resolve()
    synth_script = root / "src" / "synthesis.py"
    generated_dir = root / "generated_images"

    if not synth_script.exists():
        logger.error(f"Synthesis script not found at {synth_script}")
        raise FileNotFoundError(synth_script)

    # Automatically drop the polyfill next to the synthesis script before execution
    inject_pkg_resources_polyfill(synth_script.parent)

    with tempfile.TemporaryDirectory(prefix="dcface_style_") as temp_dir:
        style_dir = Path(temp_dir)
        # DCFace synthesis.py expects a directory of style images
        shutil.copy2(args.style_image, style_dir / Path(args.style_image).name)
        
        before_set = set(generated_dir.rglob("*")) if generated_dir.exists() else set()
        
        cmd = [
            "python", str(synth_script),
            "--id_images_root", str(Path(args.id_image).resolve()),
            "--style_images_root", str(style_dir)
        ]
        
        result = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
        
        if result.returncode != 0:
            logger.error(f"DCFace synthesis failed:\n{result.stderr}")
            raise SystemExit(result.returncode)

        target_path = Path(args.output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        new_image = find_newest_image(generated_dir, before_set)
        shutil.copy2(new_image, target_path)

if __name__ == "__main__":
    main()
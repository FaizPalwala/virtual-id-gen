"""Generate one identity-preserving image with the InstantID SDXL pipeline.

The adapter is intentionally a small CLI boundary around the current Diffusers
InstantID community pipeline.  It accepts both hyphenated and underscore CLI
options so it can be used directly and from the dataset pipeline.  It does not
require DCFace or a style image.

Example:
    python instantidadapter.py --idimage seed.jpg --outputpath output.png \
        --prompt "studio headshot, three-quarter view" --seed 42
"""
from __future__ import annotations

import argparse
import inspect
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import cv2
import numpy as np
from PIL import Image

LOGGER = logging.getLogger("instantid_adapter")
INSTANTID_REPOSITORY = "InstantX/InstantID"
ANTELOPE_REPOSITORY = "DIAMONIK7777/antelopev2"
DEFAULT_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_PROMPT = (
    "professional portrait photo, photorealistic, detailed skin, natural lighting"
)
DEFAULT_NEGATIVE_PROMPT = "blurry, deformed, mutated, low quality, watermark, text"


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution backend and precision selected for the current host."""

    device: str
    dtype: Any
    providers: list[str]


def configure_logging(verbose: bool) -> None:
    """Configure concise CLI logging once for direct module execution."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def get_runtime_config():
    """Select a safe Torch dtype and InsightFace ONNX providers."""
    import torch

    if torch.cuda.is_available():
        return RuntimeConfig(
            device="cuda",
            dtype=torch.float16,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    if torch.backends.mps.is_available():
        return RuntimeConfig(
            device="mps",
            dtype=torch.float32,
            providers=["CPUExecutionProvider"],
        )
    return RuntimeConfig(
        device="cpu",
        dtype=torch.float32,
        providers=["CPUExecutionProvider"],
    )


def get_antelope_root(cache_dir: Path | None = None) -> str:
    """Return the InsightFace model root, downloading antelopev2 if needed."""
    from huggingface_hub import snapshot_download

    root = cache_dir or Path.home() / ".cache" / "instantid" / "insightface"
    target = root / "models" / "antelopev2"
    required_file = target / "scrfd_10g_bnkps.onnx"
    if not required_file.exists():
        LOGGER.info("Downloading InsightFace antelopev2 weights to %s", target)
        snapshot_download(ANTELOPE_REPOSITORY, local_dir=str(target))
    return str(root)


def get_instantid_paths(cache_dir: Path | None = None) -> tuple[str, str]:
    """Download/cache and return the InstantID IP-Adapter and ControlNet paths."""
    from huggingface_hub import hf_hub_download, snapshot_download

    adapter_path = hf_hub_download(
        INSTANTID_REPOSITORY,
        "ip-adapter.bin",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    repository_path = snapshot_download(
        INSTANTID_REPOSITORY,
        allow_patterns=["ControlNetModel/*"],
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    controlnet_path = Path(repository_path) / "ControlNetModel"
    if not controlnet_path.exists():
        raise FileNotFoundError(
            f"InstantID ControlNet was not cached at {controlnet_path}"
        )
    return adapter_path, str(controlnet_path)


def draw_keypoints(image: Image.Image, keypoints: np.ndarray) -> Image.Image:
    """Render InstantID's five facial landmarks as a ControlNet conditioning image."""
    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape != (5, 2):
        raise ValueError(f"Expected five (x, y) landmarks, received {points.shape}.")
    canvas = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for first, second in ((0, 2), (1, 2), (3, 2), (4, 2)):
        x_values, y_values = points[[first, second], 0], points[[first, second], 1]
        length = max(
            1, int(np.hypot(x_values[0] - x_values[1], y_values[0] - y_values[1]) / 2)
        )
        angle = int(
            math.degrees(
                math.atan2(y_values[0] - y_values[1], x_values[0] - x_values[1])
            )
        )
        polygon = cv2.ellipse2Poly(
            (int(x_values.mean()), int(y_values.mean())), (length, 4), angle, 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, polygon, colours[first])
    canvas = (canvas * 0.6).astype(np.uint8)
    for index, (x_value, y_value) in enumerate(points):
        cv2.circle(canvas, (int(x_value), int(y_value)), 10, colours[index], -1)
    return Image.fromarray(canvas)


def detect_primary_face(
    analyser, image_bgr: np.ndarray, source: str, ctx_id: int
) -> Any:
    """Detect the largest face, retrying padded copies for tightly cropped seeds."""
    height, width = image_bgr.shape[:2]
    for ratio in (0.0, 0.25, 0.50):
        padding_y, padding_x = int(height * ratio), int(width * ratio)
        candidate = (
            image_bgr
            if ratio == 0
            else cv2.copyMakeBorder(
                image_bgr,
                padding_y,
                padding_y,
                padding_x,
                padding_x,
                cv2.BORDER_REFLECT_101,
            )
        )
        for det_size in ((640, 640), (320, 320)):
            analyser.prepare(ctx_id=ctx_id, det_size=det_size)
            faces = analyser.get(candidate)
            if not faces:
                continue
            # Never mutate InsightFace objects returned for a padded retry.
            face = max(
                faces,
                key=lambda item: float(
                    (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
                ),
            )
            face.kps = np.asarray(face.kps, dtype=np.float32).copy()
            face.kps[:, 0] -= padding_x
            face.kps[:, 1] -= padding_y
            return face
    raise ValueError(f"No usable face detected in {source}; tried padded copies.")


def patch_legacy_instantid_check_inputs(pipe) -> bool:
    """Bridge the legacy positional InstantID call only when the installed API needs it.

    Returns whether a patch was installed.  The wrapper forwards modern keyword
    invocations unchanged and makes the old 14-argument invocation explicit.
    """
    original = pipe.check_inputs
    if getattr(original, "_instantid_compat_patch", False):
        return False

    def check_inputs(self, *args, **kwargs):
        if kwargs or len(args) != 14:
            return original(*args, **kwargs)
        names = (
            "prompt",
            "prompt_2",
            "image",
            "callback_steps",
            "negative_prompt",
            "negative_prompt_2",
            "prompt_embeds",
            "negative_prompt_embeds",
            "pooled_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "controlnet_conditioning_scale",
            "control_guidance_start",
            "control_guidance_end",
            "callback_on_step_end_tensor_inputs",
        )
        values = dict(zip(names, args, strict=True))
        values["controlnet_conditioning_scale"] = float(
            values["controlnet_conditioning_scale"]
        )
        return original(**values)

    check_inputs._instantid_compat_patch = True
    pipe.check_inputs = MethodType(check_inputs, pipe)
    LOGGER.info("Installed legacy InstantID check_inputs compatibility shim")
    return True


def build_pipeline(args: argparse.Namespace, runtime: RuntimeConfig):
    """Load ControlNet, the InstantID community pipeline, and IP-Adapter weights."""
    from diffusers import ControlNetModel, DiffusionPipeline

    adapter_path, controlnet_path = get_instantid_paths(args.cache_dir)
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=runtime.dtype
    )
    pipeline = DiffusionPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=runtime.dtype,
        custom_pipeline="pipeline_stable_diffusion_xl_instantid",
    ).to(runtime.device)
    patch_legacy_instantid_check_inputs(pipeline)
    pipeline.load_ip_adapter_instantid(adapter_path)
    pipeline.set_ip_adapter_scale(float(args.ip_adapter_scale))
    return pipeline


def make_generator(seed: int | None, device: str):
    """Return a reproducibly seeded Torch generator, or None for stochastic runs."""
    if seed is None:
        return None
    import torch

    return torch.Generator(device=device).manual_seed(seed)


def validate_arguments(args: argparse.Namespace) -> None:
    """Fail early for configurations that SDXL cannot satisfy."""
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive.")
    if args.width % 8 or args.height % 8:
        raise ValueError("width and height must be divisible by 8.")
    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive.")
    if args.ip_adapter_scale < 0 or args.controlnet_conditioning_scale < 0:
        raise ValueError("conditioning scales must be non-negative.")


def generate_image(args: argparse.Namespace) -> Path:
    """Generate and save one InstantID image from a full-resolution seed image."""
    validate_arguments(args)
    runtime = get_runtime_config()
    LOGGER.info("Using device=%s, dtype=%s", runtime.device, runtime.dtype)
    source = Path(args.idimage).expanduser().resolve()
    image_bgr = cv2.imread(str(source))
    if image_bgr is None:
        raise FileNotFoundError(source)
    from insightface.app import FaceAnalysis

    analyser = FaceAnalysis(
        name="antelopev2",
        root=get_antelope_root(args.cache_dir),
        providers=runtime.providers,
    )
    face = detect_primary_face(analyser, image_bgr, str(source), args.face_ctx_id)
    pipeline = build_pipeline(args, runtime)
    rgb_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    keypoint_image = draw_keypoints(rgb_image, face.kps)
    result = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image_embeds=face.embedding,
        image=keypoint_image,
        width=args.width,
        height=args.height,
        controlnet_conditioning_scale=float(args.controlnet_conditioning_scale),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=float(args.guidance_scale),
        generator=make_generator(args.seed, runtime.device),
    ).images[0]
    output = Path(args.outputpath).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    LOGGER.info("Saved generated image to %s", output)
    return output


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI; option aliases preserve old and new callers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--idimage", "--id-image", "--id_image", dest="idimage", required=True
    )
    parser.add_argument(
        "--outputpath",
        "--output-path",
        "--output_path",
        dest="outputpath",
        required=True,
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--negativeprompt",
        "--negative-prompt",
        "--negative_prompt",
        dest="negative_prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
    )
    parser.add_argument(
        "--base-model", "--base_model", dest="base_model", default=DEFAULT_BASE_MODEL
    )
    parser.add_argument(
        "--ip-adapter-scale",
        "--ip_adapter_scale",
        dest="ip_adapter_scale",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--controlnet-conditioning-scale",
        "--controlnet_conditioning_scale",
        dest="controlnet_conditioning_scale",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--num-inference-steps",
        "--num_inference_steps",
        dest="num_inference_steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--guidance-scale",
        "--guidance_scale",
        dest="guidance_scale",
        type=float,
        default=5.0,
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--face-ctx-id",
        type=int,
        default=0,
        help="InsightFace context ID; use -1 for CPU.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    """Parse command-line arguments and generate one output image."""
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    generate_image(args)


if __name__ == "__main__":
    main()

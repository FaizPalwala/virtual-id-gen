"""InstantID adapter compatible with current diffusers/transformers/huggingface_hub.

This adapter deliberately does not pin or downgrade Hugging Face dependencies.
It applies one narrow compatibility shim for the legacy InstantID community
pipeline, whose check_inputs call uses positional arguments from an older
Diffusers API.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from types import MethodType

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, DiffusionPipeline
from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("InstantIDAdapter")

INSTANTID_REPO = "InstantX/InstantID"


def hardware() -> tuple[str, torch.dtype, list[str]]:
    if torch.cuda.is_available():
        logger.info("Hardware detected: CUDA")
        return "cuda", torch.float16, ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if torch.backends.mps.is_available():
        logger.info("Hardware detected: Apple Silicon MPS; using float32 for stability")
        return "mps", torch.float32, ["CPUExecutionProvider"]
    logger.info("Hardware detected: CPU")
    return "cpu", torch.float32, ["CPUExecutionProvider"]


def antelope_root() -> str:
    root = Path.home() / ".cache" / "instantid" / "insightface"
    target = root / "models" / "antelopev2"
    if not (target / "scrfd_10g_bnkps.onnx").exists():
        logger.info("Downloading InsightFace antelopev2 weights")
        snapshot_download("DIAMONIK7777/antelopev2", local_dir=str(target))
    return str(root)


def instantid_paths() -> tuple[str, str]:
    adapter = hf_hub_download(INSTANTID_REPO, "ip-adapter.bin")
    controlnet = snapshot_download(INSTANTID_REPO, allow_patterns=["ControlNetModel/*"])
    return adapter, str(Path(controlnet) / "ControlNetModel")


def draw_kps(image: Image.Image, kps: np.ndarray) -> Image.Image:
    kps = np.asarray(kps)
    canvas = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for a, b in ((0, 2), (1, 2), (3, 2), (4, 2)):
        x, y = kps[[a, b], 0], kps[[a, b], 1]
        length = int(np.hypot(x[0] - x[1], y[0] - y[1]) / 2)
        angle = int(math.degrees(math.atan2(y[0] - y[1], x[0] - x[1])))
        poly = cv2.ellipse2Poly((int(x.mean()), int(y.mean())), (length, 4), angle, 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, colors[a])
    canvas = (canvas * 0.6).astype(np.uint8)
    for i, (x, y) in enumerate(kps):
        cv2.circle(canvas, (int(x), int(y)), 10, colors[i], -1)
    return Image.fromarray(canvas)


def detect_face(app: FaceAnalysis, image: np.ndarray, source: str):
    """Handle tightly cropped faces by retrying with reflected context."""
    h, w = image.shape[:2]
    for ratio in (0.0, 0.25, 0.50):
        ph, pw = int(h * ratio), int(w * ratio)
        candidate = image if ratio == 0 else cv2.copyMakeBorder(
            image, ph, ph, pw, pw, cv2.BORDER_REFLECT_101
        )
        for det_size in ((640, 640), (320, 320)):
            app.prepare(ctx_id=0, det_size=det_size)
            faces = app.get(candidate)
            if faces:
                for face in faces:
                    face["kps"][:, 0] -= pw
                    face["kps"][:, 1] -= ph
                    face["bbox"][[0, 2]] -= pw
                    face["bbox"][[1, 3]] -= ph
                return max(faces, key=lambda f: (f["bbox"][2]-f["bbox"][0])*(f["bbox"][3]-f["bbox"][1]))
    raise ValueError(f"No usable face detected in {source}; tried unpadded and padded copies.")


def patch_legacy_instantid_check_inputs(pipe) -> None:
    """Bridge InstantID's old positional call to modern Diffusers' keyword API."""
    original = pipe.check_inputs

    def check_inputs(self, *args, **kwargs):
        if kwargs or len(args) != 14:
            return original(*args, **kwargs)
        (
            prompt, prompt_2, image, callback_steps, negative_prompt,
            negative_prompt_2, prompt_embeds, negative_prompt_embeds,
            pooled_prompt_embeds, negative_pooled_prompt_embeds,
            controlnet_conditioning_scale, control_guidance_start,
            control_guidance_end, callback_on_step_end_tensor_inputs,
        ) = args
        return original(
            prompt=prompt, prompt_2=prompt_2, image=image,
            callback_steps=callback_steps, negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

    pipe.check_inputs = MethodType(check_inputs, pipe)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one identity-preserving SDXL image with InstantID")
    parser.add_argument("--id_image", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--style_image", default=None, help="Accepted for orchestrator compatibility; unused")
    parser.add_argument("--dcface_root", default=None, help="Accepted for orchestrator compatibility; unused")
    parser.add_argument("--prompt", default="professional portrait photo, photorealistic, detailed skin, natural lighting")
    parser.add_argument("--negative_prompt", default="blurry, deformed, mutated, low quality, watermark, text")
    parser.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--ip_adapter_scale", type=float, default=0.8)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.8)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None)
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning("Ignoring extra orchestrator arguments: %s", unknown)

    device, dtype, providers = hardware()
    image_bgr = cv2.imread(args.id_image)
    if image_bgr is None:
        raise FileNotFoundError(args.id_image)
    analyser = FaceAnalysis(name="antelopev2", root=antelope_root(), providers=providers)
    face = detect_face(analyser, image_bgr, args.id_image)
    adapter_path, controlnet_path = instantid_paths()
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)

    # This is the maintained Diffusers community implementation, dynamically cached by Diffusers.
    pipe = DiffusionPipeline.from_pretrained(
        args.base_model, controlnet=controlnet, torch_dtype=dtype,
        custom_pipeline="pipeline_stable_diffusion_xl_instantid",
    ).to(device)
    patch_legacy_instantid_check_inputs(pipe)
    pipe.load_ip_adapter_instantid(adapter_path)
    pipe.set_ip_adapter_scale(float(args.ip_adapter_scale))

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    kps_image = draw_kps(Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)), face["kps"])
    result = pipe(
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        image_embeds=face["embedding"], image=kps_image,
        controlnet_conditioning_scale=float(args.controlnet_conditioning_scale),
        num_inference_steps=args.num_inference_steps, guidance_scale=float(args.guidance_scale),
        generator=generator,
    ).images[0]
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    logger.info("Saved %s", output)


if __name__ == "__main__":
    main()
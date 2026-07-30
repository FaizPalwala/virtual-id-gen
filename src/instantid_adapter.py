"""
instantid_adapter.py
InstantID SDXL adapter for generating identity-preserving images from a seed.
Generate one identity-preserving image with the InstantID SDXL pipeline.

The adapter is intentionally a small CLI boundary around the current Diffusers
InstantID community pipeline.  It accepts both hyphenated and underscore CLI
options so it can be used directly and from the dataset pipeline.  

Example:
    python instantid_adapter.py --idimage seed.jpg --outputpath output.png \
        --prompt "studio headshot, three-quarter view" --seed 42
"""
from __future__ import annotations

import argparse
import inspect
import io
import logging
import math
from contextlib import redirect_stderr, redirect_stdout
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
    # Silence InsightFace noisy ``set det-size`` stdout spam during detection.
    _devnull = io.StringIO()
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
            with redirect_stdout(_devnull), redirect_stderr(_devnull):
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
        """Bridge legacy 14-arg positional calls to modern keyword API."""
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


@dataclass(frozen=True)
class EncodedIdentity:
    """Cached InstantID conditioning derived once from a source seed image."""

    source_path: str
    face_embedding: np.ndarray
    keypoint_image: Image.Image


def validate_generation_settings(
    width: int,
    height: int,
    steps: int,
    ip_adapter_scale: float,
    controlnet_scale: float,
) -> None:
    """Fail early for image dimensions and conditioning values SDXL cannot use."""
    if width <= 0 or height <= 0 or width % 8 or height % 8:
        raise ValueError("width and height must be positive and divisible by 8.")
    if steps <= 0:
        raise ValueError("num_inference_steps must be positive.")
    if ip_adapter_scale < 0 or controlnet_scale < 0:
        raise ValueError("conditioning scales must be non-negative.")


class InstantIDGeneratorSession:
    """Long-lived InstantID worker that loads all models once per process.

    Construct one session for a Slurm generation job.  Call ``encode_identity``
    once per source seed, then ``generate`` for its many candidate variations.
    This avoids repeatedly starting Python and reloading SDXL, ControlNet,
    InstantID, and InsightFace for every generated candidate.
    """

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        ip_adapter_scale: float = 0.90,
        controlnet_conditioning_scale: float = 0.80,
        cache_dir: Path | None = None,
        require_cuda: bool = True,
    ):
        """Load SDXL pipeline, ControlNet, InstantID adapter, and InsightFace.

        All models are loaded once and reused for the session lifetime.
        """
        validate_generation_settings(
            8, 8, 1, ip_adapter_scale, controlnet_conditioning_scale
        )
        self.runtime = get_runtime_config()
        if require_cuda and self.runtime.device != "cuda":
            raise RuntimeError(
                "CUDA is required for generation but PyTorch selected "
                f"{self.runtime.device!r}. Check the Slurm allocation and "
                "CUDA-enabled PyTorch installation."
            )
        # import onnxruntime as ort

        # if (
        #     self.runtime.device == "cuda"
        #     and "CUDAExecutionProvider" not in ort.get_available_providers()
        # ):
        #     raise RuntimeError(
        #         "ONNX Runtime lacks CUDAExecutionProvider. Available: "
        #         f"{ort.get_available_providers()}. Install onnxruntime-gpu."
        #     )
        from insightface.app import FaceAnalysis
        from diffusers import ControlNetModel, DiffusionPipeline

        self.face_ctx_id = 0 if self.runtime.device == "cuda" else -1
        self.controlnet_conditioning_scale = float(controlnet_conditioning_scale)
        self.analyser = FaceAnalysis(
            name="antelopev2",
            root=get_antelope_root(cache_dir),
            providers=self.runtime.providers,
        )
        adapter_path, controlnet_path = get_instantid_paths(cache_dir)
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=self.runtime.dtype
        )
        # Load pipeline components from SDXL base (text encoder, VAE, scheduler).
        # Fine-tuned models like Juggernaut-XL-v9 only ship UNet weights;
        # loading the full pipeline from them fails with missing text_encoder.
        from diffusers import UNet2DConditionModel

        self.pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=self.runtime.dtype,
            custom_pipeline="pipeline_stable_diffusion_xl_instantid",
        )
        if base_model != DEFAULT_BASE_MODEL:
            # Fine-tuned models like Juggernaut-XL-v9 are distributed as
            # single .safetensors checkpoints, not diffusers folder format.
            from huggingface_hub import hf_hub_download, list_repo_files

            repo_files = list_repo_files(base_model)
            ckpt_files = [f for f in repo_files if f.endswith(".safetensors")]
            if not ckpt_files:
                raise FileNotFoundError(
                    f"No .safetensors checkpoint in {base_model}. "
                    f"Files: {repo_files[:5]}"
                )
            ckpt_path = hf_hub_download(base_model, ckpt_files[0])
            unet = UNet2DConditionModel.from_single_file(
                ckpt_path, torch_dtype=self.runtime.dtype
            ).to(self.runtime.device)
            self.pipeline.unet = unet
        self.pipeline = self.pipeline.to(self.runtime.device)
        # Clear cache after loading model bodies to reduce fragmentation
        # before the IP-Adapter allocation.
        torch.cuda.empty_cache()
        patch_legacy_instantid_check_inputs(self.pipeline)
        self.pipeline.load_ip_adapter_instantid(adapter_path)
        self.pipeline.set_ip_adapter_scale(float(ip_adapter_scale))

        # ------------------------------------------------------------------
        # Attention optimisation (xFormers or SDPA — choose best available)
        # ------------------------------------------------------------------
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
            LOGGER.info("xFormers memory-efficient attention enabled")
        except (ImportError, ModuleNotFoundError):
            self.pipeline.enable_attention_slicing(slice_size="auto")
            LOGGER.info("xFormers unavailable; fell back to attention slicing")

        # ------------------------------------------------------------------
        # VAE memory/throughput tuning
        # ------------------------------------------------------------------
        self.pipeline.vae.enable_slicing()
        self.pipeline.vae.enable_tiling()

        # ------------------------------------------------------------------
        # Scheduler: DPM-Solver++ with Karras sigmas for faster convergence
        # ------------------------------------------------------------------
        from diffusers import DPMSolverMultistepScheduler

        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
            solver_order=2,
        )

        # ------------------------------------------------------------------
        # CUDA kernel auto-tuning (Ampere+ tensor cores, TF32, cuDNN)
        # ------------------------------------------------------------------
        import torch

        if self.runtime.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            LOGGER.info(
                "CUDA optimisations enabled: cudnn benchmark, TF32 matmul, "
                "high matmul precision"
            )
        # ------------------------------------------------------------------
        # Suppress diffusers library noise: VAE dtype warnings and
        # per-step denoising tqdm bars (we have our own identity-level bar).
        # ------------------------------------------------------------------
        import diffusers.utils.logging as _dlog

        _dlog.set_verbosity_error()
        try:
            self.pipeline.set_progress_bar_config(disable=True)
        except (AttributeError, TypeError):
            pass
        LOGGER.info(
            "Persistent InstantID session ready: device=%s, dtype=%s",
            self.runtime.device,
            self.runtime.dtype,
        )

    def encode_identity(self, source_path: str | Path) -> EncodedIdentity:
        """Detect one seed face and cache its embedding and landmark image."""
        source = Path(source_path).expanduser().resolve()
        image_bgr = cv2.imread(str(source))
        if image_bgr is None:
            raise FileNotFoundError(source)
        face = detect_primary_face(
            self.analyser, image_bgr, str(source), self.face_ctx_id
        )
        rgb_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        return EncodedIdentity(
            str(source),
            np.asarray(face.embedding, dtype=np.float32),
            draw_keypoints(rgb_image, face.kps),
        )

    def precompute_prompt_embeddings(
        self,
        prompts: list[str],
        negative_prompt: str,
        num_images_per_prompt: int = 1,
    ) -> tuple[dict[str, tuple[Any, Any, Any, Any]], tuple[Any, Any, Any, Any]]:
        """Pre-encode every unique prompt + negative prompt once.

        SDXL's text encoder runs twice per generation (prompt + negative prompt).
        For 400 identities × 39 variations that is 31 200 encoder forward passes.
        This method reduces that to 40 calls total (39 prompts + 1 negative).

        Returns
        -------
        prompt_cache : dict[str, (prompt_embeds, negative_prompt_embeds, pooled, neg_pooled)]
            Keyed by the verbatim prompt string.
        neg_embeds : (prompt_embeds, negative_prompt_embeds, pooled, neg_pooled)
            Shared negative‑prompt embedding tuple (IDENTICAL for all prompts).
        """
        unique_prompts = sorted(set(prompts))
        import torch

        prompt_cache: dict[str, tuple] = {}
        LOGGER.info(
            "Precomputing embeddings for %d unique prompts + negative…",
            len(unique_prompts),
        )
        for prompt_text in unique_prompts:
            with torch.inference_mode():
                (
                    prompt_embeds,
                    neg_embeds,
                    pooled_embeds,
                    neg_pooled_embeds,
                ) = self.pipeline.encode_prompt(
                    prompt=prompt_text,
                    negative_prompt=negative_prompt,
                    device=self.runtime.device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=True,
                )
            prompt_cache[prompt_text] = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": neg_embeds,
                "pooled_prompt_embeds": pooled_embeds,
                "negative_pooled_prompt_embeds": neg_pooled_embeds,
            }

        # Build the deduped negative embedding (identical across all prompts).
        first_entry = prompt_cache[unique_prompts[0]]
        neg_tuple = (
            first_entry["negative_prompt_embeds"],
            first_entry["pooled_prompt_embeds"],
            first_entry["negative_pooled_prompt_embeds"],
        )
        LOGGER.info("Embedding cache built: %d prompts", len(prompt_cache))
        return prompt_cache, neg_tuple

    def generate_cached(
        self,
        identity: EncodedIdentity,
        output_path: str | Path,
        *,
        prompt_embeddings: dict[str, Any],
        prompt_text: str,
        neg_embeds_tuple: tuple[Any, Any, Any, Any],
        seed: int,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 35,
        guidance_scale: float = 5.5,
    ) -> Path:
        """Generate one candidate using precomputed SDXL embeddings."""
        validate_generation_settings(
            width, height, num_inference_steps, 0.0, self.controlnet_conditioning_scale
        )
        import torch

        embeds = prompt_embeddings[prompt_text]

        generator = torch.Generator(device=self.runtime.device).manual_seed(seed)
        with torch.inference_mode():
            result = self.pipeline(
                prompt_embeds=embeds["prompt_embeds"],
                negative_prompt_embeds=embeds["negative_prompt_embeds"],
                pooled_prompt_embeds=embeds["pooled_prompt_embeds"],
                negative_pooled_prompt_embeds=embeds[
                    "negative_pooled_prompt_embeds"
                ],
                image_embeds=identity.face_embedding,
                image=identity.keypoint_image,
                width=width,
                height=height,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps,
                guidance_scale=float(guidance_scale),
                generator=generator,
                disable_progress_bar=True,
            ).images[0]
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output)
        return output

    def generate(
        self,
        identity: EncodedIdentity,
        output_path: str | Path,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 35,
        guidance_scale: float = 5.5,
    ) -> Path:
        """Generate one candidate without reinitialising the loaded models."""
        validate_generation_settings(
            width, height, num_inference_steps, 0.0, self.controlnet_conditioning_scale
        )
        import torch

        generator = torch.Generator(device=self.runtime.device).manual_seed(seed)
        with torch.inference_mode():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_embeds=identity.face_embedding,
                image=identity.keypoint_image,
                width=width,
                height=height,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps,
                guidance_scale=float(guidance_scale),
                generator=generator,
                disable_progress_bar=True,
            ).images[0]
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output)
        return output

    def close(self) -> None:
        """Release model references and return unused CUDA memory to the driver."""
        if hasattr(self, "pipeline"):
            del self.pipeline
        if hasattr(self, "analyser"):
            del self.analyser
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate CLI settings with the same rules used by a persistent session."""
    validate_generation_settings(
        args.width,
        args.height,
        args.num_inference_steps,
        args.ip_adapter_scale,
        args.controlnet_conditioning_scale,
    )


def generate_image(args: argparse.Namespace) -> Path:
    """CLI-compatible one-image wrapper around the persistent session API."""
    validate_arguments(args)
    session = InstantIDGeneratorSession(
        base_model=args.base_model,
        ip_adapter_scale=args.ip_adapter_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        cache_dir=args.cache_dir,
        require_cuda=False,
    )
    try:
        identity = session.encode_identity(args.idimage)
        return session.generate(
            identity,
            args.outputpath,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
    finally:
        session.close()


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

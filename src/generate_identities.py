"""
generate_identities.py
Generate varied raw InstantID candidate pools from full-resolution SFHQ seeds.

Each identity receives a deterministic variation plan.  A plan changes only
non-identity attributes (pose, expression, lighting, camera/background) while
the source image remains the sole InstantID identity condition.  Every trial's
plan, prompt, and seed are written to ``raw_candidate_manifest.csv``.

Example:
    python generate_identities.py \\
        --seedsdir ../data/seeds \\
        --outputdir ../data/identities \\
        --nidentities 600 \\
        --variantsperidentity 85 \\
        --randomstate 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import get_image_paths
from extract_embeddings import (
    get_embedding_and_attributes_robust,
    load_arcface_model,
)
from prompts import (
    NEGATIVE_PROMPT,
    PHOTOREALISM_PREFIX,
    VariationSpec,
    build_variation_plan,
)

LOGGER = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as ``1h23m`` or ``45s``."""
    if seconds < 0:
        seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# Model-specific hyperparameter presets.  Config defaults are set for
# Juggernaut-XL-v9 (the primary model).  Switching base_model in config.yaml
# automatically applies the matching preset for params not explicitly overridden.
MODEL_DEFAULTS: dict[str, dict[str, float | int]] = {
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "guidance_scale": 5.5,
        "ip_adapter_scale": 0.90,
        "num_inference_steps": 25,
    },
    "RunDiffusion/Juggernaut-XL-v9": {
        "guidance_scale": 3.0,
        "ip_adapter_scale": 0.85,
        "num_inference_steps": 30,
    },
}

def generate_identities(
    seedsdir: str,
    outputdir: str,
    nidentities: int = 400,
    candidatesperidentity: int = 39,
    ctxid: int = 0,
    randomstate: int = 42,
    min_similarity_raw: float = 0.40,
    instantid_config: dict | None = None,
) -> str:
    """Generate candidate pools until exactly ``nidentities`` valid seeds exist.

    Source images are iterated in shuffled order.  A source is skipped if
    robust ArcFace validation or InstantID seed encoding cannot find a face.
    Generation stops when ``nidentities`` clusters are completed.
    """
    from instantid_adapter import InstantIDGeneratorSession

    if nidentities <= 0:
        raise ValueError("nidentities must be positive.")
    if candidatesperidentity <= 0:
        raise ValueError("candidatesperidentity must be positive.")

    instantid_config = instantid_config or {}
    base_model = instantid_config.get("base_model") or (
        "stabilityai/stable-diffusion-xl-base-1.0"
    )
    # Resolve hyperparams: CLI override > model preset > hardcoded fallback.
    _preset = MODEL_DEFAULTS.get(base_model, {})
    _gs = float(instantid_config.get("guidance_scale", _preset.get("guidance_scale", 5.5)))
    _ips = float(instantid_config.get("ip_adapter_scale", _preset.get("ip_adapter_scale", 0.90)))
    _steps = int(instantid_config.get("num_inference_steps", _preset.get("num_inference_steps", 25)))
    sources = get_image_paths(seedsdir)
    if len(sources) < nidentities:
        raise ValueError(
            f"Need at least {nidentities} source images; found {len(sources)}."
        )
    rng = random.Random(randomstate)
    rng.shuffle(sources)
    attempted_sources = sources

    output = Path(outputdir)
    candidates_root = output / "candidates"
    rejected_root = output / "rejected"
    candidates_root.mkdir(parents=True, exist_ok=True)
    variations = build_variation_plan(candidatesperidentity)
    validation_app = load_arcface_model(ctxid)
    records: list[dict] = []
    skipped_seeds: list[dict] = []
    completed = 0
    session = InstantIDGeneratorSession(
        base_model=base_model,
        ip_adapter_scale=_ips,
        controlnet_conditioning_scale=float(
            instantid_config.get("controlnet_conditioning_scale", 0.80)
        ),
        cache_dir=Path(instantid_config["cache_dir"])
        if instantid_config.get("cache_dir")
        else None,
        require_cuda=bool(instantid_config.get("require_cuda", True)),
    )
    try:
        # ------------------------------------------------------------------
        # Precompute SDXL text embeddings for the 39 variation prompts + the
        # shared negative prompt.  Without this the text encoder runs twice
        # (prompt + negative) per generation — 31 200 times for a 400-identity
        # run.  With the cache it runs exactly 40 times (39 prompts + 1 neg).
        # ------------------------------------------------------------------
        prompt_texts = [var.prompt() for var in variations]
        prompt_cache, neg_embeds_tuple = session.precompute_prompt_embeddings(
            prompt_texts, NEGATIVE_PROMPT
        )

        progress = tqdm(
            total=nidentities,
            desc="Identities",
            unit="id",
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )
        t_start = time.monotonic()
        for attempt_index, seed_path in enumerate(attempted_sources, start=1):
            if completed >= nidentities:
                break
            seed_image = cv2.imread(str(seed_path))
            seed_embedding, _, _ = get_embedding_and_attributes_robust(
                validation_app, seed_image, ctxid
            )
            if seed_embedding is None:
                reason = "validation_face_not_detected"
                LOGGER.warning("Skipping seed %s: %s", seed_path, reason)
                skipped_seeds.append(
                    {
                        "seedpath": str(seed_path),
                        "attempt": attempt_index,
                        "reason": reason,
                    }
                )
                continue
            try:
                identity = session.encode_identity(seed_path)
            except (FileNotFoundError, ValueError) as error:
                reason = f"instantid_seed_encoding_failed: {error}"
                LOGGER.warning("Skipping seed %s: %s", seed_path, reason)
                skipped_seeds.append(
                    {
                        "seedpath": str(seed_path),
                        "attempt": attempt_index,
                        "reason": reason,
                    }
                )
                continue

            cluster_id = completed
            cluster_dir = candidates_root / f"identity_{cluster_id:03d}"
            cluster_dir.mkdir(exist_ok=True)
            for trial, variation in enumerate(variations):
                generation_seed = (
                    randomstate + cluster_id * candidatesperidentity + trial
                )
                candidate_path = cluster_dir / f"candidate_{trial:03d}.png"
                session.generate_cached(
                    identity,
                    candidate_path,
                    prompt_embeddings=prompt_cache,
                    prompt_text=variation.prompt(),
                    neg_embeds_tuple=neg_embeds_tuple,
                    seed=generation_seed,
                    width=int(instantid_config.get("width", 1024)),
                    height=int(instantid_config.get("height", 1024)),
                    num_inference_steps=_steps,
                    guidance_scale=_gs,
                )
                # Deferred validation: ArcFace quality gating is handled by
                # the preprocessing stage (preprocess.py), which already
                # performs detection confidence, sharpness, and similarity
                # checks.  Skipping per-generation validation removes ~15 600
                # redundant ArcFace forward passes.
                records.append(
                    {
                        "identityid": cluster_id,
                        "clusterid": cluster_id,
                        "trial": trial,
                        "seedpath": str(seed_path),
                        "seed_attempt": attempt_index,
                        "generationseed": generation_seed,
                        "prompt": variation.prompt(),
                        "negative_prompt": NEGATIVE_PROMPT,
                        "raw_candidatepath": str(candidate_path),
                        "raw_arcface_similarity": None,
                        "raw_status": "unvalidated",
                        **asdict(variation),
                    }
                )
            completed += 1
            progress.update(1)
            elapsed = time.monotonic() - t_start
            rate = elapsed / completed if completed else 0
            LOGGER.info(
                "identity %3d/%-3d | seed=%s | skipped=%d | elapsed=%s | "
                "%.1f min/id | ETA %s",
                completed,
                nidentities,
                Path(seed_path).name,
                len(skipped_seeds),
                _fmt_duration(elapsed),
                rate / 60,
                _fmt_duration(rate * (nidentities - completed)),
            )
    finally:
        session.close()

    pd.DataFrame(records).to_csv(output / "raw_candidate_manifest.csv", index=False)
    pd.DataFrame(skipped_seeds, columns=("seedpath", "attempt", "reason")).to_csv(
        output / "skipped_seed_manifest.csv", index=False
    )
    summary = {
        "requested_identities": nidentities,
        "completed_identities": completed,
        "seed_attempts": min(len(attempted_sources), completed + len(skipped_seeds)),
        "available_source_images": len(sources),
        "skipped_seeds": len(skipped_seeds),
        "generated_variants_per_identity": candidatesperidentity,
        "source_image_in_final_cluster": False,
        "randomstate": randomstate,
        "min_similarity_raw": min_similarity_raw,
    }
    (output / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    if completed < nidentities:
        raise RuntimeError(
            f"Completed {completed}/{nidentities} identities after exhausting "
            f"all available source images. "
            f"See {output / 'skipped_seed_manifest.csv'} for skipped seeds."
        )
    return str(output)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for standalone generation-stage execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seedsdir", required=True)
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--nidentities", type=int, default=400)
    parser.add_argument(
        "--variantsperidentity",
        "--candidatesperidentity",
        dest="candidatesperidentity",
        type=int,
        default=39,
    )
    parser.add_argument("--ctxid", type=int, default=0)
    parser.add_argument("--randomstate", type=int, default=42)
    parser.add_argument("--minsimilarityraw", type=float, default=0.40)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate_identities(
        args.seedsdir,
        args.outputdir,
        args.nidentities,
        args.candidatesperidentity,
        args.ctxid,
        args.randomstate,
        args.minsimilarityraw,
    )

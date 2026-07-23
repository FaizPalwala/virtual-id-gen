"""
generate_identities.py
Generate varied raw InstantID candidate pools from full-resolution SFHQ seeds.

Each identity receives a deterministic variation plan.  A plan changes only
non-identity attributes (pose, expression, lighting, camera/background) while
the source image remains the sole InstantID identity condition.  Every trial's
plan, prompt, and seed are written to ``raw_candidate_manifest.csv``.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import get_image_paths, normalised_cosine_similarity
from extract_embeddings import (
    get_embedding_and_attributes_robust,
    load_arcface_model,
)

LOGGER = logging.getLogger(__name__)

PHOTOREALISM_PREFIX = (
    "RAW photo, photorealistic DSLR portrait of the same person, realistic "
    "skin texture and pores, realistic hair strands, natural colour grading, "
    "sharp eyes, high detail"
)

NEGATIVE_PROMPT = (
    "painting, illustration, drawing, CGI, 3D render, cartoon, anime, "
    "monochrome, grayscale, sepia, desaturated, waxy skin, airbrushed skin, "
    "lowres, blurry, out of focus, distorted face, malformed face, extra face, "
    "duplicate face, text, watermark, logo"
)


@dataclass(frozen=True)
class VariationSpec:
    """A non-identity rendering instruction for one generated candidate."""

    variation_id: int
    pose: str
    expression: str
    lighting: str
    setting: str
    camera: str

    def prompt(self) -> str:
        """Render the structured plan as a prompt without changing identity traits."""
        return ", ".join(
            (
                PHOTOREALISM_PREFIX,
                self.pose,
                self.expression,
                self.lighting,
                self.setting,
                self.camera,
            )
        )


def build_variation_plan(count: int = 39) -> list[VariationSpec]:
    """Return deterministic, diverse non-identity prompts.

    The first 39 entries form a balanced 13 x 3 design: 13 pose/expression/
    setting compositions crossed with three lighting treatments.  Counts above
    39 repeat the composition plan with an additional deterministic lighting
    cycle; this supports candidate oversampling without a fixed portrait prompt.
    """
    if count <= 0:
        raise ValueError("variantsperidentity must be positive.")
    compositions = [
        (
            "frontal head-and-shoulders pose",
            "neutral relaxed expression",
            "clean dark-blue studio backdrop",
            "85mm portrait lens",
        ),
        (
            "frontal head-and-shoulders pose",
            "gentle closed-mouth smile",
            "clean neutral studio backdrop",
            "85mm portrait lens",
        ),
        (
            "three-quarter view facing left",
            "neutral relaxed expression",
            "softly blurred indoor background",
            "85mm portrait lens",
        ),
        (
            "three-quarter view facing right",
            "gentle closed-mouth smile",
            "softly blurred indoor background",
            "85mm portrait lens",
        ),
        (
            "slight head turn to the left",
            "calm thoughtful expression",
            "subtle dark studio backdrop",
            "85mm portrait lens",
        ),
        (
            "slight head turn to the right",
            "calm thoughtful expression",
            "subtle dark studio backdrop",
            "85mm portrait lens",
        ),
        (
            "upright seated head-and-shoulders pose",
            "neutral relaxed expression",
            "minimal professional office background",
            "85mm portrait lens",
        ),
        (
            "upright seated head-and-shoulders pose",
            "gentle closed-mouth smile",
            "minimal professional office background",
            "85mm portrait lens",
        ),
        (
            "natural candid head-and-shoulders pose",
            "soft relaxed expression",
            "softly blurred outdoor greenery",
            "85mm portrait lens",
        ),
        (
            "looking slightly above the camera",
            "neutral relaxed expression",
            "softly blurred outdoor shade",
            "85mm portrait lens",
        ),
        (
            "looking slightly past the camera",
            "calm thoughtful expression",
            "muted indoor background",
            "85mm portrait lens",
        ),
        (
            "straight-on close portrait",
            "subtle relaxed smile",
            "simple warm-grey studio backdrop",
            "85mm portrait lens",
        ),
        (
            "three-quarter editorial portrait",
            "confident neutral expression",
            "softly blurred editorial interior",
            "85mm portrait lens",
        ),
    ]
    lighting = [
        "soft natural window light",
        "soft professional studio key light with gentle fill",
        "open-shade daylight with balanced exposure",
    ]
    return [
        VariationSpec(
            variation_id=index,
            pose=compositions[(index // len(lighting)) % len(compositions)][0],
            expression=compositions[(index // len(lighting)) % len(compositions)][1],
            setting=compositions[(index // len(lighting)) % len(compositions)][2],
            camera=compositions[(index // len(lighting)) % len(compositions)][3],
            lighting=lighting[index % len(lighting)],
        )
        for index in range(count)
    ]


def generate_identities(
    rawdir: str,
    outputdir: str,
    nidentities: int = 400,
    candidatesperidentity: int = 39,
    ctxid: int = 0,
    randomstate: int = 42,
    min_similarity_raw: float = 0.40,
    instantid_config: dict | None = None,
    max_seed_attempts: int | None = None,
) -> str:
    """Generate candidate pools until exactly ``nidentities`` valid seeds exist.

    Source images are shuffled deterministically, then considered from a pool
    larger than the requested identity count.  A source is skipped if either
    robust ArcFace validation or InstantID seed encoding cannot find a face.
    Generation stops immediately when ``nidentities`` clusters are completed.
    It raises a useful error only when all available sources (or the optional
    ``max_seed_attempts`` cap) are exhausted first.
    """
    from instantid_adapter import InstantIDGeneratorSession

    if nidentities <= 0:
        raise ValueError("nidentities must be positive.")
    if candidatesperidentity <= 0:
        raise ValueError("candidatesperidentity must be positive.")
    if max_seed_attempts is not None and max_seed_attempts <= 0:
        raise ValueError("max_seed_attempts must be positive when provided.")

    instantid_config = instantid_config or {}
    sources = get_image_paths(rawdir)
    if len(sources) < nidentities:
        raise ValueError(
            f"Need at least {nidentities} source images; found {len(sources)}."
        )
    rng = random.Random(randomstate)
    rng.shuffle(sources)
    attempted_sources = sources[:max_seed_attempts] if max_seed_attempts else sources

    output = Path(outputdir)
    candidates_root = output / "candidates"
    rejected_root = output / "rejected"
    candidates_root.mkdir(parents=True, exist_ok=True)
    rejected_root.mkdir(parents=True, exist_ok=True)
    variations = build_variation_plan(candidatesperidentity)
    validation_app = load_arcface_model(ctxid)
    records: list[dict] = []
    skipped_seeds: list[dict] = []
    completed = 0
    session = InstantIDGeneratorSession(
        base_model=instantid_config.get("base_model")
        or "stabilityai/stable-diffusion-xl-base-1.0",
        ip_adapter_scale=float(instantid_config.get("ip_adapter_scale", 0.90)),
        controlnet_conditioning_scale=float(
            instantid_config.get("controlnet_conditioning_scale", 0.80)
        ),
        cache_dir=Path(instantid_config["cache_dir"])
        if instantid_config.get("cache_dir")
        else None,
        require_cuda=bool(instantid_config.get("require_cuda", True)),
    )
    try:
        progress = tqdm(
            attempted_sources, desc="Selecting seeds and generating candidates"
        )
        for attempt_index, seed_path in enumerate(progress, start=1):
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
                session.generate(
                    identity,
                    candidate_path,
                    prompt=variation.prompt(),
                    negative_prompt=NEGATIVE_PROMPT,
                    seed=generation_seed,
                    width=int(instantid_config.get("width", 1024)),
                    height=int(instantid_config.get("height", 1024)),
                    num_inference_steps=int(
                        instantid_config.get("num_inference_steps", 35)
                    ),
                    guidance_scale=float(instantid_config.get("guidance_scale", 5.5)),
                )
                image = cv2.imread(str(candidate_path))
                embedding, _, _ = get_embedding_and_attributes_robust(
                    validation_app, image, ctxid
                )
                similarity = (
                    None
                    if embedding is None
                    else normalised_cosine_similarity(seed_embedding, embedding)
                )
                status = (
                    "accepted_raw"
                    if similarity is not None and similarity >= min_similarity_raw
                    else "rejected_raw"
                )
                stored_path = candidate_path
                if status == "rejected_raw" and candidate_path.exists():
                    stored_path = (
                        rejected_root
                        / f"identity_{cluster_id:03d}_trial_{trial:03d}.png"
                    )
                    candidate_path.replace(stored_path)
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
                        "raw_candidatepath": str(stored_path),
                        "raw_arcface_similarity": similarity,
                        "raw_status": status,
                        **asdict(variation),
                    }
                )
            completed += 1
            progress.set_postfix(
                completed=f"{completed}/{nidentities}", skipped=len(skipped_seeds)
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
        "max_seed_attempts": max_seed_attempts,
        "skipped_seeds": len(skipped_seeds),
        "generated_variants_per_identity": candidatesperidentity,
        "source_image_in_final_cluster": False,
        "randomstate": randomstate,
        "min_similarity_raw": min_similarity_raw,
    }
    (output / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    if completed < nidentities:
        cap = (
            f"the configured max_seed_attempts={max_seed_attempts}"
            if max_seed_attempts
            else "all available source images"
        )
        raise RuntimeError(
            f"Completed {completed}/{nidentities} identities after exhausting {cap}. "
            f"See {output / 'skipped_seed_manifest.csv'} for skipped seeds."
        )
    return str(output)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for standalone generation-stage execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rawdir", required=True)
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
    parser.add_argument(
        "--maxseedattempts",
        type=int,
        default=None,
        help="Optional cap on shuffled raw seeds considered",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate_identities(
        args.rawdir,
        args.outputdir,
        args.nidentities,
        args.candidatesperidentity,
        args.ctxid,
        args.randomstate,
        args.minsimilarityraw,
        max_seed_attempts=args.maxseedattempts,
    )

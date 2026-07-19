"""
generate_identities.py
Generate raw InstantID candidate pools from full-resolution SFHQ seeds.

This stage deliberately does not create the final dataset manifest.  It records
all candidate-level provenance in ``raw_candidate_manifest.csv``; the following
preprocessing stage owns final selection and writes ``identitymanifest.csv``.

Generate varied raw InstantID candidate pools from full-resolution SFHQ seeds.

Each identity receives a deterministic variation plan.  A plan changes only
non-identity attributes (pose, expression, lighting, camera/background) while
the source image remains the sole InstantID identity condition.  Every trial's
plan, prompt, and seed are written to ``raw_candidate_manifest.csv``.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import get_image_paths, normalised_cosine_similarity
from extract_embeddings import get_embedding_and_attributes, load_arcface_model

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


def invoke_generator(
    command_template: str,
    seed_path: Path,
    output_path: Path,
    generation_seed: int,
    variation: VariationSpec,
) -> None:
    """Execute InstantID for one planned variation and require its output file."""
    command = command_template.format(
        idimage=seed_path.resolve(),
        outputpath=output_path.resolve(),
        generationseed=generation_seed,
        prompt=variation.prompt(),
        negativeprompt=NEGATIVE_PROMPT,
    )
    result = subprocess.run(
        shlex.split(command), text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"Generator failed for {seed_path}: {result.stderr[-2000:]}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"Generator did not create a non-empty output at {output_path}"
        )


def generate_identities(
    rawdir: str,
    outputdir: str,
    nidentities: int = 400,
    candidatesperidentity: int = 39,
    generatorcmd: str = "",
    ctxid: int = 0,
    randomstate: int = 42,
    min_similarity_raw: float = 0.40,
) -> str:
    """Generate planned candidate variants for each full-resolution source seed.

    ``candidatesperidentity=39`` means 39 *generated* images per seed.  The
    source image is deliberately not copied into a virtual identity cluster.
    Set the final processed cluster size to at most 39, or generate more than
    39 candidates if post-processing must retain 40 images per cluster.
    """
    required = {"{idimage}", "{outputpath}", "{generationseed}", "{prompt}"}
    placeholders = set(re.findall(r"\{[^}]+\}", generatorcmd))
    if not required.issubset(placeholders):
        raise ValueError(
            "generatorcmd needs {idimage}, {outputpath}, {generationseed}, and {prompt}."
        )
    sources = get_image_paths(rawdir)
    if len(sources) < nidentities:
        raise ValueError(f"Need {nidentities} source images; found {len(sources)}.")
    rng = random.Random(randomstate)
    rng.shuffle(sources)
    output = Path(outputdir)
    candidates_root = output / "candidates"
    rejected_root = output / "rejected"
    candidates_root.mkdir(parents=True, exist_ok=True)
    rejected_root.mkdir(parents=True, exist_ok=True)
    variations = build_variation_plan(candidatesperidentity)
    app = load_arcface_model(ctxid)
    records: list[dict] = []
    for cluster_id, seed_path in enumerate(
        tqdm(sources[:nidentities], desc="Generating varied candidates")
    ):
        seed_image = cv2.imread(str(seed_path))
        seed_embedding, _, _ = get_embedding_and_attributes(app, seed_image)
        if seed_embedding is None:
            raise RuntimeError(f"Seed has no detectable face: {seed_path}")
        cluster_dir = candidates_root / f"identity_{cluster_id:03d}"
        cluster_dir.mkdir(exist_ok=True)
        for trial, variation in enumerate(variations):
            generation_seed = randomstate + cluster_id * candidatesperidentity + trial
            candidate_path = cluster_dir / f"candidate_{trial:03d}.png"
            invoke_generator(
                generatorcmd, seed_path, candidate_path, generation_seed, variation
            )
            image = cv2.imread(str(candidate_path))
            embedding, _, _ = (
                get_embedding_and_attributes(app, image)
                if image is not None
                else (None, None, None)
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
                    rejected_root / f"identity_{cluster_id:03d}_trial_{trial:03d}.png"
                )
                candidate_path.replace(stored_path)
            records.append(
                {
                    "identityid": cluster_id,
                    "clusterid": cluster_id,
                    "trial": trial,
                    "seedpath": str(seed_path),
                    "generationseed": generation_seed,
                    "prompt": variation.prompt(),
                    "negative_prompt": NEGATIVE_PROMPT,
                    "raw_candidatepath": str(stored_path),
                    "raw_arcface_similarity": similarity,
                    "raw_status": status,
                    **asdict(variation),
                }
            )
    pd.DataFrame(records).to_csv(output / "raw_candidate_manifest.csv", index=False)
    summary = {
        "nidentities": nidentities,
        "generated_variants_per_identity": candidatesperidentity,
        "source_image_in_final_cluster": False,
        "randomstate": randomstate,
        "min_similarity_raw": min_similarity_raw,
    }
    (output / "generation_summary.json").write_text(json.dumps(summary, indent=2))
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
    parser.add_argument("--generatorcmd", required=True)
    parser.add_argument("--ctxid", type=int, default=0)
    parser.add_argument("--randomstate", type=int, default=42)
    parser.add_argument("--minsimilarityraw", type=float, default=0.40)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate_identities(
        args.rawdir,
        args.outputdir,
        args.nidentities,
        args.candidatesperidentity,
        args.generatorcmd,
        args.ctxid,
        args.randomstate,
        args.minsimilarityraw,
    )

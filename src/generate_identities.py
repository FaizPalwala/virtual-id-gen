"""
generate_identities.py
Generate raw InstantID candidate pools from full-resolution SFHQ seeds.

This stage deliberately does not create the final dataset manifest.  It records
all candidate-level provenance in ``raw_candidate_manifest.csv``; the following
preprocessing stage owns final selection and writes ``identitymanifest.csv``.
"""
from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import get_image_paths, normalised_cosine_similarity
from extract_embeddings import get_embedding_and_attributes, load_arcface_model

DEFAULT_PROMPTS = [
    "professional headshot, frontal pose, neutral expression, natural lighting",
    "professional portrait, three-quarter view, soft daylight, detailed skin",
    "studio head-and-shoulders portrait, slight smile, clean neutral background",
    "high-quality portrait photograph, candid expression, indoor ambient light",
]


def invoke_generator(
    command_template: str,
    seed_path: Path,
    output_path: Path,
    generation_seed: int,
    prompt: str,
) -> None:
    """Run the configured generator command and require its declared output."""
    command = command_template.format(
        idimage=seed_path.resolve(),
        outputpath=output_path.resolve(),
        generationseed=generation_seed,
        prompt=prompt,
    )
    result = subprocess.run(
        shlex.split(command), text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"Generator failed for {seed_path}: {result.stderr[-2000:]}")
    if not output_path.exists():
        raise RuntimeError(f"Generator did not create {output_path}")


def generate_identities(
    rawdir: str,
    outputdir: str,
    nidentities: int = 400,
    candidatesperidentity: int = 70,
    generatorcmd: str = "",
    ctxid: int = 0,
    randomstate: int = 42,
    min_similarity_raw: float = 0.40,
    prompts: list[str] | None = None,
) -> str:
    """Create raw candidate pools using unprocessed SFHQ source images as seeds."""
    required = {"{idimage}", "{outputpath}", "{generationseed}"}
    if not required.issubset(set(__import__("re").findall(r"\{[^}]+\}", generatorcmd))):
        raise ValueError(
            "generatorcmd must contain {idimage}, {outputpath}, and {generationseed}."
        )
    sources = get_image_paths(rawdir)
    if len(sources) < nidentities:
        raise ValueError(
            f"Need {nidentities} detectable source candidates; found {len(sources)}."
        )
    rng = random.Random(randomstate)
    rng.shuffle(sources)
    output = Path(outputdir)
    candidates = output / "candidates"
    rejected = output / "rejected"
    for directory in (candidates, rejected):
        directory.mkdir(parents=True, exist_ok=True)
    app = load_arcface_model(ctxid)
    records = []
    prompts = prompts or DEFAULT_PROMPTS
    for cluster_id, seed_path in enumerate(
        tqdm(sources[:nidentities], desc="Generating candidate pools")
    ):
        seed_image = cv2.imread(str(seed_path))
        seed_embedding, _, _ = get_embedding_and_attributes(app, seed_image)
        if seed_embedding is None:
            raise RuntimeError(f"Seed has no detectable face: {seed_path}")
        cluster_dir = candidates / f"identity_{cluster_id:03d}"
        cluster_dir.mkdir(exist_ok=True)
        for trial in range(candidatesperidentity):
            generation_seed = randomstate + cluster_id * candidatesperidentity + trial
            prompt = prompts[trial % len(prompts)]
            candidate_path = cluster_dir / f"candidate_{trial:03d}.png"
            invoke_generator(
                generatorcmd, seed_path, candidate_path, generation_seed, prompt
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
            final_path = candidate_path
            if status == "rejected_raw" and candidate_path.exists():
                final_path = (
                    rejected / f"identity_{cluster_id:03d}_trial_{trial:03d}.png"
                )
                candidate_path.replace(final_path)
            records.append(
                {
                    "identityid": cluster_id,
                    "clusterid": cluster_id,
                    "trial": trial,
                    "seedpath": str(seed_path),
                    "generationseed": generation_seed,
                    "prompt": prompt,
                    "raw_candidatepath": str(final_path),
                    "raw_arcface_similarity": similarity,
                    "raw_status": status,
                }
            )
    pd.DataFrame(records).to_csv(output / "raw_candidate_manifest.csv", index=False)
    (output / "generation_summary.json").write_text(
        json.dumps(
            {
                "nidentities": nidentities,
                "candidates_per_identity": candidatesperidentity,
                "randomstate": randomstate,
                "min_similarity_raw": min_similarity_raw,
            },
            indent=2,
        )
    )
    return str(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rawdir", required=True)
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--nidentities", type=int, default=400)
    parser.add_argument("--candidatesperidentity", type=int, default=70)
    parser.add_argument("--generatorcmd", required=True)
    parser.add_argument("--ctxid", type=int, default=0)
    parser.add_argument("--randomstate", type=int, default=42)
    parser.add_argument("--minsimilarityraw", type=float, default=0.40)
    args = parser.parse_args()
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

"""Hydra orchestrator for raw seed -> candidates -> final crop -> dataset."""
from __future__ import annotations
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run selected independent pipeline stages in their required order."""
    nidentities = cfg.dataset.nidentities
    forget_steps = cfg.dataset.forget_steps
    forget_pct = cfg.dataset.forget_pct

    nforget = int(nidentities * forget_pct)
    holdout_frac = cfg.dataset.holdout_frac
    min_holdout = cfg.dataset.min_holdout

    # Clamp: every step must have ≥ 1 identity.
    if nforget < forget_steps:
        forget_steps = max(5, nforget)

    if nforget >= nidentities:
        raise ValueError(
            f"forget ({nforget}) >= nidentities ({nidentities})"
        )

    # Minimum size guards for download and build (generate handles sharding).
    if nidentities < 100 and (cfg.steps.download or cfg.steps.build):
        raise ValueError(f"nidentities must be ≥ 100 (got {nidentities})")
    if forget_steps < 5 and cfg.steps.build:
        raise ValueError(f"forget_steps must be ≥ 5 (got {forget_steps})")

    root = Path(cfg.dataset.dataroot)
    seeds = root / "seeds"
    identities = root / "identities"
    processed = root / "processed"
    embeddings = root / "embeddings"
    dataset = root / "dataset"
    if cfg.steps.download:
        from download import download_sfhq

        download_sfhq(
            cfg.dataset.part,
            str(seeds),
            num_images=nidentities,
            skip_download=cfg.dataset.get("skip_download", False),
        )
    if cfg.steps.generate:
        from generate_identities import generate_identities

        generate_identities(
            str(seeds),
            str(identities),
            nidentities,
            cfg.dataset.candidatesperidentity,
            cfg.pipeline.ctxid,
            cfg.dataset.seed,
            cfg.pipeline.min_similarity_raw,
            dict(cfg.pipeline.instantid),
        )
    if cfg.steps.preprocess:
        from preprocess import preprocess_identity_candidates

        preprocess_identity_candidates(
            str(identities),
            str(processed),
            cfg.dataset.imagesperidentity,
            cfg.pipeline.imgsize,
            cfg.pipeline.blurthreshold,
            cfg.pipeline.confthreshold,
            cfg.pipeline.ctxid,
        )
    if cfg.steps.extract:
        from extract_embeddings import extract_embeddings

        # candidate_manifest is inferred by extract_embeddings (lives next to
        # identities/candidates by pipeline convention).
        extract_embeddings(
            str(identities / "candidates"),
            str(embeddings),
            cfg.pipeline.ctxid,
        )
    if cfg.steps.build:
        from build_dataset import build_dataset

        LOGGER.info(
            "Dataset created at %s",
            build_dataset(
                str(processed),
                str(embeddings),
                str(dataset),
                nforget,
                forget_steps,
                cfg.dataset.seed,
                cfg.dataset.imagesperidentity,
                holdout_frac,
                min_holdout,
            ),
        )

        # Build imbalanced variant for unlearning stress-testing.
        from build_dataset import build_imbalanced_dataset

        LOGGER.info(
            "Imbalanced dataset created at %s",
            build_imbalanced_dataset(
                str(processed),
                str(embeddings),
                str(dataset),
                nforget,
                forget_steps,
                cfg.dataset.seed,
                holdout_frac,
                min_holdout,
                cfg.dataset.imagesperidentity,
                cfg.dataset.candidatesperidentity,
            ),
        )

        # Build full-resolution 1024×1024 candidate dataset (max-size,
        # no splits — plain embedding mapping for general-purpose release).
        from build_dataset import build_raw_dataset

        LOGGER.info(
            "Raw dataset created at %s",
            build_raw_dataset(
                str(root),
                str(embeddings),
                str(dataset),
                cfg.dataset.seed,
            ),
        )


if __name__ == "__main__":
    main()

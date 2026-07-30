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
    ntest = int(nidentities * cfg.dataset.test_pct)

    # Clamp: every step must have ≥ 1 identity.
    if nforget < forget_steps:
        forget_steps = max(5, nforget)

    if nforget + ntest >= nidentities:
        raise ValueError(
            f"forget ({nforget}) + test ({ntest}) >= nidentities ({nidentities})"
        )

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
            cfg.pipeline.min_similarity_final,
            cfg.pipeline.ctxid,
        )
    if cfg.steps.extract:
        from extract_embeddings import extract_embeddings

        extract_embeddings(
            str(processed / "images"), str(embeddings), cfg.pipeline.ctxid
        )
    if cfg.steps.build:
        # Minimum dataset size guards (only enforced during build).
        if nidentities < 100:
            raise ValueError(f"nidentities must be ≥ 100 (got {nidentities})")
        if forget_steps < 5:
            raise ValueError(f"forget_steps must be ≥ 5 (got {forget_steps})")

        from build_dataset import build_dataset

        LOGGER.info(
            "Dataset created at %s",
            build_dataset(
                str(processed),
                str(embeddings),
                str(dataset),
                nforget,
                ntest,
                forget_steps,
                cfg.dataset.seed,
            ),
        )


if __name__ == "__main__":
    main()

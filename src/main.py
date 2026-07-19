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
    if cfg.dataset.nforget + cfg.dataset.ntest >= cfg.dataset.nidentities:
        raise ValueError("Forget and test splits leave no retain identities.")
    root = Path(cfg.dataset.dataroot)
    raw = root / "raw"
    identities = root / "identities"
    processed = root / "processed"
    embeddings = root / "embeddings"
    dataset = root / "dataset"
    if cfg.steps.download:
        from download import download_sfhq

        download_sfhq(cfg.dataset.part, str(raw))
    if cfg.steps.generate:
        from generate_identities import generate_identities

        generate_identities(
            str(raw),
            str(identities),
            cfg.dataset.nidentities,
            cfg.dataset.candidatesperidentity,
            cfg.pipeline.generatorcmd,
            cfg.pipeline.ctxid,
            cfg.dataset.seed,
            cfg.pipeline.min_similarity_raw,
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
        from build_dataset import build_dataset

        LOGGER.info(
            "Dataset created at %s",
            build_dataset(
                str(processed),
                str(embeddings),
                str(dataset),
                cfg.dataset.nforget,
                cfg.dataset.ntest,
                cfg.dataset.seed,
            ),
        )


if __name__ == "__main__":
    main()

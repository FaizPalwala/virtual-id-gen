"""
main.py

Orchestrator for the SFHQ dataset Virtual Identity Pipeline.
Supports modular execution for HPC job scheduling.
"""

import logging
import time
from pathlib import Path
from typing import Callable, Any

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def run_step(label: str, fn: Callable, **kwargs: Any) -> Any:
    logger.info("=" * 70)
    logger.info(f"STARTING: {label}")
    logger.info("=" * 70)

    start_time = time.time()
    value = fn(**kwargs)
    elapsed = time.time() - start_time

    logger.info(f"[DONE] {label} completed in {elapsed:.1f}s")
    return value


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.dataset.n_forget + cfg.dataset.n_test >= cfg.dataset.n_identities:
        logger.error(
            "n_forget + n_test must be strictly less than n_identities.")
        raise ValueError("Invalid split configuration.")

    root = Path(cfg.dataset.data_root)
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    identities_dir = root / "identities"
    embeddings_dir = root / "embeddings"
    dataset_dir = root / "dataset"

    if cfg.steps.download:
        from download import download_sfhq
        run_step("STEP 1: Download", 
                 download_sfhq, 
                 part=1, 
                 output_dir=str(raw_dir))

    if cfg.steps.preprocess:
        from preprocess import preprocess_dataset
        run_step("STEP 2: Align and Preprocess", 
                 preprocess_dataset, 
                 input_dir=str(raw_dir), 
                 output_dir=str(processed_dir), 
                 max_images=cfg.pipeline.max_images)

    if cfg.steps.generate:
        from generate_identities import generate_identities
        # adapter_path = Path(__file__).with_name("dcface_adapter.py")
        # Switch the adapter target to InstantID
        adapter_path = Path(__file__).with_name("instantid_adapter.py")
        command_template = f"python {adapter_path} --dcface_root {Path(cfg.pipeline.dcface_root).resolve()} --id_image {{id_image}} --style_image {{style_image}} --output_path {{output_path}}"
        run_step("STEP 3: Generate Identities", 
                 generate_identities, 
                 processed_dir=str(processed_dir), 
                 output_dir=str(identities_dir), 
                 n_identities=cfg.dataset.n_identities, 
                 images_per_identity=cfg.dataset.images_per_identity,
                 candidates_per_identity=cfg.dataset.candidates_per_identity, 
                 generator_cmd=cfg.pipeline.generator_cmd, 
                 ctx_id=cfg.pipeline.ctx_id, 
                 random_state=cfg.dataset.seed, 
                 visualise=not cfg.pipeline.skip_vis, 
                 min_similarity=cfg.pipeline.min_similarity, 
                 blur_threshold=cfg.pipeline.blur_threshold)

    if cfg.steps.extract:
        from extract_embeddings import extract_embeddings
        run_step("STEP 4: Extract Attributes", 
                 extract_embeddings, 
                 input_dir=str(identities_dir / "images"), 
                 output_dir=str(embeddings_dir), ctx_id=cfg.pipeline.ctx_id)

    if cfg.steps.build:
        from build_dataset import build_dataset
        dataset_csv = run_step("STEP 5: Build Final Dataset", 
                               build_dataset, 
                               identity_dir=str(identities_dir), 
                               embeddings_dir=str(embeddings_dir), 
                               output_dir=str(dataset_dir), 
                               n_forget=cfg.dataset.n_forget, 
                               n_test=cfg.dataset.n_test, 
                               random_state=cfg.dataset.seed)
        logger.info(f"\n[PIPELINE COMPLETE] Dataset generated at: {dataset_csv}")


if __name__ == "__main__":
    main()

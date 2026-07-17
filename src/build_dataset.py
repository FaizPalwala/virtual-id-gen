"""
build_dataset.py

Constructs the final Phase 2 dataset mapping by merging generated identity 
manifests with extracted embedding attributes. Applies the train/test/forget splits.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AGE_GROUPS = {
    0: 'Young (0-24)', 
    1: 'Adult (25-44)', 
    2: 'Middle-Aged (45-64)', 
    3: 'Senior (65+)'
}

def build_dataset(
    identity_dir: str, 
    embeddings_dir: str, 
    output_dir: str, 
    n_forget: int = 40, 
    n_test: int = 60, 
    random_state: int = 42
) -> str:
    """
    Builds the final sfhq_dataset.csv file.
    
    Args:
        identity_dir (str): Path to the generated identities manifest.
        embeddings_dir (str): Path to the extracted .npy arrays.
        output_dir (str): Path to save the final dataset.
        n_forget (int): Number of identities to assign to the forget split.
        n_test (int): Number of identities to assign to the test split.
        random_state (int): Seed for reproducible shuffles.
        
    Returns:
        str: Path to the generated CSV file.
    """
    manifest_path = Path(identity_dir) / 'identity_manifest.csv'
    emb_path = Path(embeddings_dir)
    
    logger.info("Loading manifests and attributes...")
    manifest = pd.read_csv(manifest_path)
    
    attributes = pd.DataFrame({
        'image_path': np.load(emb_path / 'image_paths.npy').astype(str),
        'age_group': np.load(emb_path / 'age_groups.npy'),
        'age': np.load(emb_path / 'ages.npy'),
        'gender': np.load(emb_path / 'genders.npy')
    })
    
    # Merge on image_path
    df = manifest[['image_path', 'cluster_id']].merge(
        attributes, on='image_path', how='inner', validate='one_to_one'
    )
    
    if len(df) != len(manifest):
        raise RuntimeError(f"Attribute extraction is missing {len(manifest) - len(df)} generated images.")
        
    unique_ids = sorted(df.cluster_id.unique())
    if n_forget + n_test >= len(unique_ids):
        raise ValueError("Invalid split sizes: n_forget + n_test exceeds total identities.")
        
    # Reproducible shuffle
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_ids)
    
    forget_ids = unique_ids[:n_forget]
    test_ids = unique_ids[n_forget : n_forget + n_test]
    retain_ids = unique_ids[n_forget + n_test :]
    
    split_map = {**{i: 'forget' for i in forget_ids},
                 **{i: 'test' for i in test_ids},
                 **{i: 'retain' for i in retain_ids}}
                 
    steps_map = {idx: step for step, idx in enumerate(forget_ids)}
    
    df['split'] = df.cluster_id.map(split_map)
    df['forget_step'] = df.cluster_id.map(steps_map).fillna(-1).astype(int)
    
    # Filter out unknown age groups
    df = df[df.age_group != -1].copy()
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    csv_file = out_path / 'sfhq_dataset.csv'
    df[['image_path', 'cluster_id', 'age_group', 'split', 'forget_step']].to_csv(csv_file, index=False)
    
    _save_summary(df, unique_ids, forget_ids, test_ids, retain_ids, n_forget, out_path)
    
    logger.info(f"Final dataset built successfully: {csv_file}")
    return str(csv_file)

def _save_summary(
    df: pd.DataFrame, 
    all_ids: list, 
    forget_ids: list, 
    test_ids: list, 
    retain_ids: list, 
    n_forget: int, 
    out_path: Path
) -> None:
    """Helper function to compile and save dataset metadata."""
    summary = {
        'total_images': len(df),
        'n_clusters': len(all_ids),
        'identity_source': 'identity-conditioned synthetic variants',
        'split_sizes': {k: int(v) for k, v in df.groupby('split').size().items()},
        'n_forget_steps': n_forget,
        'forget_clusters': [int(x) for x in forget_ids],
        'test_clusters': [int(x) for x in test_ids],
        'retain_clusters': [int(x) for x in retain_ids],
        'label_distribution_retain': {
            AGE_GROUPS[int(k)]: int(v) 
            for k, v in df[df.split == 'retain'].age_group.value_counts().items()
        }
    }
    
    (out_path / 'dataset_summary.json').write_text(json.dumps(summary, indent=2))
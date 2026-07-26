#!/bin/bash
# ==========================================
# hpc_merge.sh — Merge 4 shard outputs into unified identities/
# ==========================================
#SBATCH --job-name=msc_merge
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ------------------------------------------------------------------
# Reads $DATA_DIR/shard_{0..3}/ and writes:
#   $DATA_DIR/merged/identities/raw_candidate_manifest.csv
#   $DATA_DIR/merged/identities/candidates/
#
# Subsequent steps point dataset.dataroot=$DATA_DIR/merged so that
# identities/ resolves to the merged output.
# ------------------------------------------------------------------

# ==========================================
# 1. Environment Setup
# ==========================================
module purge
module load miniforge
conda activate data_gen

# ==========================================
# 2. Path Variables
# ==========================================
REPO_DIR="$SLURM_SUBMIT_DIR"
PARENT_DIR=$(dirname "$REPO_DIR")
DATA_DIR="/scratch/$USER/datagen/data"

# ==========================================
# 3. Run Merge
# ==========================================
echo "[$(date)] Merging shard outputs..."

SHARD_ROOT="$DATA_DIR"                          # where shard_0/ ... shard_3/ live
MERGED_DIR="$DATA_DIR/merged/identities"        # unified identities/ directory

mkdir -p "$MERGED_DIR"

python "$REPO_DIR/scripts/merge_shards.py" \
    --outputdir "$SHARD_ROOT" \
    --mergeddir "$MERGED_DIR" \
    --shardcount 4

EXIT_CODE=$?
echo "[$(date)] Merge finished (exit $EXIT_CODE)"
exit $EXIT_CODE

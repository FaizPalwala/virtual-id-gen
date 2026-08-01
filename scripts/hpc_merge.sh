#!/bin/bash
# ==========================================
# hpc_merge.sh — Merge 12 shard outputs into unified identities/
# ==========================================
#SBATCH --job-name=msc_merge
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ------------------------------------------------------------------
# Reads $DATA_DIR/shard_{0..11}/ and writes:
#   $DATA_DIR/identities/raw_candidate_manifest.csv
#   $DATA_DIR/identities/candidates/
#
# Subsequent steps use dataset.dataroot=$DATA_DIR.
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

SHARD_COUNT=12
IDENTITIES_PER_SHARD=50

# ==========================================
# 3. Run Merge
# ==========================================
echo "[$(date)] Merging $SHARD_COUNT shard outputs ($IDENTITIES_PER_SHARD ids each)..."

SHARD_ROOT="$DATA_DIR"
MERGED_DIR="$DATA_DIR/identities"

mkdir -p "$MERGED_DIR"

python "$REPO_DIR/scripts/merge_shards.py" \
    --outputdir "$SHARD_ROOT" \
    --identitiesdir "$MERGED_DIR" \
    --seedsdir "$DATA_DIR/seeds" \
    --shardcount "$SHARD_COUNT" \
    --identitiespershard "$IDENTITIES_PER_SHARD"

EXIT_CODE=$?
echo "[$(date)] Merge finished (exit $EXIT_CODE)"
exit $EXIT_CODE

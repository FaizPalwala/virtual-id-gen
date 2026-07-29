#!/bin/bash
# ==========================================
# hpc_build.sh — Phase 4: Assemble Final Dataset CSV
# ==========================================
#SBATCH --job-name=msc_build
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ------------------------------------------------------------------
# Pure-CPU step: merges the final identity manifest with extracted
# embeddings/attributes and applies train/test/forget splits.
#
# Input:  $DATA_DIR/merged/processed/identitymanifest.csv
#         $DATA_DIR/merged/embeddings/
# Output: $DATA_DIR/merged/dataset/sfhqdataset.csv
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
# 3. Run Build
# ==========================================
echo "[$(date)] Assembling final dataset..."

cd "$REPO_DIR/src"

python main.py --config-name step5_build \
    dataset.dataroot="$DATA_DIR" \
    > "$REPO_DIR/logs/build_${SLURM_JOB_ID}.log" 2>&1

EXIT_CODE=$?
echo "[$(date)] Build finished (exit $EXIT_CODE)"
exit $EXIT_CODE

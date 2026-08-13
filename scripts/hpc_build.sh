#!/bin/bash
# ==========================================
# hpc_build.sh — Phase 4: Assemble Final Dataset CSV
# ==========================================
#SBATCH --job-name=msc_build
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ------------------------------------------------------------------
# Pure-CPU step: merges the final identity manifest with extracted
# embeddings/attributes and applies train/test/forget splits.
#
# Input:  $DATA_DIR/processed/identitymanifest.csv
#         $DATA_DIR/embeddings/
# Output: $DATA_DIR/dataset/dataset.csv
#
# ALSO converts raw candidates to JPEG q95 (4:4:4) IN PLACE — UNCONDITIONAL
# inside the raw build (verify-then-delete, RELEASE_TODO Phase D2 P1-4) so
# the off-node transfer bundle is ~22 GB not ~100 GB of PNG.  8 CPUs for
# the parallel PNG→JPEG workers (memory-bound, 4 workers optimal).
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

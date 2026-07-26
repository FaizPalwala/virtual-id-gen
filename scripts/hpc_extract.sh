#!/bin/bash
# ==========================================
# hpc_extract.sh — Phase 3: Extract ArcFace Embeddings & Demographics
# ==========================================
#SBATCH --job-name=msc_extract
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ------------------------------------------------------------------
# Runs extract_embeddings on the final preprocessed images:
#   - ArcFace feature vectors (512-d identity embeddings)
#   - Demographic proxy attributes (age group, gender)
#
# Input:  $DATA_DIR/merged/processed/images/
# Output: $DATA_DIR/merged/embeddings/
# ------------------------------------------------------------------

# ==========================================
# 1. Environment Setup
# ==========================================
module purge
module load miniforge
module load cuda
conda activate data_gen

# ==========================================
# 2. Path Variables & Caching
# ==========================================
REPO_DIR="$SLURM_SUBMIT_DIR"
PARENT_DIR=$(dirname "$REPO_DIR")
DATA_DIR="/scratch/$USER/datagen/data"
CACHE_DIR="$PARENT_DIR/model_cache"

export INSIGHTFACE_HOME="$CACHE_DIR/insightface"
mkdir -p "$INSIGHTFACE_HOME"

# ==========================================
# 3. GPU Preflight
# ==========================================
echo "[$(date)] Running GPU Preflight..."
bash "$REPO_DIR/scripts/gpu_preflight.sh"

# ==========================================
# 4. Run Extract
# ==========================================
echo "[$(date)] Starting embedding extraction..."

cd "$REPO_DIR/src"

python main.py --config-name step4_extract \
    dataset.dataroot="$DATA_DIR/merged" \
    > "$REPO_DIR/logs/extract_${SLURM_JOB_ID}.log" 2>&1

EXIT_CODE=$?
echo "[$(date)] Extract finished (exit $EXIT_CODE)"
exit $EXIT_CODE

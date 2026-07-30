#!/bin/bash
# ==========================================
# hpc_preprocess.sh — Phase 2: Align, Quality-Filter & Select Final Images
# ==========================================
#SBATCH --job-name=msc_preprocess
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ------------------------------------------------------------------
# Runs preprocess_identity_candidates on the merged identities:
#   - MTCNN face detection + alignment
#   - Sharpness filtering (Laplacian variance)
#   - ArcFace similarity gating (keeps images close to the source identity)
#   - Final per-identity selection (keeps the best N images)
#
# Input:  $DATA_DIR/merged/identities/
# Output: $DATA_DIR/merged/processed/
# ------------------------------------------------------------------

# ==========================================
# 1. Environment Setup
# ==========================================
module purge
module load miniforge
module load cuda/12.6.2
conda activate data_gen
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ==========================================
# 2. Path Variables & Caching
# ==========================================
REPO_DIR="$SLURM_SUBMIT_DIR"
PARENT_DIR=$(dirname "$REPO_DIR")
DATA_DIR="/scratch/$USER/datagen/data"
CACHE_DIR="$PARENT_DIR/model_cache"

# Thread pinning
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Model cache (InsightFace models for ArcFace + MTCNN)
export INSIGHTFACE_HOME="$CACHE_DIR/insightface"
mkdir -p "$INSIGHTFACE_HOME"

# ==========================================
# 3. GPU Preflight
# ==========================================
echo "[$(date)] Running GPU Preflight..."
bash "$REPO_DIR/scripts/gpu_preflight.sh"

# ==========================================
# 4. Run Preprocess
# ==========================================
echo "[$(date)] Starting preprocessing (face alignment + quality filtering)..."

cd "$REPO_DIR/src"

python main.py --config-name step3_preprocess \
    dataset.dataroot="$DATA_DIR" \
    > "$REPO_DIR/logs/preprocess_${SLURM_JOB_ID}.log" 2>&1

EXIT_CODE=$?
echo "[$(date)] Preprocess finished (exit $EXIT_CODE)"
exit $EXIT_CODE

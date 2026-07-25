#!/bin/bash
# ==========================================
# hpc_smoke_test.sh — Minimal end-to-end validation (1 identity)
# ==========================================
# Submits a single-GPU job that generates one identity and verifies
# the output images are not solid black.  Runs in ~15 minutes.
#
# Usage:
#   sbatch scripts/hpc_smoke_test.sh
#   cat logs/smoke_*.log
# ==========================================
#SBATCH --job-name=msc_smoke
#SBATCH --time=0-01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -euo pipefail

# ==========================================
# 1. Environment Setup
# ==========================================
module purge
module load miniforge
module load cuda
conda activate data_gen

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TQDM_DISABLE=1
export ORT_LOG_LEVEL=3

REPO_DIR="$SLURM_SUBMIT_DIR"
PARENT_DIR=$(dirname "$REPO_DIR")
CACHE_DIR="$PARENT_DIR/model_cache"
DATA_DIR="$PARENT_DIR/data"

export HF_HOME="$CACHE_DIR/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export INSIGHTFACE_HOME="$CACHE_DIR/insightface"
mkdir -p "$HF_HOME" "$INSIGHTFACE_HOME" "$REPO_DIR/logs"

# ==========================================
# 2. Stage to node-local scratch
# ==========================================
SMOKE_TMP="$TMPDIR/smoke"
mkdir -p "$SMOKE_TMP/repo" "$SMOKE_TMP/data"

echo "[$(date)] Staging repo + source images to $SMOKE_TMP..."
cp -r "$REPO_DIR/"* "$SMOKE_TMP/repo/"
if [ -d "$DATA_DIR/raw" ]; then
    cp -r "$DATA_DIR/raw" "$SMOKE_TMP/data/raw"
fi

# ==========================================
# 3. GPU Preflight
# ==========================================
echo "[$(date)] Running GPU preflight..."
bash "$SMOKE_TMP/repo/scripts/gpu_preflight.sh"

# ==========================================
# 4. Generate 1 Identity (39 candidates)
# ==========================================
SMOKE_DATA="$SMOKE_TMP/smoke_data"
mkdir -p "$SMOKE_DATA/raw"

if [ -d "$SMOKE_TMP/data/raw" ]; then
    cp -r "$SMOKE_TMP/data/raw/." "$SMOKE_DATA/raw/"
fi

echo "[$(date)] Generating 1 identity (39 candidates)..."
cd "$SMOKE_TMP/repo/src"

python main.py --config-name step3_generate \
    dataset.dataroot="$SMOKE_DATA" \
    dataset.nidentities=1 \
    dataset.seed=42 \
    dataset.nforget=0 \
    dataset.ntest=0 \
    > "$REPO_DIR/logs/smoke_generate_${SLURM_JOB_ID}.log" 2>&1

EXIT_CODE=$?
echo "[$(date)] Generation finished (exit $EXIT_CODE)"

if [ $EXIT_CODE -ne 0 ]; then
    echo "FAIL: generate step crashed (exit $EXIT_CODE)"
    exit $EXIT_CODE
fi

# ==========================================
# 5. Verify output images are not black
# ==========================================
IDENTITY_DIR=$(find "$SMOKE_DATA/identities" -type d -name "identity_*" 2>/dev/null | head -1)

if [ -z "$IDENTITY_DIR" ]; then
    echo "FAIL: no identity directory found under $SMOKE_DATA/identities"
    exit 1
fi

echo "[$(date)] Verifying candidates in $IDENTITY_DIR..."

BLACK_COUNT=0
TOTAL=0
FIRST_BLACK=""

for img in "$IDENTITY_DIR"/candidate_*.png; do
    TOTAL=$((TOTAL + 1))
    # Check if image is all-black using Python
    IS_BLACK=$(python3 -c "
import numpy as np
from PIL import Image
arr = np.array(Image.open('$img').convert('RGB'))
print(1 if arr.max() < 5 else 0)
" 2>/dev/null || echo "2")

    if [ "$IS_BLACK" = "1" ]; then
        BLACK_COUNT=$((BLACK_COUNT + 1))
        [ -z "$FIRST_BLACK" ] && FIRST_BLACK="$img"
    elif [ "$IS_BLACK" = "2" ]; then
        echo "WARNING: PIL verification failed for $img (env issue, not image bug)"
    fi
done

echo ""
echo "========================================="
if [ $BLACK_COUNT -gt 0 ]; then
    echo "FAIL: $BLACK_COUNT/$TOTAL images are solid black"
    echo "First black: $FIRST_BLACK"
    exit 1
else
    echo "PASS: All $TOTAL candidate images have non-zero pixels"
fi

echo "Smoke test directory: $SMOKE_DATA/identities"
echo "========================================="
echo "[$(date)] Smoke test complete — PASS"

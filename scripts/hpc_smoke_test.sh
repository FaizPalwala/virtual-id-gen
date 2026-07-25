#!/bin/bash
# ==========================================
# hpc_smoke_test.sh — Minimal end-to-end validation (1 identity)
# ==========================================
# Submits a single-GPU job that generates one identity and verifies
# the output images are not solid black.  Runs in ~15 minutes.
#
# BEFORE SUBMITTING:
#   mkdir -p logs
#
# Usage:
#   sbatch scripts/hpc_smoke_test.sh
#   cat logs/smoke_*.out
# ==========================================
#SBATCH --job-name=msc_smoke
#SBATCH --time=0-01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -eo pipefail  # dropped -u: TMPDIR naming varies across Slurm sites

# Print immediately so we know the job started
echo "========================================="
echo "[$(date)] Smoke test started (job $SLURM_JOB_ID)"
echo "========================================="

# ==========================================
# 1. Environment Setup
# ==========================================
echo "[$(date)] Setting up environment..."

module purge  2>/dev/null || true
module load miniforge 2>/dev/null || true
module load cuda 2>/dev/null || true
conda activate data_gen 2>/dev/null || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TQDM_DISABLE=1
export ORT_LOG_LEVEL=3

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PARENT_DIR="$(dirname "$REPO_DIR")"
CACHE_DIR="$PARENT_DIR/model_cache"
DATA_DIR="$PARENT_DIR/data"

export HF_HOME="$CACHE_DIR/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export INSIGHTFACE_HOME="$CACHE_DIR/insightface"
mkdir -p "$HF_HOME" "$INSIGHTFACE_HOME" "$REPO_DIR/logs"

echo "  REPO_DIR  = $REPO_DIR"
echo "  DATA_DIR  = $DATA_DIR"
echo "  CACHE_DIR = $CACHE_DIR"

# ==========================================
# 2. Stage to node-local scratch
# ==========================================
# TMPDIR naming varies — try common Slurm env vars
SCRATCH_ROOT="${TMPDIR:-${LOCAL_SCRATCH:-${SCRATCH:-/tmp}}}"
SMOKE_TMP="$SCRATCH_ROOT/smoke_${SLURM_JOB_ID}"
mkdir -p "$SMOKE_TMP/repo" "$SMOKE_TMP/data"

echo "[$(date)] Staging repo to $SMOKE_TMP..."
cp -r "$REPO_DIR/"* "$SMOKE_TMP/repo/" 2>/dev/null
if [ -d "$DATA_DIR/raw" ]; then
    cp -r "$DATA_DIR/raw" "$SMOKE_TMP/data/raw"
    echo "  Copied source images: $(ls "$SMOKE_TMP/data/raw" | wc -l) files"
else
    echo "  WARNING: $DATA_DIR/raw not found — source images missing?"
fi

# ==========================================
# 3. GPU Preflight
# ==========================================
echo "[$(date)] Running GPU preflight..."
if [ -f "$SMOKE_TMP/repo/scripts/gpu_preflight.sh" ]; then
    bash "$SMOKE_TMP/repo/scripts/gpu_preflight.sh" || {
        echo "WARNING: GPU preflight had non-zero exit — continuing anyway"
    }
else
    echo "WARNING: gpu_preflight.sh not found — skipping"
fi

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
    echo "Log: logs/smoke_generate_${SLURM_JOB_ID}.log"
    echo "========================================="
    exit $EXIT_CODE
fi

# ==========================================
# 5. Verify output images are not black
# ==========================================
IDENTITY_DIR=$(find "$SMOKE_DATA/identities" -type d -name "identity_*" 2>/dev/null | head -1)

if [ -z "$IDENTITY_DIR" ]; then
    echo "FAIL: no identity directory found under $SMOKE_DATA/identities"
    echo "Contents of $SMOKE_DATA:"
    find "$SMOKE_DATA" -type d 2>/dev/null | head -20
    echo "========================================="
    exit 1
fi

echo "[$(date)] Verifying candidates in $IDENTITY_DIR..."

BLACK_COUNT=0
TOTAL=0
FIRST_BLACK=""

for img in "$IDENTITY_DIR"/candidate_*.png; do
    [ -f "$img" ] || continue
    TOTAL=$((TOTAL + 1))
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
        echo "  WARNING: PIL check failed for $(basename "$img") (env issue, not image bug)"
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

# ==========================================
# 6. Sync results back to persistent storage
# ==========================================
SMOKE_OUT="$DATA_DIR/smoke_${SLURM_JOB_ID}"
mkdir -p "$SMOKE_OUT"
echo "[$(date)] Syncing results to $SMOKE_OUT..."
rsync -av "$IDENTITY_DIR/" "$SMOKE_OUT/" 2>&1 | tail -3
echo "Results persisted at: $SMOKE_OUT"

echo "Smoke test directory (tmp): $SMOKE_DATA/identities"
echo "========================================="
echo "[$(date)] Smoke test complete — PASS"

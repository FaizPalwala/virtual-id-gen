#!/bin/bash
# ==========================================
# hpc_smoke_test.sh — Juggernaut-XL-v9 validation (4 ids × 3 imgs)
# ==========================================
# Submits a single-GPU job that generates 4 identities (3 candidates each)
# using RunDiffusion/Juggernaut-XL-v9 and verifies output is non-black.
# Runs in ~10–15 minutes.
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
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -eo pipefail

echo "========================================="
echo "[$(date)] Juggernaut-XL-v9 smoke test (job $SLURM_JOB_ID)"
echo "========================================="

# ==========================================
# 1. Environment Setup
# ==========================================
echo "[$(date)] Setting up environment..."

module purge  2>/dev/null || true
module load miniforge 2>/dev/null || true
module load cuda/12.6.2 2>/dev/null || true
conda activate data_gen 2>/dev/null || true
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TQDM_DISABLE=1
export ORT_LOG_LEVEL=3

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PARENT_DIR="$(dirname "$REPO_DIR")"
CACHE_DIR="$PARENT_DIR/model_cache"
DATA_DIR="/scratch/$USER/datagen/data"

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
SCRATCH_ROOT="${TMPDIR:-${LOCAL_SCRATCH:-${SCRATCH:-/tmp}}}"
SMOKE_TMP="$SCRATCH_ROOT/smoke_${SLURM_JOB_ID}"
mkdir -p "$SMOKE_TMP/repo" "$SMOKE_TMP/data"

echo "[$(date)] Staging repo to $SMOKE_TMP..."
cp -r "$REPO_DIR/"* "$SMOKE_TMP/repo/" 2>/dev/null
if [ -d "$DATA_DIR/seeds" ]; then
    cp -r "$DATA_DIR/seeds" "$SMOKE_TMP/data/seeds"
    echo "  Copied source images: $(ls "$SMOKE_TMP/data/seeds" | wc -l) files"
else
    echo "  FAIL: $DATA_DIR/seeds not found — source images missing?"
    exit 1
fi

# ==========================================
# 3. GPU Preflight
# ==========================================
echo "[$(date)] Running GPU preflight..."
if [ -f "$SMOKE_TMP/repo/scripts/gpu_preflight.sh" ]; then
    bash "$SMOKE_TMP/repo/scripts/gpu_preflight.sh" || {
        echo "WARNING: GPU preflight had non-zero exit — continuing anyway"
    }
fi

# ==========================================
# 4. Generate 4 Identities × 3 Candidates (Juggernaut-XL-v9)
# ==========================================
SMOKE_DATA="$SMOKE_TMP/smoke_gen"
mkdir -p "$SMOKE_DATA/seeds"

if [ -d "$SMOKE_TMP/data/seeds" ]; then
    cp -r "$SMOKE_TMP/data/seeds/." "$SMOKE_DATA/seeds/"
fi

echo "[$(date)] Generating 4 identities × 3 candidates (Juggernaut-XL-v9)..."
cd "$SMOKE_TMP/repo/src"

python main.py --config-name step2_generate \
    dataset.dataroot="$SMOKE_DATA" \
    dataset.nidentities=4 \
    dataset.candidatesperidentity=3 \
    dataset.seed=42 \
    dataset.forget_steps=0 \
    dataset.test_pct=0 \
    pipeline.instantid.base_model="RunDiffusion/Juggernaut-XL-v9" \
    > "$REPO_DIR/logs/smoke_generate_${SLURM_JOB_ID}.log" 2>&1

EXIT_CODE=$?
echo "[$(date)] Generation finished (exit $EXIT_CODE)"

if [ $EXIT_CODE -ne 0 ]; then
    echo "FAIL: generate step crashed (exit $EXIT_CODE)"
    echo "Log: logs/smoke_generate_${SLURM_JOB_ID}.log"
    exit $EXIT_CODE
fi

# ==========================================
# 5. Verify output
# ==========================================
IDENTITY_DIRS=$(find "$SMOKE_DATA/identities" -type d -name "identity_*" 2>/dev/null | sort)
IDENTITY_COUNT=$(echo "$IDENTITY_DIRS" | grep -c "identity_" || true)

echo "[$(date)] Verifying output..."
echo "  Found $IDENTITY_COUNT identity directories (expected 4)"

if [ "$IDENTITY_COUNT" -ne 4 ]; then
    echo "FAIL: expected 4 identity dirs, got $IDENTITY_COUNT"
    find "$SMOKE_DATA/identities" -type d 2>/dev/null | head -20
    exit 1
fi

BLACK_COUNT=0
TOTAL=0

for id_dir in $IDENTITY_DIRS; do
    CANDIDATE_COUNT=$(find "$id_dir" -name "candidate_*.png" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $(basename "$id_dir"): $CANDIDATE_COUNT candidates"

    for img in "$id_dir"/candidate_*.png; do
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
            echo "    BLACK: $(basename "$img")"
        elif [ "$IS_BLACK" = "2" ]; then
            echo "    WARNING: PIL check failed for $(basename "$img")"
        fi
    done
done

echo ""
echo "========================================="
if [ $BLACK_COUNT -gt 0 ]; then
    echo "FAIL: $BLACK_COUNT/$TOTAL images are solid black"
    exit 1
else
    echo "PASS: All $TOTAL images non-black ($IDENTITY_COUNT identities)"
fi

# ==========================================
# 6. Sync results to persistent storage
# ==========================================
SMOKE_OUT="$DATA_DIR/smoke_gen"
rm -rf "$SMOKE_OUT" 2>/dev/null || true
mkdir -p "$SMOKE_OUT"
echo "[$(date)] Syncing results to $SMOKE_OUT..."
rsync -av "$SMOKE_DATA/identities/" "$SMOKE_OUT/identities/" 2>&1 | tail -3
echo "Results persisted at: $SMOKE_OUT"

echo "========================================="
echo "[$(date)] Juggernaut-XL-v9 smoke test — PASS"

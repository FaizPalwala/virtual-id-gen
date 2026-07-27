#!/bin/bash
# ==========================================
# hpc_preprocess_smoke.sh — Smoke test the preprocessing pipeline
# ==========================================
# Processes only 70 candidates (1 identity) and reports rejection
# reasons without enforcing cluster cardinality.  Runs in ~5 minutes.
#
# Usage:
#   sbatch scripts/hpc_preprocess_smoke.sh
#   cat logs/preprocess_smoke_*.out
#   cat /scratch/$USER/datagen/data/merged/processed/preprocessing_rejection_manifest.csv
# ==========================================
#SBATCH --job-name=msc_ppsmoke
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/preprocess_smoke_%j.out
#SBATCH --error=logs/preprocess_smoke_%j.err

set -eo pipefail

echo "========================================="
echo "[$(date)] Preprocess smoke test started (job $SLURM_JOB_ID)"
echo "========================================="

module purge  2>/dev/null || true
module load miniforge 2>/dev/null || true
module load cuda 2>/dev/null || true
conda activate data_gen 2>/dev/null || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PARENT_DIR="$(dirname "$REPO_DIR")"
CACHE_DIR="$PARENT_DIR/model_cache"
DATA_DIR="/scratch/$USER/datagen/data"

export INSIGHTFACE_HOME="$CACHE_DIR/insightface"
mkdir -p "$REPO_DIR/logs" "$INSIGHTFACE_HOME"

echo "  DATA_DIR  = $DATA_DIR"
echo "  CACHE_DIR = $CACHE_DIR"

echo "[$(date)] Running GPU preflight..."
bash "$REPO_DIR/scripts/gpu_preflight.sh" 2>/dev/null || echo "  (preflight skipped)"

echo "[$(date)] Running smoke preprocess (70 candidates, 1 identity)..."
cd "$REPO_DIR/src"

# Direct invocation — no Hydra config pollution for debug flags
python preprocess.py \
    --identitydir "$DATA_DIR/merged/identities" \
    --processeddir "$DATA_DIR/merged/processed" \
    --imagesperidentity 70 \
    --max-candidates 70 \
    > "$REPO_DIR/logs/preprocess_smoke_${SLURM_JOB_ID}.log" 2>&1

EXIT_CODE=$?

echo ""
echo "========================================="
echo "[$(date)] Smoke test finished (exit $EXIT_CODE)"
echo ""
echo "Check rejection manifest:"
echo "  cat $DATA_DIR/merged/processed/preprocessing_rejection_manifest.csv | cut -d, -f13 | sort | uniq -c | sort -rn"
echo ""
if [ -d "$DATA_DIR/merged/processed/images" ]; then
    ACCEPTED=$(find "$DATA_DIR/merged/processed/images" -type f | wc -l)
    echo "Accepted images: $ACCEPTED"
else
    echo "No accepted images (all candidates rejected)"
fi
echo "========================================="

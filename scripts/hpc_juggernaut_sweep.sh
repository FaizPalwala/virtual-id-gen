#!/bin/bash
# ==========================================
# hpc_juggernaut_sweep.sh — Juggernaut-XL-v9 parameter sweep
# ==========================================
# Tests guidance_scale × ip_adapter_scale combinations on a single
# identity (3 candidates each).  Outputs grid-labeled directories
# so you can visually compare photorealism vs identity fidelity.
#
# Usage:
#   sbatch scripts/hpc_juggernaut_sweep.sh
#   # or interactive:  bash scripts/hpc_juggernaut_sweep.sh
#
# Output: data/sweep_juggernaut/<guidance>_<ip>/candidates/
# ==========================================
#SBATCH --job-name=jugg_sweep
#SBATCH --time=0-00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

set -eo pipefail

# ── Parameter grid (edit these arrays) ──
GUIDANCE_SCALES=(3.0 4.0 5.5)
IP_ADAPTER_SCALES=(0.75 0.85 0.90)
NUM_STEPS=30
NIDENTITIES=1
CANDIDATES_PER_ID=3
SEED=42

# ── Environment ──
echo "========================================="
echo "[$(date)] Juggernaut-XL-v9 parameter sweep"
echo "========================================="
echo "  guidance_scale:      ${GUIDANCE_SCALES[*]}"
echo "  ip_adapter_scale:    ${IP_ADAPTER_SCALES[*]}"
echo "  num_inference_steps: $NUM_STEPS"
echo "  combos:              $((${#GUIDANCE_SCALES[@]} * ${#IP_ADAPTER_SCALES[@]}))"
echo ""

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

# ── Stage to node-local scratch ──
SCRATCH_ROOT="${TMPDIR:-${LOCAL_SCRATCH:-${SCRATCH:-/tmp}}}"
SWEEP_TMP="$SCRATCH_ROOT/sweep_${SLURM_JOB_ID:-$$}"
mkdir -p "$SWEEP_TMP/repo" "$SWEEP_TMP/data"
echo "[$(date)] Staging to $SWEEP_TMP..."
cp -r "$REPO_DIR/"* "$SWEEP_TMP/repo/" 2>/dev/null
if [ -d "$DATA_DIR/seeds" ]; then
    cp -r "$DATA_DIR/seeds" "$SWEEP_TMP/data/seeds"
    echo "  Copied $(ls "$SWEEP_TMP/data/seeds" | wc -l) source images"
fi

# ── Run sweep ──
TOTAL=$((${#GUIDANCE_SCALES[@]} * ${#IP_ADAPTER_SCALES[@]}))
CURRENT=0

for gs in "${GUIDANCE_SCALES[@]}"; do
    for ips in "${IP_ADAPTER_SCALES[@]}"; do
        CURRENT=$((CURRENT + 1))
        LABEL="${gs}_${ips}"
        SWEEP_DATA="$SWEEP_TMP/sweep_$LABEL"
        mkdir -p "$SWEEP_DATA/seeds"

        if [ -d "$SWEEP_TMP/data/seeds" ]; then
            cp -r "$SWEEP_TMP/data/seeds/." "$SWEEP_DATA/seeds/"
        fi

        echo ""
        echo "[$(date)] [$CURRENT/$TOTAL] gs=$gs ips=$ips steps=$NUM_STEPS"

        cd "$SWEEP_TMP/repo/src"

        python main.py --config-name step2_generate \
            dataset.dataroot="$SWEEP_DATA" \
            dataset.nidentities="$NIDENTITIES" \
            dataset.candidatesperidentity="$CANDIDATES_PER_ID" \
            dataset.seed="$SEED" \
            dataset.forget_steps=0 \
            pipeline.instantid.base_model="RunDiffusion/Juggernaut-XL-v9" \
            pipeline.instantid.guidance_scale="$gs" \
            pipeline.instantid.ip_adapter_scale="$ips" \
            pipeline.instantid.num_inference_steps="$NUM_STEPS" \
            > "$REPO_DIR/logs/sweep_${LABEL}_${SLURM_JOB_ID:-$$}.log" 2>&1 \
            && echo "  DONE" || echo "  FAILED (see log)"

        # Sync individual result
        SWEEP_OUT="$DATA_DIR/sweep_juggernaut/$LABEL"
        mkdir -p "$SWEEP_OUT"
        rsync -a "$SWEEP_DATA/identities/" "$SWEEP_OUT/" 2>/dev/null || true
    done
done

# ── Summary ──
echo ""
echo "========================================="
echo "[$(date)] Sweep complete"
echo "Results: $DATA_DIR/sweep_juggernaut/"
echo ""
echo "Directory layout:"
echo "  sweep_juggernaut/"
for gs in "${GUIDANCE_SCALES[@]}"; do
    for ips in "${IP_ADAPTER_SCALES[@]}"; do
        echo "    ${gs}_${ips}/"
    done
done
echo ""
echo "To compare: open the candidate_000.png in each subdir."
echo "========================================="

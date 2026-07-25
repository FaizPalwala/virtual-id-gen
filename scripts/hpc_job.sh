#!/bin/bash
# ==========================================
# Slurm Resource Allocation — Multi-GPU Job Array
# ==========================================
#SBATCH --job-name=msc_datagen_phase1
#SBATCH --time=2-00:00:00                 # Request exactly 48 hours
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                       # 1 GPU per array task (4 in parallel)
#SBATCH --array=0-3                        # Run 4 identical copies, split by shard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_shard%a_%j.out
#SBATCH --error=logs/%x_shard%a_%j.err

# ------------------------------------------------------------------
# Each array task handles 1/4th of the identities.
# The randomstate seeds: 42, 44, 46, 48 (one per shard).
#
# Shard 0 (seed=42): identities   0– 99
# Shard 1 (seed=44): identities 100–199
# Shard 2 (seed=46): identities 200–299
# Shard 3 (seed=48): identities 300–399
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
REPO_DIR=$SLURM_SUBMIT_DIR
PARENT_DIR=$(dirname "$REPO_DIR")
DATA_DIR="$PARENT_DIR/data"
CACHE_DIR="$PARENT_DIR/model_cache"

mkdir -p "$DATA_DIR"

# Thread pinning
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Model cache on persistent storage
export HF_HOME="$CACHE_DIR/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export INSIGHTFACE_HOME="$CACHE_DIR/insightface"
mkdir -p "$HF_HOME" "$INSIGHTFACE_HOME"

# High-speed downloads
export HF_TOKEN="hf_your_actual_token_here"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Each array task gets its own TMPDIR subfolder for isolation
SHARD_TMPDIR="$TMPDIR/shard_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SHARD_TMPDIR/repo" "$SHARD_TMPDIR/data"

# ==========================================
# 3. Staging (Copy IN to $TMPDIR)
# ==========================================
echo "[$(date)] Shard ${SLURM_ARRAY_TASK_ID}: Staging to node-local scratch..."

cp -r "$REPO_DIR/"* "$SHARD_TMPDIR/repo/"

if [ "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    cp -r "$DATA_DIR/"* "$SHARD_TMPDIR/data/"
fi

# ==========================================
# 4. GPU Preflight Check
# ==========================================
echo "[$(date)] Shard ${SLURM_ARRAY_TASK_ID}: Running GPU Preflight..."
bash "$SHARD_TMPDIR/repo/scripts/gpu_preflight.sh"

# ==========================================
# 5. Run the Data Generation (one shard)
# ==========================================
echo "[$(date)] Shard ${SLURM_ARRAY_TASK_ID}: Starting Phase 1 Data Generation..."

SHARD_SEED=$((42 + SLURM_ARRAY_TASK_ID * 2))

cd "$SHARD_TMPDIR/repo/src"

# Generate 100 identities per shard with a unique random seed.
# The seeds 42,44,46,48 produce non-overlapping shuffles of the SFHQ source pool.
#
# Each shard writes into its own dataroot to avoid overwriting.
# After the job, merge with: python scripts/merge_shards.py
SHARD_DATA="$SHARD_TMPDIR/data_shard_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SHARD_DATA/raw"
# Source images are shared — symlink or copy once
if [ -d "$SHARD_TMPDIR/data/raw" ] && [ ! -d "$SHARD_DATA/raw" ] || [ -z "$(ls -A "$SHARD_DATA/raw" 2>/dev/null)" ]; then
    cp -r "$SHARD_TMPDIR/data/raw/." "$SHARD_DATA/raw/"
fi

python main.py --config-name step3_generate \
    dataset.dataroot="$SHARD_DATA" \
    dataset.nidentities=100 \
    dataset.seed="$SHARD_SEED" \
    > "$REPO_DIR/logs/datagen_shard_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.log" 2>&1

EXIT_CODE=$?

# ==========================================
# 6. Data Sync (Copy OUT to Shared Storage)
# ==========================================
echo "[$(date)] Shard ${SLURM_ARRAY_TASK_ID}: Generation finished (exit $EXIT_CODE)"
echo "[$(date)] Syncing data back..."

# Sync each shard's output under its own subdirectory for later merging
SHARD_OUT="$DATA_DIR/shard_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SHARD_OUT"
rsync -av "$SHARD_DATA/identities/" "$SHARD_OUT/"

echo "[$(date)] Shard ${SLURM_ARRAY_TASK_ID}: Sync complete."
exit $EXIT_CODE
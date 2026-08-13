#!/bin/bash
# ==========================================
# hpc_generate.sh — Phase 2: Identity Generation (15-way GPU array)
# ==========================================
#SBATCH --job-name=msc_generate
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                      # Dedicated GPU (gres isolation = no sharing)
#SBATCH --mem=64G                         # Explicit RAM (SDXL+ControlNet+InstantID+ONNX)
#SBATCH --cpus-per-task=8                 # Matches OMP_NUM_THREADS in the script
#SBATCH --array=0-14
#SBATCH --output=logs/%x_shard%a_%j.out
#SBATCH --error=logs/%x_shard%a_%j.err

# NOTE (2026-08-06): --exclusive removed deliberately.  It reserved a whole
# 4-GPU node per shard to use 1 GPU (15 of 20 GPUs wasted under the 5-job
# concurrency cap → scheduler deprioritized the array).  GPU isolation is
# guaranteed by --gres=gpu:1 alone; --mem=64G pins the CPU-RAM footprint
# (model loading peak ~30 GB, 2× headroom).  Per-shard wall time is ~8 h
# (100 cands/id); 12 h limit gives 50% headroom and stays backfill-friendly.
# If OOM resurfaces: bump --mem (CPU) or re-add --exclusive (node-level).

# CUDA memory: enable expandable segments to reduce fragmentation from
# loading SDXL + ControlNet + Juggernaut + InstantID in sequence.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Log assigned GPU for diagnostics (do NOT set CUDA_VISIBLE_DEVICES — Slurm
# manages GPU isolation on this cluster; overriding can break discovery).
echo "[INFO] Shard ${SLURM_ARRAY_TASK_ID}: GPU $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"

# ------------------------------------------------------------------
# Each array task generates 50 identities with a unique random seed.
# 750 identities across 15 shards (5 concurrent HPC job slots × 3 waves).
# Seed partition is computed IN-GENERATE: every shard independently
# shuffles the loose 750-seed pool (srand 7, same as partition_seeds.sh)
# and takes its round-robin slice — no Mac-side step, no cross-shard
# seed reuse (v1.0.0 bug: 397 unique seeds from 600 identities).
#
# Shard  0 (seed=42): identities   0– 49
# Shard  1 (seed=44): identities  50– 99
# Shard  2 (seed=46): identities 100–149
# Shard  3 (seed=48): identities 150–199
# Shard  4 (seed=50): identities 200–249
# Shard  5 (seed=52): identities 250–299
# Shard  6 (seed=54): identities 300–349
# Shard  7 (seed=56): identities 350–399
# Shard  8 (seed=58): identities 400–449
# Shard  9 (seed=60): identities 450–499
# Shard 10 (seed=62): identities 500–549
# Shard 11 (seed=64): identities 550–599
# Shard 12 (seed=66): identities 600–649
# Shard 13 (seed=68): identities 650–699
# Shard 14 (seed=70): identities 700–749
#
# Merge with hpc_merge.sh after all shards complete.
# Output lands under $DATA_DIR/shard_${TASK_ID}/ for later merging.
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

mkdir -p "$DATA_DIR"

# Thread pinning
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Silence all tqdm progress bars (diffusers denoising + VAE decoder).
# Identity-level progress is logged via LOGGER.info() lines instead.
export TQDM_DISABLE=1

# Suppress ONNX Runtime thread-affinity errors (HPC node quirk)
export ORT_LOG_LEVEL=3

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
# 3. Staging (Copy IN → node-local scratch)
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
# 4b. Prompt Plan Pre-flight (fail fast, before GPU spend)
# ==========================================
# Validates the 20x5 prompt grid: 100 unique prompts, per-dimension
# diversity floors (lighting=5, pose>=10, ...), determinism.  A silent
# collapse here costs 8 GPU-hours x 15 shards before release QA catches it
# (RELEASE_TODO Phase D2 P1-3).  Every shard re-verifies (cheap, ~1 s).
echo "[$(date)] Shard ${SLURM_ARRAY_TASK_ID}: Validating prompt plan..."
cd "$SHARD_TMPDIR/repo/src"
PYTHONPATH="$SHARD_TMPDIR/repo/src" python "$SHARD_TMPDIR/repo/scripts/validate_prompt_plan.py" \
    >> "$REPO_DIR/logs/generate_shard_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.log" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Shard ${SLURM_ARRAY_TASK_ID}: prompt plan validation FAILED — aborting before generation"
    exit 1
fi
echo "[$(date)] Shard ${SLURM_ARRAY_TASK_ID}: prompt plan OK"


# ==========================================
# 5. Generate 50 Identities (one shard)
# ==========================================
echo "[$(date)] Shard ${SLURM_ARRAY_TASK_ID}: Starting identity generation..."

SHARD_SEED=$((42 + SLURM_ARRAY_TASK_ID * 2))

cd "$SHARD_TMPDIR/repo/src"

# Each shard writes into its own dataroot so the twelve array tasks don't
# overwrite each other.  After the job finishes, merge with hpc_merge.sh.
SHARD_DATA="$SHARD_TMPDIR/data_shard_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SHARD_DATA/seeds"

# Deterministic in-generate seed partition — NO Mac-side step needed.
# Every array task independently computes the SAME shuffle (srand 7) of
# the full loose seed pool and takes the round-robin slice for its shard
# id.  Identical scheme to partition_seeds.sh, so seed→shard assignment
# is reproducible and consistent whether or not the Mac-side script ran.
# Guarantees no cross-shard seed reuse (the v1.0.0 bug: all shards shared
# the full pool, producing 397 distinct seeds for 600 identities).
NSHARDS=15
SEED_PICK=$(mktemp)
ls "$SHARD_TMPDIR/data/seeds"/seed_*.jpg 2>/dev/null | sort | perl -e '
use List::Util qw(shuffle);
srand(7);
while (<STDIN>) { chomp; push @seeds, $_; }
my @shuffled = shuffle(@seeds);
my $nshards = shift;
my $shard = shift;
for my $i (0 .. $#shuffled) {
    print "$shuffled[$i]\n" if $i % $nshards == $shard;
}
' "$NSHARDS" "$SLURM_ARRAY_TASK_ID" > "$SEED_PICK"

n_seeds=$(wc -l < "$SEED_PICK")
if [ "$n_seeds" -lt 50 ]; then
    echo "[ERROR] Shard ${SLURM_ARRAY_TASK_ID}: only $n_seeds seeds in partition "
          "(need ≥50).  Is the full 750-seed pool present at "
          "$SHARD_TMPDIR/data/seeds?"
    rm -f "$SEED_PICK"
    exit 1
fi
mkdir -p "$SHARD_DATA/seeds"
while read -r s; do cp "$s" "$SHARD_DATA/seeds/"; done < "$SEED_PICK"
rm -f "$SEED_PICK"
echo "[$(date)] Shard ${SLURM_ARRAY_TASK_ID}: $n_seeds seeds via in-generate partition"

python main.py --config-name step2_generate \
    dataset.dataroot="$SHARD_DATA" \
    dataset.nidentities=50 \
    dataset.seed="$SHARD_SEED" \
    dataset.forget_steps=0 \
    > "$REPO_DIR/logs/generate_shard_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.log" 2>&1

EXIT_CODE=$?

# ==========================================
# 6. Data Sync (Copy OUT → shared storage)
# ==========================================
echo "[$(date)] Shard ${SLURM_ARRAY_TASK_ID}: Generation finished (exit $EXIT_CODE)"
echo "[$(date)] Syncing data back..."

SHARD_OUT="$DATA_DIR/shard_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SHARD_OUT"
rsync -av "$SHARD_DATA/identities/" "$SHARD_OUT/"

echo "[$(date)] Shard ${SLURM_ARRAY_TASK_ID}: Sync complete."
exit $EXIT_CODE

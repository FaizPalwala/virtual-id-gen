#!/bin/bash
# ==========================================
# Slurm Resource Allocation
# ==========================================
#SBATCH --job-name=msc_datagen_phase1
#SBATCH --time=24:00:00                  
#SBATCH --partition=gpu                  
#SBATCH --gres=gpu:1                     
#SBATCH --cpus-per-task=8                
#SBATCH --mem=32G                        
#SBATCH --output=logs/%x_%j.out          
#SBATCH --error=logs/%x_%j.err           

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
# SLURM_SUBMIT_DIR is the repo directory where you run sbatch
REPO_DIR=$SLURM_SUBMIT_DIR

# The data and cache folders are one level up, alongside the repo
PARENT_DIR=$(dirname "$REPO_DIR")
DATA_DIR="$PARENT_DIR/data"
CACHE_DIR="$PARENT_DIR/model_cache"

# Ensure the local data directory exists for syncing later
mkdir -p "$DATA_DIR"

# Optimization: Pin threads to prevent CPU thrashing
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Optimization: Redirect caches to persistent storage (avoids re-downloading heavy models)
export HF_HOME="$CACHE_DIR/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export INSIGHTFACE_HOME="$CACHE_DIR/insightface"
mkdir -p "$HF_HOME" "$INSIGHTFACE_HOME"

# ==========================================
# 3. Staging (Copy IN to $TMPDIR)
# ==========================================
echo "[$(date)] Staging repository and data to node-local scratch..."

# Create isolated folders in TMPDIR to maintain the side-by-side structure
mkdir -p "$TMPDIR/repo"
mkdir -p "$TMPDIR/data"

# Copy the entire repository into TMPDIR/repo
cp -r "$REPO_DIR/"* "$TMPDIR/repo/"

# Copy existing data (if any) into TMPDIR/data so preprocessing has its input
if [ "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    cp -r "$DATA_DIR/"* "$TMPDIR/data/"
fi

# ==========================================
# Preflight Check
# ==========================================
echo "[$(date)] Running GPU Preflight Check..."
bash "$TMPDIR/repo/scripts/gpu_preflight.sh"

# ==========================================
# 4. Run the Data Generation
# ==========================================
echo "[$(date)] Starting Phase 1 Data Generation..."
cd "$TMPDIR/repo/src"

# Execute the pipeline.
# We dynamically override dataset.dataroot so Hydra knows exactly where the TMPDIR data is.
python main.py --config-name step3_generate dataset.dataroot="$TMPDIR/data" \
    > "$REPO_DIR/logs/datagen_execution_${SLURM_JOB_ID}.log" 2>&1

# python instantid_adapter.py \
#   --id_image ../../data/test/source.jpg \
#   --style_image ../../data/test/style.jpg \
#   --output_path ../../data/test/output.jpg \
#   --seed 42 \
#   --width 1024 \
#   --height 1024 \
#   --num_inference_steps 30 \
#   --guidance_scale 5.0 \
# for seed in 42 43 44 45
# do
#     python instantid_adapter.py \
#         --id_image ../../data/test/source.jpg \
#         --output_path "../../data/test/grid_${seed}.jpg" \
#         --seed "${seed}" \
#         --width 1024 \
#         --height 1024 \
#         --num_inference_steps 35 \
#         --guidance_scale 5.5 \
#         --ip_adapter_scale 0.90 \
#         --controlnet_conditioning_scale 0.80 \
#         --prompt "RAW photo, realistic, editorial headshot, soft daylight, natural skin texture, sharp eyes, DSLR photograph" \
#         --negative_prompt "painting, illustration, CGI, 3D render, monochrome, grayscale, desaturated, blurry, deformed, text, watermark"
# done > "${REPO_DIR}/logs/trial_${SLURM_JOB_ID}.log" 2>&1

EXIT_CODE=$?

# ==========================================
# 5. Data Sync (Copy OUT to Project Directory)
# ==========================================
echo "[$(date)] Generation finished with exit code $EXIT_CODE"
echo "[$(date)] Syncing the data folder back..."

# Rsync safely syncs the newly generated data back alongside your repo folder
rsync -av "$TMPDIR/data/" "$DATA_DIR/"

echo "[$(date)] Sync complete. Data safely stored at $DATA_DIR"
exit $EXIT_CODE
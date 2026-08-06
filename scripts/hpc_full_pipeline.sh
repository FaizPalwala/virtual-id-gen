#!/bin/bash
# ==========================================
# hpc_full_pipeline.sh — End-to-End Dataset Generation Pipeline
# ==========================================
#SBATCH --job-name=msc_pipeline
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ------------------------------------------------------------------
# Master orchestrator: submits each phase as a Slurm job chained with
# --dependency=afterok so the scheduler handles ordering and failure
# propagation automatically.  The orchestrator itself is a lightweight
# CPU job that exits after the last child is submitted.
#
#     ┌─────────────┐
#     │  generate    │  15-GPU array (hpc_generate.sh)
#     └──────┬──────┘
#            │ afterok
#     ┌──────▼──────┐
#     │   merge      │  1 CPU        (hpc_merge.sh)
#     └──────┬──────┘
#            │ afterok (BOTH)
#     ┌──────▼──────┐          ┌──────────────┐
#     │  extract     │  1 GPU  │  preprocess   │  1 GPU
#     │  (hpc_extract.sh)      │  (hpc_preprocess.sh)
#     └──────┬──────┘          └──────┬───────┘
#            │ afterok:extract,preprocess
#     ┌──────▼──────┐
#     │   build      │  1 CPU        (hpc_build.sh)   — all dataset artifacts
#     └─────────────┘
#
# Extract and preprocess are SIBLING steps: both consume the merge output
# (1024×1024 candidates + raw_candidate_manifest.csv) and neither reads the
# other's output.  Preprocess runs its own ArcFace gate (load_arcface_model
# in preprocess.py); extract writes embeddings/ + demographics.  The build
# step joins their outputs through the (identity_id, trial) key.  Running
# them on two GPUs in parallel halves the post-merge wall time.
# NOTE: demographics/embeddings come from the 1024×1024 candidates
# (InsightFace detection fails on tight 224 crops), and the 224 crops
# inherit them via identity join at build time.
#
# Each job writes its own log under logs/.
# If any phase fails, downstream jobs are cancelled by Slurm.
# ------------------------------------------------------------------

set -euo pipefail

REPO_DIR="$SLURM_SUBMIT_DIR"
SCRIPT_DIR="$REPO_DIR/scripts"
mkdir -p "$REPO_DIR/logs"

submit_job() {
    local label="$1"
    local script="$2"
    shift 2
    local dep_args=("$@")

    local job_id
    job_id=$(sbatch --parsable "${dep_args[@]}" "$script")
    echo "[$(date)] Submitted $label: $job_id"
    echo "$job_id"
}

# ---- Phase 1: Generate (15-GPU array) ----
GEN_JOB=$(submit_job "generate" "$SCRIPT_DIR/hpc_generate.sh")
echo ""

# ---- Phase 2: Merge (CPU, depends on generate) ----
MERGE_JOB=$(submit_job "merge" "$SCRIPT_DIR/hpc_merge.sh" \
    --dependency="afterok:${GEN_JOB}")
echo ""

# ---- Phase 3: Extract + Preprocess (two GPUs, both depend on merge) ----
# Sibling steps — run in PARALLEL on two GPUs.  Extract does ArcFace
# embeddings + demographics on the 1024x1024 candidates; preprocess does
# MTCNN align + quality gate to 224x224 crops.  Neither reads the other's
# output; build joins them via (identity_id, trial).
EXT_JOB=$(submit_job "extract" "$SCRIPT_DIR/hpc_extract.sh" \
    --dependency="afterok:${MERGE_JOB}")
PRE_JOB=$(submit_job "preprocess" "$SCRIPT_DIR/hpc_preprocess.sh" \
    --dependency="afterok:${MERGE_JOB}")
echo ""

# ---- Phase 4: Build (CPU, depends on BOTH extract and preprocess) ----
BLD_JOB=$(submit_job "build" "$SCRIPT_DIR/hpc_build.sh" \
    --dependency="afterok:${EXT_JOB}:${PRE_JOB}")
echo ""

echo "=============================================="
echo "Pipeline submitted.  Summary:"
echo "  generate    : $GEN_JOB"
echo "  merge       : $MERGE_JOB  (after $GEN_JOB)"
echo "  extract     : $EXT_JOB    (after $MERGE_JOB, parallel)"
echo "  preprocess  : $PRE_JOB    (after $MERGE_JOB, parallel)"
echo "  build       : $BLD_JOB    (after extract AND preprocess)"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "=============================================="

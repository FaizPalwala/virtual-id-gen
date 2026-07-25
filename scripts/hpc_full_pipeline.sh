#!/bin/bash
# ==========================================
# hpc_full_pipeline.sh — End-to-End Dataset Generation Pipeline
# ==========================================
#SBATCH --job-name=msc_pipeline
#SBATCH --time=3-00:00:00
#SBATCH --partition=cpu
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
#     │  generate    │  4-GPU array  (hpc_generate.sh)
#     └──────┬──────┘
#            │ afterok
#     ┌──────▼──────┐
#     │   merge      │  1 CPU        (hpc_merge.sh)
#     └──────┬──────┘
#            │ afterok
#     ┌──────▼──────┐
#     │ preprocess   │  1 GPU        (hpc_preprocess.sh)
#     └──────┬──────┘
#            │ afterok
#     ┌──────▼──────┐
#     │  extract     │  1 GPU        (hpc_extract.sh)
#     └──────┬──────┘
#            │ afterok
#     ┌──────▼──────┐
#     │   build      │  1 CPU        (hpc_build.sh)
#     └─────────────┘
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

# ---- Phase 1: Generate (4-GPU array) ----
GEN_JOB=$(submit_job "generate" "$SCRIPT_DIR/hpc_generate.sh")
echo ""

# ---- Phase 2: Merge (CPU, depends on generate) ----
MERGE_JOB=$(submit_job "merge" "$SCRIPT_DIR/hpc_merge.sh" \
    --dependency="afterok:${GEN_JOB}")
echo ""

# ---- Phase 3: Preprocess (GPU, depends on merge) ----
PRE_JOB=$(submit_job "preprocess" "$SCRIPT_DIR/hpc_preprocess.sh" \
    --dependency="afterok:${MERGE_JOB}")
echo ""

# ---- Phase 4: Extract (GPU, depends on preprocess) ----
EXT_JOB=$(submit_job "extract" "$SCRIPT_DIR/hpc_extract.sh" \
    --dependency="afterok:${PRE_JOB}")
echo ""

# ---- Phase 5: Build (CPU, depends on extract) ----
BLD_JOB=$(submit_job "build" "$SCRIPT_DIR/hpc_build.sh" \
    --dependency="afterok:${EXT_JOB}")
echo ""

echo "=============================================="
echo "Pipeline submitted.  Summary:"
echo "  generate    : $GEN_JOB"
echo "  merge       : $MERGE_JOB  (after $GEN_JOB)"
echo "  preprocess  : $PRE_JOB    (after $MERGE_JOB)"
echo "  extract     : $EXT_JOB    (after $PRE_JOB)"
echo "  build       : $BLD_JOB    (after $EXT_JOB)"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "=============================================="

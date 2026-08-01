# HPC CookBook (Aire Setup Cheat Sheet)

## `data_gen` Envirnoment Setup:

```bash
# 1. Start fresh and load the module
module purge
module load miniforge

# 2. Create and activate the specific environment
conda create -n data_gen python=3.10 -y
conda activate data_gen

# 3. Install PyTorch for CUDA 12.4 (compatible with driver >=12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install xformers with a CUDA 12.4-matched build
pip install xformers --index-url https://download.pytorch.org/whl/cu124

# 5. Install remaining dependencies
pip install -r requirements.txt

```

Once that finishes, your environment is fully primed.

### Quick start
```bash
# Full pipeline (all six phases, chained with --dependency):
sbatch scripts/hpc_full_pipeline.sh

# Or individual phases (note: extract runs BEFORE preprocess):
sbatch scripts/hpc_generate.sh       # 12-GPU array, ~12 hr
sbatch scripts/hpc_merge.sh          # CPU, ~30 min (after generate)
sbatch scripts/hpc_extract.sh        # 1 GPU, ~2 hr (ArcFace on 1024 candidates)
sbatch scripts/hpc_preprocess.sh     # 1 GPU, ~4 hr (MTCNN align + gate → 224 crops)
sbatch scripts/hpc_build.sh          # CPU, ~15 min (builds all four dataset pairs)
```

Extract runs on the raw 1024×1024 candidates **before** preprocess so
InsightFace detection works on full portraits (it fails on tight 224 crops).
The four dataset pairs (Bench balanced/imbalanced + Full balanced/imbalanced)
are all produced by the build step from one pipeline run.
# HPC CookBook (Aire Setup Cheat Sheet)

## `data_gen` Envirnoment Setup:

```bash
# 1. Start fresh and load the module
module purge
module load miniforge

# 2. Create and activate the specific environment
conda create -n data_gen python=3.10 -y
conda activate data_gen

# 3. Install PyTorch via Conda (matches your torch>=2.0 and torchvision>=0.15 requirements)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. Install everything else directly from your file
pip install -r requirements.txt

```

Once that finishes, your environment is fully primed. You can submit the `HPC_Job.sh` Slurm script, and it will pick up all these dependencies natively.
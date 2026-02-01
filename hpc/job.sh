#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=ealstm_streamflow_run       # Job name (shows up in queue)
#SBATCH --time=12:00:00                        # Max run time (HH:MM:SS) - 12 hours is safe
#SBATCH --nodes=1                              # We only need 1 computer node
#SBATCH --ntasks=1                             # We run 1 main task
#SBATCH --cpus-per-task=4                      # CPU cores (Matches num_workers in loader)
#SBATCH --mem=32G                              # RAM (32 GB is plenty for this dataset)
#SBATCH --gres=gpu:1                           # Request 1 GPU
#SBATCH --partition=gpu                        # Ensure we are on the GPU partition

# --- EMAIL NOTIFICATIONS ---
#SBATCH --mail-type=BEGIN,END,FAIL             # Email on start, finish, and crash

# --- LOGGING ---
#SBATCH --output=logs/train_%j.out             # Standard Output (%j = job ID)
#SBATCH --error=logs/train_%j.err              # Standard Error

# NOTE: --account and --mail-user are passed via command line in submit.sh

# ---------------------------

# 1. Setup Environment
echo "Setting up job environment on $(hostname)..."
module purge
module load intel-oneapi-compilers/2023.1.0
module load python/3.11.6
source venv/bin/activate

# 2. Debug Info (Optional but helpful)
echo "Python path: $(which python)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 3. Run Training
# Unbuffered (-u) ensures logs are written immediately, not held in memory
echo "Starting Training Script..."
python -u run_training.py

echo "Job Finished."
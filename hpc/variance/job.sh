#!/bin/bash

# --- SLURM CONFIGURATION ---
# Job 2: post-hoc variance head (Ch3 MVE / Student-t). Trains ONE small network
# over the frozen, precomputed hidden states + ensemble mean — NOT a 10x LSTM
# array. So: no --array, short wall time, modest RAM. One GPU is plenty (and the
# job is small enough it could even run on CPU if the queue is tight).
#SBATCH --job-name=mve_head                    # Job name (shows up in queue)
#SBATCH --time=02:00:00                        # Short: MLP over cached 256-d vectors
#SBATCH --nodes=1                              # Single node
#SBATCH --ntasks=1                             # Single task (no array)
#SBATCH --cpus-per-task=4                      # CPU cores for the loader
#SBATCH --mem=16G                              # RAM (contract + hidden states fit easily)
#SBATCH --gpus=1

# --- EMAIL NOTIFICATIONS ---
#SBATCH --mail-type=BEGIN,END,FAIL             # Email on start, finish, and crash

# --- LOGGING ---
# %j is the job ID (no array here, so no %A/%a)
#SBATCH --output=logs/mve_%j.out
#SBATCH --error=logs/mve_%j.err

# NOTE: --account and --mail-user are passed via command line in submit.sh

# ---------------------------

# 1. Setup Environment (shares the one venv built by ../setup_env.sh)
echo "Setting up job environment on $(hostname)..."
module purge
module load intel-oneapi-compilers/2023.1.0
module load python/3.11.6
module load cuda                 # CUDA runtime for the GPU torch build
source venv/bin/activate

# 2. Debug Info
echo "Python path: $(which python)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 3. Step 0 — (re)build the index contract from the ensemble outputs.
#    Fast, CPU-only. Safe to run every time; it's the seam between Job 1 and Job 2.
#    TODO(uncomment once Job 1 outputs + states are present on the cluster):
# echo "Building index contract for: $EXP_NAME / $MODEL_TYPE"
python -m src.build_index --exp_name $EXP_NAME --model_type $MODEL_TYPE

# 4. Step 1 — train the variance head off the frozen mean + hidden states.
#    TODO(fill in once run_variance_head.py exists; flag contract not finalized):
echo "Starting Variance Head for Experiment: $EXP_NAME | Model: $MODEL_TYPE"
python -u run_variance_head.py --exp_name $EXP_NAME --model_type $MODEL_TYPE

echo "Job Finished."

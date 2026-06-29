#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=ealstm_ensemble             # Job name (shows up in queue)
#SBATCH --time=48:00:00                        # Max run time (HH:MM:SS)
#SBATCH --nodes=1                              # We only need 1 computer node
#SBATCH --ntasks=1                             # We run 1 main task
#SBATCH --cpus-per-task=4                      # CPU cores (Matches num_workers in loader)
#SBATCH --mem=24G                              # RAM
#SBATCH --gpus=1
#SBATCH --array=0-9%5                          # Creates 10 jobs (IDs 0-9) while running a max of 5 at a time

# --- EMAIL NOTIFICATIONS ---
#SBATCH --mail-type=BEGIN,END,FAIL             # Email on start, finish, and crash

# --- LOGGING ---
# %A is the array master job ID, %a is the specific task ID (0-9)
#SBATCH --output=logs/train_%A_%a.out 
#SBATCH --error=logs/train_%A_%a.err

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
# Experiment name, model type, member ID, and any optional flags (EXTRA_FLAGS, e.g.
# --save_hidden) are all resolved in submit.sh and forwarded via --export.
echo "Starting Training Script for Experiment: $EXP_NAME | Member: $SLURM_ARRAY_TASK_ID | Flags: ${EXTRA_FLAGS:-none}"
python -u run_training.py --exp_name $EXP_NAME --model_type $MODEL_TYPE --member_id $SLURM_ARRAY_TASK_ID $EXTRA_FLAGS

echo "Job Finished."
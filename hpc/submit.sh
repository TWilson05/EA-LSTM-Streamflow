#!/bin/bash
set -e

# Require the user to specify the experiment name
if [ -z "$1" ]; then
    echo "Error: You must provide an experiment name and a model type!"
    echo "Usage: ./submit.sh <experiment_name> <model_type> [save_hidden]"
    echo "Valid options: baseline, area, topographic, phase-split"
    echo "Valid models: ealstm, lstm"
    echo "Optional 3rd arg 'save_hidden' also emits LSTM hidden states (Ch3 VE head)"
    exit 1
fi

EXP_NAME=$1
MODEL_TYPE=$2
SAVE_HIDDEN=$3   # optional: pass "save_hidden" to also emit h_T for the variance head

# 1. Load Secrets
if [ -f "secrets.env" ]; then
    # Silent fix: Remove \r characters if they exist
    sed -i 's/\r//' secrets.env
    source secrets.env
else
    echo "Error: secrets.env not found!"
    echo "Please create 'secrets.env' with EMAIL=... and ACCOUNT=..."
    exit 1
fi

# 2. Automate Log Folder Creation
if [ ! -d "logs" ]; then
    mkdir -p logs
    echo "Created 'logs' directory."
fi

# 3. Submit to SLURM
echo "Submitting 10-member ensemble job for experiment: $EXP_NAME and model: $MODEL_TYPE"

sbatch \
    --account=$ACCOUNT \
    --mail-user=$EMAIL \
    --export=ALL,EXP_NAME=$EXP_NAME,MODEL_TYPE=$MODEL_TYPE,SAVE_HIDDEN=$SAVE_HIDDEN \
    hpc/job.sh
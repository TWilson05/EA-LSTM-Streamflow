#!/bin/bash
set -e

# Require the experiment name + model type
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: You must provide an experiment name and a model type!"
    echo "Usage: ./hpc/variance/submit.sh <experiment_name> <model_type>"
    echo "Valid options: baseline, area, topographic, phase-split"
    echo "Valid models: ealstm, lstm"
    echo "NOTE: run this only AFTER manually validating the ensemble in EXP5."
    exit 1
fi

EXP_NAME=$1
MODEL_TYPE=$2

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

# 3. Submit to SLURM (single task — no array; the head is one small network)
echo "Submitting variance-head job for experiment: $EXP_NAME and model: $MODEL_TYPE"

sbatch \
    --account=$ACCOUNT \
    --mail-user=$EMAIL \
    --export=ALL,EXP_NAME=$EXP_NAME,MODEL_TYPE=$MODEL_TYPE \
    hpc/variance/job.sh

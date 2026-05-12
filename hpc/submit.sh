#!/bin/bash
set -e

# Require the user to specify the experiment name
if [ -z "$1" ]; then
    echo "Error: You must provide an experiment name!"
    echo "Usage: ./submit.sh <experiment_name>"
    echo "Valid options: baseline, area, topographic, phase-split"
    exit 1
fi

EXP_NAME=$1

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
echo "Submitting 10-member ensemble job for experiment: $EXP_NAME"

sbatch \
    --account=$ACCOUNT \
    --mail-user=$EMAIL \
    --export=ALL,EXP_NAME=$EXP_NAME \
    hpc/job.sh
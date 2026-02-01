#!/bin/bash
set -e

# 1. Load Secrets
if [ -f "secrets.env" ]; then
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
# We pass the secrets as overrides (--account, --mail-user)
# We execute from the current directory
echo "Submitting job for account: $ACCOUNT"

sbatch \
    --account=$ACCOUNT \
    --mail-user=$EMAIL \
    hpc/job.sh
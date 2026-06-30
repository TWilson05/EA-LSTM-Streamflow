#!/bin/bash

# Stop script on any error
set -e

echo "Starting Environment Setup..."

# 1. Load System Modules (Sockeye-specific)
# We purge old modules to ensure a clean slate
module purge
module load intel-oneapi-compilers/2023.1.0
module load python/3.11.6
module load cuda                 # CUDA runtime on PATH for the GPU torch build
# If you need geospatial libs that require system binaries, load them here too
# e.g., module load gdal/3.5.3 (Only if pip install fails later)

# 2. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python -m venv venv
else
    echo "'venv' already exists. Updating existing environment."
fi

# 3. Activate
source venv/bin/activate

# 4. Install Dependencies
echo "Installing dependencies..."
pip install --upgrade pip
# CUDA-enabled PyTorch FIRST, from the CUDA wheel index (V100 GPUs -> cu118).
# The bare `torch` in requirements.txt can resolve to a CPU-ONLY wheel, which
# silently trains on CPU (torch.cuda.is_available() == False on the GPU node).
# Installing the cu118 build first means the later `-r requirements.txt` sees torch
# already satisfied and won't replace it. Kept out of requirements.txt so the local
# (macOS) install path still works — there are no CUDA wheels for macOS.
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 5. Install the local project package
echo "Installing local 'src' package..."
pip install -e .

# 6. Verify the torch BUILD (not availability): on the login node there is no GPU,
# so torch.cuda.is_available() is False here regardless — a non-None CUDA build is
# the correct login-node check. is_available() flips to True on the compute node.
python -c "import torch; print('torch', torch.__version__, '| CUDA build:', torch.version.cuda)"

echo "Setup Complete! You can now submit jobs."

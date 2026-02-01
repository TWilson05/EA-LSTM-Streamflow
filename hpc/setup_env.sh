#!/bin/bash

# Stop script on any error
set -e

echo "Starting Environment Setup..."

# 1. Load System Modules (Sockeye-specific)
# We purge old modules to ensure a clean slate
module purge
module load intel-oneapi-compilers/2023.1.0
module load python/3.11.6
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
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Install the local project package
echo "Installing local 'src' package..."
pip install -e .

echo "Setup Complete! You can now submit jobs."

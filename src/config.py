from pathlib import Path

# This finds the directory containing config.py (src/), then gets its parent (root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define standard subdirectories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ERA5 Specific Paths
ERA5_DIR = DATA_DIR / "raw" / "era5"
ERA5_PRECIP_DIR = ERA5_DIR / "precipitation"
ERA5_TEMP_DIR = ERA5_DIR / "temperature"

# Area: [North, West, South, East]
ERA5_BOUNDS = [62, -129, 47, -105] 

# Create directories if they don't exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, ERA5_PRECIP_DIR, ERA5_TEMP_DIR]:
    path.mkdir(parents=True, exist_ok=True)

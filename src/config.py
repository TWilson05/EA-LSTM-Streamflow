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

# Spatial Data Paths
DRAINAGE_DIR = DATA_DIR / "raw" / "drainage_areas"
GLACIER_SHP_PATH = DATA_DIR / "raw" / "RGI-western-canada" / "02_rgi60_WesternCanadaUS.shp"
MASS_BALANCE_PATH = DATA_DIR / "raw" / "mass_balance" / "ts_monthly_const_area_fnn_cluster.csv"

# Drainage Files (List them if you have multiple regions)
DRAINAGE_FILES = [
    DRAINAGE_DIR / "MDA_ADP_05.gpkg",
    DRAINAGE_DIR / "MDA_ADP_07.gpkg"
]

OUTPUT_STATIC_ATTR = PROCESSED_DATA_DIR / "static_attributes.csv"
OUTPUT_GLACIER_VOL = PROCESSED_DATA_DIR / "glacier_volume_change.csv"

# Area bounds lat/lons: [North, West, South, East]
# Used for ERA5 download
ERA5_BOUNDS = [62, -129, 47, -105] 

# Create directories if they don't exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, ERA5_PRECIP_DIR, ERA5_TEMP_DIR]:
    path.mkdir(parents=True, exist_ok=True)

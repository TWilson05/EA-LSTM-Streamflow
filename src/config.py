from pathlib import Path

# This finds the directory containing config.py (src/), then gets its parent (root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define standard subdirectories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Define start and end years
DATA_START_YEAR = 1980
DATA_END_YEAR = 2022

# Define train/validation/test splits
TRAIN_START_YEAR = DATA_START_YEAR
TRAIN_END_YEAR = 2005
VAL_START_YEAR = 2006
VAL_END_YEAR = 2012
TEST_START_YEAR = 2013
TEST_END_YEAR = DATA_END_YEAR

### RAW DATA

# ERA5 Specific Paths
ERA5_DIR = RAW_DATA_DIR / "era5"
ERA5_PRECIP_DIR = ERA5_DIR / "precipitation"
ERA5_TEMP_DIR = ERA5_DIR / "temperature"
ERA5_RAD_DIR = ERA5_DIR / "radiation"
# processed ERA5 data
CLIMATE_OUTPUT_DIR = PROCESSED_DATA_DIR / "climate"

# Used for dem and era5 download ranges
SPATIAL_BOUNDS = RAW_DATA_DIR / "spatial_bounds.csv"

# Drainage Files
DRAINAGE_DIR = RAW_DATA_DIR / "drainage_areas"
DRAINAGE_FILES = [
    DRAINAGE_DIR / "MDA_ADP_05.gpkg",
    DRAINAGE_DIR / "MDA_ADP_07.gpkg",
    DRAINAGE_DIR / "MDA_ADP_08.gpkg"
]

# Elevation data files
ELEVATION_DIR = RAW_DATA_DIR / "dem_data"

# Glacier data files
GLACIER_SHP_PATH = RAW_DATA_DIR / "RGI-western-canada" / "02_rgi60_WesternCanadaUS.shp"
GLACIER_AREA_PATH = RAW_DATA_DIR / "mass_balance" / "area_dyn.csv"
MASS_BALANCE_PATH = RAW_DATA_DIR / "mass_balance" / "ts_monthly_const_area_lstm.csv"

# Static attributes
OUTPUT_STATIC_ATTR = PROCESSED_DATA_DIR / "static_attributes.csv"
OUTPUT_GLACIER_VOL = PROCESSED_DATA_DIR / "glacier_volume_change.csv"

# Lake cover
LAKE_COVER = RAW_DATA_DIR / "lake_cover.nc"


# Create directories if they don't exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
             ERA5_PRECIP_DIR, ERA5_TEMP_DIR, ERA5_RAD_DIR,
             CLIMATE_OUTPUT_DIR, ELEVATION_DIR, OUTPUT_DATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

import math
import warnings
import pandas as pd
import geopandas as gpd
from src.config import RAW_DATA_DIR, DRAINAGE_FILES

def filter_stations_by_annual_completeness(df, max_missing_pct=40.0):
    """
    Filters out stations that have more than max_missing_pct missing data 
    in ANY of the years present in the dataset.
    """
    df.index = pd.to_datetime(df.index)
    
    # 1. Group by year and calculate the percentage of NaNs per year per station
    # This creates a DataFrame where index=Year and columns=Station IDs
    annual_nan_pct = df.groupby(df.index.year).apply(lambda x: x.isna().mean() * 100)
    
    # 2. Identify stations where the maximum annual missingness is below the threshold
    # .max() checks across all years for each station
    stations_to_keep = annual_nan_pct.columns[annual_nan_pct.max() <= max_missing_pct]
    stations_to_drop = annual_nan_pct.columns[annual_nan_pct.max() > max_missing_pct]
    
    print(f"Filtering at {max_missing_pct}% annual threshold:")
    print(f" - Keeping {len(stations_to_keep)} stations.")
    print(f" - Dropping {len(stations_to_drop)} stations due to incomplete years.")

    return df[stations_to_keep]

def filter_stations_by_mda(df, mda_codes=["05", "07"]):
    """
    Filters columns (stations) to only include those starting with specific
    MDA (Major Drainage Area) codes.
    
    Args:
        df: DataFrame with station IDs as columns.
        mda_codes: List of strings (e.g. ["05", "07"]).
    """
    # Create a tuple for startswith (it requires a tuple, not a list)
    prefixes = tuple(mda_codes)
    
    # Identify matching columns
    valid_stations = [col for col in df.columns if col.startswith(prefixes)]
    dropped_stations = set(df.columns) - set(valid_stations)
    
    print(f"Filtering by MDA codes {mda_codes}:")
    print(f" - Keeping {len(valid_stations)} stations.")
    print(f" - Dropping {len(dropped_stations)} stations (wrong region).")
    
    return df[valid_stations]

def compute_and_save_bounds(stations_list=None):
    """
    1. Loads all basin shapefiles.
    2. Filters for the requested stations (if provided).
    3. Computes the total bounding box (North, South, East, West).
    4. Saves these bounds to a CSV for ERA5 and DEM scripts to use.
    """
    print("⏳ Computing spatial bounds from basin files...")
    
    basins_list = []
    for path in DRAINAGE_FILES:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                # Flexible layer reading
                gdf = gpd.read_file(path, layer='DrainageBasin_BassinDeDrainage')
            
            # Normalize ID column
            id_col = next((c for c in gdf.columns if c.lower() in ['stationnum', 'id', 'station_id']), None)
            if id_col:
                gdf = gdf[[id_col, 'geometry']].rename(columns={id_col: 'station_id'})
                basins_list.append(gdf)
        except Exception as e:
            print(f"⚠️ Warning: Could not load {path.name}: {e}")

    if not basins_list:
        raise ValueError("❌ No basin files loaded. Check DRAINAGE_FILES in config.")

    gdf_all = pd.concat(basins_list, ignore_index=True)
    
    # Filter if specific stations requested
    if stations_list:
        gdf_all['station_id'] = gdf_all['station_id'].astype(str).str.strip()
        requested = set([s.strip() for s in stations_list])
        gdf_all = gdf_all[gdf_all['station_id'].isin(requested)]

    if gdf_all.empty:
        raise ValueError("❌ No basins matched the station list.")

    # Convert to WGS84 to get Lat/Lon
    gdf_wgs84 = gdf_all.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    
    # Round bounds (Floor min, Ceil max) + Buffer (0.5 deg) for safety
    bounds = {
        'north': math.ceil(maxy + 0.5),
        'south': math.floor(miny - 0.5),
        'east': math.ceil(maxx + 0.5),
        'west': math.floor(minx - 0.5)
    }
    
    # Save to CSV for other scripts
    bounds_path = RAW_DATA_DIR / "spatial_bounds.csv"
    pd.DataFrame([bounds]).to_csv(bounds_path, index=False)
    print(f"✅ Spatial bounds saved to {bounds_path}")
    print(f"   Bounds: {bounds}")
    
    return bounds

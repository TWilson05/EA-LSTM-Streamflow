import geopandas as gpd
import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR, OUTPUT_STATIC_ATTR, OUTPUT_GLACIER_VOL

# Standard Equal Area projection for Western Canada
CANADA_ALBERS_CRS = "+proj=aea +lat_1=50 +lat_2=70 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"

def load_study_stations():
    """Reads headers from combined_streamflow to get the station list."""
    path = RAW_DATA_DIR / "combined_streamflow.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    
    # Read only the header
    df = pd.read_csv(path, nrows=0)
    # Assuming the first column is Date, rest are IDs
    return [col for col in df.columns if col.lower() not in ['date', 'day', 'time']]

def process_spatial_attributes(basin_gpkg_paths, glacier_shp_path, mass_balance_path):
    """
    Main workflow to calculate static basin attributes and glacier volume changes.
    """
    # ---------------------------------------------------------
    # 1. LOAD AND MERGE BASINS
    # ---------------------------------------------------------
    print("⏳ Loading and merging basin files...")
    study_stations = load_study_stations()
    
    basins_list = []
    for path in basin_gpkg_paths:
        # Load file
        gdf = gpd.read_file(path)
        # Standardize ID column (Look for 'StationNum', 'ID', etc)
        id_col = next((c for c in gdf.columns if c.lower() in ['stationnum', 'id', 'station_id']), None)
        if id_col:
            gdf = gdf.rename(columns={id_col: 'station_id'})
            basins_list.append(gdf)
        else:
            print(f"⚠️ Warning: No ID column found in {path.name}")

    if not basins_list:
        raise ValueError("No valid basin data loaded.")

    gdf_basins = pd.concat(basins_list, ignore_index=True)
    
    # ---------------------------------------------------------
    # 2. FILTER & CHECK MISSING STATIONS
    # ---------------------------------------------------------
    # Clean whitespace
    gdf_basins['station_id'] = gdf_basins['station_id'].astype(str).str.strip()
    study_stations = [s.strip() for s in study_stations]

    # Check for missing
    found_stations = set(gdf_basins['station_id'])
    requested_stations = set(study_stations)
    missing = requested_stations - found_stations
    
    if missing:
        print(f"⚠️ Warning: {len(missing)} stations from streamflow CSV have no drainage polygon.")
        print(f"   Examples: {list(missing)[:5]}")
    
    # Filter to only study stations
    gdf_basins = gdf_basins[gdf_basins['station_id'].isin(requested_stations)].copy()
    print(f"✅ Processing {len(gdf_basins)} basins.")

    # ---------------------------------------------------------
    # 3. LOAD GLACIERS & REPROJECT
    # ---------------------------------------------------------
    print("⏳ Loading glaciers and reprojecting...")
    gdf_glaciers = gpd.read_file(glacier_shp_path)
    
    # Ensure RGIId exists
    rgi_col = next((c for c in gdf_glaciers.columns if 'rgiid' in c.lower()), 'RGIId')
    gdf_glaciers = gdf_glaciers.rename(columns={rgi_col: 'RGIId'})

    # Reproject to Albers Equal Area
    gdf_basins = gdf_basins.to_crs(CANADA_ALBERS_CRS)
    gdf_glaciers = gdf_glaciers.to_crs(CANADA_ALBERS_CRS)

    # ---------------------------------------------------------
    # 4. CALCULATE BASIN AREAS
    # ---------------------------------------------------------
    # Calculate total basin area in km² (geometry.area is in m²)
    gdf_basins['basin_area_km2'] = gdf_basins.geometry.area / 1e6
    
    # ---------------------------------------------------------
    # 5. SPATIAL INTERSECTION (The Fast Way)
    # ---------------------------------------------------------
    print("⏳ Calculating glacier-basin intersections (this may take a moment)...")
    
    # overlay is much faster than iterating row-by-row
    intersection = gpd.overlay(
        gdf_glaciers[['RGIId', 'geometry']], 
        gdf_basins[['station_id', 'geometry']], 
        how='intersection'
    )
    
    # Calculate area of the intersection pieces (Glacier parts inside basins)
    intersection['glacier_area_km2'] = intersection.geometry.area / 1e6

    # ---------------------------------------------------------
    # 6. STATIC ATTRIBUTES OUTPUT
    # ---------------------------------------------------------
    # Sum glacier area per basin
    glacier_sums = intersection.groupby('station_id')['glacier_area_km2'].sum()
    
    # Merge back to the main basin list (left join to keep 0% glaciers)
    static_df = gdf_basins[['station_id', 'basin_area_km2']].set_index('station_id')
    static_df['glacier_area_km2'] = glacier_sums
    static_df['glacier_area_km2'] = static_df['glacier_area_km2'].fillna(0)
    
    # Calculate percentage
    static_df['glacier_pct'] = (static_df['glacier_area_km2'] / static_df['basin_area_km2']) * 100
    
    static_df.to_csv(OUTPUT_STATIC_ATTR)
    print(f"✅ Static attributes saved to {OUTPUT_STATIC_ATTR}")

    # ---------------------------------------------------------
    # 7. VOLUME CHANGE CALCULATION
    # ---------------------------------------------------------
    print("⏳ Calculating volume changes...")
    
    # Pivot: Index=RGIId, Col=StationID, Value=Area_km2
    # This creates the "Area Matrix" (A)
    area_matrix = intersection.pivot_table(
        index='RGIId', 
        columns='station_id', 
        values='glacier_area_km2', 
        aggfunc='sum',
        fill_value=0
    )

    # Load Mass Balance Data (M)
    # Rows=RGIId, Cols=Dates (YYYY-MM)
    mb_df = pd.read_csv(mass_balance_path, index_col=0)
    
    # Align Indices (Keep only glaciers present in both files)
    common_glaciers = area_matrix.index.intersection(mb_df.index)
    print(f"   Matched {len(common_glaciers)} glaciers between Shapefile and Mass Balance CSV.")
    
    A = area_matrix.loc[common_glaciers]
    M = mb_df.loc[common_glaciers]
    
    # Math: Volume Change = (MassBalance * Area)
    # Units: MB is likely in Meters (m w.e.). Area is in km² (10^6 m²).
    # Result = m * 10^6 m² = 10^6 m³ (Million Cubic Meters - MCM)
    
    # We want result: Index=Date, Col=Station
    # M.T is (Time x Glaciers)
    # A   is (Glaciers x Stations)
    # Dot Product -> (Time x Stations)
    
    vol_change = M.T.dot(A)
    
    # Format index to datetime
    vol_change.index = pd.to_datetime(vol_change.index)
    
    vol_change.to_csv(OUTPUT_GLACIER_VOL)
    print(f"✅ Monthly volume changes (MCM) saved to {OUTPUT_GLACIER_VOL}")

    return static_df, vol_change

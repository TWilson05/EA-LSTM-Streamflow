import geopandas as gpd
import pandas as pd
import warnings
from src.config import OUTPUT_STATIC_ATTR, OUTPUT_GLACIER_VOL

# Standard Equal Area projection for Western Canada
CANADA_ALBERS_CRS = "+proj=aea +lat_1=50 +lat_2=70 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"

def process_spatial_attributes(basin_gpkg_paths, glacier_shp_path, mass_balance_path, stations_list):
    """
    Main workflow to calculate static basin attributes and glacier volume changes.
    Args:
        basin_gpkg_paths: List of paths to .gpkg files.
        glacier_shp_path: Path to RGI shapefile.
        mass_balance_path: Path to mass balance CSV.
        stations_list: List of station IDs (str) to analyze.
    """
    # ---------------------------------------------------------
    # 1. LOAD AND MERGE BASINS (With Warning Fixes)
    # ---------------------------------------------------------
    print("⏳ Loading and merging basin files...")
    
    basins_list = []
    for path in basin_gpkg_paths:
        try:
            # FIX: Specify the layer to avoid "Multiple layers" warning
            # FIX: Suppress RuntimeWarnings for weird date columns in HYDAT
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                gdf = gpd.read_file(path, layer='DrainageBasin_BassinDeDrainage')
                
            # Standardize ID column
            id_col = next((c for c in gdf.columns if c.lower() in ['stationnum', 'id', 'station_id']), None)
            
            if id_col:
                # Keep only what we need: ID and Geometry
                gdf = gdf[[id_col, 'geometry']].rename(columns={id_col: 'station_id'})
                basins_list.append(gdf)
            else:
                print(f"⚠️ Warning: No ID column found in {path.name}")
                
        except Exception as e:
            print(f"❌ Error loading {path.name}: {e}")

    if not basins_list:
        raise ValueError("No valid basin data loaded.")

    gdf_basins = pd.concat(basins_list, ignore_index=True)
    
    # ---------------------------------------------------------
    # 2. FILTER TO MATCH STUDY STATIONS
    # ---------------------------------------------------------
    # Clean whitespace
    gdf_basins['station_id'] = gdf_basins['station_id'].astype(str).str.strip()
    requested_stations = set([s.strip() for s in stations_list])

    # Check for missing polygons
    found_stations = set(gdf_basins['station_id'])
    missing_polygons = requested_stations - found_stations
    
    if missing_polygons:
        print(f"⚠️ Warning: {len(missing_polygons)} stations in your list have no drainage polygon.")
        # Optional: Print first 5 missing to help debug
        # print(f"   Examples: {list(missing_polygons)[:5]}")
    
    # Filter: Keep only basins that are in our requested list
    gdf_basins = gdf_basins[gdf_basins['station_id'].isin(requested_stations)].copy()
    print(f"✅ Processing {len(gdf_basins)} basins.")

    # ---------------------------------------------------------
    # 3. LOAD GLACIERS & REPROJECT
    # ---------------------------------------------------------
    print("⏳ Loading glaciers and reprojecting...")
    gdf_glaciers = gpd.read_file(glacier_shp_path)
    
    rgi_col = next((c for c in gdf_glaciers.columns if 'rgiid' in c.lower()), 'RGIId')
    gdf_glaciers = gdf_glaciers.rename(columns={rgi_col: 'RGIId'})

    gdf_basins = gdf_basins.to_crs(CANADA_ALBERS_CRS)
    gdf_glaciers = gdf_glaciers.to_crs(CANADA_ALBERS_CRS)

    # ---------------------------------------------------------
    # 4. CALCULATE BASIN AREAS & INTERSECTIONS
    # ---------------------------------------------------------
    gdf_basins['basin_area_km2'] = gdf_basins.geometry.area / 1e6
    
    print("⏳ Calculating glacier-basin intersections...")
    intersection = gpd.overlay(
        gdf_glaciers[['RGIId', 'geometry']], 
        gdf_basins[['station_id', 'geometry']], 
        how='intersection'
    )
    
    intersection['glacier_area_km2'] = intersection.geometry.area / 1e6

    # ---------------------------------------------------------
    # 5. STATIC ATTRIBUTES OUTPUT
    # ---------------------------------------------------------
    glacier_sums = intersection.groupby('station_id')['glacier_area_km2'].sum()
    
    static_df = gdf_basins[['station_id', 'basin_area_km2']].set_index('station_id')
    static_df['glacier_area_km2'] = glacier_sums
    static_df['glacier_area_km2'] = static_df['glacier_area_km2'].fillna(0)
    static_df['glacier_pct'] = (static_df['glacier_area_km2'] / static_df['basin_area_km2']) * 100
    
    # Save
    static_df.to_csv(OUTPUT_STATIC_ATTR)
    print(f"✅ Static attributes saved to {OUTPUT_STATIC_ATTR}")

    # ---------------------------------------------------------
    # 6. VOLUME CHANGE CALCULATION
    # ---------------------------------------------------------
    print("⏳ Calculating volume changes...")
    
    area_matrix = intersection.pivot_table(
        index='RGIId', columns='station_id', values='glacier_area_km2', 
        aggfunc='sum', fill_value=0
    )

    mb_df = pd.read_csv(mass_balance_path, index_col=0)
    
    # Align indices
    common_glaciers = area_matrix.index.intersection(mb_df.index)
    
    if len(common_glaciers) == 0:
        print("❌ Error: No common glaciers found between Mass Balance CSV and Shapefiles.")
        return static_df, None
        
    print(f"   Matched {len(common_glaciers)} glaciers.")
    
    vol_change = mb_df.loc[common_glaciers].T.dot(area_matrix.loc[common_glaciers])
    vol_change.index = pd.to_datetime(vol_change.index)
    
    vol_change.to_csv(OUTPUT_GLACIER_VOL)
    print(f"✅ Monthly volume changes (MCM) saved to {OUTPUT_GLACIER_VOL}")

    return static_df, vol_change
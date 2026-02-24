import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import warnings
from rasterstats import zonal_stats
from src.config import (
    DRAINAGE_FILES, 
    ELEVATION_DIR, 
    GLACIER_SHP_PATH,
    MASS_BALANCE_PATH,
    OUTPUT_STATIC_ATTR,
    OUTPUT_GLACIER_VOL
)

# Standard Equal Area projection for Western Canada
CANADA_ALBERS_CRS = "+proj=aea +lat_1=50 +lat_2=70 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"

def generate_slope_raster(dem_path, slope_path):
    """
    Reads the DEM in small chunks (windows) to prevent memory errors,
    calculates the topographic slope, and saves it to a new GeoTIFF.
    """
    if slope_path.exists():
        print("   ℹ️ Using cached slope raster.")
        return slope_path
        
    print("   ⏳ Generating slope raster via memory-safe chunking...")
    
    with rasterio.open(dem_path) as src:
        kwargs = src.meta.copy()
        # Force output to 32-bit float to save disk space
        kwargs.update(dtype=rasterio.float32, nodata=-9999.0)
        dx, dy = src.res
        
        with rasterio.open(slope_path, 'w', **kwargs) as dst:
            # Process the raster in small memory-safe blocks
            for ji, window in src.block_windows(1):
                # Explicitly cast to float32
                elev = src.read(1, window=window).astype(np.float32)
                
                # gradient requires at least a 2x2 array
                if elev.shape[0] < 2 or elev.shape[1] < 2:
                    slope = np.zeros_like(elev)
                else:
                    dy_grad, dx_grad = np.gradient(elev, dy, dx)
                    slope = np.arctan(np.sqrt(dx_grad**2 + dy_grad**2)) * (180.0 / np.pi)
                
                # Apply nodata mask
                if src.nodata is not None:
                    slope[elev == src.nodata] = -9999.0
                    
                dst.write(slope.astype(rasterio.float32), 1, window=window)
                
    print(f"   ✅ Saved slope raster to {slope_path}")
    return slope_path

def process_spatial_attributes(stations_list):
    """
    Computes static attributes and glacier volume changes using:
    1. Pre-downloaded DEM (ELEVATION_DIR/western_canada_dem.tif)
    2. Basin Shapefiles (DRAINAGE_FILES)
    3. Glacier Shapefiles (GLACIER_SHP_PATH)
    """
    
    # --- 1. Load Basins ---
    print("⏳ Loading and merging basin files...")
    basins_list = []
    for path in DRAINAGE_FILES:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                gdf = gpd.read_file(path, layer='DrainageBasin_BassinDeDrainage')
                
            id_col = next((c for c in gdf.columns if c.lower() in ['stationnum', 'id', 'station_id']), None)
            if id_col:
                gdf = gdf[[id_col, 'geometry']].rename(columns={id_col: 'station_id'})
                basins_list.append(gdf)
        except Exception as e:
            print(f"❌ Error loading {path.name}: {e}")

    if not basins_list:
        raise RuntimeError("No basin files loaded.")

    gdf_basins = pd.concat(basins_list, ignore_index=True)
    
    # Filter to requested stations
    gdf_basins['station_id'] = gdf_basins['station_id'].astype(str).str.strip()
    requested_stations = set([s.strip() for s in stations_list])
    gdf_basins = gdf_basins[gdf_basins['station_id'].isin(requested_stations)].copy()
    print(f"✅ Processing {len(gdf_basins)} basins.")

    # --- 2. Compute Elevation & Slope ---
    print("⏳ Computing Basin Elevations and Slopes...")
    dem_path = ELEVATION_DIR / "western_canada_dem.tif"
    slope_path = ELEVATION_DIR / "western_canada_slope.tif"
    
    if not dem_path.exists():
        raise FileNotFoundError(f"❌ DEM not found at {dem_path}. Run data_ingestion.download_aws_dem first!")

    # Generate slope raster if it doesn't exist yet
    generate_slope_raster(dem_path, slope_path)

    # Reproject Basins to DEM CRS (EPSG:3857 for AWS tiles)
    basins_proj = gdf_basins.to_crs("EPSG:3857")
    
    # Calculate Elevation Stats (Mean & Std)
    elev_stats = zonal_stats(basins_proj, str(dem_path), stats="mean std")
    gdf_basins['mean_elev'] = [s['mean'] for s in elev_stats]
    gdf_basins['std_elev'] = [s['std'] for s in elev_stats]

    # Calculate Slope Stats (Mean & Std)
    slope_stats = zonal_stats(basins_proj, str(slope_path), stats="mean std")
    gdf_basins['mean_slope'] = [s['mean'] for s in slope_stats]
    gdf_basins['std_slope'] = [s['std'] for s in slope_stats]

    # Fill Missing Data with medians across all stations
    for col in ['mean_elev', 'std_elev', 'mean_slope', 'std_slope']:
        missing = gdf_basins[col].isna().sum()
        if missing > 0:
            print(f"   ⚠️ Warning: {missing} basins missing {col}. Filling with median.")
            gdf_basins[col] = gdf_basins[col].fillna(gdf_basins[col].median())

    # --- 3. Compute Areas (Reproject to Albers) ---
    gdf_basins = gdf_basins.to_crs(CANADA_ALBERS_CRS)
    gdf_basins['basin_area_km2'] = gdf_basins.geometry.area / 1e6

    # --- 4. Glacier Intersection ---
    print("⏳ Intersecting Glaciers...")
    gdf_glaciers = gpd.read_file(GLACIER_SHP_PATH)
    rgi_col = next((c for c in gdf_glaciers.columns if 'rgiid' in c.lower()), 'RGIId')
    gdf_glaciers = gdf_glaciers.rename(columns={rgi_col: 'RGIId'})
    gdf_glaciers = gdf_glaciers.to_crs(CANADA_ALBERS_CRS)

    intersection = gpd.overlay(
        gdf_glaciers[['RGIId', 'geometry']], 
        gdf_basins[['station_id', 'geometry']], 
        how='intersection'
    )
    intersection['glacier_area_km2'] = intersection.geometry.area / 1e6

    # --- 5. Save Static Attributes ---
    glacier_sums = intersection.groupby('station_id')['glacier_area_km2'].sum()
    
    # Update dataframe to include the new columns
    static_df = gdf_basins[[
        'station_id', 'basin_area_km2', 'mean_elev', 'std_elev', 'mean_slope', 'std_slope'
    ]].set_index('station_id')
    
    static_df['glacier_area_km2'] = glacier_sums
    static_df['glacier_area_km2'] = static_df['glacier_area_km2'].fillna(0)
    static_df['glacier_pct'] = (static_df['glacier_area_km2'] / static_df['basin_area_km2']) * 100
    
    static_df.to_csv(OUTPUT_STATIC_ATTR)
    print(f"✅ Static attributes saved to {OUTPUT_STATIC_ATTR}")

    # --- 6. Compute Volume Change ---
    print("⏳ Calculating volume changes...")
    area_matrix = intersection.pivot_table(
        index='RGIId', columns='station_id', values='glacier_area_km2', 
        aggfunc='sum', fill_value=0
    )

    mb_df = pd.read_csv(MASS_BALANCE_PATH, index_col=0)
    common_glaciers = area_matrix.index.intersection(mb_df.index)
    
    if len(common_glaciers) > 0:
        vol_change = mb_df.loc[common_glaciers].T.dot(area_matrix.loc[common_glaciers])
        vol_change.index = pd.to_datetime(vol_change.index)
        vol_change.to_csv(OUTPUT_GLACIER_VOL)
        print(f"✅ Volume changes saved to {OUTPUT_GLACIER_VOL}")
        return static_df, vol_change
    else:
        print("⚠️ No common glaciers found for volume calculation.")
        return static_df, None

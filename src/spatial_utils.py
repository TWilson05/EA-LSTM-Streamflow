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
    Reads the DEM in manual square chunks to prevent memory errors,
    calculates the 2D topographic slope, and saves it.
    """
    if slope_path.exists():
        print("   ℹ️ Using cached slope raster.")
        return slope_path
        
    print("   ⏳ Generating slope raster via 2D chunking...")
    
    with rasterio.open(dem_path) as src:
        kwargs = src.meta.copy()
        kwargs.update(dtype=rasterio.float32, nodata=-9999.0)
        dx, dy = src.res
        
        width = src.width
        height = src.height
        chunk_size = 2048 # ~16MB per chunk in memory
        
        with rasterio.open(slope_path, 'w', **kwargs) as dst:
            
            # Manually loop over the image in 2D squares
            for i in range(0, height, chunk_size):
                for j in range(0, width, chunk_size):
                    
                    w = min(chunk_size, width - j)
                    h = min(chunk_size, height - i)
                    window = rasterio.windows.Window(j, i, w, h)
                    
                    elev = src.read(1, window=window).astype(np.float32)
                    
                    if src.nodata is not None:
                        elev[elev == src.nodata] = np.nan
                        
                    if h < 2 or w < 2:
                        slope = np.zeros_like(elev)
                    else:
                        dy_grad, dx_grad = np.gradient(elev, dy, dx)
                        slope = np.arctan(np.sqrt(dx_grad**2 + dy_grad**2)) * (180.0 / np.pi)
                    
                    slope[np.isnan(slope)] = -9999.0
                    dst.write(slope.astype(rasterio.float32), 1, window=window)
                    
    print(f"   ✅ Saved slope raster to {slope_path}")
    return slope_path


def process_spatial_attributes(stations_list):
    """
    Computes static attributes and glacier volume changes.
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

    generate_slope_raster(dem_path, slope_path)

    basins_proj = gdf_basins.to_crs("EPSG:3857")
    
    elev_stats = zonal_stats(basins_proj, str(dem_path), stats="mean min max range")
    gdf_basins['mean_elev'] = [s['mean'] for s in elev_stats]
    gdf_basins['min_elev'] = [s['min'] for s in elev_stats]
    gdf_basins['max_elev'] = [s['max'] for s in elev_stats]
    gdf_basins['elev_range'] = [s['range'] for s in elev_stats]

    slope_stats = zonal_stats(basins_proj, str(slope_path), stats="mean")
    gdf_basins['mean_slope'] = [s['mean'] for s in slope_stats]

    # --- BATHYMETRY FIX ---
    # Coastal basins overlapping the ocean will pick up negative bathymetry values. 
    # We clip anything below 0 back to sea level.
    for col in ['min_elev', 'mean_elev']:
        gdf_basins[col] = gdf_basins[col].clip(lower=0.0)
        
    # --- RANGE CORRECTION ---
    # Recalculate elevation range to reflect the new clipped minimum
    gdf_basins['elev_range'] = gdf_basins['max_elev'] - gdf_basins['min_elev']

    for col in ['mean_elev', 'min_elev', 'max_elev', 'elev_range', 'mean_slope']:
        missing = gdf_basins[col].isna().sum()
        if missing > 0:
            print(f"   ⚠️ Warning: {missing} basins missing {col}. Filling with median.")
            gdf_basins[col] = gdf_basins[col].fillna(gdf_basins[col].median())

    # --- 3. Compute Areas (Reproject to Albers) ---
    gdf_basins = gdf_basins.to_crs(CANADA_ALBERS_CRS)
    gdf_basins['basin_area_km2'] = gdf_basins.geometry.area / 1e6

    # --- 4. Fast Glacier Intersection ---
    print("⏳ Intersecting Glaciers (Optimized sjoin method)...")
    gdf_glaciers = gpd.read_file(GLACIER_SHP_PATH)
    rgi_col = next((c for c in gdf_glaciers.columns if 'rgiid' in c.lower()), 'RGIId')
    gdf_glaciers = gdf_glaciers.rename(columns={rgi_col: 'RGIId'})
    gdf_glaciers = gdf_glaciers.to_crs(CANADA_ALBERS_CRS)

    gdf_glaciers['geometry'] = gdf_glaciers.geometry.simplify(10.0)
    basins_simplified = gdf_basins.copy()
    basins_simplified['geometry'] = basins_simplified.geometry.simplify(10.0)

    candidates = gpd.sjoin(
        gdf_glaciers[['RGIId', 'geometry']], 
        basins_simplified[['station_id', 'geometry']], 
        how='inner', 
        predicate='intersects'
    )

    candidates = candidates.merge(
        basins_simplified[['station_id', 'geometry']], 
        on='station_id', 
        suffixes=('_glac', '_basin')
    )

    glac_geoms = gpd.GeoSeries(candidates['geometry_glac'])
    basin_geoms = gpd.GeoSeries(candidates['geometry_basin'])
    candidates['glacier_area_km2'] = glac_geoms.intersection(basin_geoms).area / 1e6
    intersection = candidates[['RGIId', 'station_id', 'glacier_area_km2']]

    # --- 5. Save Static Attributes ---
    glacier_sums = intersection.groupby('station_id')['glacier_area_km2'].sum()
    
    # Update dataframe to reflect the removed lake_cover column
    static_df = gdf_basins[[
        'station_id', 'basin_area_km2', 'mean_elev', 'min_elev', 
        'max_elev', 'elev_range', 'mean_slope'
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

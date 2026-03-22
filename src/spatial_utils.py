import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import Affine
import xarray as xr
import warnings
from rasterstats import zonal_stats
from src.config import (
    DRAINAGE_FILES, 
    ELEVATION_DIR, 
    GLACIER_SHP_PATH,
    MASS_BALANCE_PATH,
    OUTPUT_STATIC_ATTR,
    OUTPUT_GLACIER_VOL,
    RAW_DATA_DIR # Added to locate lake_cover.nc
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

def get_lake_cover_stats(basins_gdf, nc_path):
    """
    Extracts mean lake cover fraction from an ERA5 NetCDF file for each basin.
    """
    if not nc_path.exists():
        print(f"   ⚠️ Warning: Lake cover file not found at {nc_path}. Filling with 0.")
        return [0.0] * len(basins_gdf)

    with xr.open_dataset(nc_path, engine='netcdf4') as ds:
        lat_name = next((v for v in ['latitude', 'lat'] if v in ds.coords), None)
        lon_name = next((v for v in ['longitude', 'lon'] if v in ds.coords), None)
        # Typically 'cl' or 'lake_cover'
        var_name = next((v for v in ds.data_vars), None) 
        
        lats = ds[lat_name].values
        lons = ds[lon_name].values
        data = ds[var_name].values

        # ERA5 longitudes are often 0-360. Convert to -180 to 180
        if lons.max() > 180:
            lons = np.where(lons > 180, lons - 360, lons)

        # Drop time dimension if it exists (assuming static file shape is 1, lat, lon)
        if data.ndim == 3:
            data = data[0]

        # Calculate Affine transform for rasterstats
        # ERA5 typically has descending latitudes and ascending longitudes
        dlon = lons[1] - lons[0]
        dlat = lats[1] - lats[0] 
        transform = Affine.translation(lons[0] - dlon/2, lats[0] - dlat/2) * Affine.scale(dlon, dlat)

        # Reproject basins to Lat/Lon to match ERA5
        basins_4326 = basins_gdf.to_crs("EPSG:4326")
        
        # Calculate mean lake cover
        stats = zonal_stats(basins_4326, data, affine=transform, stats="mean", nodata=np.nan)
        
        # Multiply by 100 to make it a percentage (matching glacier_pct)
        return [s['mean'] * 100 if s['mean'] is not None else 0.0 for s in stats]

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

    for col in ['mean_elev', 'min_elev', 'max_elev', 'elev_range', 'mean_slope']:
        missing = gdf_basins[col].isna().sum()
        if missing > 0:
            print(f"   ⚠️ Warning: {missing} basins missing {col}. Filling with median.")
            gdf_basins[col] = gdf_basins[col].fillna(gdf_basins[col].median())

    # --- 3. Compute Lake Cover ---
    print("⏳ Computing Lake Cover...")
    lake_nc_path = RAW_DATA_DIR / "lake_cover.nc"
    gdf_basins['lake_cover_pct'] = get_lake_cover_stats(gdf_basins, lake_nc_path)

    # --- 4. Compute Areas (Reproject to Albers) ---
    gdf_basins = gdf_basins.to_crs(CANADA_ALBERS_CRS)
    gdf_basins['basin_area_km2'] = gdf_basins.geometry.area / 1e6

    # --- 5. Glacier Intersection ---
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

    # --- 6. Save Static Attributes ---
    glacier_sums = intersection.groupby('station_id')['glacier_area_km2'].sum()
    
    # Update dataframe to include the new columns
    static_df = gdf_basins[[
        'station_id', 'basin_area_km2', 'mean_elev', 'min_elev', 
        'max_elev', 'elev_range', 'mean_slope', 'lake_cover_pct'
    ]].set_index('station_id')
    
    static_df['glacier_area_km2'] = glacier_sums
    static_df['glacier_area_km2'] = static_df['glacier_area_km2'].fillna(0)
    static_df['glacier_pct'] = (static_df['glacier_area_km2'] / static_df['basin_area_km2']) * 100
    
    static_df.to_csv(OUTPUT_STATIC_ATTR)
    print(f"✅ Static attributes saved to {OUTPUT_STATIC_ATTR}")

    # --- 7. Compute Volume Change ---
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

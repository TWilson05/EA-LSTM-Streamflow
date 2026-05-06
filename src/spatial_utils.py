import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
import warnings

# Use your exact config setup
from src.config import (
    DRAINAGE_FILES, 
    ELEVATION_DIR, 
    GLACIER_SHP_PATH,
    MASS_BALANCE_PATH,
    OUTPUT_STATIC_ATTR,
    OUTPUT_GLACIER_VOL,
    RAW_DATA_DIR
)
STATION_METADATA_PATH = RAW_DATA_DIR / "station_metadata.csv"

# Standard Equal Area projection for Western Canada
CANADA_ALBERS_CRS = "+proj=aea +lat_1=50 +lat_2=70 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"

def process_spatial_attributes(stations_list):
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

    # --- 2. Basin-by-Basin Topography (In-Memory) ---
    print("⏳ Extracting Elevation and Topography Basin-by-Basin (Zero Disk Storage)...")
    dem_raw = ELEVATION_DIR / "western_canada_dem.tif"
    
    if not dem_raw.exists():
        raise FileNotFoundError(f"❌ DEM not found at {dem_raw}. Run data_ingestion.download_aws_dem first!")

    stats = {
        'mean_elev': [], 'min_elev': [], 'max_elev': [], 'elev_range': [],
        'median_elev': [], 'lower_quartile_elev': [], 'upper_quartile_elev': [],
        'mean_slope': [], 'mean_sin_aspect': [], 'mean_cos_aspect': [],
        'north_facing_pct': [], 'south_facing_pct': []
    }

    with rasterio.open(dem_raw) as src:
        # Match basins to the exact projection of the DEM
        basins_proj = gdf_basins.to_crs(src.crs)
        dx, dy = src.res

        # Suppress Numpy runtime warnings for empty arrays (e.g. perfectly flat basins)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            for idx, row in basins_proj.iterrows():
                try:
                    # 1. Cookie-cutter the DEM to exactly this basin's boundary
                    # Everything outside the basin becomes 'nodata'
                    out_image, _ = mask(src, [row.geometry], crop=True)
                    elev = out_image[0].astype(np.float32)

                    # 2. Scrub Bathymetry and Nodata
                    if src.nodata is not None:
                        elev[elev == src.nodata] = np.nan
                    elev[elev < 0] = np.nan  # Drop the ocean floor

                    # Check if the basin successfully captured data
                    valid_mask = ~np.isnan(elev)
                    if not valid_mask.any():
                        for k in stats: stats[k].append(np.nan)
                        continue

                    # 3. Compute Elevation Stats instantly from the RAM array
                    valid_elev = elev[valid_mask]
                    stats['mean_elev'].append(np.mean(valid_elev))
                    stats['min_elev'].append(np.min(valid_elev))
                    stats['max_elev'].append(np.max(valid_elev))
                    stats['elev_range'].append(np.max(valid_elev) - np.min(valid_elev))
                    stats['median_elev'].append(np.median(valid_elev))
                    stats['lower_quartile_elev'].append(np.percentile(valid_elev, 25))
                    stats['upper_quartile_elev'].append(np.percentile(valid_elev, 75))

                    # 4. Compute Topography Math in RAM
                    dy_grad, dx_grad = np.gradient(elev, dy, dx)
                    
                    slope_arr = np.arctan(np.sqrt(dx_grad**2 + dy_grad**2)) * (180.0 / np.pi)
                    aspect_rad = np.arctan2(-dx_grad, dy_grad)
                    aspect_deg = (np.degrees(aspect_rad) + 360) % 360

                    sin_arr = np.sin(np.radians(aspect_deg))
                    cos_arr = np.cos(np.radians(aspect_deg))
                    
                    north_arr = ((aspect_deg >= 315) | (aspect_deg < 45)).astype(np.float32)
                    south_arr = ((aspect_deg >= 135) & (aspect_deg < 225)).astype(np.float32)

                    # Apply Flat Mask (Drop slopes < 0.1 deg from aspect stats)
                    flat_mask = slope_arr < 0.1
                    sin_arr[flat_mask] = np.nan
                    cos_arr[flat_mask] = np.nan
                    north_arr[flat_mask] = np.nan
                    south_arr[flat_mask] = np.nan

                    # 5. Extract Topo Stats
                    # We use a combined mask because np.gradient creates a 1-pixel border of NaNs
                    valid_slope_mask = valid_mask & ~np.isnan(slope_arr)
                    
                    stats['mean_slope'].append(np.nanmean(slope_arr[valid_slope_mask]))
                    stats['mean_sin_aspect'].append(np.nanmean(sin_arr[valid_slope_mask]))
                    stats['mean_cos_aspect'].append(np.nanmean(cos_arr[valid_slope_mask]))
                    
                    # Mean of binary array * 100 = Percentage
                    stats['north_facing_pct'].append(np.nanmean(north_arr[valid_slope_mask]) * 100)
                    stats['south_facing_pct'].append(np.nanmean(south_arr[valid_slope_mask]) * 100)

                except Exception as e:
                    print(f"   ⚠️ Error processing topography for basin {row['station_id']}: {e}")
                    for k in stats: stats[k].append(np.nan)

    # Attach stats back to the main dataframe
    for k in stats:
        gdf_basins[k] = stats[k]

    # Fill any straggling NaNs with medians
    topo_cols = list(stats.keys())
    for col in topo_cols:
        missing = gdf_basins[col].isna().sum()
        if missing > 0:
            print(f"   ⚠️ Warning: {missing} basins missing {col}. Filling with median.")
            gdf_basins[col] = gdf_basins[col].fillna(gdf_basins[col].median())

    # --- 3. Merge Station Metadata (Lat / Lon) ---
    print("⏳ Merging Station Coordinates...")
    try:
        metadata_df = pd.read_csv(STATION_METADATA_PATH)
        
        # We know the exact column names, so we just declare them directly
        id_col = 'Station Number'
        lat_col = 'Latitude'
        lon_col = 'Longitude'
        
        # Verify those columns actually exist in the CSV to prevent crashes
        missing_cols = [col for col in [id_col, lat_col, lon_col] if col not in metadata_df.columns]
        
        if not missing_cols:
            # Clean the IDs to ensure they match perfectly (removes invisible spaces)
            metadata_df[id_col] = metadata_df[id_col].astype(str).str.strip()
            
            gdf_basins = gdf_basins.merge(
                metadata_df[[id_col, lat_col, lon_col]], 
                left_on='station_id', right_on=id_col, how='left'
            )
            
            # Rename for standardization across your project
            gdf_basins = gdf_basins.rename(columns={lat_col: 'latitude', lon_col: 'longitude'})
            
            # Drop the duplicate ID column we just merged in
            if id_col != 'station_id':
                gdf_basins = gdf_basins.drop(columns=[id_col])
        else:
            print(f"   ⚠️ Could not find these exact columns in the metadata: {missing_cols}")
            
    except Exception as e:
        print(f"   ⚠️ Failed to load or merge metadata: {e}")

    # --- 4. Compute Areas (Reproject to Albers) ---
    gdf_basins = gdf_basins.to_crs(CANADA_ALBERS_CRS)
    gdf_basins['basin_area_km2'] = gdf_basins.geometry.area / 1e6

    # --- 5. Fast Glacier Intersection ---
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
        how='inner', predicate='intersects'
    )

    candidates = candidates.merge(
        basins_simplified[['station_id', 'geometry']], 
        on='station_id', suffixes=('_glac', '_basin')
    )

    glac_geoms = gpd.GeoSeries(candidates['geometry_glac'])
    basin_geoms = gpd.GeoSeries(candidates['geometry_basin'])
    candidates['glacier_area_km2'] = glac_geoms.intersection(basin_geoms).area / 1e6
    intersection = candidates[['RGIId', 'station_id', 'glacier_area_km2']]

    # --- 6. Save Static Attributes ---
    glacier_sums = intersection.groupby('station_id')['glacier_area_km2'].sum()
    
    final_cols = ['station_id', 'basin_area_km2'] + topo_cols
    if 'latitude' in gdf_basins.columns and 'longitude' in gdf_basins.columns:
        final_cols.extend(['latitude', 'longitude'])
    
    static_df = gdf_basins[final_cols].set_index('station_id')
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

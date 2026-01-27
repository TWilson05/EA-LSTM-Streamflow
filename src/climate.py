import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from tqdm import tqdm
from src.config import CLIMATE_OUTPUT_DIR, ERA5_PRECIP_DIR, ERA5_TEMP_DIR
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='cfgrib')

def get_grid_info(sample_path, engine=None):
    """
    Extracts 1D latitude and longitude arrays.
    detects if latitude is descending (typical for ERA5).
    """
    try:
        ds = xr.open_dataset(sample_path, engine=engine)
    except Exception:
        ds = xr.open_dataset(sample_path, engine='cfgrib')

    lat_name = next((v for v in ['latitude', 'lat'] if v in ds.coords), None)
    lon_name = next((v for v in ['longitude', 'lon'] if v in ds.coords), None)

    if not lat_name or not lon_name:
        raise ValueError(f"Could not find lat/lon in {sample_path}")

    lats = ds[lat_name].values
    lons = ds[lon_name].values
    
    ds.close()
    return lats, lons

def normalize_longitudes(lons):
    """
    ERA5 is often 0-360. Shapefiles are often -180 to 180.
    This converts 0-360 lons to -180 to 180 for matching.
    """
    lons_norm = lons.copy()
    mask = lons_norm > 180
    if np.any(mask):
        print("ℹ️ Detected 0-360° longitude format in ERA5. Converting to -180/180° for matching.")
        lons_norm[mask] = lons_norm[mask] - 360
    return lons_norm

def compute_spatial_weights(basins_gdf, lats, lons):
    """
    Computes intersection weights with CRS and Longitude checks.
    """
    print(f"⏳ Computing spatial weights for {len(basins_gdf)} basins...")
    
    # 1. Normalize Grid Longitudes (0-360 -> -180-180) for intersection logic
    lons_matching = normalize_longitudes(lons)
    
    weights_lookup = {}
    
    # Grid resolution
    lat_res = abs(lats[1] - lats[0])
    lon_res = abs(lons[1] - lons[0])

    empty_basins = 0

    for _, row in tqdm(basins_gdf.iterrows(), total=len(basins_gdf), desc="Mapping Grid"):
        station_id = row['station_id']
        basin_geom = row.geometry
        
        minx, miny, maxx, maxy = basin_geom.bounds
        
        # Optimized Masking
        lat_min, lat_max = min(lats), max(lats)
        
        # Check if basin is even inside the global grid
        if (minx > max(lons_matching)) or (maxx < min(lons_matching)) or \
           (miny > lat_max) or (maxy < lat_min):
            empty_basins += 1
            continue

        # Find candidate indices
        lat_mask = (lats >= miny - lat_res) & (lats <= maxy + lat_res)
        lon_mask = (lons_matching >= minx - lon_res) & (lons_matching <= maxx + lon_res)
        
        valid_lat_indices = np.where(lat_mask)[0]
        valid_lon_indices = np.where(lon_mask)[0]
        
        basin_weights = {}
        
        for i in valid_lat_indices:
            for j in valid_lon_indices:
                # Construct cell polygon using the MATCHING longitude
                y_c, x_c = lats[i], lons_matching[j]
                
                cell = box(x_c - lon_res/2, y_c - lat_res/2, 
                           x_c + lon_res/2, y_c + lat_res/2)
                
                if basin_geom.intersects(cell):
                    inter_area = basin_geom.intersection(cell).area
                    if inter_area > 0:
                        # Save weight mapping to original indices (i, j)
                        basin_weights[(i, j)] = inter_area / cell.area

        if basin_weights:
            total_w = sum(basin_weights.values())
            weights_lookup[station_id] = {k: v/total_w for k, v in basin_weights.items()}
        else:
            empty_basins += 1

    if empty_basins > 0:
        print(f"⚠️ Warning: {empty_basins} basins failed intersection.")
    
    return weights_lookup

def process_daily_precip(files, weights_map):
    """
    Handles Daily NetCDF files.
    """
    records = []
    print(f"--- Processing {len(files)} Precipitation Files ---")
    
    for fpath in tqdm(files, desc="Precip Files"):
        try:
            with xr.open_dataset(fpath) as ds:
                var = next((v for v in ['tp', 'total_precipitation', 'precip'] if v in ds.variables), None)
                if not var: continue
                
                # Load Data (Time, Lat, Lon)
                data = ds[var].values
                times = pd.to_datetime(ds['valid_time'].values) # Or 'valid_time'
                
                for t_idx, time in enumerate(times):
                    if data.ndim == 3:
                        daily_slice = data[t_idx, :, :]
                    else:
                        continue 
                    
                    row = {'datetime': time}
                    has_data = False
                    
                    for station, w_dict in weights_map.items():
                        val = 0.0
                        for (r, c), w in w_dict.items():
                            # Simple bounds check just in case
                            if r < daily_slice.shape[0] and c < daily_slice.shape[1]:
                                val += daily_slice[r, c] * w
                                has_data = True
                        row[station] = val

                    if has_data:
                        records.append(row)

        except Exception as e:
            print(f"❌ Error in {fpath.name}: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index('datetime').sort_index()
        df = df * 1000  # Convert m to mm
        
        # FIX: Group by date index to merge duplicates if any exist
        df = df.groupby(df.index).first()
        
    return df

def process_hourly_temp(files, weights_map):
    """
    Handles Hourly GRIB files. Aggregates to Daily Min/Max.
    Removed Mean calculation for speed.
    """
    daily_stats = {'min': [], 'max': []}
    print(f"--- Processing {len(files)} Temperature Files (Hourly) ---")
    
    for fpath in tqdm(files, desc="Temp Files"):
        try:
            with xr.open_dataset(fpath, engine='cfgrib') as ds:
                var = next((v for v in ['t2m', '2t'] if v in ds.variables), None)
                if not var: continue

                data = ds[var].values 
                times = pd.to_datetime(ds.coords['valid_time'].values if 'valid_time' in ds.coords else ds.coords['time'].values)
                
                # Local Time Adjustment
                times = times - pd.Timedelta(hours=7)
                
                # Pre-allocate array for speed: (Time, Stations)
                n_times = len(times)
                stations = list(weights_map.keys())
                n_stations = len(stations)
                
                # (Time x Stations) matrix
                temp_matrix = np.zeros((n_times, n_stations))
                
                # Fill matrix - iterating stations is faster than iterating rows if w_dict is small
                for s_idx, station in enumerate(stations):
                    w_dict = weights_map[station]
                    # Extract weighted sum for this station across ALL times at once?
                    # Hard with irregular weights. Standard loop is safest for irregular grid.
                    # Optimization: Iterate times, then stations.
                    pass

                # --- Optimized Inner Loop ---
                # Instead of appending dicts to a list (slow), let's use numpy
                
                file_dates = times.date 
                unique_dates = np.unique(file_dates)
                
                # We need to compute station values for each timestep
                # To speed this up, we can pre-calculate the indices/weights arrays for dot product?
                # For now, let's just stick to the robust loop but clean it up.
                
                file_records = []
                for t_idx, time in enumerate(times):
                    if data.ndim == 3:
                        hourly_slice = data[t_idx, :, :]
                    else: 
                        continue

                    # Create row dict
                    row = {'date': time.date()}
                    
                    # Inner loop optimization: 
                    # Avoid strict bounds checks if we trust compute_spatial_weights
                    for station, w_dict in weights_map.items():
                        val = 0.0
                        for (r, c), w in w_dict.items():
                            val += hourly_slice[r, c] * w
                        row[station] = val
                    
                    file_records.append(row)
                
                # Aggregate THIS file immediately
                df_chunk = pd.DataFrame(file_records)
                grouped = df_chunk.groupby('date')
                
                daily_stats['min'].append(grouped.min())
                daily_stats['max'].append(grouped.max())
                
        except Exception as e:
            print(f"❌ Error in {fpath.name}: {e}")

    final_dfs = {}
    for stat in ['min', 'max']:
        if daily_stats[stat]:
            full_df = pd.concat(daily_stats[stat])
            
            # --- FIX: MERGE DUPLICATES ---
            # Group by index (Date) and take min/max again
            # This handles the overlap where 2021-12-31 exists in both 2021 and 2022 files
            if stat == 'min':
                full_df = full_df.groupby(level=0).min()
            else:
                full_df = full_df.groupby(level=0).max()
                
            full_df = full_df.sort_index()
            final_dfs[stat] = full_df - 273.15 # Kelvin to Celsius
        else:
            final_dfs[stat] = None

    return final_dfs['min'], final_dfs['max']

def process_era5_basin_data(basin_gpkg_list, stations_list):
    """
    Main Orchestrator
    """
    # 1. Load Basins
    print("Step 1/4: Loading Basins...")
    gdfs = []
    for p in basin_gpkg_list:
        gdf = gpd.read_file(p, layer='DrainageBasin_BassinDeDrainage')
        id_col = next(c for c in gdf.columns if c.lower() in ['stationnum', 'id', 'station_id'])
        gdf = gdf.rename(columns={id_col: 'station_id'})
        gdfs.append(gdf)
    
    full_gdf = pd.concat(gdfs)
    full_gdf['station_id'] = full_gdf['station_id'].str.strip()
    filtered_gdf = full_gdf[full_gdf['station_id'].isin(stations_list)].copy()
    
    # Force Reprojection
    if filtered_gdf.crs != "EPSG:4326":
        print("   🔄 Reprojecting basins to EPSG:4326 (Lat/Lon)...")
        filtered_gdf = filtered_gdf.to_crs("EPSG:4326")
    
    # 2. Map Weights
    print("Step 2/4: Mapping Spatial Weights...")
    precip_files = sorted(list(ERA5_PRECIP_DIR.glob("*.nc")))
    if not precip_files: raise FileNotFoundError("No ERA5 Precip files found")
    
    lats, lons = get_grid_info(precip_files[0])
    weights = compute_spatial_weights(filtered_gdf, lats, lons)
    
    if not weights:
        raise ValueError("Weights Dictionary is empty.")
    
    # 3. Process Precip
    print("\nStep 3/4: Processing Precipitation...")
    df_precip = process_daily_precip(precip_files, weights)
    df_precip.to_csv(CLIMATE_OUTPUT_DIR / "daily_precipitation.csv")
    
    # 4. Process Temp
    print("\nStep 4/4: Processing Temperature...")
    temp_files = sorted(list(ERA5_TEMP_DIR.glob("*.grib")) + list(ERA5_TEMP_DIR.glob("*.grib2")))
    
    if temp_files:
        # Note: Only min and max returned now
        df_min, df_max = process_hourly_temp(temp_files, weights)
        
        if df_min is not None:
            df_min.to_csv(CLIMATE_OUTPUT_DIR / "daily_temp_min.csv")
            df_max.to_csv(CLIMATE_OUTPUT_DIR / "daily_temp_max.csv")
            print("✅ Climate processing complete.")
            return df_precip, df_max # Returning Max instead of Mean just to return something
            
    return df_precip, None

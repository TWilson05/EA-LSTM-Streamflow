import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
from src.config import CLIMATE_OUTPUT_DIR, ERA5_PRECIP_DIR, ERA5_TEMP_DIR
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def get_grid_info(sample_path):
    """Extracts 1D latitude and longitude arrays from an NC file."""
    # Force the netcdf4 engine
    with xr.open_dataset(sample_path, engine='netcdf4') as ds:
        lat_name = next((v for v in ['latitude', 'lat'] if v in ds.coords), None)
        lon_name = next((v for v in ['longitude', 'lon'] if v in ds.coords), None)

        if not lat_name or not lon_name:
            raise ValueError(f"Could not find lat/lon in {sample_path}")

        lats = ds[lat_name].values
        lons = ds[lon_name].values
        
    return lats, lons

def compute_spatial_weights(basins_gdf, lats, lons):
    """Computes intersection weights with CRS and Longitude checks."""
    print(f"⏳ Computing spatial weights for {len(basins_gdf)} basins...")
    
    # ERA5 is often 0-360. Convert to -180 to 180 if needed.
    lons_matching = lons.copy()
    if np.any(lons_matching > 180):
        lons_matching[lons_matching > 180] -= 360

    weights_lookup = {}
    lat_res = abs(lats[1] - lats[0])
    lon_res = abs(lons[1] - lons[0])
    empty_basins = 0

    for _, row in tqdm(basins_gdf.iterrows(), total=len(basins_gdf), desc="Mapping Grid"):
        station_id = row['station_id']
        basin_geom = row.geometry
        minx, miny, maxx, maxy = basin_geom.bounds
        
        # Candidate indices
        lat_mask = (lats >= miny - lat_res) & (lats <= maxy + lat_res)
        lon_mask = (lons_matching >= minx - lon_res) & (lons_matching <= maxx + lon_res)
        
        valid_lat_indices = np.where(lat_mask)[0]
        valid_lon_indices = np.where(lon_mask)[0]
        
        basin_weights = {}
        
        for i in valid_lat_indices:
            for j in valid_lon_indices:
                y_c, x_c = lats[i], lons_matching[j]
                cell = box(x_c - lon_res/2, y_c - lat_res/2, x_c + lon_res/2, y_c + lat_res/2)
                
                if basin_geom.intersects(cell):
                    inter_area = basin_geom.intersection(cell).area
                    if inter_area > 0:
                        basin_weights[(i, j)] = inter_area / cell.area

        if basin_weights:
            total_w = sum(basin_weights.values())
            weights_lookup[station_id] = {k: v/total_w for k, v in basin_weights.items()}
        else:
            empty_basins += 1

    if empty_basins > 0:
        print(f"⚠️ Warning: {empty_basins} basins failed intersection.")
    
    return weights_lookup

def process_era5_files(files, weights_map, var_type):
    """
    Unified function for processing Hourly NC files.
    Applies weights using vectorized NumPy advanced indexing with NaN handling.
    """
    # 1. Pre-compile the spatial indices for ultra-fast NumPy extraction
    stn_meta = {}
    for stn, w_dict in weights_map.items():
        lat_idx = [k[0] for k in w_dict.keys()]
        lon_idx = [k[1] for k in w_dict.keys()]
        w_arr = np.array(list(w_dict.values()))
        stn_meta[stn] = (lat_idx, lon_idx, w_arr)
        
    daily_dfs_1 = [] # Holds Precip OR Temp Min
    daily_dfs_2 = [] # Holds Temp Max
    
    desc = "Precip Files" if var_type == 'precip' else "Temp Files"
    
    for fpath in tqdm(files, desc=desc):
        try:
            with xr.open_dataset(fpath, engine='netcdf4') as ds:
                # Identify Variable
                if var_type == 'precip':
                    var = next((v for v in ['tp', 'total_precipitation', 'precip'] if v in ds.variables), None)
                else:
                    var = next((v for v in ['t2m', '2t'] if v in ds.variables), None)
                    
                data = ds[var].values # Shape: (time, lat, lon)
                
                # Extract Times & Adjust to Local (UTC - 7)
                time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'
                times = pd.to_datetime(ds[time_coord].values) - pd.Timedelta(hours=7)
                
                # Vectorized Spatial Averaging with NaN Safeguard
                hourly_data = {}
                for stn, (lats, lons, ws) in stn_meta.items():
                    pixel_data = data[:, lats, lons] # Extract all pixels for this basin
                    
                    # --- NEW FIX: Handle NaNs (Coastlines/Water Bodies) ---
                    if np.isnan(pixel_data).any():
                        # Create a mask where data actually exists
                        valid_mask = ~np.isnan(pixel_data)
                        
                        # Zero out weights where data is missing
                        active_weights = ws * valid_mask 
                        
                        # Find the new total weight for each time step
                        weight_sums = np.sum(active_weights, axis=1)
                        
                        # Prevent division by zero if an entire basin is off-shore
                        weight_sums[weight_sums == 0] = np.nan 
                        
                        # Calculate the NaN-safe weighted average
                        hourly_data[stn] = np.nansum(pixel_data * ws, axis=1) / weight_sums
                    else:
                        # Fast path if the basin is fully inland (no NaNs)
                        hourly_data[stn] = np.sum(pixel_data * ws, axis=1)
                    # --------------------------------------------------------
                    
                df_hourly = pd.DataFrame(hourly_data, index=times)
                
                # Temporal Aggregation (Resample to Daily)
                if var_type == 'precip':
                    # ERA5 Precip is accumulated -> Sum to get daily total, convert m to mm
                    df_daily = df_hourly.resample('D').sum() * 1000 
                    daily_dfs_1.append(df_daily)
                elif var_type == 'temp':
                    # ERA5 Temp -> Min/Max, convert Kelvin to Celsius
                    df_min = df_hourly.resample('D').min() - 273.15 
                    df_max = df_hourly.resample('D').max() - 273.15
                    daily_dfs_1.append(df_min)
                    daily_dfs_2.append(df_max)
                    
        except Exception as e:
            print(f"❌ Error processing {fpath.name}: {e}")

    # Combine all months and drop any duplicate border days
    if var_type == 'precip':
        df_final = pd.concat(daily_dfs_1).sort_index()
        return df_final[~df_final.index.duplicated(keep='first')]
    else:
        df_min_final = pd.concat(daily_dfs_1).sort_index()
        df_max_final = pd.concat(daily_dfs_2).sort_index()
        return (
            df_min_final[~df_min_final.index.duplicated(keep='first')],
            df_max_final[~df_max_final.index.duplicated(keep='first')]
        )

def process_era5_basin_data(basin_gpkg_list, stations_list):
    """Main Orchestrator"""
    
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
    
    if filtered_gdf.crs != "EPSG:4326":
        print("   🔄 Reprojecting basins to EPSG:4326 (Lat/Lon)...")
        filtered_gdf = filtered_gdf.to_crs("EPSG:4326")
    
    # 2. Map Weights
    print("Step 2/4: Mapping Spatial Weights...")
    precip_files = sorted(list(ERA5_PRECIP_DIR.glob("*.nc")))
    if not precip_files: 
        raise FileNotFoundError("No ERA5 Precip files found")
    
    lats, lons = get_grid_info(precip_files[0])
    weights = compute_spatial_weights(filtered_gdf, lats, lons)
    
    if not weights:
        raise ValueError("Weights Dictionary is empty.")
    
    # 3. Process Precip
    print("\nStep 3/4: Processing Precipitation...")
    df_precip = process_era5_files(precip_files, weights, var_type='precip')
    
    precip_out = CLIMATE_OUTPUT_DIR / "daily_precipitation.csv"
    df_precip.to_csv(precip_out)
    print(f"✅ Precipitation data saved to {precip_out}.")
    
    # 4. Process Temp
    print("\nStep 4/4: Processing Temperature...")
    temp_files = sorted(list(ERA5_TEMP_DIR.glob("*.nc")))
    
    if temp_files:
        df_min, df_max = process_era5_files(temp_files, weights, var_type='temp')
        
        df_min.to_csv(CLIMATE_OUTPUT_DIR / "daily_temp_min.csv")
        df_max.to_csv(CLIMATE_OUTPUT_DIR / "daily_temp_max.csv")
        print(f"✅ Temperature data saved to {CLIMATE_OUTPUT_DIR}.")
        
    print("✅ Climate processing complete.")

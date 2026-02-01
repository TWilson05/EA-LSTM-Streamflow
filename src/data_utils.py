import pandas as pd
import numpy as np
import json
import os
from src.config import PROCESSED_DATA_DIR, CLIMATE_OUTPUT_DIR

def load_raw_csvs():
    """Loads all raw dataframes without filtering."""
    precip = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_precipitation.csv", index_col=0, parse_dates=True)
    tmax = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_temp_max.csv", index_col=0, parse_dates=True)
    tmin = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_temp_min.csv", index_col=0, parse_dates=True)
    flow = pd.read_csv(PROCESSED_DATA_DIR / "filtered_streamflow.csv", index_col=0, parse_dates=True)
    static = pd.read_csv(PROCESSED_DATA_DIR / "static_attributes.csv", index_col=0)
    
    # Standardize Area column name
    if 'basin_area_km2' in static.columns: 
        static = static.rename(columns={'basin_area_km2': 'area_km2'})
    
    return precip, tmax, tmin, flow, static

def align_and_filter(precip, tmax, tmin, flow, static):
    """Aligns dates and stations across all dataframes."""
    # 1. Intersect Stations
    common_stations = sorted(list(set(flow.columns).intersection(precip.columns)))
    
    # 2. Master Index (Time)
    master_index = precip.index.sort_values()
    
    # 3. Filter & Reindex
    precip = precip.loc[master_index, common_stations]
    tmax = tmax.reindex(master_index)[common_stations]
    tmin = tmin.reindex(master_index)[common_stations]
    flow = flow.reindex(master_index)[common_stations]
    static = static.loc[common_stations, ['area_km2', 'glacier_pct', 'mean_elev']]
    
    return precip, tmax, tmin, flow, static, common_stations, master_index

def calculate_runoff(flow_m3s, areas_km2):
    """Converts Flow (m^3/s) -> Specific Runoff (mm/day)."""
    return (flow_m3s * 86.4) / areas_km2

def compute_and_save_scalers(dyn_array, stat_array, basin_stds, scaler_path):
    """Computes mean/std and saves to JSON."""
    scalers = {
        "dyn_mean": np.nanmean(dyn_array, axis=(0, 1)).tolist(),
        "dyn_std": np.nanstd(dyn_array, axis=(0, 1)).tolist(),
        "stat_mean": np.nanmean(stat_array, axis=0).tolist(),
        "stat_std": np.nanstd(stat_array, axis=0).tolist(),
        "basin_stds": basin_stds.tolist()
    }
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, "w") as f:
        json.dump(scalers, f)
    return scalers

def load_scalers(scaler_path):
    with open(scaler_path, "r") as f:
        return json.load(f)

def normalize(array, mean, std):
    return (array - np.array(mean)) / (np.array(std) + 1e-6)

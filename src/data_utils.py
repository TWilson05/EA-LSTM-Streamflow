import pandas as pd
import numpy as np
import json
import os
from src.config import PROCESSED_DATA_DIR, CLIMATE_OUTPUT_DIR

# Map feature names to their corresponding files
DYNAMIC_FILE_MAP = {
    'precip': "daily_precipitation.csv",
    'temp_max': "daily_temp_max.csv",
    'temp_min': "daily_temp_min.csv",
    'snow': "daily_snowfall.csv",
    'rain': "daily_rainfall.csv",
    "freeze_frac": "daily_fraction_below_zero.csv"
}

def load_raw_csvs(dynamic_cols):
    """Loads requested dynamic dataframes and all static/flow data."""
    dyn_dict = {}
    
    # Only load the dynamic features requested
    for col in dynamic_cols:
        if col not in DYNAMIC_FILE_MAP:
            raise ValueError(f"Unknown dynamic feature '{col}'. Add it to DYNAMIC_FILE_MAP in data_utils.py")
        file_path = CLIMATE_OUTPUT_DIR / DYNAMIC_FILE_MAP[col]
        dyn_dict[col] = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Always load streamflow and static attributes
    flow = pd.read_csv(PROCESSED_DATA_DIR / "combined_streamflow.csv", index_col=0, parse_dates=True)
    static = pd.read_csv(PROCESSED_DATA_DIR / "static_attributes.csv", index_col=0)
    
    # Standardize Area column name to match your config lists
    if 'area_km2' in static.columns: 
        static = static.rename(columns={'area_km2': 'basin_area_km2'})
        
    return dyn_dict, flow, static

def align_and_filter(dyn_dict, flow, static, static_cols):
    """Aligns dates and stations across all selected dataframes."""
    
    # 1. Intersect Stations across Flow and ALL dynamic dataframes
    common_stations = set(flow.columns)
    for df in dyn_dict.values():
        common_stations = common_stations.intersection(df.columns)
    common_stations = sorted(list(common_stations))
    
    # 2. Master Index (Time) from the first dynamic feature
    first_feature = list(dyn_dict.keys())[0]
    master_index = dyn_dict[first_feature].index.sort_values()
    
    # 3. Filter & Reindex Dynamic Data
    for col in dyn_dict.keys():
        dyn_dict[col] = dyn_dict[col].reindex(master_index)[common_stations]
        
    # 4. Filter Flow
    flow = flow.reindex(master_index)[common_stations]
    
    # 5. Safely Extract Static Features
    # We ensure all requested columns exist
    missing_static = [col for col in static_cols if col not in static.columns]
    if missing_static:
        raise KeyError(f"Missing static columns in CSV: {missing_static}")
        
    filtered_static = static.loc[common_stations, static_cols]
    
    return dyn_dict, flow, static, filtered_static, common_stations, master_index

def calculate_runoff(flow_m3s, areas_km2):
    """Converts Flow (m^3/s) -> Specific Runoff (mm/day)."""
    return (flow_m3s * 86.4) / areas_km2

def compute_and_save_scalers(dyn_array, stat_array, basin_stds, scaler_path, static_feature_names):
    """Computes mean/std and saves to JSON."""
    scalers = {
        "dyn_mean": np.nanmean(dyn_array, axis=(0, 1)).tolist(),
        "dyn_std": np.nanstd(dyn_array, axis=(0, 1)).tolist(),
        "stat_mean": np.nanmean(stat_array, axis=0).tolist(),
        "stat_std": np.nanstd(stat_array, axis=0).tolist(),
        "basin_stds": basin_stds.tolist(),
        "static_features": static_feature_names
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

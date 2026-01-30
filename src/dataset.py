import pandas as pd
import numpy as np
import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from src.config import PROCESSED_DATA_DIR, CLIMATE_OUTPUT_DIR

class LazyStreamflowDataset(Dataset):
    def __init__(self, dyn_array, stat_array, y_array, time_indices, sequence_length=365):
        """
        Args:
            dyn_array: (Total_Time, N_Stations, N_Dyn_Feats) - Normalized
            stat_array: (N_Stations, N_Stat_Feats) - Normalized
            y_array: (Total_Time, N_Stations) - Targets
            time_indices: List of valid integer time indices 't' for prediction
        """
        self.dyn = dyn_array
        self.stat = stat_array
        self.y = y_array
        self.time_indices = time_indices
        self.seq_len = sequence_length
        self.num_stations = dyn_array.shape[1]
        
    def __len__(self):
        # Total samples = Number of valid days * Number of stations
        return len(self.time_indices) * self.num_stations

    def __getitem__(self, idx):
        # Map linear index to (time_index, station_index)
        # Logic: We iterate through all stations for time t, then move to t+1
        time_ptr = idx // self.num_stations
        station_idx = idx % self.num_stations
        
        # Get the actual time index from the valid list
        t = self.time_indices[time_ptr]
        
        # 1. Dynamic Window [t-365 : t] for specific station
        # Slicing numpy arrays is "lazy" (view, not copy) until converted to tensor
        x_dyn = self.dyn[t - self.seq_len : t, station_idx, :] 
        
        # 2. Static Features for specific station
        x_stat = self.stat[station_idx]
        
        # 3. Target
        y_val = self.y[t, station_idx]
        
        return (torch.from_numpy(x_dyn).float(), 
                torch.from_numpy(x_stat).float(), 
                torch.tensor([y_val]).float())

def load_and_preprocess_data(sequence_length=365, batch_size=256, scaler_path="models/scalers.json"):
    print("⏳ Loading datasets...")
    
    # 1. Load Data
    precip = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_precipitation.csv", index_col=0, parse_dates=True)
    tmax = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_temp_max.csv", index_col=0, parse_dates=True)
    tmin = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_temp_min.csv", index_col=0, parse_dates=True)
    flow = pd.read_csv(PROCESSED_DATA_DIR / "filtered_streamflow.csv", index_col=0, parse_dates=True)
    static = pd.read_csv(PROCESSED_DATA_DIR / "static_attributes.csv", index_col=0)
    
    area_col = 'basin_area_km2'
        
    # Select 3 Static Features
    static = static[[area_col, 'glacier_pct', 'mean_elev']]
    
    # 2. Align
    common_stations = sorted(list(set(flow.columns).intersection(precip.columns)))
    master_index = precip.index.sort_values()
    
    tmax = tmax.reindex(master_index)
    tmin = tmin.reindex(master_index)
    flow = flow.reindex(master_index)
    
    # 3. Filter
    precip = precip[common_stations]
    tmax = tmax[common_stations]
    tmin = tmin[common_stations]
    flow = flow[common_stations]
    static = static.loc[common_stations]
    
    # 4. Convert Flow to Specific Runoff (mm/day)
    # This divides m^3/s by Area (km^2), converting it to intrinsic mm/day
    areas = static[area_col].values
    y_runoff = (flow * 86.4) / areas
    
    # 5. Normalization
    print("   Normalizing features...")
    # Keep as numpy arrays (float32 to save RAM)
    dyn_array = np.stack([precip.values, tmax.values, tmin.values], axis=2).astype(np.float32)
    
    train_mask = (master_index.year >= 1990) & (master_index.year <= 2012)
    test_mask = ((master_index.year >= 1980) & (master_index.year <= 1989)) | \
                ((master_index.year >= 2013) & (master_index.year <= 2022))

    # Fit scaler on Train
    train_slice = dyn_array[train_mask]
    
    # Compute stats
    dyn_mean = np.nanmean(train_slice, axis=(0, 1))
    dyn_std = np.nanstd(train_slice, axis=(0, 1))
    
    stat_vals = static.values.astype(np.float32)
    stat_mean = np.nanmean(stat_vals, axis=0)
    stat_std = np.nanstd(stat_vals, axis=0)
    
    # --- SAVE SCALERS ---
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scalers = {
        "dyn_mean": dyn_mean.tolist(),
        "dyn_std": dyn_std.tolist(),
        "stat_mean": stat_mean.tolist(),
        "stat_std": stat_std.tolist(),
        "static_features": [area_col, 'glacier_pct', 'mean_elev'] # Save feature names too
    }
    with open(scaler_path, "w") as f:
        json.dump(scalers, f)
    print(f"   ✅ Saved normalization scalers to {scaler_path}")
    
    # Apply Normalization
    dyn_norm = (dyn_array - dyn_mean) / (dyn_std + 1e-6)
    stat_norm = (stat_vals - stat_mean) / (stat_std + 1e-6)
    y_vals = y_runoff.values.astype(np.float32)

    # 6. Create Indices
    def get_valid_indices(mask):
        indices = np.where(mask)[0]
        # Filter for lookback
        return indices[indices >= sequence_length]

    train_indices = get_valid_indices(train_mask)
    test_indices = get_valid_indices(test_mask)
    
    print(f"   Train Days: {len(train_indices)} | Test Days: {len(test_indices)}")
    
    # 7. Create Loaders
    # num_workers=0 is safer for RAM in Colab. Increase to 2 or 4 if RAM allows.
    train_ds = LazyStreamflowDataset(dyn_norm, stat_norm, y_vals, train_indices, sequence_length)
    test_ds = LazyStreamflowDataset(dyn_norm, stat_norm, y_vals, test_indices, sequence_length)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✅ Data Ready.")
    print(f"   Train Samples: {len(train_ds)}")
    
    return train_loader, test_loader, common_stations

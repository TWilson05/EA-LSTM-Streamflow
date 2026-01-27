import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from functools import reduce
from src.config import PROCESSED_DATA_DIR, CLIMATE_OUTPUT_DIR

class StreamflowDataset(Dataset):
    def __init__(self, X_dynamic, X_static, y, sequence_length=365):
        self.X_dynamic = X_dynamic
        self.X_static = X_static
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X_dynamic[idx], self.X_static[idx], self.y[idx]

def load_and_preprocess_data(sequence_length=365, batch_size=256):
    print("⏳ Loading datasets...")
    
    # 1. Load Data
    precip = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_precipitation.csv", index_col=0, parse_dates=True)
    tmax = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_temp_max.csv", index_col=0, parse_dates=True)
    tmin = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_temp_min.csv", index_col=0, parse_dates=True)
    flow = pd.read_csv(PROCESSED_DATA_DIR / "filtered_streamflow.csv", index_col=0, parse_dates=True)
    
    # Load Static and handle potential column name differences
    static = pd.read_csv(PROCESSED_DATA_DIR / "static_attributes.csv", index_col=0) # Assuming station_id is index
    
    # Fix column name if needed
    if 'basin_area_km2' in static.columns:
        area_col = 'basin_area_km2'
    else:
        area_col = 'area_km2'

    # Filter Static Features immediately
    # We only want Area and Glacier %
    static = static[[area_col, 'glacier_pct']]
    
    # 2. Strict Alignment (The Fix for the 1979-12-31 issue)
    print("   Aligning dates and stations...")
    
    # Intersection of Stations
    station_sets = [set(df.columns) for df in [precip, tmax, tmin, flow]]
    # Also ensure stations exist in static index
    station_sets.append(set(static.index.astype(str))) 
    
    common_stations = sorted(list(set.intersection(*station_sets)))
    
    # Intersection of Dates
    date_sets = [set(df.index) for df in [precip, tmax, tmin, flow]]
    common_dates = sorted(list(set.intersection(*date_sets)))
    
    # Convert back to DatetimeIndex for slicing
    common_dates = pd.DatetimeIndex(common_dates)
    
    print(f"   Common Period: {common_dates.min().date()} to {common_dates.max().date()}")
    print(f"   Common Stations: {len(common_stations)}")

    # 3. Filter All Dataframes
    precip = precip.loc[common_dates, common_stations]
    tmax = tmax.loc[common_dates, common_stations]
    tmin = tmin.loc[common_dates, common_stations]
    flow = flow.loc[common_dates, common_stations]
    static = static.loc[common_stations]
    
    # 4. Convert Flow to Specific Runoff (mm/day)
    # Area column might be 'basin_area_km2' or 'area_km2'
    areas = static[area_col].values
    
    # (Time x Station) / (Station,) -> Result is mm/day
    y_runoff = (flow * 86.4) / areas
    
    # 5. Define Splits
    # Training: 1990-2012
    train_mask = (common_dates.year >= 1990) & (common_dates.year <= 2012)
    
    # Testing: 1980-1989 OR 2013-2022
    test_mask = ((common_dates.year >= 1980) & (common_dates.year <= 1989)) | \
                ((common_dates.year >= 2013) & (common_dates.year <= 2022))

    # 6. Normalization
    print("   Normalizing features...")
    # Stack dynamic features: (Time, Stations, 3) -> [Precip, Tmax, Tmin]
    # This stack will now work because dimensions are guaranteed to match
    dyn_array = np.stack([precip.values, tmax.values, tmin.values], axis=2)
    
    # Calculate Mean/Std from TRAINING split only
    train_slice = dyn_array[train_mask]
    dyn_mean = np.nanmean(train_slice, axis=(0, 1))
    dyn_std = np.nanstd(train_slice, axis=(0, 1))
    
    # Normalize Dynamic
    dyn_norm = (dyn_array - dyn_mean) / (dyn_std + 1e-6)
    
    # Normalize Static (Columns: Area, GlacierPct)
    stat_vals = static.values
    stat_mean = np.nanmean(stat_vals, axis=0)
    stat_std = np.nanstd(stat_vals, axis=0)
    stat_norm = (stat_vals - stat_mean) / (stat_std + 1e-6)
    
    # 7. Sliding Window Generation
    def create_sequences(date_mask):
        X_dyn_list, X_stat_list, y_list = [], [], []
        
        # Get integer indices where date_mask is True
        valid_indices = np.where(date_mask)[0]
        
        # Filter indices that are too early (need 365 days lookback)
        valid_indices = valid_indices[valid_indices >= sequence_length]
        
        if len(valid_indices) == 0:
            print("   ⚠️ Warning: No valid sequences found for this split!")
            return None
        
        # Pre-allocate lists? No, appended lists are fast enough for <100k samples usually.
        # But iterating 10,000 days is slow in Python.
        # Optimization: We can stride this, but for now loop is clear.
        
        for t_idx in valid_indices:
            # Dynamic Window: (t-365 : t)
            # Shape: (365, Stations, 3) -> We want (Stations, 365, 3)
            # We treat every station as a separate sample
            
            window = dyn_norm[t_idx-sequence_length : t_idx] # (365, N_Stations, 3)
            target = y_runoff.iloc[t_idx].values # (N_Stations,)
            
            # Transpose to (N_Stations, 365, 3)
            window = window.transpose(1, 0, 2)
            
            # Static: (N_Stations, 2)
            stat = stat_norm
            
            X_dyn_list.append(window)
            X_stat_list.append(stat)
            y_list.append(target.reshape(-1, 1))
            
        # Concatenate samples
        # Result: (Total_Days * N_Stations, Seq_Len, Feats)
        X_dyn_all = np.concatenate(X_dyn_list, axis=0)
        X_stat_all = np.concatenate(X_stat_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        
        return (torch.FloatTensor(X_dyn_all),
                torch.FloatTensor(X_stat_all),
                torch.FloatTensor(y_all))

    print("   Generating Training Sequences (this may take 1-2 mins)...")
    train_data = create_sequences(train_mask)
    
    print("   Generating Testing Sequences...")
    test_data = create_sequences(test_mask)
    
    # Check if data exists
    if train_data is None or test_data is None:
        raise ValueError("Sequence generation failed. Check splits.")

    train_loader = DataLoader(StreamflowDataset(*train_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(StreamflowDataset(*test_data), batch_size=batch_size, shuffle=False)
    
    print(f"✅ Data Ready.")
    print(f"   Train Samples: {len(train_loader.dataset)}")
    print(f"   Test Samples:  {len(test_loader.dataset)}")
    
    return train_loader, test_loader, common_stations

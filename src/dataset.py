import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
    
    # Load Static
    static = pd.read_csv(PROCESSED_DATA_DIR / "static_attributes.csv", index_col=0)
    if 'basin_area_km2' in static.columns:
        area_col = 'basin_area_km2'
    else:
        area_col = 'area_km2'
    static = static[[area_col, 'glacier_pct']]
    
    # 2. ALIGNMENT FIX (Preserve 1979 Climate Data)
    print("   Aligning dates and stations...")
    
    # A. Stations: Intersection (We can only predict stations we have flow for)
    # Note: We assume climate data covers these stations (it should if derived from shapefiles)
    common_stations = sorted(list(set(flow.columns).intersection(precip.columns)))
    print(f"   Common Stations: {len(common_stations)}")

    # B. Dates: Use CLIMATE as the Master Index
    # We want the full range of climate data (e.g., 1979-2022)
    # Precip is usually the cleanest master index
    master_index = precip.index.sort_values()
    
    # Ensure all climate vars share this index
    tmax = tmax.reindex(master_index)
    tmin = tmin.reindex(master_index)
    
    # CRITICAL: Reindex flow to master index.
    # 1979 dates will become NaN in 'flow', which is fine (we won't compute loss on them).
    # But the rows will exist, keeping the array indices aligned.
    flow = flow.reindex(master_index)
    
    print(f"   Master Index Range: {master_index.min().date()} to {master_index.max().date()}")

    # 3. Filter Columns (Stations)
    precip = precip[common_stations]
    tmax = tmax[common_stations]
    tmin = tmin[common_stations]
    flow = flow[common_stations]
    static = static.loc[common_stations]
    
    # 4. Convert Flow to Specific Runoff (mm/day)
    areas = static[area_col].values
    y_runoff = (flow * 86.4) / areas
    
    # 5. Define Splits (Based on Years)
    # Training: 1990-2012
    train_mask = (master_index.year >= 1990) & (master_index.year <= 2012)
    
    # Testing: 1980-1989 OR 2013-2022
    test_mask = ((master_index.year >= 1980) & (master_index.year <= 1989)) | \
                ((master_index.year >= 2013) & (master_index.year <= 2022))

    # 6. Normalization (Fit on Train, Apply to All)
    print("   Normalizing features...")
    dyn_array = np.stack([precip.values, tmax.values, tmin.values], axis=2)
    
    train_slice = dyn_array[train_mask]
    dyn_mean = np.nanmean(train_slice, axis=(0, 1))
    dyn_std = np.nanstd(train_slice, axis=(0, 1))
    
    dyn_norm = (dyn_array - dyn_mean) / (dyn_std + 1e-6)
    
    stat_vals = static.values
    stat_mean = np.nanmean(stat_vals, axis=0)
    stat_std = np.nanstd(stat_vals, axis=0)
    stat_norm = (stat_vals - stat_mean) / (stat_std + 1e-6)
    
    # 7. Sequence Generation
    def create_sequences(date_mask):
        X_dyn_list, X_stat_list, y_list = [], [], []
        
        # indices where date_mask is True (the days we want to PREDICT)
        target_indices = np.where(date_mask)[0]
        
        # Filter: ensure we have enough history for the lookback
        valid_indices = target_indices[target_indices >= sequence_length]
        
        if len(valid_indices) == 0:
            print("   ⚠️ Warning: No valid sequences found for this split!")
            return None
        
        print(f"   Processing {len(valid_indices)} time steps...")
        
        for t_idx in valid_indices:
            # Lookback: indices [t-365 ... t-1]
            # This allows us to use 1979 data (indices 0-364) to predict index 365 (Jan 1 1980)
            window = dyn_norm[t_idx-sequence_length : t_idx] 
            
            # Target: index t
            target = y_runoff.iloc[t_idx].values 
            
            # Transpose to (Stations, Seq_Len, Features)
            window = window.transpose(1, 0, 2)
            
            # Replicate static features
            stat = stat_norm
            
            X_dyn_list.append(window)
            X_stat_list.append(stat)
            y_list.append(target.reshape(-1, 1))
            
        X_dyn_all = np.concatenate(X_dyn_list, axis=0)
        X_stat_all = np.concatenate(X_stat_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        
        return (torch.FloatTensor(X_dyn_all),
                torch.FloatTensor(X_stat_all),
                torch.FloatTensor(y_all))

    print("   Generating Training Sequences...")
    train_data = create_sequences(train_mask)
    
    print("   Generating Testing Sequences...")
    test_data = create_sequences(test_mask)
    
    if train_data is None or test_data is None:
        raise ValueError("Sequence generation failed.")

    train_loader = DataLoader(StreamflowDataset(*train_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(StreamflowDataset(*test_data), batch_size=batch_size, shuffle=False)
    
    print(f"✅ Data Ready.")
    print(f"   Train Samples: {len(train_loader.dataset)}")
    print(f"   Test Samples:  {len(test_loader.dataset)}")
    
    return train_loader, test_loader, common_stations

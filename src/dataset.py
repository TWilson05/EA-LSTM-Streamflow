import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import PROCESSED_DATA_DIR, CLIMATE_OUTPUT_DIR

class StreamflowDataset(Dataset):
    def __init__(self, X_dynamic, X_static, y, sequence_length=365):
        """
        Args:
            X_dynamic: Tensor (Num_Samples, Seq_Len, Num_Dyn_Features)
            X_static:  Tensor (Num_Samples, Num_Static_Features)
            y:         Tensor (Num_Samples, 1)
        """
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
    # Dynamic (Climate) - [Date x Stations]
    precip = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_precipitation.csv", index_col=0, parse_dates=True)
    tmax = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_temp_max.csv", index_col=0, parse_dates=True)
    tmin = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_temp_min.csv", index_col=0, parse_dates=True)
    
    # Static (Attributes) - [Station x Features]
    static = pd.read_csv(PROCESSED_DATA_DIR / "static_attributes.csv", index_col='station_id')
    
    # Target (Streamflow) - [Date x Stations]
    flow = pd.read_csv(PROCESSED_DATA_DIR / "filtered_streamflow.csv", index_col=0, parse_dates=True)

    # 2. Align Data (Intersection of dates and stations)
    common_stations = flow.columns.intersection(precip.columns)
    common_dates = flow.index.intersection(precip.index)
    
    # Filter everything
    precip = precip.loc[common_dates, common_stations]
    tmax = tmax.loc[common_dates, common_stations]
    tmin = tmin.loc[common_dates, common_stations]
    flow = flow.loc[common_dates, common_stations]
    static = static.loc[common_stations] # Ensure order matches columns
    
    # 3. Convert Flow to Specific Runoff (mm/day)
    # Flow (m³/s) * (86400 s/day) / (Area (km²) * 10^6 m²/km²) * 1000 mm/m = mm/day
    # Simplified: Flow / Area * 86.4
    areas = static['basin_area_km2'].values
    # Broadcasting division: (Time x Station) / (Station,)
    y_runoff = (flow * 86.4) / areas
    
    # 4. Define Splits
    train_dates = (common_dates.year >= 1990) & (common_dates.year <= 2012)
    # Testing: 1980-1989 OR 2013-2022
    test_dates = ((common_dates.year >= 1980) & (common_dates.year <= 1989)) | \
                 ((common_dates.year >= 2013) & (common_dates.year <= 2022))

    # 5. Normalization (Fit on TRAIN, Apply to ALL)
    # Stack dynamic features: (Time, Stations, Features)
    dyn_array = np.stack([precip.values, tmax.values, tmin.values], axis=2)
    
    # Calculate Mean/Std from Training Period only
    train_slice = dyn_array[train_dates]
    dyn_mean = np.nanmean(train_slice, axis=(0, 1))
    dyn_std = np.nanstd(train_slice, axis=(0, 1))
    
    # Normalize Dynamic
    dyn_norm = (dyn_array - dyn_mean) / (dyn_std + 1e-6)
    
    # Normalize Static
    static_feats = static[['basin_area_km2', 'glacier_pct']].values
    stat_mean = np.nanmean(static_feats, axis=0)
    stat_std = np.nanstd(static_feats, axis=0)
    stat_norm = (static_feats - stat_mean) / (stat_std + 1e-6)
    
    # Note: We do NOT normalize targets (y) generally for physical interpretation in NSE, 
    # but some normalize for stability. Here we keep it physical (mm/day).

    # 6. Create Sliding Windows
    # We need to reshape data into (Samples, Seq_Len, Features)
    # Strategy: We iterate through every station and every valid date
    
    def create_sequences(date_mask):
        X_dyn_list, X_stat_list, y_list = [], [], []
        
        # Get indices of dates that belong to this split
        # We need to ensure we have 'sequence_length' lookback for each target
        valid_indices = np.where(date_mask)[0]
        
        # Filter indices that are too early (don't have enough history)
        valid_indices = valid_indices[valid_indices >= sequence_length]
        
        print(f"   Processing split with {len(valid_indices)} days x {len(common_stations)} basins...")
        
        for t_idx in valid_indices:
            # Slicing: (t-seq_len : t)
            # Shape: (Seq_Len, Stations, Feats) -> Transpose to (Stations, Seq_Len, Feats)
            window = dyn_norm[t_idx-sequence_length : t_idx]
            window = window.transpose(1, 0, 2) 
            
            # Static features repeated for this time step (Stations, Feats)
            stat = stat_norm
            
            # Targets for this day (Stations, 1)
            target = y_runoff.iloc[t_idx].values.reshape(-1, 1)
            
            X_dyn_list.append(window)
            X_stat_list.append(stat)
            y_list.append(target)
            
        # Concatenate: Result is (Total_Samples, ...) where Total_Samples = Days * Stations
        return (torch.FloatTensor(np.concatenate(X_dyn_list, axis=0)),
                torch.FloatTensor(np.concatenate(X_stat_list, axis=0)),
                torch.FloatTensor(np.concatenate(y_list, axis=0)))

    print("Generating Training Sequences...")
    train_data = create_sequences(train_dates)
    print("Generating Testing Sequences...")
    test_data = create_sequences(test_dates)
    
    # Create Loaders
    train_loader = DataLoader(StreamflowDataset(*train_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(StreamflowDataset(*test_data), batch_size=batch_size, shuffle=False)
    
    print(f"✅ Data Ready. Train Samples: {len(train_loader.dataset)}, Test Samples: {len(test_loader.dataset)}")
    return train_loader, test_loader, common_stations

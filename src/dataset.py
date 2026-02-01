import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import MODELS_DIR
from src.data_utils import (load_raw_csvs, align_and_filter, calculate_runoff, 
                            compute_and_save_scalers, normalize)

class LazyStreamflowDataset(Dataset):
    def __init__(self, dyn_array, stat_array, y_array, time_indices, basin_stds, sequence_length=365):
        self.dyn = dyn_array
        self.stat = stat_array
        self.y = y_array
        self.time_indices = time_indices
        self.basin_stds = basin_stds
        self.seq_len = sequence_length
        self.num_stations = dyn_array.shape[1]
        
    def __len__(self):
        return len(self.time_indices) * self.num_stations

    def __getitem__(self, idx):
        t = self.time_indices[idx // self.num_stations]
        s = idx % self.num_stations
        
        return (torch.from_numpy(self.dyn[t - self.seq_len : t, s]).float(), 
                torch.from_numpy(self.stat[s]).float(), 
                torch.tensor([self.y[t, s]]).float(),
                torch.tensor([self.basin_stds[s]]).float())

def load_and_preprocess_data(sequence_length=365, batch_size=256, num_workers=0):
    print("⏳ Loading Data...")
    p, tmax, tmin, flow, static = load_raw_csvs()
    p, tmax, tmin, flow, static, stations, index = align_and_filter(p, tmax, tmin, flow, static)
    
    # Prepare Arrays
    dyn_array = np.stack([p.values, tmax.values, tmin.values], axis=2).astype(np.float32)
    y_vals = calculate_runoff(flow.values, static['area_km2'].values).astype(np.float32)
    stat_vals = static.values.astype(np.float32)
    
    # Masks
    train_mask = (index.year >= 1990) & (index.year <= 2008)
    val_mask = (index.year >= 2009) & (index.year <= 2012)
    test_mask = ((index.year >= 1980) & (index.year <= 1989)) | (index.year >= 2013)
    
    # Compute Normalization (Train Only)
    print("   Computing Norms...")
    train_dyn = dyn_array[train_mask]
    basin_stds = np.nanstd(y_vals[train_mask], axis=0)
    basin_stds[basin_stds < 1e-4] = 1.0
    
    static_feature_names = ['area_km2', 'glacier_pct', 'mean_elev']
    scalers = compute_and_save_scalers(train_dyn,
                                       stat_vals,
                                       basin_stds,
                                       MODELS_DIR / "scalers.json",
                                       static_feature_names)
    
    # Apply Normalization
    dyn_norm = normalize(dyn_array, scalers['dyn_mean'], scalers['dyn_std'])
    stat_norm = normalize(stat_vals, scalers['stat_mean'], scalers['stat_std'])
    
    # Create Loaders
    def make_loader(mask, shuffle):
        indices = np.where(mask)[0]
        valid_indices = indices[indices >= sequence_length]
        ds = LazyStreamflowDataset(dyn_norm, stat_norm, y_vals, valid_indices, basin_stds, sequence_length)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return (make_loader(train_mask, True), 
            make_loader(val_mask, False), 
            make_loader(test_mask, False), 
            stations)

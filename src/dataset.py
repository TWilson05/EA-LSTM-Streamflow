import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import (MODELS_DIR,
                        TRAIN_START_YEAR, TRAIN_END_YEAR,
                        VAL_START_YEAR, VAL_END_YEAR,
                        TEST_START_YEAR, TEST_END_YEAR)
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
 
def load_and_preprocess_data(dynamic_cols, static_cols, exp_name, sequence_length=365, batch_size=256, num_workers=0):
    print("⏳ Loading Data...")
    
    # 1. Load and Align
    dyn_dict_raw, flow_raw, static_raw = load_raw_csvs(dynamic_cols)
    dyn_dict, flow, static_full, static, stations, index = align_and_filter(
        dyn_dict_raw, flow_raw, static_raw, static_cols
    )
    
    # 2. Dynamically Stack Arrays (Preserves the order of dynamic_cols!)
    dyn_list = [dyn_dict[col].values for col in dynamic_cols]
    dyn_array = np.stack(dyn_list, axis=2).astype(np.float32)
    
    # 3. Safely calculate runoff
    # We pull area from static_full just in case 'basin_area_km2' isn't in static_cols
    basin_areas = static_full.loc[stations, 'basin_area_km2'].values
    y_vals = calculate_runoff(flow.values, basin_areas).astype(np.float32)
    
    stat_vals = static.values.astype(np.float32)
    
    # 4. Time Masks
    train_mask = (index.year >= TRAIN_START_YEAR) & (index.year <= TRAIN_END_YEAR)
    val_mask = (index.year >= VAL_START_YEAR) & (index.year <= VAL_END_YEAR)
    test_mask = (index.year >= TEST_START_YEAR) & (index.year <= TEST_END_YEAR)
    
    # 5. Compute Normalization (Train Only)
    print("   Computing Norms...")
    train_dyn = dyn_array[train_mask]
    basin_stds = np.nanstd(y_vals[train_mask], axis=0)
    basin_stds[basin_stds < 1e-4] = 1.0
    
    # Pass the dynamic static_cols variable instead of the hardcoded list
    scaler_path = MODELS_DIR / f"{exp_name}_scalers.json"
    scalers = compute_and_save_scalers(train_dyn,
                                       stat_vals,
                                       basin_stds,
                                       scaler_path,
                                       static_cols)
    
    # 6. Apply Normalization
    dyn_norm = normalize(dyn_array, scalers['dyn_mean'], scalers['dyn_std'])
    stat_norm = normalize(stat_vals, scalers['stat_mean'], scalers['stat_std'])
    
    # 7. Create Loaders
    def make_loader(mask, shuffle):
        indices = np.where(mask)[0]
        valid_indices = indices[indices >= sequence_length]
        ds = LazyStreamflowDataset(dyn_norm, stat_norm, y_vals, valid_indices, basin_stds, sequence_length)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return (make_loader(train_mask, True), 
            make_loader(val_mask, False), 
            make_loader(test_mask, False), 
            stations)

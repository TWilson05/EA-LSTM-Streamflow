import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from src.config import PROCESSED_DATA_DIR, CLIMATE_OUTPUT_DIR, MODELS_DIR

def predict_and_save_test_results(model, device, output_file="test_predictions.csv", sequence_length=365):
    print("⏳ Generating Test Predictions CSV...")
    
    # 1. Load Raw Data
    precip = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_precipitation.csv", index_col=0, parse_dates=True)
    tmax = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_temp_max.csv", index_col=0, parse_dates=True)
    tmin = pd.read_csv(CLIMATE_OUTPUT_DIR / "daily_temp_min.csv", index_col=0, parse_dates=True)
    static = pd.read_csv(PROCESSED_DATA_DIR / "static_attributes.csv", index_col=0)
    
    # Align
    common_stations = sorted(list(set(precip.columns).intersection(static.index.astype(str))))
    master_index = precip.index.sort_values()
    
    # 2. Load Scalers (Ensure we normalize exactly like training)
    scaler_path = MODELS_DIR / "scalers.json"
    with open(scaler_path, "r") as f:
        scalers = json.load(f)
        
    dyn_mean = np.array(scalers["dyn_mean"])
    dyn_std = np.array(scalers["dyn_std"])
    stat_mean = np.array(scalers["stat_mean"])
    stat_std = np.array(scalers["stat_std"])
    static_feats = scalers["static_features"] # e.g. ['area_km2', 'glacier_pct', 'mean_elev']

    # 3. Prepare Normalized Data
    print("   Normalizing data...")
    # Dynamic (Time, Stations, 3)
    dyn_array = np.stack([precip[common_stations].values, 
                          tmax[common_stations].values, 
                          tmin[common_stations].values], axis=2).astype(np.float32)
    dyn_norm = (dyn_array - dyn_mean) / (dyn_std + 1e-6)
    
    # Static (Stations, 3)
    stat_vals = static.loc[common_stations, static_feats].values.astype(np.float32)
    stat_norm = (stat_vals - stat_mean) / (stat_std + 1e-6)

    # 4. Define Test Indices
    # Matches the test_mask logic from dataset.py
    test_mask = ((master_index.year >= 1980) & (master_index.year <= 1989)) | \
                ((master_index.year >= 2013) & (master_index.year <= 2022))
    
    # We need indices relative to the master_index to slice the window
    all_indices = np.arange(len(master_index))
    test_indices = all_indices[test_mask]
    
    # Filter for lookback (cannot predict if t < 365)
    valid_test_indices = test_indices[test_indices >= sequence_length]
    valid_test_dates = master_index[valid_test_indices]
    
    # 5. Prediction Loop (Station by Station)
    # Why Station by Station? It's easier to fill DataFrame columns.
    # We can batch the *time steps* for a single station to speed it up.
    
    model.eval()
    results_dict = {}
    
    print(f"   Predicting for {len(common_stations)} stations over {len(valid_test_dates)} days...")
    
    for i, station_id in enumerate(tqdm(common_stations)):
        # A. Prepare Station Inputs
        # Static: (1, 3) -> repeat to (Num_Test_Days, 3)
        stat_input = torch.tensor(stat_norm[i]).float().to(device).unsqueeze(0)
        stat_batch = stat_input.repeat(len(valid_test_indices), 1)
        
        # Dynamic: Need to gather windows [t-365 : t] for all t in valid_test_indices
        # To avoid massive RAM usage, we batch this process
        station_preds = []
        batch_size = 512
        
        # Loop through time in chunks
        for k in range(0, len(valid_test_indices), batch_size):
            batch_indices = valid_test_indices[k : k+batch_size]
            
            # Construct Batch of Windows
            # Shape: (Batch, 365, 3)
            windows = []
            for t in batch_indices:
                windows.append(dyn_norm[t-sequence_length : t, i, :])
            
            dyn_batch = torch.tensor(np.array(windows)).float().to(device)
            curr_stat_batch = stat_batch[:len(batch_indices)] # Slice static to match
            
            with torch.no_grad():
                # Predict
                y_pred = model(dyn_batch, curr_stat_batch)
                station_preds.extend(y_pred.cpu().numpy().flatten())
        
        # Store in dict
        results_dict[station_id] = station_preds

    # 6. Build DataFrame
    df_results = pd.DataFrame(results_dict, index=valid_test_dates)
    
    # Save
    df_results.to_csv(output_file)
    print(f"✅ Predictions saved to {output_file}")
    return df_results

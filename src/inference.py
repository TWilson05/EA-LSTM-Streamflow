# src/inference.py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.config import MODELS_DIR, DATA_START_YEAR, DATA_END_YEAR
from src.data_utils import load_raw_csvs, align_and_filter, load_scalers, normalize

SEQUENCE_LENGTH = 365

def predict_and_save_full_results(model, device, output_file, dynamic_cols, static_cols, batch_size=256, force_zero_glacier=False):
    """
    Generates predictions for the ENTIRE dataset (Train + Val + Test) and saves to CSV.
    Rows: Dates, Columns: Station IDs.
    """
    print(f"⏳ Generating Full Dataset Predictions -> {output_file.name}")
    
    # 1. Load and Align Data
    dyn_dict_raw, flow_raw, static_raw = load_raw_csvs(dynamic_cols)
    dyn_dict, flow, static_full, static, stations, master_index = align_and_filter(
        dyn_dict_raw, flow_raw, static_raw, static_cols
    )
    
    # 2. Load Scalers
    scaler_path = MODELS_DIR / "scalers.json"
    scalers = load_scalers(scaler_path)
    
    # 3. Normalize Dynamic Features
    print("   Normalizing dynamic data...")
    dyn_list = [dyn_dict[col].values for col in dynamic_cols]
    dyn_array = np.stack(dyn_list, axis=2).astype(np.float32)
    dyn_norm = normalize(dyn_array, scalers['dyn_mean'], scalers['dyn_std'])
    
    # 4. Normalize Static Features
    if force_zero_glacier and 'glacier_pct' in static_cols:
        print("   ❄️ OVERRIDE: Forcing glacier_pct to 0.0 for counterfactual analysis...")
        static['glacier_pct'] = 0.0
        
    stat_vals = static.values.astype(np.float32)
    stat_norm = normalize(stat_vals, scalers['stat_mean'], scalers['stat_std'])
    
    # 5. Define Full Dataset Indices
    # --- CHANGED: Now spans from the start of training to the end of testing ---
    full_mask = (master_index >= DATA_START_YEAR) & (master_index <= DATA_END_YEAR)
    
    all_indices = np.arange(len(master_index))
    target_indices = all_indices[full_mask]
    
    # Filter for lookback (cannot predict if t < sequence_length)
    valid_indices = target_indices[target_indices >= SEQUENCE_LENGTH]
    valid_dates = master_index[valid_indices]
    
    print(f"   Predicting for {len(stations)} stations over {len(valid_dates)} days...")
    
    # 6. Prediction Loop
    model.eval()
    results_dict = {}
    
    for i, station_id in enumerate(tqdm(stations, desc="Stations")):
        station_static = torch.tensor(stat_norm[i]).float().to(device).unsqueeze(0)
        station_preds = []
        
        for k in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[k : k + batch_size]
            current_batch_size = len(batch_indices)
            
            windows = []
            for t in batch_indices:
                windows.append(dyn_norm[t-SEQUENCE_LENGTH : t, i, :])
            
            dyn_batch = torch.tensor(np.array(windows)).float().to(device)
            stat_batch = station_static.repeat(current_batch_size, 1)
            
            with torch.no_grad():
                y_pred = model(dyn_batch, stat_batch)
                station_preds.extend(y_pred.cpu().numpy().flatten())
        
        results_dict[station_id] = station_preds

    # 7. Build and Save
    print("   Constructing DataFrame...")
    df_results = pd.DataFrame(results_dict, index=valid_dates)
    df_results.to_csv(output_file)
    print(f"✅ Predictions saved to {output_file}")
    
    return df_results

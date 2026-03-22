# src/inference.py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.config import MODELS_DIR, TEST_START_YEAR, TEST_END_YEAR
from src.data_utils import load_raw_csvs, align_and_filter, load_scalers, normalize

SEQUENCE_LENGTH = 365

def predict_and_save_test_results(model, device, output_file, dynamic_cols, static_cols, batch_size=256, force_zero_glacier=False):
    """
    Generates predictions for the test set and saves them to a CSV.
    Rows: Dates, Columns: Station IDs.
    """
    print("⏳ Generating Predictions...")
    
    # 1. Load and Align Data (Using the new dynamic lists)
    dyn_dict_raw, flow_raw, static_raw = load_raw_csvs(dynamic_cols)
    dyn_dict, flow, static_full, static, stations, master_index = align_and_filter(
        dyn_dict_raw, flow_raw, static_raw, static_cols
    )
    
    # 2. Load Scalers
    scaler_path = MODELS_DIR / "scalers.json"
    scalers = load_scalers(scaler_path)
    
    # 3. Normalize Dynamic Features
    print("   Normalizing data...")
    # Dynamically stack based on the requested features
    dyn_list = [dyn_dict[col].values for col in dynamic_cols]
    dyn_array = np.stack(dyn_list, axis=2).astype(np.float32)
    dyn_norm = normalize(dyn_array, scalers['dyn_mean'], scalers['dyn_std'])
    
    # 4. Normalize Static Features
    # 'static' is already filtered to 'static_cols' from align_and_filter

    # --- COUNTERFACTUAL OVERRIDE ---
    if force_zero_glacier and 'glacier_pct' in static_cols:
        print("   ❄️ OVERRIDE: Forcing glacier_pct to 0.0 for counterfactual analysis...")
        static['glacier_pct'] = 0.0
    elif force_zero_glacier and 'glacier_area_km2' in static_cols:
        print("   ❄️ OVERRIDE: Forcing glacier_area_km2 to 0.0 for counterfactual analysis...")
        static['glacier_area_km2'] = 0.0
    # -------------------------------
    
    stat_vals = static.values.astype(np.float32)
    stat_norm = normalize(stat_vals, scalers['stat_mean'], scalers['stat_std'])
    
    # 5. Define Test Indices using specific dates
    test_mask = (master_index.year >= TEST_START_YEAR) & (master_index.year <= TEST_END_YEAR)
    
    # Convert boolean mask to integer indices relative to the master_index
    all_indices = np.arange(len(master_index))
    test_indices = all_indices[test_mask]
    
    # Filter for lookback (cannot predict if t < sequence_length)
    valid_test_indices = test_indices[test_indices >= SEQUENCE_LENGTH]
    valid_test_dates = master_index[valid_test_indices]
    
    print(f"   Predicting for {len(stations)} stations over {len(valid_test_dates)} days...")
    
    # 6. Prediction Loop (Station-by-Station)
    model.eval()
    results_dict = {}
    
    # We iterate stations one by one to keep the resulting dictionary clean
    for i, station_id in enumerate(tqdm(stations, desc="Stations")):
        
        # A. Prepare Static Input for this Station
        # Shape: (1, Num_Static_Features)
        station_static = torch.tensor(stat_norm[i]).float().to(device).unsqueeze(0)
        
        station_preds = []
        
        # B. Batch over Time
        for k in range(0, len(valid_test_indices), batch_size):
            batch_indices = valid_test_indices[k : k + batch_size]
            current_batch_size = len(batch_indices)
            
            # Construct the Dynamic Batch: (Batch, Seq_Len, Features)
            windows = []
            for t in batch_indices:
                # Extract [t-365 : t] for station i
                windows.append(dyn_norm[t-SEQUENCE_LENGTH : t, i, :])
            
            # Convert to Tensor
            dyn_batch = torch.tensor(np.array(windows)).float().to(device)
            
            # Expand static features to match this batch size
            stat_batch = station_static.repeat(current_batch_size, 1)
            
            # Predict
            with torch.no_grad():
                y_pred = model(dyn_batch, stat_batch)
                station_preds.extend(y_pred.cpu().numpy().flatten())
        
        # Store results for this station
        results_dict[station_id] = station_preds

    # 7. Build and Save DataFrame
    print("   Constructing DataFrame...")
    df_results = pd.DataFrame(results_dict, index=valid_test_dates)
    
    df_results.to_csv(output_file)
    print(f"✅ Predictions saved to {output_file}")
    
    return df_results

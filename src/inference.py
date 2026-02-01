import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.config import MODELS_DIR
from src.data_utils import load_raw_csvs, align_and_filter, load_scalers, normalize

def predict_and_save_test_results(model, device, output_file, sequence_length=365, batch_size=256):
    """
    Generates predictions for the test set and saves them to a CSV.
    Rows: Dates, Columns: Station IDs.
    """
    print("⏳ Generating Predictions...")
    
    # 1. Load and Align Data (Reusing shared logic)
    # Note: We pass 'p' (precip) as dummy flow because we don't need actual flow targets for inference
    p, tmax, tmin, flow, static_df = load_raw_csvs()
    p, tmax, tmin, _, static_df, stations, master_index = align_and_filter(p, tmax, tmin, p, static_df)
    
    # 2. Load Scalers
    scaler_path = MODELS_DIR / "scalers.json"
    scalers = load_scalers(scaler_path)
    
    # 3. Normalize Dynamic Features
    print("   Normalizing data...")
    # Shape: (Total_Time, N_Stations, 3)
    dyn_array = np.stack([p.values, tmax.values, tmin.values], axis=2).astype(np.float32)
    dyn_norm = normalize(dyn_array, scalers['dyn_mean'], scalers['dyn_std'])
    
    # 4. Normalize Static Features
    # Ensure we pick the exact same columns used during training
    static_feats = scalers["static_features"] # e.g. ['area_km2', 'glacier_pct', 'mean_elev']
    stat_vals = static_df[static_feats].values.astype(np.float32)
    stat_norm = normalize(stat_vals, scalers['stat_mean'], scalers['stat_std'])
    
    # 5. Define Test Indices
    # Matches the test split logic: 1980-1989 OR 2013-Present
    test_mask = ((master_index.year >= 1980) & (master_index.year <= 1989)) | \
                (master_index.year >= 2013)
    
    # Convert boolean mask to integer indices relative to the master_index
    all_indices = np.arange(len(master_index))
    test_indices = all_indices[test_mask]
    
    # Filter for lookback (cannot predict if t < sequence_length)
    valid_test_indices = test_indices[test_indices >= sequence_length]
    valid_test_dates = master_index[valid_test_indices]
    
    print(f"   Predicting for {len(stations)} stations over {len(valid_test_dates)} days...")
    
    # 6. Prediction Loop (Station-by-Station)
    model.eval()
    results_dict = {}
    
    # We iterate stations one by one to keep the resulting dictionary clean
    for i, station_id in enumerate(tqdm(stations, desc="Stations")):
        
        # A. Prepare Static Input for this Station
        # Shape: (1, 3) -> We will repeat this to match the batch size later
        # stat_norm[i] gets the static features for the i-th station
        station_static = torch.tensor(stat_norm[i]).float().to(device).unsqueeze(0)
        
        station_preds = []
        
        # B. Batch over Time
        # Instead of predicting 1 day at a time, we predict 'batch_size' days at once for this station
        for k in range(0, len(valid_test_indices), batch_size):
            batch_indices = valid_test_indices[k : k + batch_size]
            current_batch_size = len(batch_indices)
            
            # Construct the Dynamic Batch: (Batch, Seq_Len, Features)
            # We slice the continuous time series 'dyn_norm' for the specific station 'i'
            windows = []
            for t in batch_indices:
                # Extract [t-365 : t] for station i
                windows.append(dyn_norm[t-sequence_length : t, i, :])
            
            # Convert to Tensor
            dyn_batch = torch.tensor(np.array(windows)).float().to(device)
            
            # Expand static features to match this batch size
            stat_batch = station_static.repeat(current_batch_size, 1)
            
            # Predict
            with torch.no_grad():
                y_pred = model(dyn_batch, stat_batch)
                
                # Move to CPU and flatten
                # y_pred is (Batch, 1) -> flatten to (Batch,)
                station_preds.extend(y_pred.cpu().numpy().flatten())
        
        # Store results for this station
        results_dict[station_id] = station_preds

    # 7. Build and Save DataFrame
    print("   Constructing DataFrame...")
    df_results = pd.DataFrame(results_dict, index=valid_test_dates)
    
    df_results.to_csv(output_file)
    print(f"✅ Predictions saved to {output_file}")
    
    return df_results

# src/inference.py
import json
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.config import DATA_START_YEAR, DATA_END_YEAR
from src.data_utils import load_raw_csvs, align_and_filter, load_scalers, normalize, scaler_json_path

SEQUENCE_LENGTH = 365

def predict_and_save_full_results(model, device, output_file, dynamic_cols, static_cols, exp_name, batch_size=256, force_zero_glacier=False, hidden_out_file=None):
    """
    Generates predictions for the ENTIRE dataset (Train + Val + Test) and saves to CSV.
    Rows: Dates, Columns: Station IDs.

    If `hidden_out_file` is given, the final-timestep LSTM hidden state h_T is
    collected in the SAME forward pass (identical scalers / windowing / eval
    determinism as the prediction — this is what guarantees h_T corresponds to
    its prediction f_m) and saved as a float16 array of shape
    (n_dates, n_stations, H), with a companion `hidden_index.json` written
    alongside it. Requires a model whose forward accepts `return_hidden=True`.
    """
    print(f"⏳ Generating Full Dataset Predictions -> {output_file.name}")
    
    # 1. Load and Align Data
    dyn_dict_raw, flow_raw, static_raw = load_raw_csvs(dynamic_cols)
    dyn_dict, flow, static_full, static, stations, master_index = align_and_filter(
        dyn_dict_raw, flow_raw, static_raw, static_cols
    )
    
    # 2. Load Scalers (architecture-qualified name from the model itself, with a
    #    legacy fallback so older single-architecture runs still resolve)
    model_type = getattr(model, "MODEL_TYPE", None)
    scaler_path = scaler_json_path(exp_name, model_type)
    if not scaler_path.exists():
        scaler_path = scaler_json_path(exp_name)  # legacy {exp_name}_scalers.json
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
    full_mask = (master_index.year >= DATA_START_YEAR) & (master_index.year <= DATA_END_YEAR)
    
    all_indices = np.arange(len(master_index))
    target_indices = all_indices[full_mask]
    
    # Filter for lookback (cannot predict if t < sequence_length)
    valid_indices = target_indices[target_indices >= SEQUENCE_LENGTH]
    valid_dates = master_index[valid_indices]
    
    print(f"   Predicting for {len(stations)} stations over {len(valid_dates)} days...")
    
    # 6. Prediction Loop
    model.eval()
    results_dict = {}

    save_hidden = hidden_out_file is not None
    if save_hidden:
        H = model.fc.in_features                       # 256; the dim of h_T feeding fc
        # Pre-allocate the full (dates, stations, H) array as float16 (~2.2 GB for
        # 15706 x 269 x 256) and fill it in place to bound peak memory.
        hidden_arr = np.empty((len(valid_dates), len(stations), H), dtype=np.float16)

    for i, station_id in enumerate(tqdm(stations, desc="Stations")):
        station_static = torch.tensor(stat_norm[i]).float().to(device).unsqueeze(0)
        station_preds = []
        station_h = [] if save_hidden else None

        for k in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[k : k + batch_size]
            current_batch_size = len(batch_indices)

            windows = []
            for t in batch_indices:
                windows.append(dyn_norm[t-SEQUENCE_LENGTH : t, i, :])

            dyn_batch = torch.tensor(np.array(windows)).float().to(device)
            stat_batch = station_static.repeat(current_batch_size, 1)

            with torch.no_grad():
                if save_hidden:
                    y_pred, h_T = model(dyn_batch, stat_batch, return_hidden=True)
                    station_h.append(h_T.cpu().numpy().astype(np.float16))
                else:
                    y_pred = model(dyn_batch, stat_batch)
                station_preds.extend(y_pred.cpu().numpy().flatten())

        results_dict[station_id] = station_preds
        if save_hidden:
            hidden_arr[:, i, :] = np.concatenate(station_h, axis=0)

    # 7. Build and Save predictions
    print("   Constructing DataFrame...")
    df_results = pd.DataFrame(results_dict, index=valid_dates)
    df_results.index.name = 'Date'
    df_results.to_csv(output_file)
    print(f"✅ Predictions saved to {output_file}")

    # 8. Save hidden states + shared index (identical grid across members)
    if save_hidden:
        hidden_out_file = Path(hidden_out_file)
        hidden_out_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(hidden_out_file, hidden_arr)
        index_path = hidden_out_file.parent / "hidden_index.json"
        with open(index_path, "w") as f:
            json.dump({
                "stations": list(stations),
                "dates": [d.strftime("%Y-%m-%d") for d in valid_dates],
                "shape": list(hidden_arr.shape),         # (n_dates, n_stations, H)
                "axes": ["date", "station", "hidden"],
                "dtype": "float16",
                "hidden_size": int(H),
            }, f)
        print(f"✅ Hidden states {hidden_arr.shape} saved to {hidden_out_file}")
        print(f"   Shared index -> {index_path}")

    return df_results

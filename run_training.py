import argparse
import torch
from src.dataset import load_and_preprocess_data
from src.models import EALSTM
from src.training import train_epoch, evaluate
from src.inference import predict_and_save_full_results
from src.config import MODELS_DIR, OUTPUT_DATA_DIR

# --- EXPERIMENT FEATURE MAPPINGS ---
EXPERIMENT_CONFIGS = {
    "baseline": {
        "dynamic": ['temp_max', 'temp_min', 'precip'],
        "static":  ['glacier_pct']
    },
    "topographic": {
        "dynamic": ['temp_max', 'temp_min', 'precip'],
        "static":  ['basin_area_km2', 'mean_elev', 'elev_range', 'mean_slope', 'glacier_pct']
    },
    "phase-split": {
        "dynamic": ['temp_max', 'temp_min', 'rain', 'snow'],
        "static":  ['basin_area_km2', 'mean_elev', 'elev_range', 'mean_slope', 'glacier_pct']
    }
}

def main():
    # --- 1. Command Line Arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, choices=EXPERIMENT_CONFIGS.keys(), help="Name of the experiment to run")
    parser.add_argument("--member_id", type=int, required=True, help="Ensemble member ID (0-9)")
    args = parser.parse_args()

    # --- 2. Configuration ---
    # SLURM isolates the GPU, so it will always be cuda:0 for this specific Python process
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCHS = 50
    HIDDEN_DIM = 256
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 512
    NUM_WORKERS = 4
    PATIENCE = 5

    # Retrieve features for the chosen experiment
    DYNAMIC_FEATURES = EXPERIMENT_CONFIGS[args.exp_name]["dynamic"]
    STATIC_FEATURES = EXPERIMENT_CONFIGS[args.exp_name]["static"]

    print(f"🚀 Job started on {DEVICE} | Experiment: {args.exp_name} | Member: {args.member_id}")
    print(f"📊 Features -> Dynamic: {len(DYNAMIC_FEATURES)} | Static: {len(STATIC_FEATURES)}")

    # --- 3. Load Data ---
    train_loader, val_loader, test_loader, stations = load_and_preprocess_data(
        dynamic_cols=DYNAMIC_FEATURES,
        static_cols=STATIC_FEATURES,
        sequence_length=365,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # File naming scheme
    member_model_path = MODELS_DIR / f"{args.exp_name}_member_{args.member_id}.pth"
    member_pred_path = OUTPUT_DATA_DIR / f"{args.exp_name}_preds_member_{args.member_id}.csv"
    member_cf_path = OUTPUT_DATA_DIR / f"{args.exp_name}_preds_noglacier_member_{args.member_id}.csv"

    # --- 4. Initialize Model ---
    model = EALSTM(input_dim_dyn=len(DYNAMIC_FEATURES), 
                   input_dim_stat=len(STATIC_FEATURES),
                   hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"Starting Training for Member {args.member_id}...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = evaluate(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0 
            torch.save(model.state_dict(), member_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n🛑 Early stopping triggered!")
                break

    # --- 6. Final Benchmark ---
    print(f"\n--- Final Evaluation: Member {args.member_id} ---")
    model.load_state_dict(torch.load(member_model_path, weights_only=True))

    test_loss = evaluate(model, test_loader, DEVICE)
    print(f"Test Set Basin-Averaged Loss: {test_loss:.4f}")

    print("Generating Standard Predictions CSV...")
    predict_and_save_full_results(
        model, DEVICE, output_file=member_pred_path,
        dynamic_cols=DYNAMIC_FEATURES, static_cols=STATIC_FEATURES,
        batch_size=BATCH_SIZE, force_zero_glacier=False
    )

    if 'glacier_pct' in STATIC_FEATURES:
        print("Generating Counterfactual Predictions CSV (0% Glaciation)...")
        predict_and_save_full_results(
            model, DEVICE, output_file=member_cf_path,
            dynamic_cols=DYNAMIC_FEATURES, static_cols=STATIC_FEATURES,
            batch_size=BATCH_SIZE, force_zero_glacier=True
        )

    print(f"\n🎉 Member {args.member_id} Complete.")

if __name__ == "__main__":
    main()

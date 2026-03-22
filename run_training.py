# run_training.py
import torch
from src.dataset import load_and_preprocess_data
from src.models import EALSTM
from src.training import train_epoch, evaluate
from src.inference import predict_and_save_test_results
from src.config import MODELS_DIR, OUTPUT_DATA_DIR

def main():
    # 1. Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 50
    HIDDEN_DIM = 256
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 512
    NUM_WORKERS = 4
    PATIENCE = 5

    # FEATURE TOGGLES: Comment out variables to exclude them from the run
    DYNAMIC_FEATURES = [
        # 'precip',
        'temp_max',
        'temp_min',
        'rain',
        'snow',
        'freeze_frac'
    ]

    STATIC_FEATURES = [
        'basin_area_km2',
        'mean_elev',
        'glacier_pct',
        'elev_range',
        'mean_slope'
    ]

    print(f"🚀 Job started on {DEVICE}")
    print(f"📊 Features -> Dynamic: {len(DYNAMIC_FEATURES)} | Static: {len(STATIC_FEATURES)}")

    # 2. Load Data
    train_loader, val_loader, test_loader, stations = load_and_preprocess_data(
        dynamic_cols=DYNAMIC_FEATURES,
        static_cols=STATIC_FEATURES,
        sequence_length=365,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
        )

    # 3. Initialize Model
    model = EALSTM(input_dim_dyn=len(DYNAMIC_FEATURES), 
                input_dim_stat=len(STATIC_FEATURES),
                hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Starting Training...")
    for epoch in range(EPOCHS):
        # Train (1990-2008)
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        
        # Validate (2009-2012)
        val_loss = evaluate(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Best Model based on Validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0 # Reset counter
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")
            print("  --> Saved new best model")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement for {epochs_no_improve} epochs")
            
            # Early Stopping Check
            if epochs_no_improve >= PATIENCE:
                print(f"\n🛑 Early stopping triggered! Validation loss hasn't improved in {PATIENCE} epochs.")
                print("Restoring best weights and moving to final evaluation...")
                break

    # 5. Final Benchmark
    print("\n--- Final Evaluation ---")
    # Load the best weights (crucial step!)
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pth", weights_only=True))

    # A. Quantitative Score
    test_loss = evaluate(model, test_loader, DEVICE)
    print(f"Test Set Basin-Averaged Loss: {test_loss:.4f}")

    # B. Generate CSV Predictions
    print("Generating CSV...")
    predict_and_save_test_results(
        model,
        DEVICE,
        output_file=OUTPUT_DATA_DIR / "test_set_predictions.csv",
        dynamic_cols=DYNAMIC_FEATURES,
        static_cols=STATIC_FEATURES,
        batch_size=BATCH_SIZE,
        force_zero_glacier=False
    )

    # C. Generate Counterfactual (0% Glaciation) Predictions
    if 'glacier_pct' in STATIC_FEATURES:
        print("\nGenerating Counterfactual Predictions CSV (0% Glaciation)...")
        predict_and_save_test_results(
            model,
            DEVICE,
            output_file=OUTPUT_DATA_DIR / "test_set_predictions_no_glacier.csv",
            dynamic_cols=DYNAMIC_FEATURES,
            static_cols=STATIC_FEATURES,
            batch_size=BATCH_SIZE,
            force_zero_glacier=True
        )

    print("Done.")

if __name__ == "__main__":
    main()

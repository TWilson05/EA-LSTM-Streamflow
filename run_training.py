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
    NUM_WORKERS = 8

    print(f"🚀 Job started on {DEVICE}")

    # 2. Load Data
    train_loader, val_loader, test_loader, stations = load_and_preprocess_data(
        sequence_length=365,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS)

    # 3. Initialize Model
    # Dynamic Features: Precip, Tmax, Tmin (3)
    # Static Features: Area, MeanElev, Glacier% (3)
    model = EALSTM(input_dim_dyn=3, 
                input_dim_stat=3, 
                hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    best_val_loss = float('inf')

    print("Starting Training...")
    for epoch in range(EPOCHS):
        # Train (1990-2008)
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        
        # Validate (2009-2012)
        val_loss = evaluate(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss (NSE*): {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Best Model based on Validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")
            print("   --> Saved new best model")

    # 5. Final Benchmark
    print("\n--- Final Evaluation ---")
    # Load the best weights (crucial step!)
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pth"))

    # A. Quantitative Score
    test_loss = evaluate(model, test_loader, DEVICE)
    print(f"Test Set Basin-Averaged Loss: {test_loss:.4f}")

    # B. Generate CSV Predictions
    print("Generating CSV...")
    predict_and_save_test_results(
        model,
        DEVICE,
        output_file=OUTPUT_DATA_DIR / "test_set_predictions.csv",
        batch_size=BATCH_SIZE
    )
    print("Done.")

if __name__ == "__main__":
    main()

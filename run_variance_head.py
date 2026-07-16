"""Job 2 entry point — train the post-hoc variance head. Twin of run_training.py.

Root runner for the MVE variance head. Mirrors run_training.py but trains ONE small head
over the frozen, precomputed hidden states (no 10-member array). It reads frozen artifacts
via the index contract (data/output/results_MVE/) and never imports Job 1 training internals.

Split discipline (differs from the mean model): the mean model used the fixed TEMPORAL
splits (fit on train, early-stop on val). Here val IS the fit data (train-year residuals
are optimistic), so val is carved 80/20 into fit/select with a fixed seed; test is untouched
here and is read at inference via the contract.
"""
import argparse

import torch
from torch.utils.data import DataLoader, random_split

from run_training import EXPERIMENT_CONFIGS          # DRY: reuse the experiment keys
from src.config import MODELS_DIR, OUTPUT_DATA_DIR
from src.mve_dataset import load_variance_data
from src.mve_head import VarianceHead, SingleALDHead
from src.mve_training import fit_variance_head
from src.mve_inference import predict_and_save_variance


def main():
    # --- 1. Command line arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["ealstm", "lstm"], default="lstm",
                        help="Mean-model architecture whose hidden states/preds we read")
    parser.add_argument("--exp_name", type=str, default="topographic",
                        choices=EXPERIMENT_CONFIGS.keys())
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac_fit", type=float, default=0.8,
                        help="fraction of the val split used to FIT (rest = early-stop/select)")
    parser.add_argument("--head", type=str, default="gaussian", choices=["gaussian", "ald"],
                        help="aleatoric head: gaussian (VarianceHead) or ald (SingleALDHead, skew-aware)")
    args = parser.parse_args()

    # --- 2. Configuration ---
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    HIDDEN_DIM = 256

    print(f"🚀 Variance head | {DEVICE} | exp: {args.exp_name} | mean model: {args.model_type}")

    # --- 3. Data (only the val split; head fits on val, test read at inference) ---
    _, val_loader, _, stations = load_variance_data(
        exp_name=args.exp_name, model_type=args.model_type,
        batch_size=args.batch_size, num_workers=NUM_WORKERS, splits=("val",),
    )

    # --- 4. Carve val -> fit / select (80/20, fixed seed) ---
    val_ds = val_loader.dataset
    n = len(val_ds)
    n_fit = int(args.val_frac_fit * n)
    n_sel = n - n_fit
    gen = torch.Generator().manual_seed(args.seed)
    fit_ds, sel_ds = random_split(val_ds, [n_fit, n_sel], generator=gen)
    fit_loader = DataLoader(fit_ds, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    sel_loader = DataLoader(sel_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    print(f"val carved: fit {n_fit} / select {n_sel}  (seed {args.seed}, {args.val_frac_fit:.0%}/"
          f"{1 - args.val_frac_fit:.0%})")

    # --- 5. Head + fit (criterion auto-selected by head type in fit_variance_head) ---
    HeadCls = SingleALDHead if args.head == "ald" else VarianceHead
    head = HeadCls(hidden_dim=HIDDEN_DIM)
    save_path = MODELS_DIR / f"{args.exp_name}_{args.model_type}_{head.MODEL_TYPE.lower()}_varhead.pth"
    head, history = fit_variance_head(
        head, fit_loader, sel_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        device=DEVICE, save_path=save_path,
    )
    print(f"best sel NLL {min(history['sel_nll']):.4f} | head saved -> {save_path}")

    # --- 6. Inference: emit the predictive variance grid (reads the full grid via contract) ---
    # ALD writes to a SEPARATE dir so the Gaussian results (results_MVE/variance) are kept for
    # the before/after comparison; Gaussian keeps the default dir.
    out_dir = None if args.head == "gaussian" else (OUTPUT_DATA_DIR / "results_MVE" / "variance_ald")
    predict_and_save_variance(
        exp_name=args.exp_name, model_type=args.model_type,
        state_path=save_path, device=DEVICE, hidden_dim=HIDDEN_DIM, out_dir=out_dir,
    )

    print("🎉 Variance head complete.")


if __name__ == "__main__":
    main()

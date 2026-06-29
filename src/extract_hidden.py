"""Post-hoc hidden-state extraction for the Ch3 variance head (pipeline item 1).

For an ensemble whose members were ALREADY trained (before --save_hidden existed),
this re-runs inference on each frozen member to emit the final-timestep hidden
state h_T without retraining. A normal pipeline run started with
`run_training.py --save_hidden` produces the same artifacts inline; this is the
catch-up path for existing checkpoints.

mu-hat and h_T are regenerated in one pass, so h_T corresponds to its prediction
(invariant #2). The regenerated predictions (written to results_MVE/states/), not
the original CSVs, are the canonical f_m for the VE stage.

Run from the project root as a module:

    python -m src.extract_hidden                 # topographic_lstm, members 0-9
    python -m src.extract_hidden --exp_name topographic --n_members 10

Needs ~2.2 GB disk per member (float16 hidden states).
"""
import argparse

import numpy as np
import pandas as pd
import torch

from src.config import MODELS_DIR, OUTPUT_DATA_DIR
from src.models import StandardLSTM
from src.inference import predict_and_save_full_results
# Single source of truth for the per-experiment feature lists (no drift vs training).
from run_training import EXPERIMENT_CONFIGS

HIDDEN_SIZE = 256
BATCH_SIZE = 512


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default="topographic", choices=EXPERIMENT_CONFIGS.keys())
    p.add_argument("--n_members", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = p.parse_args()

    model_type = StandardLSTM.MODEL_TYPE                       # "lstm"
    dynamic_features = EXPERIMENT_CONFIGS[args.exp_name]["dynamic"]
    static_features = EXPERIMENT_CONFIGS[args.exp_name]["static"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states_dir = OUTPUT_DATA_DIR / "results_MVE" / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    # Canonical f_m used by EXP5 today (for the determinism sanity check, item 2)
    existing_preds_dir = OUTPUT_DATA_DIR / f"{args.exp_name}_{model_type}"

    print(f"Device: {device} | {args.exp_name}_{model_type} | "
          f"{args.n_members} members -> {states_dir}")

    for m in range(args.n_members):
        ckpt = MODELS_DIR / f"{args.exp_name}_{model_type}_member_{m}.pth"
        model = StandardLSTM(
            dyn_input_size=len(dynamic_features),
            stat_input_size=len(static_features),
            hidden_size=HIDDEN_SIZE,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()

        pred_out = states_dir / f"{args.exp_name}_{model_type}_preds_member_{m}.csv"
        hidden_out = states_dir / f"hidden_member_{m}.npy"

        print(f"\n=== member {m} ===")
        df = predict_and_save_full_results(
            model, device, output_file=pred_out,
            dynamic_cols=dynamic_features, static_cols=static_features,
            exp_name=args.exp_name, batch_size=args.batch_size,
            hidden_out_file=hidden_out,
        )

        # Item 2 — determinism sanity check vs the existing canonical CSV.
        # NOTE: exact bit-match is NOT guaranteed across hardware (cuDNN/CPU vs the
        # HPC GPU). A large deviation means a real pipeline bug (scaler/window drift);
        # a tiny one is benign — the regenerated preds above are what the VE stage
        # should use, since they are the ones h_T was produced with.
        existing = existing_preds_dir / f"{args.exp_name}_{model_type}_preds_member_{m}.csv"
        if existing.exists():
            old = pd.read_csv(existing, index_col=0, parse_dates=True)
            a, b = df.align(old, join="inner")
            mask = np.isfinite(a.values) & np.isfinite(b.values)
            max_abs = np.nanmax(np.abs(a.values[mask] - b.values[mask]))
            status = "OK" if max_abs < 1e-4 else "DRIFT — investigate"
            print(f"   determinism vs saved CSV: max|Δ| = {max_abs:.2e}  [{status}]")
        else:
            print(f"   (no existing CSV at {existing} to compare)")

    print(f"\nDone. Hidden states + hidden_index.json in {states_dir}")


if __name__ == "__main__":
    main()

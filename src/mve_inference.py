"""Variance-head inference / prediction. Twin of src/inference.py (Job 2).

Turns a trained head into per-cell aleatoric scale on the (date, station) grid,
combines it with the ensemble epistemic variance, and writes the predictive layer
aligned to the index contract.

THE CRUX (standardized -> physical): the head predicts sigma_a of the STANDARDIZED
residual z = r/q_std, so its output is dimensionless. Physical aleatoric scale is
q_std * sigma_a (per station). Total predictive variance is
    (q_std * sigma_a)^2 + sigma_e^2
i.e. the sigma_a^2 + sigma_e^2 accounting; sigma_e^2 comes from the ensemble and is
already physical (mm/day^2). This conversion lives ONLY here -- training stays in
standardized z-space throughout.

sigma_a is finite on the FULL grid (h exists for every cell), so nothing is NaN-scattered
here. valid_mask (in the contract) is kept for EVALUATION subsetting (cells with obs),
not for the sigma product. Inference runs under no_grad, so the softplus fragility that
bites during training does not apply -- softplus is always finite/positive here.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.config import OUTPUT_DATA_DIR, MODELS_DIR
from src.mve_dataset import load_contract, _discover_hidden_members, _pool_hidden, STATES_DIR
from src.mve_head import VarianceHead


def load_head(state_path, hidden_dim=256, device=None):
    """Instantiate VarianceHead, load its state_dict, eval(). Returns (head, device)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    head = VarianceHead(hidden_dim=hidden_dim)
    head.load_state_dict(torch.load(state_path, map_location=device))
    head.to(device).eval()
    return head, device


def predict_sigma_a(head, contract, device, chunk=2048):
    """Run the head over every cell and return sigma_a on the (n_dates, n_stations) grid
    in STANDARDIZED units. h is pooled the same way as in mve_dataset (mean over members)
    and processed in date-row chunks to cap memory (the raw hidden_member_*.npy are ~2 GB
    each). All cells are finite (h exists everywhere)."""
    D, S = len(contract["dates"]), len(contract["stations"])
    member_paths = [p for _, p in _discover_hidden_members(STATES_DIR)]

    sigma_a = np.empty((D, S), dtype=np.float32)
    head.eval()
    with torch.no_grad():
        for start in range(0, D, chunk):
            rows = np.arange(start, min(start + chunk, D))
            h = _pool_hidden(member_paths, rows, "mean")          # (r, S, H) float32
            r, _, H = h.shape
            ht = torch.from_numpy(h.reshape(-1, H)).float().to(device)
            s = head(ht).squeeze(-1).cpu().numpy()                # (r*S,)
            sigma_a[rows] = s.reshape(r, S)
    return sigma_a


def predict_and_save_variance(exp_name="topographic", model_type="lstm", state_path=None,
                              out_dir=None, device=None, hidden_dim=256):
    """Full inference: sigma_a -> physical -> combine with sigma_e^2 -> persist, grid-aligned
    to the contract. Returns a dict of the grid arrays.

    Calibration (PIT / coverage on test, stratified by regime) is the acceptance test -- run
    it in EXP5 against the saved grids; it is what judges whether these sigmas are honest.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if state_path is None:
        state_path = MODELS_DIR / f"{exp_name}_{model_type}_{VarianceHead.MODEL_TYPE.lower()}_varhead.pth"
    head, device = load_head(state_path, hidden_dim=hidden_dim, device=device)

    c = load_contract(exp_name, model_type)                       # q_std, sigma_e2, mu_hat, valid
    sigma_a_std = predict_sigma_a(head, c, device)                # (D, S) standardized

    # THE conversion: standardized scale of z -> physical scale of the residual (mm/day)
    sigma_a_phys = c["q_std"][None, :] * sigma_a_std              # (D, S) mm/day
    sigma_a2 = sigma_a_phys ** 2
    var_total = sigma_a2 + c["sigma_e2"]                          # sigma_a^2 + sigma_e^2

    out_dir = Path(out_dir) if out_dir else (OUTPUT_DATA_DIR / "results_MVE" / "variance")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sigma_a.npy", sigma_a_phys)
    np.savez(out_dir / "predictive.npz", mu_hat=c["mu_hat"], sigma_e2=c["sigma_e2"],
             sigma_a2=sigma_a2, var_total=var_total)
    meta = {
        "exp_name": exp_name, "model_type": model_type,
        "head_type": VarianceHead.MODEL_TYPE, "state_path": str(state_path),
        "units": "mm/day (sigma_a, sigma; mm/day^2 for variances)",
        "standardization": "sigma_a_phys = q_std * softplus(head(h)); head trained on z = r/q_std",
        "accounting": "var_total = (q_std*sigma_a)^2 + sigma_e^2  (sigma_a^2 + sigma_e^2)",
        "grid": "full (date, station); finite everywhere (h exists). valid_mask (contract) "
                "marks cells with obs, for evaluation only.",
    }
    with open(out_dir / "variance_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    v = c["valid"]
    print(f"Variance head inference [{exp_name}_{model_type}] -> {out_dir}")
    print(f"  grid {sigma_a_phys.shape} | on valid cells: "
          f"median sigma_a {np.median(sigma_a_phys[v]):.3f}, "
          f"median sigma_e^2 {np.median(c['sigma_e2'][v]):.4f}, "
          f"median var_total {np.median(var_total[v]):.4f} mm/day^2")
    return {"sigma_a": sigma_a_phys, "sigma_a2": sigma_a2,
            "sigma_e2": c["sigma_e2"], "var_total": var_total, "valid": v}

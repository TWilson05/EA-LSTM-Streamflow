"""Variance-head dataset — assemble training tensors from the index contract.
Twin of src/dataset.py (Job 2).

Responsibility:
  - Load frozen artifacts from data/output/results_MVE/ (hidden states, y_obs_mm,
    valid_mask, q_std, index_dates) + the member prediction CSVs for mu_hat / sigma_e^2.
  - Build the head's training target: standardized residual z = (y - mu_hat) / q_std.
  - Apply the train/val/test split from index_dates.csv (fit on val, report on test).
  - Source of truth for the grid is build_index's contract — do NOT re-derive it here.

Scaffold only — bodies intentionally left unimplemented (owner to populate).

TODO:
  - hidden-state pooling policy (mean-pool across members vs per-member) — pending decision
  - a Dataset/loader (or plain tensors) yielding (h_T, z), masked to valid & split
"""

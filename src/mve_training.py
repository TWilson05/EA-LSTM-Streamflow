"""Variance-head training loop. Twin of src/training.py (Job 2).

Responsibility:
  - NLL training loop over the mve_dataset tensors.
  - Fit on val years, early-stop/select on val, never touch test during fit.
  - REUSE src/training.py helpers (device setup, checkpoint save/load, early stopping) —
    do not fork them.
  - Save the head separately (e.g. {exp}_{model_type}_varhead.pth); never overwrite the
    frozen mean checkpoints.

Scaffold only — bodies intentionally left unimplemented (owner to populate).

TODO:
  - def train_variance_head(...): the fit loop returning the trained head + history
"""

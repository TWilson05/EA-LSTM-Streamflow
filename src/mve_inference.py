"""Variance-head inference / prediction. Twin of src/inference.py (Job 2).

Responsibility:
  - Load a trained varhead, emit per-cell aleatoric scale sigma_a (and nu) on the grid.
  - Combine with the ensemble between-member variance sigma_e^2 -> total predictive
    variance (sigma_a^2 + sigma_e^2), per the 1/M accounting in chapter3.md.
  - Write outputs to data/output/results_MVE/variance/ (sigma_a, nu, predictive, meta),
    grid-aligned to the index contract.

Scaffold only — bodies intentionally left unimplemented (owner to populate).

TODO:
  - def predict_and_save_variance(...): forward pass + assemble + persist
"""

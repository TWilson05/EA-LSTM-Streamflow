"""Variance head model + likelihood. Twin of src/models.py (the module) plus the
loss role of src/training.py (Job 2).

Responsibility:
  - VarianceHead(nn.Module): small MLP on the frozen hidden state
    (hidden_dim -> var_hidden -> Student-t params: scale, and nu global-or-input-dependent).
    Positivity via softplus; scale is of the STANDARDIZED residual z (physical:
    sigma_phys = q_std * scale).
  - Student-t NLL (coupled to the head, kept in this file).

Scaffold only — bodies intentionally left unimplemented (owner to populate).

TODO:
  - class VarianceHead(nn.Module): __init__, forward
  - class StudentTNLL(nn.Module) (or a functional nll): forward(params, z, mask)
  - decision: Gaussian first vs Student-t; nu global vs nu(x)
"""

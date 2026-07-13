"""Variance head model + likelihood (Job 2). Twin of models.py, plus the loss role of
training.py.

STRAIGHT prediction space (mm/day). The mean predicts streamflow straight and this head
models the STRAIGHT-space standardized residual. We do NOT log-transform streamflow;
log space is a Branch-B ESCALATION, taken only if calibration (PIT) shows a tail-shape
failure the straight-space head can't cover.

Consumes the (h, z) batches from mve_dataset:
  h : (B, 256) frozen hidden state (mean-pooled over members)
  z : (B, 1)   standardized residual (y - mu_hat)/q_std, STRAIGHT space -- q_std is applied
               upstream, so nothing here touches q_std / y_true again.

To implement:
  1. VarianceHead(nn.Module)
       __init__ : layer(s) 256 -> [var_hidden] -> 1. Linear (256 -> 1) is the start; add a
                  hidden layer only if linear underfits the empirical binned residual
                  variance (escalation ladder).
       forward(h) -> the predicted SCALE (sigma > 0) of the straight-space residual z,
                  via softplus for positivity; sigma is used directly by the NLL. (No statics
                  passed -- they are already inside h. NOT a log-transform of streamflow.)
  2. gaussian_nll(head_out, z) -> scalar
       NLL of z under N(0, sigma^2) (sigma from head_out), meaned over the batch. Fit on
       the val split; report on test.

Escalation ONLY (not the default; gated on a PIT tail-failure): add a Student-t `nu`
output (a separate parameter, not a transform of sigma), and/or move the target to log
space (Branch B -> CMAL/UMAL).
"""

import math
import torch
import torch.nn as nn

class VarianceHead(nn.Module):
    MODEL_TYPE = "Linear"

    def __init__(self, hidden_dim=256):
        super().__init__()

        # Train a linear unit from hidden_dim (256, ) to (1, ), then positivity transform
        self.linear = nn.Linear(in_features=hidden_dim, out_features=1)
        self.softplus = nn.Softplus()

    def forward(self, h):
        """
        h:      (B, hidden_dim) frozen hidden state (mean-pooled over members). The statics
                are already baked into h (StandardLSTM concatenates them into the LSTM input),
                so nothing else is passed -- this mirrors the mean head, which is fc(h_T) alone.
        returns sigma (B, 1): predicted scale (> 0) of the STRAIGHT-space standardized
                residual z. softplus gives positivity; sigma is used directly by the NLL
                (no exp; no log-transform of streamflow -- z stays straight).
        """
        raw = self.linear(h)          # (B, 256) -> (B, 1), unconstrained
        sigma = self.softplus(raw)    # (B, 1), positive scale
        return sigma

class GaussianNLL(nn.Module):
    MODEL_TYPE = "GaussianNLL"

    def __init__(self, eps=1e-6, full=True):
        super().__init__()
        self.eps = eps
        self.full = full

    def forward(self, sigma, z):
        """
        sigma: (B, 1) positive scale straight from head + softplus
        z:     (B, 1) standardized residual from dataset (may contain NaN)
        """
        # z carries NaN (mve_dataset passes all cells through). Mask and SELECT before the
        # math so NaNs never enter the graph / poison gradients (mirrors BasinAveragedNSELoss).
        mask = ~torch.isnan(z)
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=sigma.device)

        s = sigma[mask] + self.eps        # select valid entries FIRST
        zz = z[mask]

        nll = torch.log(s) + 0.5 * (zz / s) ** 2
        if self.full:                     # add the 2*pi constant only when reporting/comparing
            nll = nll + 0.5 * math.log(2 * math.pi)

        return torch.mean(nll)
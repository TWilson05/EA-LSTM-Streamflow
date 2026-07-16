"""Variance head models (Job 2). Twin of models.py.

Heads ONLY. The NLL losses live in mve_training (matching training.py, which defines
BasinAveragedNSELoss / MaskedMSELoss there); this file holds the head nn.Modules and the
distribution math they own.

STRAIGHT prediction space (mm/day). The mean predicts streamflow straight and each head
models the STRAIGHT-space standardized residual. We do NOT log-transform streamflow;
log space is a Branch-B ESCALATION, taken only if calibration (PIT) shows a tail-shape
failure the straight-space head can't cover.

Consumes the (h, z) batches from mve_dataset:
  h : (B, 256) frozen hidden state (mean-pooled over members)
  z : (B, 1)   standardized residual (y - mu_hat)/q_std, STRAIGHT space -- q_std is applied
               upstream, so nothing here touches q_std / y_true again.

Heads:
  1. VarianceHead   — symmetric Gaussian: forward(h) -> sigma > 0 (softplus). Paired with
                      GaussianNLL (in mve_training).
  2. SingleALDHead  — skew-aware asymmetric-Laplace (Branch B escalation): forward(h) ->
                      (b > 0, tau in (0,1)). Paired with ALD_NLL (in mve_training). Drop-in
                      swap for VarianceHead — fit_variance_head auto-selects its loss and
                      mve_inference auto-detects it from the checkpoint. The head owns the ALD
                      moment math (meanshift / var_factor); the loss consumes it.

Escalation ladder: Linear -> add a hidden layer only if the linear head underfits;
symmetric -> ALD if PIT shows a tail-shape failure; ALD -> CMAL/UMAL (mixture of ALDs)
and/or a log-space target.
"""

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

class SingleALDHead(nn.Module):
    """Skew-aware aleatoric head: frozen h -> (scale b>0, asymmetry tau in (0,1)).

    Same freeze as VarianceHead: two LINEAR readouts of the frozen, mean-pooled h; the mean
    model is never touched. Its loss (ALD_NLL, in mve_training) mean-recenters the ALD so
    E[y]=mu_hat. The head OWNS the ALD moment math below, so the loss and mve_inference share
    one definition.

    Asymmetric-Laplace, quantile parameterization ALD(m, b, tau):
        f(z) = (tau(1-tau)/b) * exp(-rho_tau((z-m)/b)),   rho_tau(u) = u*(tau - 1{u<0})
        mean = m + b*(1-2tau)/(tau(1-tau));  var = b^2*(1-2tau+2tau^2)/(tau^2(1-tau)^2)
    Mean-FROZEN => fix the location m = -b*meanshift(tau) so E[z]=0. Variance is
    translation-invariant, so sigma_a^2 uses only (b, tau). tau<0.5 => fat UPPER tail
    (matches the right-skewed streamflow residuals). meanshift / var_factor are pure
    arithmetic, so they also run on numpy arrays (mve_inference calls them on grids).
    """
    MODEL_TYPE = "ALD"
    TAU_MIN = 0.01                                   # keep tau off {0,1} for finite -log(tau(1-tau))

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.scale = nn.Linear(hidden_dim, 1)        # -> softplus -> b > 0
        self.asym = nn.Linear(hidden_dim, 1)         # -> squashed sigmoid -> tau in (TAU_MIN, 1-TAU_MIN)
        self.softplus = nn.Softplus()

    def forward(self, h):
        """h: (B, hidden_dim) frozen state. Returns (B, 2) = [b, tau]. b is the ALD scale of
        the STRAIGHT-space standardized residual z; tau is the asymmetry (tau<0.5 = fat upper
        tail). No log-transform; statics are already inside h (mirrors the mean head)."""
        b = self.softplus(self.scale(h)) + 1e-6
        tau = self.TAU_MIN + (1.0 - 2.0 * self.TAU_MIN) * torch.sigmoid(self.asym(h))
        return torch.cat([b, tau], dim=-1)

    @staticmethod
    def split(out):
        """(B,2) head output -> (b, tau), each (B,1). Shared by ALD_NLL and inference."""
        return out[..., 0:1], out[..., 1:2]

    @staticmethod
    def meanshift(tau):
        """(E[z]-m)/b for the ALD; the loss fixes m = -b*meanshift(tau) so E[z]=0."""
        return (1.0 - 2.0 * tau) / (tau * (1.0 - tau))

    @staticmethod
    def var_factor(tau):
        """Var(z) / b^2 for the ALD; sigma_a = b*sqrt(var_factor(tau))."""
        return (1.0 - 2.0 * tau + 2.0 * tau ** 2) / (tau ** 2 * (1.0 - tau) ** 2)

    @staticmethod
    def aleatoric_sigma(out):
        """std of the ALD (z-units) from the head output: b*sqrt(var_factor(tau)). Feeds the
        sigma_a^2 + sigma_e^2 accounting; the location recenter does not affect it (variance
        is translation-invariant)."""
        b, tau = SingleALDHead.split(out)
        return b * torch.sqrt(SingleALDHead.var_factor(tau))
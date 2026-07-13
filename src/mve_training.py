"""Variance-head training loop. Twin of src/training.py (Job 2).

Fits the (frozen-mean) variance head by Gaussian NLL on (h, z) batches from
mve_dataset. The head is the ONLY trainable object here -- there is no mean model in
this job, just the frozen hidden state h. Fit on the val split (train-year residuals
are optimistic), early-stop on a held-out selection set, report on test. Save the head
separately; never touch the mean checkpoints.

Structure mirrors training.py: a shared _run_epoch(..., optimizer=None) core with
train_epoch / evaluate wrappers. The only changes are the batch shape -- (h, z) with the
call head(h), vs Job 1's (x_dyn, x_stat, y, q_std) with model(x_dyn, x_stat) -- and the
loss (GaussianNLL, imported from mve_head, not defined here). fit_variance_head adds the
epoch loop + early-stop + save (Job 1 keeps that in run_training.py).
"""
import copy

import torch
from tqdm import tqdm

from src.mve_head import GaussianNLL


def _run_epoch(head, loader, device, criterion, optimizer=None):
    """Shared core for training and evaluation (twin of training.py._run_epoch).

    Batch is (h, z). z carries NaN (mve_dataset passes all cells through), so we mask per
    batch to get num_valid for weighting + to skip all-NaN batches; the criterion
    (GaussianNLL) masks NaN internally when it computes the loss. Returns the
    num_valid-weighted mean NLL over the epoch.
    """
    is_train = optimizer is not None
    head.train() if is_train else head.eval()

    total_w_loss = 0.0
    total_valid = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        pbar = tqdm(loader, desc="Train" if is_train else "Eval")
        for h, z in pbar:
            h, z = h.to(device), z.to(device)

            num_valid = (~torch.isnan(z)).sum().item()
            if num_valid == 0:
                continue

            if is_train:
                optimizer.zero_grad()

            sigma = head(h)
            loss = criterion(sigma, z)

            if is_train:
                loss.backward()
                optimizer.step()

            # num_valid-weighted accumulation (batches differ in valid count)
            total_w_loss += loss.item() * num_valid
            total_valid += num_valid

            pbar.set_postfix({'nll': f"{loss.item():.4f}"})

    return total_w_loss / total_valid if total_valid > 0 else 0.0


def train_epoch(head, loader, optimizer, criterion, device):
    return _run_epoch(head, loader, device, criterion, optimizer)


def evaluate(head, loader, criterion, device):
    return _run_epoch(head, loader, device, criterion, optimizer=None)


def fit_variance_head(head, fit_loader, sel_loader, *, criterion=None, epochs=50, lr=1e-3,
                      patience=5, device=None, save_path=None):
    """Train `head` on fit_loader, early-stop / select on sel_loader, keep the best state,
    optionally save it to save_path. Returns (head, history).

    `criterion` defaults to GaussianNLL(); pass a Student-t NLL here later to swap the
    likelihood without touching the loop (the Branch-B escalation).

    Split discipline (chapter3.md): fit = val split, select = held-out slice of val (or
    k-fold within val); test is NEVER used to fit or select. save_path naming is set by the
    caller, e.g. MODELS_DIR / f"{exp}_{model_type}_{head.MODEL_TYPE.lower()}_varhead.pth".
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    head.to(device)
    criterion = criterion or GaussianNLL()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    history = {"fit_nll": [], "sel_nll": []}
    best_sel = float("inf")
    best_state = copy.deepcopy(head.state_dict())
    no_improve = 0

    for epoch in range(epochs):
        fit_nll = train_epoch(head, fit_loader, optimizer, criterion, device)
        sel_nll = evaluate(head, sel_loader, criterion, device)
        history["fit_nll"].append(fit_nll)
        history["sel_nll"].append(sel_nll)
        print(f"Epoch {epoch + 1:>3}/{epochs} | fit NLL {fit_nll:.4f} | sel NLL {sel_nll:.4f}")

        if sel_nll < best_sel:
            best_sel = sel_nll
            best_state = copy.deepcopy(head.state_dict())
            no_improve = 0
            if save_path is not None:
                torch.save(best_state, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no sel improvement in {patience}).")
                break

    head.load_state_dict(best_state)   # restore best regardless of save_path
    return head, history

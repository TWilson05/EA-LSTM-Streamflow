"""Variance-head dataset — assemble training tensors from the index contract.
Twin of src/dataset.py (Job 2).

Unlike dataset.py (which builds 365-day forcing windows for the LSTM), this reads the
FROZEN artifacts written by Job 1 + build_index and produces (h_T, z) pairs, where
h_T is the precomputed hidden state and z = (y - mu_hat) / q_std is the standardized
residual the head models. No forcing, no sequence windowing, no scaler computation —
all of that is frozen upstream. The (date, station) grid is taken verbatim from the
contract; nothing is re-derived here (single source of truth).

Public API (mirrors dataset.load_and_preprocess_data):
    load_variance_data(exp_name, model_type=...) -> (train_loader, val_loader,
                                                     test_loader, stations)
    load_contract(exp_name, model_type=...)      -> dict of full grid arrays
                                                     (for inference / diagnostics)
    ensemble_moments(...)                         -> (mu_hat, sigma_e2, M) on the grid

Fit the head on the VAL split (train-year residuals are over-optimistic); report on
TEST. All three loaders are returned for flexibility.

Memory note: raw hidden_member_*.npy are ~2 GB each; they are mmap'd and pooled one
split at a time, so peak RAM tracks the split size (val/test are small; the train
split is the largest — avoid building it unless you need it).
"""
import json
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import OUTPUT_DATA_DIR
from src.models import StandardLSTM

RESULTS_MVE = OUTPUT_DATA_DIR / "results_MVE"
STATES_DIR = RESULTS_MVE / "states"


# --- locating Job 1 outputs --------------------------------------------------------
def _member_id(path):
    return int(re.search(r"member_(\d+)", path.stem).group(1))


def _discover_member_csvs(exp_name, model_type):
    """Sorted member prediction CSVs, tolerating the run_training location and the
    manually-nested subdir (mirrors build_index._find_member0)."""
    pat = f"{exp_name}_{model_type}_preds_member_*.csv"
    for d in (OUTPUT_DATA_DIR, OUTPUT_DATA_DIR / f"{exp_name}_{model_type}", STATES_DIR):
        hits = list(d.glob(pat))
        if hits:
            return sorted(hits, key=_member_id)
    raise FileNotFoundError(
        f"No member prediction CSVs matching '{pat}' under {OUTPUT_DATA_DIR}. "
        "Run the ensemble (Job 1) first.")


def _discover_hidden_members(states_dir):
    """Sorted (member_id, path) for the extracted hidden states."""
    hits = sorted(states_dir.glob("hidden_member_*.npy"), key=_member_id)
    if not hits:
        raise FileNotFoundError(
            f"No hidden_member_*.npy under {states_dir}. Run training/inference with "
            "--save_hidden (Job 1) before building the variance-head dataset.")
    return [(_member_id(p), p) for p in hits]


# --- ensemble moments (mu_hat, sigma_e^2) from the member CSVs ---------------------
def ensemble_moments(exp_name, model_type, dates, stations):
    """Ensemble mean mu_hat and between-member variance sigma_e^2 (ddof=1, M-1) on the
    (dates, stations) grid, reduced over the member prediction CSVs (mm/day)."""
    csvs = _discover_member_csvs(exp_name, model_type)
    contract_dates = pd.DatetimeIndex(dates)
    mats = []
    for p in csvs:
        # Reindex each member's sim streamflow onto the contract's (date, station) axis.
        # THIS is the agreement: everything is pinned to ONE axis, positionally. Imperfect
        # coverage is TOLERATED, not fatal — a (date, station) a member never predicted
        # (warm-up, data gaps) NaN-fills here and drops out later via valid & isfinite(z),
        # exactly how the ensemble loss masks NaN obs (src/training.py) instead of
        # dropping dates.
        df = pd.read_csv(p, index_col=0, parse_dates=True).reindex(
            index=contract_dates, columns=stations)
        mats.append(df.values.astype(np.float32))
    stack = np.stack(mats, axis=0)                 # (M, D, S)
    mu_hat = stack.mean(axis=0)                    # (D, S); NaN if ANY member lacks the cell
    sigma_e2 = stack.var(axis=0, ddof=1)           # (D, S), M-1
    n_missing = int(np.isnan(mu_hat).sum())
    if n_missing:
        print(f"   ensemble_moments: {n_missing:,}/{mu_hat.size:,} cells "
              f"({n_missing / mu_hat.size:.1%}) unpredicted by >=1 member -> NaN, masked out.")
    return mu_hat, sigma_e2, len(csvs)


# --- hidden-state pooling ----------------------------------------------------------
def _pool_hidden(member_paths, rows, pool):
    """Pool the per-member hidden states over the given date-row indices.
    Returns float32 (len(rows), n_stations, H). mmap keeps only the selected rows
    resident. `pool='mean'` matches training the head off the frozen ensemble mean;
    per-member pooling is a future option (raises for now)."""
    if pool != "mean":
        raise NotImplementedError(
            f"pool={pool!r} not implemented; only 'mean' (mean-pool across members). "
            "Per-member is the deferred prototype path.")
    acc = None
    for p in member_paths:
        arr = np.load(p, mmap_mode="r")            # (D, S, H) float16, on disk
        sub = np.asarray(arr[rows], dtype=np.float32)
        acc = sub if acc is None else acc + sub
    acc /= len(member_paths)
    return acc


# --- the Dataset -------------------------------------------------------------------
class VarianceHeadDataset(Dataset):
    """(h_T, z) pairs for ALL cells of one split, flattened over (date, station).
    z carries NaN where obs/mu_hat are missing -- those are masked at LOSS time
    (~isnan(z)), matching how dataset.py + training.py handle NaN (pass-through, mask in
    the loss). h stored float16 (halves RAM), served as float32; z is the standardized
    residual."""

    def __init__(self, h, z):
        self.h = torch.from_numpy(h)               # (N, H) float16
        self.z = torch.from_numpy(z)               # (N, 1) float32

    def __len__(self):
        return self.h.shape[0]

    def __getitem__(self, i):
        return self.h[i].float(), self.z[i]


# --- contract loader (full grid) ---------------------------------------------------
def load_contract(exp_name="topographic", model_type=StandardLSTM.MODEL_TYPE):
    """Load and align every contract array on the canonical (date, station) grid.
    Returns a dict: dates, stations, split (per-date), q_std (per-station), y, mu_hat,
    sigma_e2, valid (all (D, S) except q_std/split). Used by both the loaders below and
    by inference (which needs sigma_e2 + the grid). Everything is asserted to align to
    states/hidden_index.json."""
    idx = pd.read_csv(RESULTS_MVE / "index_dates.csv", parse_dates=["date"])
    dates = pd.DatetimeIndex(idx["date"])
    split = idx["split"].to_numpy()

    hidx = json.load(open(STATES_DIR / "hidden_index.json"))
    stations = list(hidx["stations"])
    assert pd.DatetimeIndex(pd.to_datetime(hidx["dates"])).equals(dates), \
        "hidden_index dates != index_dates.csv — contract is out of sync"

    y = np.load(RESULTS_MVE / "y_obs_mm.npy")                 # (D, S) f32
    valid = np.load(RESULTS_MVE / "valid_mask.npy")           # (D, S) bool
    qdf = pd.read_csv(RESULTS_MVE / "q_std.csv")
    assert list(qdf["station_id"].astype(str)) == [str(s) for s in stations], \
        "q_std station order != hidden_index station order — positional map would misalign"
    q_std = qdf["q_std"].to_numpy(dtype=np.float32)           # (S,)

    mu_hat, sigma_e2, _ = ensemble_moments(exp_name, model_type, dates, stations)
    assert y.shape == mu_hat.shape == valid.shape, "grid shape mismatch across contract arrays"

    return {"dates": dates, "stations": stations, "split": split, "q_std": q_std,
            "y": y, "mu_hat": mu_hat, "sigma_e2": sigma_e2, "valid": valid}


# --- public entry (twin of load_and_preprocess_data) -------------------------------
def load_variance_data(exp_name="topographic", model_type=StandardLSTM.MODEL_TYPE,
                       batch_size=512, num_workers=0, pool="mean",
                       splits=("train", "val", "test")):
    """Build (train, val, test) DataLoaders of (h_T, z) for the variance head.

    z = (y - mu_hat) / q_std (standardized residual, dimensionless). ALL cells are served;
    z carries NaN where obs/mu_hat are missing and is masked at LOSS time (~isnan(z)),
    matching dataset.py + training.py. Fit on val, report on test.

    `splits` selects which loaders to actually build (pooling a split is the expensive
    step); unrequested splits come back as None. The variance head only needs val, so the
    orchestrator passes splits=("val",) to skip pooling the large train split.
    """
    c = load_contract(exp_name, model_type)
    dates, stations, split = c["dates"], c["stations"], c["split"]
    y, mu_hat, q_std = c["y"], c["mu_hat"], c["q_std"]

    # Agreement guard: obs (y) and sim streamflow (mu_hat) are both pinned to the contract's
    # date axis (ensemble_moments reindexes onto it; gaps NaN-fill and get masked, not
    # dropped). This asserts the assembled grids are positionally aligned to those dates.
    assert y.shape[0] == mu_hat.shape[0] == len(dates), \
        "grid rows disagree with contract dates — obs/sim streamflow not on one axis"

    z = (y - mu_hat) / q_std[None, :]                         # (D, S), NaN where obs/mu_hat missing
    finite = np.isfinite(z)                                   # for reporting only; masking is at loss time

    member_paths = [p for _, p in _discover_hidden_members(STATES_DIR)]

    def make_loader(split_name, shuffle):
        rows = np.where(split == split_name)[0]
        if rows.size == 0:
            return None
        h = _pool_hidden(member_paths, rows, pool)            # (R, S, H) f32
        H = h.shape[-1]
        # Pass ALL cells through (matching dataset.py): z carries NaN where invalid; NaNs
        # are masked at loss time (~isnan(z)), NOT filtered here.
        h_flat = h.reshape(-1, H).astype(np.float16)          # store f16
        z_flat = z[rows].reshape(-1, 1).astype(np.float32)    # may contain NaN
        ds = VarianceHeadDataset(h_flat, z_flat)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    train_loader = make_loader("train", True) if "train" in splits else None
    val_loader = make_loader("val", False) if "val" in splits else None
    test_loader = make_loader("test", False) if "test" in splits else None

    def _finite(split_name):
        return int(finite[np.where(split == split_name)[0]].sum())
    print(f"Variance-head data [{exp_name}_{model_type}] | pool={pool} | "
          f"cells served (finite, masked at loss): "
          f"train {len(train_loader.dataset) if train_loader else 0} ({_finite('train')}), "
          f"val {len(val_loader.dataset) if val_loader else 0} ({_finite('val')}), "
          f"test {len(test_loader.dataset) if test_loader else 0} ({_finite('test')})")

    return train_loader, val_loader, test_loader, stations

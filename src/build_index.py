"""Build the canonical (station, date) index contract for the Ch3 variance head
(pipeline items 4 + 5; folds in the q_std export, item 3, and meta, items 7/8).

The head consumes several arrays that must share ONE (date, station) grid with
identical ordering: the hidden states h_m (results_MVE/states/hidden_member_*.npy),
the ensemble mean mu_hat and between-member variance sigma_e^2 (from the member
prediction CSVs), the target y, and the per-basin q_std. This module pins that
grid and emits the contract everything joins against, so the new NN pulls its
y-structure (residual r = y - mu_hat) from the pre-trained ensemble consistently.

Grid source of truth: results_MVE/states/hidden_index.json (written by the
--save_hidden / extraction pass) -> the contract aligns with h by construction.
Falls back to a member prediction CSV if the hidden index is not present yet, so
the contract can be built and checked before the HPC extraction run.

Outputs (data/output/results_MVE/):
  index_dates.csv   date, year, split             (per-date; 15706 rows)
  y_obs_mm.npy      float32 (n_dates, n_stations)  observed runoff, grid-aligned
  valid_mask.npy    bool    (n_dates, n_stations)  obs present & pred finite
  q_std.csv         station_id, q_std             (labelled training q_std)
  meta.json         transform space, split, sigma_e^2 + q_std provenance

Axis labels live in states/hidden_index.json (or are re-emitted on fallback);
every array above is positionally aligned to it as arr[date_i, station_j], and
station_j == q_std row j == member-CSV column j.

Run locally (fast, no GPU/LSTM):
    python -m src.build_index
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.config import (
    OUTPUT_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_STATIC_ATTR,
    TRAIN_START_YEAR, TRAIN_END_YEAR, VAL_START_YEAR, VAL_END_YEAR,
    TEST_START_YEAR, TEST_END_YEAR,
)
from src.data_utils import load_scalers, scaler_json_path
from src.models import StandardLSTM

# Cited OOD event windows (EXP5 cells 13a/13b) — must land in the held-out period.
EVENT_WINDOWS = {
    "heat_dome_2021": ("2021-06-01", "2021-07-31"),
    "ar_flood_2021":  ("2021-10-20", "2021-12-10"),
}


def _split_for_year(y):
    if TRAIN_START_YEAR <= y <= TRAIN_END_YEAR:
        return "train"
    if VAL_START_YEAR <= y <= VAL_END_YEAR:
        return "val"
    if TEST_START_YEAR <= y <= TEST_END_YEAR:
        return "test"
    return "none"


def _find_member0(exp_name, model_type):
    """Locate a member-0 prediction CSV (grid + pred-finiteness), tolerating both
    the run_training output location and the manually-nested subdir."""
    name = f"{exp_name}_{model_type}_preds_member_0.csv"
    for cand in (OUTPUT_DATA_DIR / name,
                 OUTPUT_DATA_DIR / f"{exp_name}_{model_type}" / name,
                 OUTPUT_DATA_DIR / "results_MVE" / "states" / name):
        if cand.exists():
            return cand
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default="topographic")
    p.add_argument("--model_type", type=str, default=StandardLSTM.MODEL_TYPE)  # "lstm"
    args = p.parse_args()

    out_dir = OUTPUT_DATA_DIR / "results_MVE"
    states_dir = out_dir / "states"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Canonical grid (stations order + dates order) ------------------------
    hidden_index = states_dir / "hidden_index.json"
    member0 = _find_member0(args.exp_name, args.model_type)
    if hidden_index.exists():
        idx = json.load(open(hidden_index))
        stations = list(idx["stations"])
        dates = pd.to_datetime(idx["dates"])
        grid_source = "states/hidden_index.json"
    elif member0 is not None:
        df0 = pd.read_csv(member0, index_col=0, parse_dates=True)
        stations = list(df0.columns)
        dates = df0.index
        grid_source = f"fallback: {member0.name}"
        print("⚠️  hidden_index.json absent — using member-0 CSV for the grid. "
              "Re-run after --save_hidden so the contract is keyed to the hidden states.")
    else:
        raise FileNotFoundError(
            "No grid source: neither states/hidden_index.json nor a member-0 "
            f"prediction CSV for {args.exp_name}_{args.model_type} was found.")

    # Sorted-order invariant: align_and_filter sorts the station intersection, so
    # member-CSV columns == sorted stations == q_std (basin_stds) order. Assert it,
    # because q_std is mapped to stations by position below.
    assert stations == sorted(stations), "station axis is not sorted — q_std mapping would misalign"
    n_dates, n_stations = len(dates), len(stations)
    print(f"Grid: {n_dates} dates x {n_stations} stations  [{grid_source}]")

    # --- 2. Observed target y on the grid (m3/s -> mm/day) -----------------------
    obs_raw = pd.read_csv(PROCESSED_DATA_DIR / "combined_streamflow.csv",
                          index_col=0, parse_dates=True)
    static = pd.read_csv(OUTPUT_STATIC_ATTR)
    static = static.set_index(static.columns[0])              # station_id is the first column
    area_col = "basin_area_km2" if "basin_area_km2" in static.columns else "area_km2"
    area = static[area_col]

    missing = [s for s in stations if s not in area.index]
    assert not missing, f"{len(missing)} stations missing basin area: {missing[:5]}"

    obs_mm = (obs_raw.reindex(index=dates, columns=stations)
              .mul(86.4 / area.reindex(stations), axis=1))      # mm/day, grid-aligned
    y = obs_mm.values.astype(np.float32)

    # valid = obs present AND prediction finite (preds are finite by construction;
    # AND-ing member-0 makes that explicit when the CSV is available)
    valid = np.isfinite(y)
    if member0 is not None:
        pred0 = (pd.read_csv(member0, index_col=0, parse_dates=True)
                 .reindex(index=dates, columns=stations).values)
        valid &= np.isfinite(pred0)

    # --- 3. Per-date split + event-held-out assertion (item 4) -------------------
    years = dates.year
    split = pd.Series([_split_for_year(int(y_)) for y_ in years], index=dates, name="split")
    for ev, (a, b) in EVENT_WINDOWS.items():
        win = split.loc[a:b]
        assert (win == "test").all(), \
            f"event {ev} ({a}..{b}) not fully in test split: {win.value_counts().to_dict()}"
    print("Split:", split.value_counts().reindex(["train", "val", "test", "none"]).dropna().to_dict(),
          "| events held out: OK")

    # --- 4. Labelled training q_std (item 3) -------------------------------------
    scaler_path = scaler_json_path(args.exp_name, args.model_type)
    if not scaler_path.exists():
        scaler_path = scaler_json_path(args.exp_name)
    basin_stds = load_scalers(scaler_path)["basin_stds"]
    assert len(basin_stds) == n_stations, \
        f"q_std length {len(basin_stds)} != {n_stations} stations"
    q_std = pd.DataFrame({"station_id": stations, "q_std": np.asarray(basin_stds, dtype=float)})

    # --- 5. Persist the contract -------------------------------------------------
    index_dates = pd.DataFrame({"date": dates, "year": years, "split": split.values})
    index_dates.to_csv(out_dir / "index_dates.csv", index=False)
    np.save(out_dir / "y_obs_mm.npy", y)
    np.save(out_dir / "valid_mask.npy", valid)
    q_std.to_csv(out_dir / "q_std.csv", index=False)

    # Re-emit axis labels on fallback so downstream has them even without hidden_index
    if not hidden_index.exists():
        states_dir.mkdir(parents=True, exist_ok=True)
        with open(states_dir / "hidden_index.json", "w") as f:
            json.dump({"stations": stations,
                       "dates": [d.strftime("%Y-%m-%d") for d in dates],
                       "shape": [n_dates, n_stations, 256],
                       "axes": ["date", "station", "hidden"],
                       "dtype": "float16", "hidden_size": 256,
                       "note": "axis labels only (emitted by build_index fallback; "
                               "hidden states not yet extracted)"}, f)

    meta = {
        "exp_name": args.exp_name,
        "model_type": args.model_type,
        "grid_source": grid_source,
        "grid_shape": {"n_dates": n_dates, "n_stations": n_stations},
        "n_members": 10,
        "target_space": "mm/day (specific runoff, m3/s * 86.4 / basin_area_km2); NO log transform",
        "residual": "r = y - mu_hat, with mu_hat = mean over the 10 member prediction CSVs",
        "head_standardization": "z = r / q_std (per basin); decompose un-standardizes by the same q_std",
        "q_std": {
            "source": f"{scaler_path.name} ['basin_stds']",
            "definition": "nanstd of training-period (1980-2005) runoff per basin, floored at 1.0",
            "units": "mm/day", "order": "aligned to the station axis (sorted)",
        },
        "sigma_e2": {
            "definition": "between-member variance, ddof=1 (M-1, M=10), over the member prediction CSVs",
            "grid": "same (date, station) grid as this contract",
        },
        "variance_accounting": "T = MSR - bias^2 = sigma_a^2 + sigma_e^2/M; "
                               "sigma_a^2 = T - sigma_e^2/M; total = sigma_a^2 + sigma_e^2",
        "split": {"train": [TRAIN_START_YEAR, TRAIN_END_YEAR],
                  "val": [VAL_START_YEAR, VAL_END_YEAR],
                  "test": [TEST_START_YEAR, TEST_END_YEAR]},
        "events_in_test": EVENT_WINDOWS,
        "valid_mask": "obs present (finite mm/day) AND prediction finite",
        "alignment": "all arrays positional arr[date_i, station_j]; labels in states/hidden_index.json",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    valid_frac = valid.mean()
    print(f"Wrote contract -> {out_dir}")
    print(f"  index_dates.csv  ({n_dates} rows)")
    print(f"  y_obs_mm.npy     {y.shape} float32")
    print(f"  valid_mask.npy   {valid.shape} bool  ({valid.sum():,} valid = {valid_frac:.1%})")
    print(f"  q_std.csv        ({n_stations} stations)")
    print(f"  meta.json")


if __name__ == "__main__":
    main()

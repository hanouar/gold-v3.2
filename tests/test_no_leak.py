"""Leakage tests.

Truncation test: features of session D computed on the FULL history must be
identical to features computed when the data stops exactly at the end of session D.
If any feature changes, it used information from the future.

Run:  python -m pytest tests -q
  or: python tests/test_no_leak.py --xauusd data/your.csv [--macro data/macro.csv] [session args]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ModelConfig  # noqa: E402
from data_pipeline import add_data_args, config_from_args, load_inputs  # noqa: E402
from features import build_daily_dataset, add_labels  # noqa: E402
from model_utils import feature_columns  # noqa: E402


def truncation_check(base, macro, cfg, n_checks=8, seed=0, verbose=True):
    full = build_daily_dataset(base, macro, cfg)
    feats = feature_columns(full)
    rng = np.random.default_rng(seed)
    candidates = full.index[260:-2]
    bad = {}
    for d in rng.choice(candidates, size=min(n_checks, len(candidates)), replace=False):
        d = pd.Timestamp(d)
        end = full.at[d, "session_end_utc"]
        cut = base[base.index < end]
        cut.attrs.update(base.attrs)
        part = build_daily_dataset(cut, macro, cfg)
        a, b = full.loc[d, feats].astype(float), part.loc[d, feats].astype(float)
        diff = ~np.isclose(a.values, b.values, rtol=1e-9, atol=1e-12, equal_nan=True)
        if diff.any():
            bad[str(d.date())] = list(np.array(feats)[diff])
        if verbose:
            print(f"{d.date()}  features={len(feats)}  changed={int(diff.sum())}")
    return bad


def synthetic_bars(days=420, minutes=15, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-02 22:00", periods=days * 24 * 60 // minutes, freq=f"{minutes}min", tz="UTC")
    local = idx.tz_convert("America/New_York")
    open_mask = ~((local.dayofweek == 5) | ((local.dayofweek == 4) & (local.hour >= 17)) |
                  ((local.dayofweek == 6) & (local.hour < 18)) | (local.hour == 17))
    idx = idx[open_mask]
    r = rng.normal(0, 0.0012, len(idx))
    close = 1800 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0005, len(idx))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0005, len(idx))))
    df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                       "volume": rng.integers(50, 500, len(idx)).astype(float)}, index=idx)
    df.index.name = "timestamp"
    df.attrs["bar_minutes"] = minutes
    return df


def test_truncation_synthetic():
    cfg = ModelConfig()
    base = synthetic_bars()
    bad = truncation_check(base, None, cfg, n_checks=4, verbose=False)
    assert not bad, f"Future information leaked into features: {bad}"


def test_no_sunday_sessions():
    cfg = ModelConfig()
    base = synthetic_bars()
    ds = build_daily_dataset(base, None, cfg)
    assert set(ds.index.dayofweek) <= {0, 1, 2, 3, 4}


def test_labels_are_next_session():
    cfg = ModelConfig()
    ds = add_labels(build_daily_dataset(synthetic_bars(), None, cfg), cfg=cfg)
    i = len(ds) // 2
    exp = float(ds["d1_high"].iloc[i + 1] > ds["d1_high"].iloc[i])
    assert ds["y_pdh_sweep"].iloc[i] == exp


def test_ny_plus_7_clock():
    from data_pipeline import _parse_ts
    utc = _parse_ts(pd.Series(["2026-03-16 00:00:00", "2026-01-05 00:00:00"]), "NY+7")
    ny = utc.dt.tz_convert("America/New_York")
    assert list(ny.dt.hour) == [17, 17]  # server midnight = 17:00 NY in EDT and EST


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_data_args(ap)
    ap.add_argument("--n-checks", type=int, default=8)
    args = ap.parse_args()
    cfg = config_from_args(args)
    base, macro = load_inputs(args, cfg)
    bad = truncation_check(base, macro, cfg, args.n_checks)
    if bad:
        print("\nLEAK DETECTED:")
        for k, v in bad.items():
            print(" ", k, v[:15])
        sys.exit(1)
    print("\nOK: no feature uses data from after its session end.")

"""Evaluate a saved multitask model on another dataset / period (e.g. another provider).

The session definition of the NEW file is given with the usual data args
(--data-tz, --session-tz, --session-start); features are those stored in the model.
Only sessions after the model's training end are scored unless --start is given.
"""
from __future__ import annotations

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from data_pipeline import add_data_args, config_from_args, load_inputs
from features import add_labels, build_daily_dataset
from multitask_models import CLASS_TARGETS, REG_TARGETS, evaluate_multitask

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ap = argparse.ArgumentParser()
    add_data_args(ap)
    ap.add_argument("--model", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    pack = joblib.load(args.model)
    cfg = config_from_args(args)
    base, macro = load_inputs(args, cfg)
    ds = add_labels(build_daily_dataset(base, macro, cfg), cfg=cfg)
    feats = pack["features"]
    missing = [f for f in feats if f not in ds.columns]
    if missing:
        raise SystemExit(f"Dataset lacks model features (need a finer base timeframe or macro?): {missing[:10]}")
    start = pd.Timestamp(args.start) if args.start else pd.Timestamp(pack["models"]["_meta"]["train_end"]) + pd.Timedelta(days=1)
    ds = ds[ds.index >= start]
    if args.end:
        ds = ds[ds.index <= args.end]
    ds = ds.dropna(subset=list(CLASS_TARGETS.values()) + list(REG_TARGETS.values()))
    # Rows need a full indicator warm-up (EMA200 on D1 etc.).
    ds = ds[ds[feats].notna().mean(axis=1) > 0.9]
    if len(ds) < 20:
        raise SystemExit(f"Only {len(ds)} scorable sessions after warm-up; need a longer file.")
    m = evaluate_multitask(pack["models"], ds, feats)
    print(json.dumps({"period": f"{ds.index.min().date()}..{ds.index.max().date()}", **m}, indent=2,
                     default=lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else x))


if __name__ == "__main__":
    main()

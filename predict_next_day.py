"""Secondary diagnostic: 3-class next-session direction (weak target, see RESEARCH_RESULTS)."""
from __future__ import annotations

import argparse
import json
import warnings

import joblib
import numpy as np

from config import ModelConfig
from data_pipeline import load_macro, load_ohlcv
from features import build_daily_dataset

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xauusd", required=True)
    ap.add_argument("--macro", default=None)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    pack = joblib.load(args.model)
    cfg = ModelConfig()
    for k, v in pack["config"].items():
        setattr(cfg, k, tuple(tuple(x) for x in v) if k == "intraday_timeframes" else v)
    base = load_ohlcv(args.xauusd, cfg)
    macro = load_macro(args.macro, cfg) if args.macro else None
    ds = build_daily_dataset(base, macro, cfg)
    if ds["n_bars"].iloc[-1] < 0.9 * ds["n_bars"].tail(40).median():
        ds = ds.iloc[:-1]  # last session incomplete
    row = ds.iloc[[-1]]

    feats, classes = pack["features"], pack["classes"]
    missing = [c for c in feats if c not in row.columns]
    if missing:
        raise ValueError(f"Current dataset is missing model features: {missing[:20]}")
    p = pack["pipeline"].predict_proba(row[feats])[0]
    probs = {int(c): float(v) for c, v in zip(classes, p)}
    labels = {-1: "BEARISH", 0: "RANGE", 1: "BULLISH"}
    print(json.dumps({
        "based_on_session": str(row.index[-1].date()),
        "prediction": labels[int(classes[int(np.argmax(p))])],
        "probabilities": {"bearish": probs.get(-1), "range": probs.get(0), "bullish": probs.get(1)},
        "model_train_end": pack.get("train_end"),
        "note": "Direction is near-random out of sample; prefer predict_multitask.py.",
    }, indent=2))


if __name__ == "__main__":
    main()

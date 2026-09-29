"""Predict the NEXT session from the last COMPLETED session.

The session definition (tz / start) stored in the model is reused, so live
features are computed exactly as in training. An unfinished session is never
used unless --allow-partial is given.
"""
from __future__ import annotations

import argparse
import json
import warnings

import joblib
import pandas as pd

from config import ModelConfig
from data_pipeline import load_macro, load_ohlcv
from features import build_daily_dataset
from multitask_models import predict_multitask

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xauusd", required=True)
    ap.add_argument("--macro", default=None)
    ap.add_argument("--macro-lag-hours", type=float, default=0.0)
    ap.add_argument("--model", default="models/xauusd_multitask.joblib")
    ap.add_argument("--asof", default=None, help="Predict from this session date instead of the latest")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    pack = joblib.load(args.model)
    cfg = ModelConfig()
    for k, v in pack["config"].items():
        if k == "intraday_timeframes":
            v = tuple(tuple(x) for x in v)
        setattr(cfg, k, v)
    if pack.get("uses_macro") and not args.macro:
        raise SystemExit("This model was trained with macro features: pass --macro.")

    base = load_ohlcv(args.xauusd, cfg)
    macro = load_macro(args.macro, cfg, args.macro_lag_hours) if args.macro else None
    ds = build_daily_dataset(base, macro, cfg)
    if args.asof:
        ds = ds[ds.index <= pd.Timestamp(args.asof)]

    warnings_out = []
    last = ds.iloc[-1]
    typical = ds["n_bars"].tail(40).median()
    session_over = pd.Timestamp.now(tz="UTC") >= last["session_end_utc"] or args.asof
    if (last["n_bars"] < 0.9 * typical) and not args.allow_partial:
        warnings_out.append(f"Session {ds.index[-1].date()} has {int(last['n_bars'])} bars vs typical "
                            f"{int(typical)}: treated as INCOMPLETE, using the previous session.")
        ds = ds.iloc[:-1]
    elif not session_over and not args.allow_partial:
        warnings_out.append("Latest session still in progress: using the previous completed session.")
        ds = ds.iloc[:-1]

    row = ds.iloc[[-1]]
    r = row.iloc[0]
    res = predict_multitask(pack["models"], row, pack["features"])
    ph, pl = res["pdh_sweep_probability"], res["pdl_sweep_probability"]
    if pl - ph >= 0.10:
        bias = "SELL_SIDE_LIQUIDITY_BIAS"
    elif ph - pl >= 0.10:
        bias = "BUY_SIDE_LIQUIDITY_BIAS"
    else:
        bias = "BALANCED_LIQUIDITY_RISK"

    a, c = float(r["d1_atr_abs"]), float(r["d1_close"])
    q = int(cfg.upper_quantile * 100)
    levels = {
        "session_high_PDH": float(r["d1_high"]),
        "session_low_PDL": float(r["d1_low"]),
        "session_close": c,
        "atr14": a,
        "median_high_projection": c + res["expected_up_excursion_atr"] * a,
        "median_low_projection": c - res["expected_down_excursion_atr"] * a,
        f"p{q}_high_projection": c + res[f"up_excursion_atr_p{q}"] * a,
        f"p{q}_low_projection": c - res[f"down_excursion_atr_p{q}"] * a,
    }
    print(json.dumps({
        "based_on_session": str(row.index[-1].date()),
        "session_end_utc": str(r["session_end_utc"]),
        "model_trained_through": pack["models"]["_meta"]["train_end"],
        **{k: round(v, 4) for k, v in res.items()},
        "first_liquidity_bias": bias,
        "price_levels": {k: round(v, 2) for k, v in levels.items()},
        "warnings": warnings_out,
        "note": ("Sweep probabilities are walk-forward calibrated. Excursion medians have ~no skill over a constant "
                 "in tests: use them as a volatility envelope, not a target."),
    }, indent=2))


if __name__ == "__main__":
    main()

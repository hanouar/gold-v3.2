"""Turn a FRED download (fetch_web_data.py fred) into a leak-aware macro CSV.

Output `timestamp` = time the value became KNOWN (observation date + lag), which is
what data_pipeline.merge_macro_asof() expects. Derived features (changes, z-scores)
are computed inside features.py, so this file only stores levels + spreads.

For revised releases (CPI/PCE/payrolls) use ALFRED vintages and a release calendar.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fred", required=True, help="CSV with observation_date (or timestamp) + series columns")
    ap.add_argument("--out", default="data/macro.csv")
    ap.add_argument("--lag-hours", type=float, default=36.0,
                    help="observation date -> available-at (FRED posts daily series next business day)")
    args = ap.parse_args()

    df = pd.read_csv(args.fred)
    dcol = "observation_date" if "observation_date" in df.columns else "timestamp"
    obs = pd.to_datetime(df.pop(dcol), utc=True).dt.normalize()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if {"DGS10", "DFII10"}.issubset(df.columns):
        df["breakeven10"] = df["DGS10"] - df["DFII10"]
    if {"DGS10", "DGS2"}.issubset(df.columns):
        df["curve_10s2s"] = df["DGS10"] - df["DGS2"]
    df.insert(0, "timestamp", obs + pd.Timedelta(hours=args.lag_hours))
    df = df.dropna(how="all", subset=[c for c in df.columns if c != "timestamp"])

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"saved {len(df):,} rows -> {p} (timestamp = observation date + {args.lag_hours:g}h)")


if __name__ == "__main__":
    main()

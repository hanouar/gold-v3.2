"""Convert the public_data_bootstrap.py downloads into files this pipeline reads.

Outputs (in data/):
  xauusd_h4_2013_2026.csv      one provider, H4, bar OPEN time, fixed 21:00 day start
                               -> use: --session-tz UTC --session-start 21:00
  xauusd_m15_mt5_2025_2026.csv MT5 broker M15, server time = New York + 7h
                               -> use: --data-tz NY+7
  xauusd_m15_utc_2026.csv      M15, UTC                      -> default session args
  macro.csv                    DGS10, DGS2, VIX, 10s2s with available-at timestamps
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/public_bootstrap")
    ap.add_argument("--out-dir", default="data")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out_dir)

    h4 = pd.read_csv(src / "gold_h4_long.csv")
    h4 = h4[h4["Time"] >= "2013-10-01"]  # before mid-2013 the file only holds daily bars
    h4 = h4.rename(columns={"Time": "timestamp"}).rename(columns=str.lower)[["timestamp", "open", "high", "low", "close"]]
    h4.to_csv(out / "xauusd_h4_2013_2026.csv", index=False)

    mt5 = pd.read_csv(src / "xau_m15_mt5.csv").rename(columns={"datetime": "timestamp"}).drop(columns=["spread"], errors="ignore")
    mt5.to_csv(out / "xauusd_m15_mt5_2025_2026.csv", index=False)

    utc = pd.read_csv(src / "gold_m15_recent_sample.csv").rename(columns={"datetime": "timestamp"})
    utc.to_csv(out / "xauusd_m15_utc_2026.csv", index=False)

    m = None
    for f, c in (("dgs10", "DGS10"), ("dgs2", "DGS2"), ("vix", "VIXCLS")):
        x = pd.read_csv(src / f"{f}.csv")
        x.columns = ["observation_date", c]
        x = x.set_index("observation_date")
        m = x if m is None else m.join(x, how="outer")
    m[m.index >= "2010"].reset_index().to_csv(out / "fred_sample.csv", index=False)
    print("Wrote H4 / M15 files and fred_sample.csv. Now run:\n"
          "  python build_macro_features.py --fred data/fred_sample.csv --out data/macro.csv")


if __name__ == "__main__":
    main()

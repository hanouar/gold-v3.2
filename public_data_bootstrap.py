"""Bootstrap a research dataset from public web sources.

This deliberately separates provider-quality data from development fallbacks.
Production research should prefer original providers (Dukascopy/Exness/FRED/CFTC)
and respect each source's terms. GitHub mirrors are convenience fallbacks.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import requests
import pandas as pd

URLS = {
    "gold_daily_long": "https://raw.githubusercontent.com/tohaitrieu/market-history/master/data/commodities/GOLD/GOLD-D1.csv",
    "gold_h4_long": "https://raw.githubusercontent.com/tohaitrieu/market-history/master/data/commodities/GOLD/GOLD-H4.csv",
    "xau_m15_mt5": "https://raw.githubusercontent.com/Gianniskas/mt5-m1-bars-us500-eurusd-xauusd/main/XAUUSD_m15.csv",
    "gold_daily_recent_broker": "https://raw.githubusercontent.com/Gianniskas/mt5-m1-bars-us500-eurusd-xauusd/main/XAUUSD_d1.csv",
    "gold_m15_recent_sample": "https://raw.githubusercontent.com/getdata-finance/xauusd-15m-ohlcv-metals-historical-data/main/XAUUSD_15m.csv",
    "dgs10": "https://raw.githubusercontent.com/TGRADEA/gradea-fred-archive/main/DGS10.csv",
    "dgs2": "https://raw.githubusercontent.com/TGRADEA/gradea-fred-archive/main/DGS2.csv",
    "vix": "https://raw.githubusercontent.com/chaltik/market_data/master/VIXCLS.csv",
    "weekly_context": "https://raw.githubusercontent.com/thayroncarlessi/bluequant-regime-lab/main/data/market_weekly.csv",
}


def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)
    print(f"{path.name}: {len(r.content):,} bytes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/public_bootstrap")
    args = ap.parse_args()
    out = Path(args.out_dir)
    for name, url in URLS.items():
        download(url, out / f"{name}.csv")

    print("\nBootstrap complete. Validate provenance and source terms before production use.")
    print("Next: python prepare_public_data.py")

if __name__ == "__main__":
    main()

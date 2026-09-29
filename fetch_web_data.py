"""Download public/low-cost web data for the XAUUSD research model.

This script is intended to be run on your own PC with internet access.
Environment variables:
  FRED_API_KEY         optional but recommended for FRED/ALFRED
  TWELVE_DATA_API_KEY  optional for XAUUSD API download
  EODHD_API_KEY        optional for economic calendar actual/estimate/previous
  CFTC_APP_TOKEN       optional; public calls work with lower limits

Examples:
  python fetch_web_data.py fred --out data/fred_daily.csv
  python fetch_web_data.py cftc --out data/cftc_gold.csv
  python fetch_web_data.py twelve --symbol "XAU/USD" --interval 15min --start 2020-01-01 --out data/xauusd_m15.csv
  python fetch_web_data.py eodhd-calendar --start 2020-01-01 --end 2026-09-29 --country US --out data/us_calendar.csv
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
import requests
import pandas as pd


def _save(df: pd.DataFrame, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"saved {len(df):,} rows -> {p}")


def fetch_fred(series_ids: list[str], api_key: str | None) -> pd.DataFrame:
    """Fetch latest-vintage daily observations from FRED.

    For revised macro series used in backtests, use a separate ALFRED/vintage process.
    This function is best for market series such as DGS10, DGS2, DFII10, VIXCLS.
    """
    if not api_key:
        raise SystemExit("Set FRED_API_KEY. A free key is available from FRED.")
    frames = []
    for sid in series_ids:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": sid,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": "1990-01-01",
        }
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        obs = r.json()["observations"]
        df = pd.DataFrame(obs)[["date", "value"]].rename(columns={"value": sid})
        df[sid] = pd.to_numeric(df[sid], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], utc=True)
        frames.append(df.set_index("date"))
        time.sleep(0.15)
    # Keep the OBSERVATION date: it is not the time the value became known.
    # load_macro()/build_macro_features.py convert it to an available-at timestamp.
    out = pd.concat(frames, axis=1).sort_index().reset_index().rename(columns={"date": "observation_date"})
    out["observation_date"] = out["observation_date"].dt.date
    return out


def fetch_cftc_gold(app_token: str | None = None) -> pd.DataFrame:
    """Fetch CFTC Disaggregated Futures Only rows for gold.

    Dataset ID 72hh-3qpy is the CFTC Disaggregated Futures Only dataset.
    The publication lag MUST be handled separately when creating model features.
    """
    base = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
    params = {
        "$limit": 50000,
        "$where": "commodity_name='GOLD'",
        "$order": "report_date_as_yyyy_mm_dd asc",
    }
    headers = {"X-App-Token": app_token} if app_token else {}
    r = requests.get(base, params=params, headers=headers, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if "report_date_as_yyyy_mm_dd" in df:
        df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], utc=True)
    return df


def fetch_twelve(symbol: str, interval: str, start: str, end: str | None, api_key: str) -> pd.DataFrame:
    """Fetch Twelve Data time series in chunks of <= 5000 records.

    For very long M1 history, prefer Dukascopy/HistData and use Twelve Data for convenience/recent updates.
    """
    if not api_key:
        raise SystemExit("Set TWELVE_DATA_API_KEY")
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "start_date": start,
        "apikey": api_key,
        "format": "JSON",
        "order": "ASC",
        "timezone": "UTC",
    }
    if end:
        params["end_date"] = end
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    js = r.json()
    if js.get("status") == "error":
        raise RuntimeError(js)
    vals = js.get("values", [])
    df = pd.DataFrame(vals)
    if df.empty:
        return df
    df = df.rename(columns={"datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df:
        df["volume"] = 0.0
    return df.sort_values("timestamp")


def fetch_eodhd_calendar(start: str, end: str, country: str, api_key: str) -> pd.DataFrame:
    if not api_key:
        raise SystemExit("Set EODHD_API_KEY")
    url = "https://eodhd.com/api/economic-events"
    params = {
        "api_token": api_key,
        "fmt": "json",
        "from": start,
        "to": end,
        "country": country,
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if "date" in df:
        df["timestamp"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fred")
    p.add_argument("--out", default="data/fred_daily.csv")
    p.add_argument("--series", nargs="*", default=[
        "DGS10", "DGS2", "DFII10", "VIXCLS", "SP500", "NASDAQCOM",
        "DCOILWTICO", "DCOILBRENTEU", "DTWEXBGS"
    ])

    p = sub.add_parser("cftc")
    p.add_argument("--out", default="data/cftc_gold.csv")

    p = sub.add_parser("twelve")
    p.add_argument("--symbol", default="XAU/USD")
    p.add_argument("--interval", default="15min")
    p.add_argument("--start", required=True)
    p.add_argument("--end", default=None)
    p.add_argument("--out", default="data/xauusd_twelve.csv")

    p = sub.add_parser("eodhd-calendar")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--country", default="US")
    p.add_argument("--out", default="data/us_calendar.csv")

    args = ap.parse_args()
    if args.cmd == "fred":
        df = fetch_fred(args.series, os.getenv("FRED_API_KEY"))
    elif args.cmd == "cftc":
        df = fetch_cftc_gold(os.getenv("CFTC_APP_TOKEN"))
    elif args.cmd == "twelve":
        df = fetch_twelve(args.symbol, args.interval, args.start, args.end, os.getenv("TWELVE_DATA_API_KEY", ""))
    elif args.cmd == "eodhd-calendar":
        df = fetch_eodhd_calendar(args.start, args.end, args.country, os.getenv("EODHD_API_KEY", ""))
    else:
        raise SystemExit(2)
    _save(df, args.out)

if __name__ == "__main__":
    main()

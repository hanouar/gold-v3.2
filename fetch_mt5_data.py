"""Fetch recent XAUUSD M15 candles from a running, logged-in MetaTrader 5 terminal.

Writes a clean file in the format the pipeline expects (UTC, bar OPEN time):
    timestamp,open,high,low,close,volume

MT5 returns bar times in broker SERVER time. Most gold brokers (FTMO, Exness, IC...)
run a "New York + 7h" clock (UTC+3 in US summer, UTC+2 in winter); that is the default
and is converted with the project's own data_pipeline._parse_ts. The script checks the
daily market break: it must reopen at 18:00 New York time, otherwise --server-tz is wrong.

Usage:
  python fetch_mt5_data.py
  python fetch_mt5_data.py --symbol XAUUSD --days 400 --out data/xauusd_mt5_m15_current.csv
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from data_pipeline import _parse_ts

CANDIDATES = ("XAUUSD", "GOLD")


def find_symbol(mt5, wanted: str | None) -> str:
    names = [s.name for s in mt5.symbols_get()]
    if wanted:
        if wanted not in names:
            raise SystemExit(f"Symbol {wanted!r} not found in this MT5 terminal.")
        return wanted
    for base in CANDIDATES:  # exact name first, then broker suffixes (XAUUSDm, XAUUSD.pro, GOLD. ...)
        if base in names:
            return base
        hits = sorted(n for n in names if n.upper().startswith(base) and not n.upper().startswith(base + "EUR"))
        if hits:
            return hits[0]
    raise SystemExit("No XAUUSD/GOLD symbol found. Pass --symbol with your broker's gold symbol.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, help="Broker symbol (auto-detected if omitted)")
    ap.add_argument("--days", type=int, default=400, help="History to fetch; >= 1 year keeps feature warm-up safe")
    ap.add_argument("--server-tz", default="NY+7", help="Broker server clock: NY+7 (default) or an IANA zone")
    ap.add_argument("--terminal", default=None, help="Path to terminal64.exe if auto-connect fails")
    ap.add_argument("--out", default="data/xauusd_mt5_m15_current.csv")
    ap.add_argument("--raw", default=None, help="Optional: also save the raw MT5 rates here")
    args = ap.parse_args()

    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise SystemExit("MetaTrader5 package missing: pip install MetaTrader5 (Windows only)")

    ok = mt5.initialize(path=args.terminal) if args.terminal else mt5.initialize()
    if not ok:
        raise SystemExit(f"MT5 connection failed: {mt5.last_error()}. Open MT5, log in, and retry "
                         "(or pass --terminal \"C:\\...\\terminal64.exe\").")
    try:
        acc = mt5.account_info()
        sym = find_symbol(mt5, args.symbol)
        mt5.symbol_select(sym, True)
        end = datetime.now(timezone.utc) + timedelta(days=2)  # MT5 compares against server-clock epochs
        rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M15, end - timedelta(days=args.days + 2), end)
        if rates is None or len(rates) == 0:
            raise SystemExit(f"No M15 data returned for {sym}: {mt5.last_error()}")
        tick = mt5.symbol_info_tick(sym)
    finally:
        mt5.shutdown()

    raw = pd.DataFrame(rates)
    if args.raw:
        raw.to_csv(args.raw, index=False)

    # Drop the candle that is still forming (its 15 minutes are not over on the server clock).
    forming = raw["time"] + 15 * 60 > tick.time
    raw = raw[~forming]

    srv = pd.to_datetime(raw["time"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame({
        "timestamp": _parse_ts(srv, args.server_tz).reset_index(drop=True),
        "open": raw["open"].values, "high": raw["high"].values,
        "low": raw["low"].values, "close": raw["close"].values,
        "volume": raw["tick_volume"].astype(float).values,
    })
    n0 = len(df)
    df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    px = df[["open", "high", "low", "close"]]
    bad = (px <= 0).any(axis=1) | (df["high"] < px.max(axis=1)) | (df["low"] > px.min(axis=1))
    df = df[~bad]

    # Time-zone check: the daily break must end at 18:00 New York time.
    ts = df["timestamp"].reset_index(drop=True)
    reopen = ts[ts.diff().between(pd.Timedelta("45min"), pd.Timedelta("3h"))]
    ny_hour = reopen.dt.tz_convert("America/New_York").dt.hour.mode()
    tz_ok = len(ny_hour) > 0 and int(ny_hour.iloc[0]) == 18

    # Windows clock check (predict_multitask.py uses the PC clock to decide if a session is over).
    server_now = _parse_ts(pd.Series([pd.to_datetime(tick.time, unit="s").strftime("%Y-%m-%d %H:%M:%S")]),
                           args.server_tz).iloc[0]
    pc_now = pd.Timestamp(time.time(), unit="s", tz="UTC")
    clock_drift_min = (pc_now - server_now).total_seconds() / 60

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    out.to_csv(args.out, index=False)

    last = df["timestamp"].iloc[-1]
    print(f"Broker:                 {acc.company if acc else '?'} ({acc.server if acc else '?'})")
    print(f"Symbol:                 {sym}")
    print(f"Server clock:           {args.server_tz} -> converted to UTC")
    print(f"First candle (UTC):     {df['timestamp'].iloc[0]}")
    print(f"Latest completed candle (UTC open time): {last}")
    print(f"Rows written:           {len(out):,}  (removed: {int(forming.sum())} forming, "
          f"{n0 - len(df) - int(bad.sum())} dup/invalid time, {int(bad.sum())} bad OHLC)")
    print(f"Output file:            {args.out}")
    if not tz_ok:
        print(f"WARNING: daily break reopens at NY hour {ny_hour.tolist()}, expected 18. "
              "Check --server-tz before using this file.")
    if clock_drift_min < -5:  # PC behind the last broker tick: the PC clock is wrong
        print(f"WARNING: Windows clock is ~{-clock_drift_min:.0f} min BEHIND broker time. "
              "Fix the Windows time/time zone setting (see instructions.txt).")
    elif clock_drift_min > 30:
        print(f"Note: last tick is {clock_drift_min:.0f} min old (market closed, or Windows clock ahead).")
    if pd.Timestamp.now(tz="America/New_York").dst() == timedelta(0):
        print("WARNING: US daylight saving time is OFF. The model's sessions start at 21:00 UTC but the market "
              "day now starts at 22:00 UTC. See instructions.txt, section 9.")


if __name__ == "__main__":
    main()

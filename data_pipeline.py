"""Loading, time-zone handling and trading-session aggregation.

Internal convention (everything downstream relies on it):
  * intraday bars are indexed by their OPEN time, tz-aware UTC;
  * each bar is assigned to a trading session ("trading_date") using a session
    clock (e.g. 17:00 New York -> 17:00 New York);
  * a daily row only ever contains information from bars of that session or
    earlier sessions.
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd

from config import ModelConfig

OHLC = ["open", "high", "low", "close"]
NY_PLUS_7 = {"NY+7", "BROKER_NY_PLUS_7"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _parse_ts(s: pd.Series, data_tz: str) -> pd.Series:
    try:
        ts = pd.to_datetime(s, errors="coerce", utc=False, format="mixed")
    except (ValueError, TypeError):  # mixed UTC offsets (e.g. +02:00 / +03:00 across DST)
        ts = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    if ts.dtype == object:
        ts = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    if getattr(ts.dt, "tz", None) is None:
        if data_tz.upper() in NY_PLUS_7:
            # NY-close broker clock (most MT4/MT5 gold servers): wall clock = New York + 7h,
            # following US DST (differs from Europe/Athens ~4 weeks a year).
            ts = (ts - pd.Timedelta(hours=7)).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
        else:
            ts = ts.dt.tz_localize(data_tz, ambiguous="NaT", nonexistent="NaT")
    return ts.dt.tz_convert("UTC")


def infer_bar_minutes(idx: pd.DatetimeIndex) -> int:
    d = pd.Series(idx).diff().dropna()
    d = d[d > pd.Timedelta(0)]
    if d.empty:
        raise ValueError("Cannot infer bar size from a single row")
    return int(d.mode().iloc[0] / pd.Timedelta(minutes=1))


def load_ohlcv(path: str, cfg: ModelConfig | None = None) -> pd.DataFrame:
    """Load OHLCV and return bars indexed by OPEN time (UTC)."""
    cfg = cfg or ModelConfig()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = cfg.timestamp_col.lower()
    if tcol not in df.columns:
        for alt in ("datetime", "time", "date", "gmt time", "local time"):
            if alt in df.columns:
                tcol = alt
                break
    required = {tcol, "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = _parse_ts(df[tcol], cfg.data_tz)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    for c in OHLC + ["volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=OHLC)[OHLC + ["volume"]]

    bar_min = infer_bar_minutes(df.index)
    if cfg.ts_convention == "close":
        df.index = df.index - pd.Timedelta(minutes=bar_min)
    elif cfg.ts_convention != "open":
        raise ValueError("ts_convention must be 'open' or 'close'")
    df.attrs["bar_minutes"] = bar_min
    return df


def load_macro(path: Optional[str], cfg: ModelConfig | None = None, lag_hours: float = 0.0) -> Optional[pd.DataFrame]:
    """Macro CSV: `timestamp` must be the moment the value became KNOWN (available_at).

    FRED-style files (observation_date,...) are accepted: a daily FRED value for date D
    is published on FRED the next business day, so it is stamped D + 36h (+ lag_hours).
    """
    if not path:
        return None
    cfg = cfg or ModelConfig()
    m = pd.read_csv(path)
    if "observation_date" in m.columns and "timestamp" not in m.columns:
        m["timestamp"] = pd.to_datetime(m.pop("observation_date")).dt.tz_localize("UTC") + pd.Timedelta(hours=36)
    if "timestamp" not in m.columns:
        raise ValueError("Macro CSV must contain 'timestamp' (available-at time)")
    m["timestamp"] = _parse_ts(m["timestamp"], cfg.data_tz) + pd.Timedelta(hours=lag_hours)
    m = m.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    m = m.set_index("timestamp")
    for c in m.columns:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    return m.dropna(how="all")


# --------------------------------------------------------------------------- #
# Session clock
# --------------------------------------------------------------------------- #
def _session_shift(session_start: str) -> pd.Timedelta:
    h, m = (int(x) for x in session_start.split(":"))
    start = pd.Timedelta(hours=h, minutes=m)
    # A session starting in the evening belongs to the NEXT calendar date.
    return pd.Timedelta(hours=24) - start if start >= pd.Timedelta(hours=12) else -start


def session_clock(idx_utc: pd.DatetimeIndex, cfg: ModelConfig) -> pd.DatetimeIndex:
    """Map UTC open times to a naive clock in which every session starts at 00:00."""
    local = idx_utc.tz_convert(cfg.session_tz).tz_localize(None)
    return local + _session_shift(cfg.session_start)


def trading_date_from_clock(clock: pd.DatetimeIndex) -> pd.DatetimeIndex:
    d = clock.normalize()
    wd = d.dayofweek
    # Weekend stubs (e.g. Sunday-evening open with a midnight session) -> Monday.
    d = d + pd.to_timedelta(np.where(wd == 5, 2, np.where(wd == 6, 1, 0)), unit="D")
    return pd.DatetimeIndex(d, name="date")


def assign_sessions(base: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    out = base.copy()
    clock = session_clock(out.index, cfg)
    out["session_clock"] = clock
    out["trading_date"] = trading_date_from_clock(clock)
    return out


def daily_bars(base: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """One row per trading session, indexed by trading_date."""
    b = assign_sessions(base, cfg)
    bar = pd.Timedelta(minutes=base.attrs.get("bar_minutes", infer_bar_minutes(base.index)))
    b["open_utc"] = b.index
    g = b.groupby("trading_date", sort=True)
    d = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
              close=("close", "last"), volume=("volume", "sum"),
              n_bars=("close", "size"), session_first_utc=("open_utc", "first"),
              session_last_utc=("open_utc", "last"))
    d["session_end_utc"] = d["session_last_utc"] + bar
    med = d["n_bars"].rolling(60, min_periods=1).median()
    d["session_frac"] = (d["n_bars"] / med).clip(upper=1.5)
    d.index.name = "date"
    return d


def intraday_bars(base: pd.DataFrame, rule: str, cfg: ModelConfig) -> Optional[pd.DataFrame]:
    """Resample on the session clock so higher-TF bars never straddle two sessions."""
    base_min = base.attrs.get("bar_minutes", infer_bar_minutes(base.index))
    if pd.Timedelta(rule) < pd.Timedelta(minutes=base_min):
        return None
    b = base.copy()
    b.index = session_clock(b.index, cfg)
    b = b[~b.index.duplicated(keep="last")]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = b.resample(rule, label="left", closed="left", origin="start_day").agg(agg).dropna(subset=OHLC)
    out["trading_date"] = trading_date_from_clock(out.index)
    return out


# --------------------------------------------------------------------------- #
# Macro
# --------------------------------------------------------------------------- #
def merge_macro_asof(ds: pd.DataFrame, macro: Optional[pd.DataFrame], cfg: ModelConfig) -> pd.DataFrame:
    """Attach the latest macro value KNOWN at each session end (never later)."""
    if macro is None or macro.empty:
        return ds
    left = ds[["session_end_utc"]].reset_index().sort_values("session_end_utc")
    cols = {}
    for c in macro.columns:  # column by column: each series has its own last valid print
        s = macro[c].dropna()
        if s.empty:
            continue
        right = pd.DataFrame({"macro_ts": s.index, c: s.values}).sort_values("macro_ts")
        m = pd.merge_asof(left, right, left_on="session_end_utc", right_on="macro_ts", direction="backward",
                          tolerance=pd.Timedelta(days=cfg.macro_max_age_days))
        cols[f"mac_{c}"] = pd.Series(m[c].values, index=m["date"].values)
    return ds.join(pd.DataFrame(cols).rename_axis("date"))


# --------------------------------------------------------------------------- #
# Shared CLI helpers
# --------------------------------------------------------------------------- #
def add_data_args(ap: argparse.ArgumentParser) -> None:
    d = ModelConfig()
    ap.add_argument("--xauusd", required=True, help="XAUUSD OHLCV CSV (M1..H4)")
    ap.add_argument("--macro", default=None, help="Macro CSV with timestamp = available-at time")
    ap.add_argument("--macro-lag-hours", type=float, default=0.0)
    ap.add_argument("--ts-convention", choices=["open", "close"], default=d.ts_convention)
    ap.add_argument("--data-tz", default=d.data_tz, help="TZ of naive CSV timestamps: UTC, an IANA zone, or NY+7 for NY-close MT4/MT5 servers")
    ap.add_argument("--session-tz", default=d.session_tz)
    ap.add_argument("--session-start", default=d.session_start, help="HH:MM in session-tz")


def config_from_args(args) -> ModelConfig:
    cfg = ModelConfig()
    for k in ("ts_convention", "data_tz", "session_tz", "session_start"):
        if hasattr(args, k):
            setattr(cfg, k, getattr(args, k))
    for k in ("classifier",):
        if getattr(args, k, None):
            setattr(cfg, k, getattr(args, k))
    return cfg


def load_inputs(args, cfg: ModelConfig):
    base = load_ohlcv(args.xauusd, cfg)
    macro = load_macro(args.macro, cfg, getattr(args, "macro_lag_hours", 0.0))
    return base, macro

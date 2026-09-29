"""Leak-free daily dataset: one row per trading session, features known at session end.

Every feature is scale-free (returns, ATR units, ratios) so a model trained with gold
at $1,200 still makes sense with gold at $4,000.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ModelConfig
from data_pipeline import daily_bars, intraday_bars, merge_macro_asof


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).rolling(n, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def _safe(x: pd.Series) -> pd.Series:
    return x.replace(0, np.nan)


def bar_features(df: pd.DataFrame, p: str, atr_n: int = 14) -> pd.DataFrame:
    """Scale-free candle/trend features for any timeframe."""
    out = pd.DataFrame(index=df.index)
    a = _safe(atr(df, atr_n))
    rng = _safe(df["high"] - df["low"])
    c = df["close"]
    out[f"{p}ret1"] = c.pct_change()
    out[f"{p}ret3"] = c.pct_change(3)
    out[f"{p}ret5"] = c.pct_change(5)
    out[f"{p}ret1_atr"] = c.diff() / a
    out[f"{p}range_atr"] = (df["high"] - df["low"]) / a
    out[f"{p}body_atr"] = (c - df["open"]) / a
    out[f"{p}upper_wick_atr"] = (df["high"] - df[["open", "close"]].max(axis=1)) / a
    out[f"{p}lower_wick_atr"] = (df[["open", "close"]].min(axis=1) - df["low"]) / a
    out[f"{p}clv"] = (c - df["low"]) / rng                     # 0 = closed on low, 1 = on high
    out[f"{p}close_to_high_atr"] = (df["high"] - c) / a
    out[f"{p}close_to_low_atr"] = (c - df["low"]) / a
    out[f"{p}atr_pct"] = a / c
    out[f"{p}atr_ratio_5_20"] = atr(df, 5) / _safe(atr(df, 20))
    out[f"{p}rsi14"] = rsi(c, 14)
    for n in (20, 50, 100, 200):
        out[f"{p}ema{n}_dist_atr"] = (c - ema(c, n)) / a
    out[f"{p}hh20_dist_atr"] = (c - df["high"].rolling(20).max().shift(1)) / a
    out[f"{p}ll20_dist_atr"] = (c - df["low"].rolling(20).min().shift(1)) / a
    if df["volume"].fillna(0).abs().sum() > 0:
        v = df["volume"]
        out[f"{p}volume_z20"] = (v - v.rolling(20).mean()) / _safe(v.rolling(20).std())
    return out


# --------------------------------------------------------------------------- #
# Daily dataset
# --------------------------------------------------------------------------- #
RAW_COLS = ["d1_open", "d1_high", "d1_low", "d1_close", "d1_atr_abs",
            "prev_day_high", "prev_day_low", "prev_day_close"]
META_COLS = ["n_bars", "session_first_utc", "session_last_utc", "session_end_utc"]


def daily_features(d: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    f = bar_features(d, "d1_", cfg.atr_window)
    a = _safe(atr(d, cfg.atr_window))
    ph, pl, pc = d["high"].shift(1), d["low"].shift(1), d["close"].shift(1)

    f["dist_pdh_atr"] = (d["close"] - ph) / a
    f["dist_pdl_atr"] = (d["close"] - pl) / a
    f["gap_from_pdc_atr"] = (d["open"] - pc) / a
    rng = d["high"] - d["low"]
    adr20 = rng.rolling(20).mean()
    f["adr20_pct"] = adr20 / d["close"]
    f["adr_usage"] = rng / _safe(adr20)

    # Liquidity behaviour of the session that just closed.
    swept_h = (d["high"] > ph).astype(float).where(ph.notna())
    swept_l = (d["low"] < pl).astype(float).where(pl.notna())
    f["swept_pdh_today"] = swept_h
    f["swept_pdl_today"] = swept_l
    f["rejected_pdh_today"] = ((d["high"] > ph) & (d["close"] < ph)).astype(float).where(ph.notna())
    f["rejected_pdl_today"] = ((d["low"] < pl) & (d["close"] > pl)).astype(float).where(pl.notna())
    f["inside_day"] = ((d["high"] <= ph) & (d["low"] >= pl)).astype(float).where(ph.notna())
    f["outside_day"] = ((d["high"] > ph) & (d["low"] < pl)).astype(float).where(ph.notna())
    for n in (5, 20):
        f[f"pdh_sweep_rate{n}"] = swept_h.rolling(n).mean()
        f[f"pdl_sweep_rate{n}"] = swept_l.rolling(n).mean()
    f["break_prev5_high"] = (d["close"] > d["high"].rolling(5).max().shift(1)).astype(float)
    f["break_prev5_low"] = (d["close"] < d["low"].rolling(5).min().shift(1)).astype(float)

    # Calendar: weekday of the next business day (known in advance, no future data).
    f["next_weekday"] = pd.Series((d.index + pd.offsets.BDay(1)).dayofweek, index=d.index).astype(float)
    gap = (pd.Series(d.index, index=d.index).shift(-1) - pd.Series(d.index, index=d.index)).dt.days
    f["session_frac"] = d["session_frac"]

    # Raw levels (NOT model features; used for labels and price projections).
    f["d1_open"], f["d1_high"], f["d1_low"], f["d1_close"] = d["open"], d["high"], d["low"], d["close"]
    f["d1_atr_abs"] = a
    f["prev_day_high"], f["prev_day_low"], f["prev_day_close"] = ph, pl, pc
    for c in META_COLS:
        f[c] = d[c]
    f["_gap_to_next_days"] = gap
    return f


def build_daily_dataset(base: pd.DataFrame, macro: pd.DataFrame | None = None,
                        cfg: ModelConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ModelConfig()
    d = daily_bars(base, cfg)
    out = daily_features(d, cfg)

    for rule, prefix in cfg.intraday_timeframes:
        tf = intraday_bars(base, rule, cfg)
        if tf is None or len(tf) < 50:
            continue
        f = bar_features(tf, prefix)
        # Last completed bar of each session: it closes at (or before) the session end.
        f["trading_date"] = tf["trading_date"]
        last = f.groupby("trading_date").last()
        last.index.name = "date"
        out = out.join(last, how="left")

    if macro is not None:
        out = merge_macro_asof(out, macro, cfg)
        mcols = [c for c in out.columns if c.startswith("mac_")]
        for c in mcols:
            s = out[c]
            out[f"{c}_chg1"] = s.diff()
            out[f"{c}_chg5"] = s.diff(5)
            out[f"{c}_z60"] = (s - s.rolling(60).mean()) / _safe(s.rolling(60).std())

    return out.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
LABEL_COLS = ["y_direction", "y_next_ret_atr", "y_pdh_sweep", "y_pdl_sweep",
              "y_up_excursion_atr", "y_down_excursion_atr",
              "y_pdh_reject", "y_pdl_reject", "y_first_sweep_high"]


def add_labels(ds: pd.DataFrame, threshold_atr: float = 0.20, cfg: ModelConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ModelConfig()
    out = ds.copy()
    nc, nh, nl = out["d1_close"].shift(-1), out["d1_high"].shift(-1), out["d1_low"].shift(-1)
    a = _safe(out["d1_atr_abs"])
    valid = out["_gap_to_next_days"].le(cfg.max_label_gap_days) & a.notna()

    r = (nc - out["d1_close"]) / a
    out["y_next_ret_atr"] = r
    out["y_direction"] = np.select([r <= -threshold_atr, r >= threshold_atr], [-1, 1], default=0).astype(float)
    out["y_pdh_sweep"] = (nh > out["d1_high"]).astype(float)
    out["y_pdl_sweep"] = (nl < out["d1_low"]).astype(float)
    out["y_up_excursion_atr"] = ((nh - out["d1_close"]) / a).clip(0, cfg.excursion_clip_atr)
    out["y_down_excursion_atr"] = ((out["d1_close"] - nl) / a).clip(0, cfg.excursion_clip_atr)
    # Research labels: sweep then close back inside (rejection) vs acceptance.
    out["y_pdh_reject"] = ((nh > out["d1_high"]) & (nc < out["d1_high"])).astype(float)
    out["y_pdl_reject"] = ((nl < out["d1_low"]) & (nc > out["d1_low"])).astype(float)
    out["y_first_sweep_high"] = np.nan  # needs intraday order; filled by add_sweep_order_labels()

    out.loc[~valid, LABEL_COLS] = np.nan
    return out


def add_sweep_order_labels(ds: pd.DataFrame, base: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """For days that sweep BOTH sides, which one was taken first (1 = high first)."""
    from data_pipeline import assign_sessions
    b = assign_sessions(base, cfg)
    out = ds.copy()
    first = {}
    dates = out.index
    nxt = pd.Series(dates[1:], index=dates[:-1])
    grp = dict(tuple(b.groupby("trading_date")))
    for d, nd in nxt.items():
        g = grp.get(nd)
        if g is None:
            continue
        hi, lo = out.at[d, "d1_high"], out.at[d, "d1_low"]
        th = g.index[g["high"] > hi]
        tl = g.index[g["low"] < lo]
        if len(th) and len(tl):
            first[d] = 1.0 if th[0] < tl[0] else (0.0 if tl[0] < th[0] else np.nan)
    out["y_first_sweep_high"] = pd.Series(first)
    out.loc[out["y_pdh_sweep"].isna(), "y_first_sweep_high"] = np.nan
    return out

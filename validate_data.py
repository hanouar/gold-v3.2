"""Data-quality report. Run this before anything else on a new data source."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data_pipeline import add_data_args, config_from_args, daily_bars, load_ohlcv


def main():
    ap = argparse.ArgumentParser()
    add_data_args(ap)
    args = ap.parse_args()
    cfg = config_from_args(args)

    df = load_ohlcv(args.xauusd, cfg)
    bar = df.attrs["bar_minutes"]
    issues, warns = [], []
    print(f"Rows: {len(df):,}   bar size: {bar} min   (timestamps treated as bar {cfg.ts_convention} time, "
          f"naive tz = {cfg.data_tz})")
    print(f"Start: {df.index.min()}   End: {df.index.max()}")

    # Geometry
    bad = df[(df["high"] < df[["open", "close", "low"]].max(axis=1)) | (df["low"] > df[["open", "close", "high"]].min(axis=1))]
    flat = (df["high"] == df["low"]).mean()
    print(f"Invalid candle geometry: {len(bad)}   flat bars (H=L): {flat:.2%}")
    if len(bad):
        issues.append(f"{len(bad)} bars with impossible OHLC")
    if flat > 0.05:
        issues.append(f"{flat:.1%} flat bars (stale feed?)")
    if (df["volume"] == 0).all():
        print("Volume: all zero (volume features disabled)")

    # Gaps
    delta = df.index.to_series().diff().dropna()
    print("\nMost common gaps:")
    print(delta.value_counts().head(6).to_string())
    long_gaps = delta[delta > pd.Timedelta(days=4)]
    if len(long_gaps):
        issues.append(f"{len(long_gaps)} gaps > 4 days (first: {long_gaps.index[0]})")

    # Provider splices / spikes: bar-to-bar jumps far beyond normal volatility
    tr = (df["close"] - df["close"].shift(1)).abs()
    scale = tr.rolling(500, min_periods=50).median()
    spikes = df[(tr > 25 * scale)]
    print(f"\nPrice jumps > 25x median bar move: {len(spikes)}")
    if len(spikes):
        print(spikes.head(5)[["open", "high", "low", "close"]].to_string())
        warns.append(f"{len(spikes)} large price jumps: news spike or bad tick/splice - inspect them")

    # Sessions
    d = daily_bars(df, cfg)
    print(f"\nSessions ({cfg.session_tz} {cfg.session_start}): {len(d):,}")
    print("Weekday counts:", d.index.day_name().value_counts().to_dict())
    typical = d["n_bars"].median()
    short = d[d["n_bars"] < 0.6 * typical]
    print(f"Typical bars/session: {typical:.0f}   short sessions (<60%): {len(short)}")
    if len(short) > 0.03 * len(d):
        issues.append(f"{len(short)} short sessions: check timezone / session settings "
                      f"(--data-tz, --session-tz, --session-start)")

    # Timezone sanity: gold's daily maintenance break should sit at the session boundary.
    loc = df.index.tz_convert(cfg.session_tz)
    hours = pd.Series(loc.hour).value_counts().reindex(range(24), fill_value=0)
    quiet = hours[hours < 0.5 * hours.median()].index.tolist()
    print(f"Hours ({cfg.session_tz}) with few bars (market break?): {quiet}")
    sh = int(cfg.session_start.split(":")[0])
    if bar < 240 and quiet and not any(abs(h - sh) <= 1 or abs(h - sh) == 23 for h in quiet):
        issues.append("daily break does not line up with session start: timestamps may be in another tz")

    if warns:
        print("\nWARNINGS:\n  - " + "\n  - ".join(warns))
    print("\n" + ("OK - no blocking issue found." if not issues else "ISSUES:\n  - " + "\n  - ".join(issues)))


if __name__ == "__main__":
    main()

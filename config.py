from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    timestamp_col: str = "timestamp"

    # ---- Input data conventions -------------------------------------------------
    # "open": timestamp = bar OPEN time (MT5, Dukascopy, HistData, Twelve Data, OANDA).
    # "close": timestamp = bar CLOSE time.
    ts_convention: str = "open"
    # Time zone of naive timestamps in the CSV (ignored if the CSV has an offset).
    data_tz: str = "UTC"

    # ---- Trading-session definition --------------------------------------------
    # Gold trades ~23h/day; the conventional daily candle runs from 17:00 New York
    # to 17:00 New York the next day. Sunday evening bars are therefore part of
    # Monday's session (no 2-hour "Sunday day").
    session_tz: str = "America/New_York"
    session_start: str = "17:00"
    # Timeframes derived from the base series (only those >= base bar size are used).
    intraday_timeframes: tuple = (("4h", "h4_"), ("1h", "h1_"), ("15min", "m15_"))

    # ---- Labels ----------------------------------------------------------------
    threshold_atr: float = 0.20          # 3-class direction (secondary diagnostic)
    atr_window: int = 14
    max_label_gap_days: int = 5          # no label across data holes longer than this
    excursion_clip_atr: float = 4.0

    # ---- Macro -----------------------------------------------------------------
    macro_max_age_days: int = 7          # older macro values become NaN (no stale ffill)

    # ---- Models ----------------------------------------------------------------
    classifier: str = "logit"            # logit | gbm | blend
    logit_C: float = 0.01               # chosen on 2017-2021 only (dev period)
    upper_quantile: float = 0.80

    # ---- Walk-forward ----------------------------------------------------------
    min_train_days: int = 750
    test_block_days: int = 63

    # ---- Transformer (secondary) -----------------------------------------------
    sequence_days: int = 40
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.15

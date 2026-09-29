# Web data plan for XAUUSD AI Model v2

This file documents a practical source stack for building the model without manually preparing every dataset.

## Tier A — Core sources (recommended)

### 1) XAUUSD intraday
Preferred order:
1. Dukascopy historical export — bid/ask + volume; good for long intraday history.
2. HistData — free XAU/USD M1/tick downloads; also offers UDX/USD, WTI/USD and Brent (BCO/USD).
3. Twelve Data API — convenient programmatic 1m/5m/15m/1h/4h access; intraday history depends on instrument/plan and is generally multi-year, not always full-history.
4. OANDA v20 candles — useful if you already have an OANDA API account/token.

Use M1 as the master raw series where possible, then resample locally to M5/M15/H1/H4/D1. This avoids different candle-boundary conventions across vendors.

### 2) US yields / real yields / equity volatility / oil / equity indices
Use FRED/ALFRED for point-in-time safe daily features.
Suggested series:
- DGS10: US 10Y Treasury yield
- DGS2: US 2Y Treasury yield
- DFII10: US 10Y real yield
- VIXCLS: VIX close
- SP500: S&P 500
- NASDAQCOM: Nasdaq Composite
- DCOILWTICO: WTI spot
- DCOILBRENTEU: Brent spot
- DTWEXBGS: Broad trade-weighted US dollar index (official proxy, not ICE DXY)

For macro variables that are revised (CPI, payrolls, GDP, PCE), use ALFRED vintage dates / realtime periods so the backtest sees only the value known at that historical time.

### 3) COT positioning
Use the CFTC Public Reporting Environment / Socrata API.
Recommended report: Disaggregated Futures Only.
Filter GOLD / COMEX and use report publication timing correctly: data are positions as of Tuesday and normally released Friday.
Never join Tuesday values into Tuesday/Wednesday/Thursday rows before the Friday publication timestamp.

### 4) Economic releases
Best source split:
- Official values: BLS / BEA / FRED/ALFRED
- Historical actual + forecast + previous in one normalized feed: EODHD Economic Events API (from 2020) or another licensed calendar vendor.

Forecast values are not generally available as a clean official US government historical series, so this is one area where a vendor is useful.

### 5) Fed expectations
CME FedWatch:
- current probabilities
- one-day / one-week / one-month comparisons
- downloadable historical probabilities for a selected meeting, up to one year in the tool

For longer history, reconstruct expectations from historical 30-Day Fed Funds futures (ZQ) using CME methodology if licensed futures history is available.

### 6) News / geopolitics
GDELT is the best free large-scale source:
- GDELT 2.0 Events
- GDELT GKG sentiment/themes/entities
- updates every 15 minutes

For a first model, do NOT feed raw text directly. Build daily features such as:
- count of gold/inflation/Fed/Iran/war/oil headlines
- average tone
- share of negative articles
- article-count z-score
- entity/theme counts

This is much less noisy and easier to backtest than embedding millions of articles.

## Tier B — Useful optional sources

### IAU / ETF holdings
IAU publishes ounces/tonnes in trust and provides downloadable data on the official iShares page.
ETF flow features should be lagged to the publication time/date actually available.

### GLD
Use the official SPDR Gold Shares holdings/history if available for your chosen workflow. Keep it optional until the exact historical publication timestamps are validated.

### CME futures volume/open interest
CME provides current daily volume/open interest and some recent tools. Deep historical datasets are available via DataMine and are often licensed/paid.

## Tier C — Premium only / do not block v1 on these

### Gold option strike OI / gamma history
Current/recent CME QuikStrike tools are useful, but robust long historical strike-level options data is generally a licensed DataMine problem.
Recommendation: do not make gamma a mandatory feature for the first model.

## Recommended production stack

### Free / low-cost model
- XAUUSD M1: Dukascopy or HistData
- DXY proxy + WTI/Brent intraday: HistData where useful
- US10Y, US2Y, real yields, VIX, SP500, Nasdaq, oil daily: FRED/ALFRED
- COT: CFTC API
- News: GDELT
- Official macro actuals: BLS/BEA/FRED/ALFRED
- Forecast/previous: optional EODHD calendar from 2020

### Premium upgrade later
- CME DataMine Gold futures volume/OI history
- CME options strike-level history / implied vol
- professional historical economic-calendar consensus data

## Leakage rules

1. Every feature must have an `available_at` timestamp.
2. Join with `merge_asof(direction="backward")` on `available_at`, not observation period.
3. COT Tuesday positions only become usable after Friday publication.
4. Revised macro data must use ALFRED vintage values.
5. Economic forecasts must be the consensus available before release, never a later revised field.
6. Daily bars cannot use the day's final high/low/close for a prediction timestamp before the close.
7. News features may only include articles timestamped before the model decision time.
8. FedWatch history must be timestamped to the snapshot date used.


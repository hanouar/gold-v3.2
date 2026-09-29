"""NOT a pytest test (no test_ functions). Same protocol for both versions: yearly expanding walk-forward, dev 2017-2021, holdout 2022+,
plus cross-provider tests. Each version builds its own dataset and uses its own default model.
usage: python tests/bench_compare.py v4|v31 out.json [path_to_v4]  (run each in its own process)
"""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VER, OUT = sys.argv[1], sys.argv[2]
ROOT = {"v4": sys.argv[3] if len(sys.argv) > 3 else "../xauusd_ai_model_v4", "v31": str(__import__("pathlib").Path(__file__).resolve().parents[1])}[VER]
sys.path.insert(0, ROOT)
D = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data") + "/"
from model_utils import feature_columns

if VER == "v4":
    from data_pipeline import load_ohlcv
    from features import build_daily_dataset, add_labels
    from multitask_models import fit_multitask

    def dataset(path, kind):
        stz, ttz, hr = {"h4": ("UTC", "UTC", 21), "mt5": ("BROKER_NY_PLUS_7", "America/New_York", 17),
                        "utc": ("UTC", "America/New_York", 17)}[kind]
        b = load_ohlcv(path, "timestamp", stz)
        ds = add_labels(build_daily_dataset(b, None, trading_timezone=ttz, session_close_hour=hr, input_stamp="open"))
        ds.index = ds.index.tz_convert(ttz).tz_localize(None).normalize()  # session date
        return ds

    def fit(tr, feats):
        return fit_multitask(tr, feats)[0]

    def predict(m, X):
        return {"p_pdh": m["pdh_sweep"].predict_proba(X)[:, 1], "p_pdl": m["pdl_sweep"].predict_proba(X)[:, 1],
                "up": m["up_excursion"].predict(X), "down": m["down_excursion"].predict(X)}
    CLV = lambda ds: (ds.d1_close - ds.d1_low) / (ds.d1_high - ds.d1_low).replace(0, np.nan)
    ATR = "d1_atr14"
else:
    from config import ModelConfig
    from data_pipeline import load_ohlcv
    from features import build_daily_dataset, add_labels
    from multitask_models import fit_multitask, predict_frame

    def dataset(path, kind):
        cfg = {"h4": ModelConfig(session_tz="UTC", session_start="21:00"),
               "mt5": ModelConfig(data_tz="NY+7"), "utc": ModelConfig()}[kind]
        b = load_ohlcv(path, cfg)
        return add_labels(build_daily_dataset(b, None, cfg), cfg=cfg)

    def fit(tr, feats):
        return fit_multitask(tr, feats)[0]

    def predict(m, X):
        pf = predict_frame(m, X)
        return {"p_pdh": pf.p_pdh_sweep.values, "p_pdl": pf.p_pdl_sweep.values,
                "up": pf.up_excursion_p50.values, "down": pf.down_excursion_p50.values}
    CLV = lambda ds: ds.d1_clv
    ATR = "d1_atr_abs"

NEED = ["y_pdh_sweep", "y_pdl_sweep", "y_up_excursion_atr", "y_down_excursion_atr"]


def core3(ds):
    rng = (ds.d1_high - ds.d1_low)
    return pd.DataFrame({"clv": CLV(ds), "r": rng / ds[ATR], "ret": ds.d1_close.diff() / ds[ATR]}, index=ds.index)


def score(y, p, clim):
    return {"auc": roc_auc_score(y, p), "acc": accuracy_score(y, p >= 0.5), "brier": brier_score_loss(y, p),
            "bss": 1 - brier_score_loss(y, p) / np.mean((y - clim) ** 2)}


def summarize(P):
    r = {"n": int(len(P))}
    for t in ("pdh", "pdl"):
        r[t] = score(P[f"y_{t}"].values, P[f"p_{t}"].values, P[f"clim_{t}"].values)
        r[t]["core3_auc"] = roc_auc_score(P[f"y_{t}"], P[f"c3_{t}"])
    for t in ("up", "down"):
        y = P[f"y_{t}"].clip(0, 4)
        r[t] = {"mae": mean_absolute_error(y, P[t]), "mae_median": mean_absolute_error(y, P[f"med_{t}"])}
    return r


def run_block(tr, te, feats):
    m = fit(tr, feats)
    pr = predict(m, te[feats])
    P = pd.DataFrame(index=te.index)
    c3tr, c3te = core3(tr), core3(te)
    ok = c3tr.notna().all(axis=1)
    for t, col in (("pdh", "y_pdh_sweep"), ("pdl", "y_pdl_sweep")):
        P[f"y_{t}"] = te[col].astype(int).values
        P[f"p_{t}"] = pr[f"p_{t}"]
        P[f"clim_{t}"] = tr[col].mean()
        lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(c3tr[ok], tr.loc[ok, col].astype(int))
        P[f"c3_{t}"] = lr.predict_proba(c3te.fillna(c3tr.median()))[:, 1]
    for t, col in (("up", "y_up_excursion_atr"), ("down", "y_down_excursion_atr")):
        P[f"y_{t}"] = te[col].values
        P[t] = np.clip(pr[t], 0, None)
        P[f"med_{t}"] = tr[col].clip(0, 4).median()
    return P


res = {"version": VER}
t0 = time.time()
ds = dataset(D + "xauusd_h4_2013_2026.csv", "h4").dropna(subset=NEED)
feats = feature_columns(ds)
res["n_features"] = len(feats)
blocks = []
for y in range(2017, int(ds.index.max().year) + 1):
    tr, te = ds[ds.index.year < y], ds[ds.index.year == y]
    blocks.append(run_block(tr, te, feats))
    print(VER, y, flush=True)
P = pd.concat(blocks)
res["wf_dev_2017_2021"] = summarize(P[P.index < "2022-01-01"])
res["wf_holdout_2022_2026"] = summarize(P[P.index >= "2022-01-01"])
res["wf_seconds"] = round(time.time() - t0, 1)

# Cross-provider: model trained on H4 only.
tr24 = ds[ds.index <= "2024-12-31"]
for name, path, kind, trn in (("mt5_m15_2025_2026", "xauusd_m15_mt5_2025_2026.csv", "mt5", tr24),
                              ("utc_m15_2026", "xauusd_m15_utc_2026.csv", "utc", ds)):
    x = dataset(D + path, kind)
    x = x[x.index > trn.index.max()].dropna(subset=NEED)
    x = x[x[feats].notna().mean(axis=1) > 0.9] if all(f in x for f in feats) else x
    missing = [f for f in feats if f not in x.columns]
    for f in missing:
        x[f] = np.nan
    res[name] = summarize(run_block(trn, x, feats))
    res[name]["missing_features"] = len(missing)
json.dump(res, open(OUT, "w"), indent=2, default=float)
print(json.dumps({k: v for k, v in res.items()}, indent=1, default=float)[:3000])

"""Strict walk-forward test of the multitask liquidity model.

For each test year Y: train on every session before Y, predict Y, never look back.
Everything is compared with two references trained the same way:
  * climatology  : the training base rate (what you know with no model)
  * core3 logit  : logistic regression on 3 features (clv, range/ATR, ret/ATR)
A model that does not beat core3 out of sample is not worth its complexity.

Use --holdout-start to keep a frozen final period that you never tune on.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import add_data_args, config_from_args, load_inputs
from features import add_labels, build_daily_dataset
from model_utils import binary_metrics, calibration_table, feature_columns, regression_metrics
from multitask_models import CLASS_TARGETS, REG_TARGETS, _logit, fit_multitask, predict_frame

warnings.filterwarnings("ignore", category=UserWarning)
CORE3 = ["d1_clv", "d1_range_atr", "d1_ret1_atr"]


def pooled(pred: pd.DataFrame, upper_q: float) -> dict:
    out = {}
    for name, tgt in CLASS_TARGETS.items():
        y = pred[tgt]
        m = binary_metrics(y, pred[f"p_{name}"], None)
        m["brier_skill"] = float(1 - m["brier"] / np.mean((y - pred[f"clim_{name}"]) ** 2))
        c3 = binary_metrics(y, pred[f"core3_{name}"], None)
        m["core3_auc"], m["core3_brier"] = c3["auc"], c3["brier"]
        m["beats_core3"] = bool(m["auc"] > c3["auc"] and m["brier"] < c3["brier"])
        out[name] = m
    for name, tgt in REG_TARGETS.items():
        out[name] = regression_metrics(pred[tgt], pred[f"{name}_p50"], pred[f"ref_{name}"], pred[f"{name}_q"], upper_q)
    return out


def main():
    ap = argparse.ArgumentParser()
    add_data_args(ap)
    ap.add_argument("--out", default="reports/multitask_walk_forward")
    ap.add_argument("--first-test-year", type=int, default=2017)
    ap.add_argument("--holdout-start", default=None, help="e.g. 2022-01-01: reported separately, never tune on it")
    ap.add_argument("--classifier", choices=["logit", "gbm", "blend"], default=None)
    ap.add_argument("--min-train-days", type=int, default=750)
    args = ap.parse_args()

    cfg = config_from_args(args)
    base, macro = load_inputs(args, cfg)
    ds = add_labels(build_daily_dataset(base, macro, cfg), cfg=cfg)
    feats = feature_columns(ds)
    need = list(CLASS_TARGETS.values()) + list(REG_TARGETS.values())
    lab = ds.dropna(subset=need)
    print(f"sessions={len(ds)}  labelled={len(lab)}  features={len(feats)}  "
          f"range={lab.index.min().date()}..{lab.index.max().date()}  classifier={cfg.classifier}")

    rows = []
    for year in range(args.first_test_year, int(lab.index.max().year) + 1):
        tr, te = lab[lab.index.year < year], lab[lab.index.year == year]
        if len(tr) < args.min_train_days or len(te) < 20:
            continue
        models, _ = fit_multitask(tr, feats, cfg)
        pf = predict_frame(models, te[feats])
        for name, tgt in CLASS_TARGETS.items():
            pf[tgt] = te[tgt]
            pf[f"clim_{name}"] = tr[tgt].mean()
            c3 = _logit(1.0).fit(tr[CORE3], tr[tgt].astype(int))
            pf[f"core3_{name}"] = c3.predict_proba(te[CORE3])[:, 1]
        for name, tgt in REG_TARGETS.items():
            pf[tgt] = te[tgt]
            pf[f"ref_{name}"] = tr[tgt].median()
        pf["year"] = year
        for extra in ("y_pdh_reject", "y_pdl_reject", "d1_atr_pct", "next_weekday"):
            pf[extra] = te[extra]
        rows.append(pf)
        m = pooled(pf, cfg.upper_quantile)
        print(f"{year}: n={len(te):3d} | PDH auc={m['pdh_sweep']['auc']:.3f} (core3 {m['pdh_sweep']['core3_auc']:.3f}) "
              f"BSS={m['pdh_sweep']['brier_skill']:+.3f} | PDL auc={m['pdl_sweep']['auc']:.3f} "
              f"(core3 {m['pdl_sweep']['core3_auc']:.3f}) BSS={m['pdl_sweep']['brier_skill']:+.3f} | "
              f"up MAE skill={m['up_excursion']['mae_skill']:+.3f} down={m['down_excursion']['mae_skill']:+.3f}",
              flush=True)

    if not rows:
        raise SystemExit("Not enough history for a walk-forward test (need > min-train-days before first test year).")
    pred = pd.concat(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "predictions.csv")

    periods = {"all": pred}
    if args.holdout_start:
        hs = pd.Timestamp(args.holdout_start)
        periods = {"development": pred[pred.index < hs], "holdout": pred[pred.index >= hs], "all": pred}
    report = {"config": {**cfg.__dict__, "intraday_timeframes": [list(t) for t in cfg.intraday_timeframes]},
              "n_features": len(feats), "features": feats,
              "periods": {k: pooled(v, cfg.upper_quantile) for k, v in periods.items() if len(v) > 20},
              "yearly": {int(y): pooled(g, cfg.upper_quantile) for y, g in pred.groupby("year")}}

    # Conditional research stats: once a side is swept, how often does it close back inside?
    report["reject_given_sweep"] = {
        "pdh": float(pred.loc[pred["y_pdh_sweep"] == 1, "y_pdh_reject"].mean()),
        "pdl": float(pred.loc[pred["y_pdl_sweep"] == 1, "y_pdl_reject"].mean()),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=str))

    # Tables
    ysum = []
    for y, m in report["yearly"].items():
        ysum.append({"year": y, "n": m["pdh_sweep"]["n"],
                     "pdh_auc": m["pdh_sweep"]["auc"], "pdh_core3_auc": m["pdh_sweep"]["core3_auc"],
                     "pdh_brier_skill": m["pdh_sweep"]["brier_skill"], "pdh_accuracy": m["pdh_sweep"]["accuracy"],
                     "pdl_auc": m["pdl_sweep"]["auc"], "pdl_core3_auc": m["pdl_sweep"]["core3_auc"],
                     "pdl_brier_skill": m["pdl_sweep"]["brier_skill"], "pdl_accuracy": m["pdl_sweep"]["accuracy"],
                     "up_mae_skill": m["up_excursion"]["mae_skill"], "down_mae_skill": m["down_excursion"]["mae_skill"]})
    pd.DataFrame(ysum).to_csv(out / "yearly_summary.csv", index=False)
    for name, tgt in CLASS_TARGETS.items():
        calibration_table(pred[tgt], pred[f"p_{name}"]).to_csv(out / f"calibration_{name}.csv", index=False)

    # Regime / calendar breakdowns
    pred["vol_regime"] = pd.qcut(pred["d1_atr_pct"], 3, labels=["low_vol", "mid_vol", "high_vol"])
    br = []
    for col in ("vol_regime", "next_weekday"):
        for k, g in pred.groupby(col, observed=True):
            if len(g) < 30:
                continue
            br.append({"split": col, "bucket": str(k), "n": len(g),
                       **{f"{n}_auc": binary_metrics(g[t], g[f"p_{n}"])["auc"] for n, t in CLASS_TARGETS.items()},
                       **{f"{n}_rate": g[t].mean() for n, t in CLASS_TARGETS.items()}})
    pd.DataFrame(br).to_csv(out / "breakdowns.csv", index=False)

    print("\n=== POOLED OUT-OF-SAMPLE ===")
    for per, m in report["periods"].items():
        print(f"[{per}] n={m['pdh_sweep']['n']}")
        for n in CLASS_TARGETS:
            x = m[n]
            print(f"  {n:10s} AUC {x['auc']:.3f} (core3 {x['core3_auc']:.3f})  acc {x['accuracy']:.3f} "
                  f"(base rate {x['base_rate']:.3f})  Brier skill {x['brier_skill']:+.3f}  "
                  f"p>=0.70: hit {x['hit_rate_p70'] or float('nan'):.3f} on {x['coverage_p70']:.0%} of days  "
                  f"beats_core3={x['beats_core3']}")
        for n in REG_TARGETS:
            x = m[n]
            q = int(cfg.upper_quantile * 100)
            print(f"  {n:14s} MAE {x['mae_atr']:.3f} ATR vs median {x['mae_ref_median']:.3f} "
                  f"(skill {x['mae_skill']:+.3f}); p{q} coverage {x[f'coverage_q{q}']:.3f}")
    print(f"\nSaved reports to {out}")


if __name__ == "__main__":
    main()

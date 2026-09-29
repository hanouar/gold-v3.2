from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib

from data_pipeline import add_data_args, config_from_args, load_inputs
from features import add_labels, build_daily_dataset
from model_utils import feature_columns
from multitask_models import evaluate_multitask, fit_multitask

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ap = argparse.ArgumentParser()
    add_data_args(ap)
    ap.add_argument("--out", default="models/xauusd_multitask.joblib")
    ap.add_argument("--classifier", choices=["logit", "gbm", "blend"], default=None)
    ap.add_argument("--train-end", default=None, help="Optional: last session used for training (ISO date)")
    args = ap.parse_args()

    cfg = config_from_args(args)
    base, macro = load_inputs(args, cfg)
    ds = add_labels(build_daily_dataset(base, macro, cfg), cfg=cfg)
    if args.train_end:
        ds = ds[ds.index <= args.train_end]
    feats = feature_columns(ds)
    models, feats = fit_multitask(ds, feats, cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg_dict = {**cfg.__dict__, "intraday_timeframes": [list(t) for t in cfg.intraday_timeframes]}
    joblib.dump({"models": models, "features": feats, "config": cfg_dict,
                 "uses_macro": bool(macro is not None)}, out)
    print(json.dumps({k: v for k, v in models["_meta"].items() if k != "features"}, indent=2))
    ins = evaluate_multitask(models, ds, feats)
    print(f"In-sample (optimistic, for sanity only): PDH auc={ins['pdh_sweep']['auc']:.3f} "
          f"PDL auc={ins['pdl_sweep']['auc']:.3f}. Judge the model with walk_forward_multitask.py.")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

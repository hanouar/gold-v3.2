from __future__ import annotations
import argparse, json, joblib
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from data_pipeline import add_data_args, config_from_args, load_inputs
from features import build_daily_dataset, add_labels
from model_utils import feature_columns, evaluate_multiclass

def main():
    ap = argparse.ArgumentParser()
    add_data_args(ap)
    ap.add_argument("--out", default="models/baseline.joblib")
    ap.add_argument("--test-start", default=None, help="ISO date. If omitted, last 20%% is held out.")
    args = ap.parse_args()

    cfg = config_from_args(args)
    base, macro = load_inputs(args, cfg)
    ds = add_labels(build_daily_dataset(base, macro, cfg), cfg.threshold_atr, cfg).dropna(subset=["y_direction"])

    features = feature_columns(ds)
    if args.test_start:
        train = ds.loc[ds.index < args.test_start]
        test = ds.loc[ds.index >= args.test_start]
    else:
        cut = int(len(ds) * 0.8)
        train, test = ds.iloc[:cut], ds.iloc[cut:]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=350,
            max_leaf_nodes=31,
            l2_regularization=2.0,
            random_state=42,
        ))
    ])

    pipe.fit(train[features], train["y_direction"].astype(int))
    proba = pipe.predict_proba(test[features])
    classes = pipe.named_steps["model"].classes_
    metrics = evaluate_multiclass(test["y_direction"].astype(int), proba, classes)

    print(json.dumps(metrics, indent=2))
    print(classification_report(test["y_direction"].astype(int), pipe.predict(test[features]), digits=3))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "features": features,
        "classes": classes.tolist(),
        "config": {**cfg.__dict__, "intraday_timeframes": [list(t) for t in cfg.intraday_timeframes]},
        "uses_macro": bool(macro is not None),
        "train_end": str(train.index.max()),
        "test_start": str(test.index.min()),
    }, out)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()

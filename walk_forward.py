from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from data_pipeline import add_data_args, config_from_args, load_inputs
from features import build_daily_dataset, add_labels
from model_utils import feature_columns, evaluate_multiclass, calibration_table

def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=2.0,
            random_state=42,
        ))
    ])

def main():
    ap = argparse.ArgumentParser()
    add_data_args(ap)
    ap.add_argument("--out", default="reports/walk_forward")
    ap.add_argument("--min-train-days", type=int, default=750)
    ap.add_argument("--test-block-days", type=int, default=63)
    args = ap.parse_args()

    cfg = config_from_args(args)
    cfg.min_train_days, cfg.test_block_days = args.min_train_days, args.test_block_days
    base, macro = load_inputs(args, cfg)
    ds = add_labels(build_daily_dataset(base, macro, cfg), cfg.threshold_atr, cfg).dropna(subset=["y_direction"])
    feats = feature_columns(ds)

    preds = []
    start = cfg.min_train_days
    while start < len(ds):
        end = min(start + cfg.test_block_days, len(ds))
        train = ds.iloc[:start]
        test = ds.iloc[start:end]

        model = make_model()
        model.fit(train[feats], train["y_direction"].astype(int))
        p = model.predict_proba(test[feats])
        classes = model.named_steps["model"].classes_.tolist()

        for i, idx in enumerate(test.index):
            row = {
                "timestamp": idx,
                "y_true": int(test.loc[idx, "y_direction"]),
                "pred": int(classes[int(np.argmax(p[i]))]),
            }
            for j, c in enumerate(classes):
                row[f"p_{c}"] = float(p[i, j])
            row["next_ret_atr"] = float(test.loc[idx, "y_next_ret_atr"])
            preds.append(row)
        start = end

    pred = pd.DataFrame(preds).set_index("timestamp")
    class_order = sorted([int(c[2:]) for c in pred.columns if c.startswith("p_")])
    proba = np.column_stack([pred[f"p_{c}"].values for c in class_order])
    metrics = evaluate_multiclass(pred["y_true"].values, proba, class_order)

    # Very simple research signal: only act when max class probability >= 0.55.
    maxp = proba.max(axis=1)
    signal = pred["pred"].where(maxp >= 0.55, 0)
    # Directional payoff proxy in ATR units. This is NOT execution-level backtesting.
    pred["strategy_r_proxy"] = signal * pred["next_ret_atr"]
    metrics["proxy_total_r"] = float(pred["strategy_r_proxy"].sum())
    metrics["proxy_mean_r"] = float(pred["strategy_r_proxy"].mean())
    metrics["proxy_active_days"] = int((signal != 0).sum())
    # Reference: always predicting the most frequent training class.
    metrics["majority_class_accuracy"] = float((pred["y_true"] == ds["y_direction"].mode().iloc[0]).mean())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "predictions.csv")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    bull_col = "p_1" if "p_1" in pred else None
    if bull_col:
        cal = calibration_table((pred["y_true"] == 1).astype(int).values, pred[bull_col].values)
        cal.to_csv(out / "calibration_bull.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved reports to: {out}")

if __name__ == "__main__":
    main()

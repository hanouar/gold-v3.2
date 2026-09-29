from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             log_loss, mean_absolute_error, roc_auc_score)

from features import LABEL_COLS, META_COLS, RAW_COLS

EXCLUDE = set(LABEL_COLS) | set(RAW_COLS) | set(META_COLS)


def feature_columns(df: pd.DataFrame, include_macro: bool = True) -> list[str]:
    cols = []
    for c in df.columns:
        if c in EXCLUDE or c.startswith(("_", "y_")):
            continue
        if not include_macro and c.startswith("mac_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].notna().sum() == 0:
            continue
        cols.append(c)
    return cols


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def binary_metrics(y, p, base_rate: float | None = None) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    br = y.mean() if base_rate is None else base_rate
    brier = brier_score_loss(y, p)
    brier_ref = brier_score_loss(y, np.full(len(y), br))
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "brier": float(brier),
        "brier_skill": float(1 - brier / brier_ref) if brier_ref > 0 else None,  # >0 = beats climatology
        "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "hit_rate_p70": float(y[p >= 0.7].mean()) if (p >= 0.7).any() else None,
        "coverage_p70": float((p >= 0.7).mean()),
        "hit_rate_p30": float(1 - y[p <= 0.3].mean()) if (p <= 0.3).any() else None,
        "coverage_p30": float((p <= 0.3).mean()),
    }


def regression_metrics(y, pred, ref_pred, q_hi=None, q=0.8) -> dict:
    y = np.asarray(y, dtype=float)
    mae, mae_ref = mean_absolute_error(y, pred), mean_absolute_error(y, ref_pred)
    out = {"n": int(len(y)), "mae_atr": float(mae), "mae_ref_median": float(mae_ref),
           "mae_skill": float(1 - mae / mae_ref) if mae_ref > 0 else None}
    if q_hi is not None:
        out[f"coverage_q{int(q*100)}"] = float((y <= np.asarray(q_hi)).mean())  # should be ~q
    return out


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    return df.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"),
                                                 realized=("y", "mean")).reset_index()


def evaluate_multiclass(y_true, proba, classes):
    pred = np.array(classes)[np.argmax(proba, axis=1)]
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=list(classes))),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(classes)).tolist(),
    }

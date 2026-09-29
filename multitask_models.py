from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ModelConfig
from model_utils import binary_metrics, feature_columns, regression_metrics

CLASS_TARGETS = {"pdh_sweep": "y_pdh_sweep", "pdl_sweep": "y_pdl_sweep"}
REG_TARGETS = {"up_excursion": "y_up_excursion_atr", "down_excursion": "y_down_excursion_atr"}


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #
def _logit(C: float):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=5000)),
    ])


def _gbm(random_state: int = 42):
    base = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=250, max_leaf_nodes=8,
                                          min_samples_leaf=40, l2_regularization=5.0, random_state=random_state)
    # Calibrate on LATER folds only (time-ordered), never on shuffled folds.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", CalibratedClassifierCV(base, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))),
    ])


def make_binary_classifier(cfg: ModelConfig | None = None, kind: str | None = None):
    cfg = cfg or ModelConfig()
    kind = kind or cfg.classifier
    if kind == "logit":
        return _logit(cfg.logit_C)
    if kind == "gbm":
        return _gbm()
    if kind == "blend":
        return VotingClassifier([("logit", _logit(cfg.logit_C)), ("gbm", _gbm())], voting="soft")
    raise ValueError(f"unknown classifier kind: {kind}")


def make_regressor(quantile: float = 0.5, kind: str = "gbm"):
    if kind == "linear":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", QuantileRegressor(quantile=quantile, alpha=0.001, solver="highs")),
        ])
    loss = "absolute_error" if quantile == 0.5 else "quantile"
    kw = {} if quantile == 0.5 else {"quantile": quantile}
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(learning_rate=0.03, max_iter=300, max_leaf_nodes=8,
                                                min_samples_leaf=40, l2_regularization=5.0,
                                                loss=loss, random_state=42, **kw)),
    ])


# --------------------------------------------------------------------------- #
# Fit / predict / evaluate
# --------------------------------------------------------------------------- #
def fit_multitask(ds: pd.DataFrame, features: list[str] | None = None, cfg: ModelConfig | None = None,
                  reg_kind: str = "gbm"):
    """Outputs:
      p(PDH sweep), p(PDL sweep),
      median + upper-quantile up/down excursion (ATR units).
    """
    cfg = cfg or ModelConfig()
    features = features or feature_columns(ds)
    clean = ds.dropna(subset=list(CLASS_TARGETS.values()) + list(REG_TARGETS.values()))
    X = clean[features]
    models = {}
    for name, tgt in CLASS_TARGETS.items():
        models[name] = make_binary_classifier(cfg).fit(X, clean[tgt].astype(int))
    for name, tgt in REG_TARGETS.items():
        y = clean[tgt]
        models[name] = make_regressor(0.5, reg_kind).fit(X, y)
        models[f"{name}_q"] = make_regressor(cfg.upper_quantile, reg_kind).fit(X, y)
    models["_meta"] = {
        "features": features,
        "train_start": str(clean.index.min().date()),
        "train_end": str(clean.index.max().date()),
        "n_train": int(len(clean)),
        "base_rates": {n: float(clean[t].mean()) for n, t in CLASS_TARGETS.items()},
        "median_excursion": {n: float(clean[t].median()) for n, t in REG_TARGETS.items()},
        "upper_quantile": cfg.upper_quantile,
    }
    return models, features


def predict_frame(models, X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    for name in CLASS_TARGETS:
        out[f"p_{name}"] = models[name].predict_proba(X)[:, 1]
    for name in REG_TARGETS:
        med = np.clip(models[name].predict(X), 0, None)
        hi = np.clip(models[f"{name}_q"].predict(X), 0, None)
        out[f"{name}_p50"] = med
        out[f"{name}_q"] = np.maximum(hi, med)  # quantile crossing guard
    return out


def predict_multitask(models, row: pd.DataFrame, features: list[str]) -> dict:
    p = predict_frame(models, row[features]).iloc[-1]
    q = int(models["_meta"]["upper_quantile"] * 100)
    return {
        "pdh_sweep_probability": float(p["p_pdh_sweep"]),
        "pdl_sweep_probability": float(p["p_pdl_sweep"]),
        "expected_up_excursion_atr": float(p["up_excursion_p50"]),
        "expected_down_excursion_atr": float(p["down_excursion_p50"]),
        f"up_excursion_atr_p{q}": float(p["up_excursion_q"]),
        f"down_excursion_atr_p{q}": float(p["down_excursion_q"]),
    }


def evaluate_multitask(models, ds: pd.DataFrame, features: list[str]) -> dict:
    clean = ds.dropna(subset=list(CLASS_TARGETS.values()) + list(REG_TARGETS.values()))
    pf = predict_frame(models, clean[features])
    meta = models["_meta"]
    out = {}
    for name, tgt in CLASS_TARGETS.items():
        out[name] = binary_metrics(clean[tgt], pf[f"p_{name}"], meta["base_rates"][name])
    for name, tgt in REG_TARGETS.items():
        ref = np.full(len(clean), meta["median_excursion"][name])
        out[name] = regression_metrics(clean[tgt], pf[f"{name}_p50"], ref, pf[f"{name}_q"], meta["upper_quantile"])
    return out

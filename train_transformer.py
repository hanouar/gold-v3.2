"""Optional sequence model. Keep it ONLY if it beats the logistic model on the same test period.

Split (time-ordered, no shuffling across time):
  train  : sessions <  --val-start       (fit)
  val    : --val-start <= s < --test-start (early stopping only)
  test   : sessions >= --test-start       (reported, never used for any decision)
Scaler/imputer are fitted on train only. Sequences use past context across split
boundaries (features only, which is allowed), labels are never shared.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from data_pipeline import add_data_args, config_from_args, load_inputs
from features import add_labels, build_daily_dataset
from model_utils import feature_columns
from multitask_models import _logit
from transformer_model import TemporalTransformer

warnings.filterwarnings("ignore", category=UserWarning)
TARGETS = {"direction": "y_direction", "pdh_sweep": "y_pdh_sweep", "pdl_sweep": "y_pdl_sweep"}


def windows(X: np.ndarray, ends: np.ndarray, L: int) -> torch.Tensor:
    return torch.from_numpy(np.stack([X[e - L + 1:e + 1] for e in ends]).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    add_data_args(ap)
    ap.add_argument("--target", choices=list(TARGETS), default="pdh_sweep")
    ap.add_argument("--val-start", default="2020-01-01")
    ap.add_argument("--test-start", default="2022-01-01")
    ap.add_argument("--out", default="models/transformer.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = config_from_args(args)
    base, macro = load_inputs(args, cfg)
    ds = add_labels(build_daily_dataset(base, macro, cfg), cfg.threshold_atr, cfg)
    feats = feature_columns(ds)
    tgt = TARGETS[args.target]
    L = cfg.sequence_days

    idx = ds.index
    tr_mask = idx < args.val_start
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    X = sc.fit(imp.fit(ds.loc[tr_mask, feats]).transform(ds.loc[tr_mask, feats])).transform(imp.transform(ds[feats]))
    X = np.clip(np.nan_to_num(X), -8, 8)

    if args.target == "direction":
        y_all = ds[tgt].map({-1: 0, 0: 1, 1: 2})
        n_classes = 3
    else:
        y_all = ds[tgt]
        n_classes = 2
    has_y = y_all.notna().values
    pos = np.arange(len(ds))
    ok = has_y & (pos >= L - 1)
    split = {
        "train": np.where(ok & (idx < args.val_start))[0],
        "val": np.where(ok & (idx >= args.val_start) & (idx < args.test_start))[0],
        "test": np.where(ok & (idx >= args.test_start))[0],
    }
    y = np.nan_to_num(y_all.values).astype(np.int64)
    print({k: len(v) for k, v in split.items()}, "features:", len(feats), "target:", tgt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformer(len(feats), cfg.d_model, cfg.nhead, cfg.num_layers, cfg.dropout, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    Xv, yv = windows(X, split["val"], L).to(device), torch.from_numpy(y[split["val"]]).to(device)

    best, best_loss, wait, patience = None, float("inf"), 0, 6
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(split["train"])
        tl = 0.0
        for i in range(0, len(perm), args.batch_size):
            e = perm[i:i + args.batch_size]
            xb, yb = windows(X, e, L).to(device), torch.from_numpy(y[e]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(e)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), yv).item()
        print(f"epoch={epoch:02d} train_loss={tl / len(perm):.4f} val_loss={vl:.4f}", flush=True)
        if vl < best_loss - 1e-4:
            best_loss, wait = vl, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break
    model.load_state_dict(best)

    # ---- Frozen test + logistic reference on the same rows
    model.eval()
    with torch.no_grad():
        pt = torch.softmax(model(windows(X, split["test"], L).to(device)), 1).cpu().numpy()
    yt = y[split["test"]]
    fit_rows = np.r_[split["train"], split["val"]]
    ref = _logit(cfg.logit_C).fit(ds.iloc[fit_rows][feats], y[fit_rows])
    pr = ref.predict_proba(ds.iloc[split["test"]][feats])
    res = {}
    for name, p in (("transformer", pt), ("logit_reference", pr)):
        r = {"accuracy": float(accuracy_score(yt, p.argmax(1))), "log_loss": float(log_loss(yt, p, labels=list(range(n_classes))))}
        if n_classes == 2:
            r["auc"] = float(roc_auc_score(yt, p[:, 1]))
            r["brier"] = float(brier_score_loss(yt, p[:, 1]))
        res[name] = r
    print(json.dumps({"test_period_from": args.test_start, "n_test": int(len(yt)), **res}, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "features": feats, "target": tgt, "n_classes": n_classes,
                "config": {**cfg.__dict__, "intraday_timeframes": [list(t) for t in cfg.intraday_timeframes]},
                "imputer_statistics": imp.statistics_, "scaler_mean": sc.mean_, "scaler_scale": sc.scale_,
                "test_metrics": res}, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

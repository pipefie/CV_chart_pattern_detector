from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import datetime as dt


def load_features(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def pick_xy(df: pd.DataFrame, target_col: str, drop_structural: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    drop_like = ["image", "image_path", "split", "symbol", "timeframe", "start_ts", "end_ts"]
    drop_cols = set(drop_like)
    drop_cols.add(target_col)
    for col in df.columns:
        if col.startswith("y_") and col != target_col:
            drop_cols.add(col)
    if drop_structural:
        prefixes = ["hs_", "dt_", "db_", "tri_"]
        for col in df.columns:
            if any(col.startswith(p) for p in prefixes):
                drop_cols.add(col)
    X = df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=[np.number]).copy()
    y = df[target_col].astype(int).copy()
    return X, y


def train_random_forest(
    train_csv: Path,
    target: str,
    params: Dict,
    out_dir: Path,
    val_csv: Path | None = None,
    drop_structural: bool = True,
) -> Dict:
    train_df = load_features(train_csv)
    X_train, y_train = pick_xy(train_df, target, drop_structural=drop_structural)

    rf_params = {
        "n_estimators": int(params.get("n_estimators", 600)),
        "max_depth": params.get("max_depth"),
        "min_samples_leaf": int(params.get("min_samples_leaf", 1)),
        "class_weight": params.get("class_weight", "balanced"),
        "n_jobs": -1,
        "random_state": int(params.get("random_state", 42)),
    }
    rf = RandomForestClassifier(**rf_params)

    scores: Dict[str, float] = {}
    feat_names = X_train.columns.tolist()
    manifest_meta = {
        "train_rows": len(train_df),
        "val_rows": len(load_features(val_csv) if val_csv else []),
        "features_used": len(feat_names),
        "drop_structural": bool(drop_structural),
        "target_positive_count": int(y_train.sum()),
        "target_negative_count": int((y_train == 0).sum()),
    }

    if val_csv:
        val_df = load_features(val_csv)
        X_val, y_val = pick_xy(val_df, target, drop_structural=drop_structural)
        rf.fit(X_train, y_train)
        probs = rf.predict_proba(X_val)[:, 1]
        scores["roc_auc"] = float(roc_auc_score(y_val, probs))
        scores["pr_auc"] = float(average_precision_score(y_val, probs))
        thresholds = np.linspace(0.05, 0.95, 19)
        f1s = [f1_score(y_val, (probs >= t).astype(int)) for t in thresholds]
        best_idx = int(np.argmax(f1s))
        scores["f1_at_best_th"] = float(f1s[best_idx])
        scores["best_threshold"] = float(thresholds[best_idx])
        try:
            perm = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=rf_params["random_state"], n_jobs=-1)
            feat_importance = [{"feature": n, "perm_importance": float(v)} for n, v in sorted(zip(feat_names, perm.importances_mean), key=lambda x: -x[1])[:30]]
        except Exception as exc:  # pragma: no cover - diagnostics only
            feat_importance = []
            scores["perm_error"] = str(exc)
    else:
        groups = train_df["symbol"].astype(str)
        gkf = GroupKFold(n_splits=5)
        aucs, pr_aucs = [], []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            rf_fold = RandomForestClassifier(
                **{**rf_params, "random_state": rf_params["random_state"] + fold},
            )
            rf_fold.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            probs = rf_fold.predict_proba(X_train.iloc[va_idx])[:, 1]
            aucs.append(roc_auc_score(y_train.iloc[va_idx], probs))
            pr_aucs.append(average_precision_score(y_train.iloc[va_idx], probs))
        scores["cv5_roc_auc_mean"] = float(np.mean(aucs))
        scores["cv5_pr_auc_mean"] = float(np.mean(pr_aucs))
        scores["best_threshold"] = 0.5
        rf.fit(X_train, y_train)
        feat_importance = [{"feature": n, "gini_importance": float(v)} for n, v in zip(feat_names, rf.feature_importances_)]

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    dump(rf, model_path)

    manifest = {
        "created_utc": dt.datetime.utcnow().isoformat(),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv) if val_csv else None,
        "target": target,
        "params": rf_params,
        "meta": manifest_meta,
        "scores": scores,
        "best_threshold": scores.get("best_threshold", 0.5),
        "feature_names": feat_names,
        "feature_importance_top": feat_importance,
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"model_path": model_path, "manifest": manifest}


__all__ = ["train_random_forest", "load_features", "pick_xy"]

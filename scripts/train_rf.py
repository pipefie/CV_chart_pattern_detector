# scripts/train_rf.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from joblib import dump
import datetime as dt

def load_features(path: str|Path) -> pd.DataFrame:
    return pd.read_csv(path)

def pick_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    # X: all numeric, excluding identifiers & other targets
    drop_like = ["image","split","symbol","start_ts","end_ts"]
    drop_cols = set(drop_like)
    drop_cols.add(target_col)
    for c in df.columns:
        if c.startswith("y_") and c != target_col:
            drop_cols.add(c)
    X = df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=[np.number]).copy()
    y = df[target_col].astype(int).copy()
    return X, y

def main():
    ap = argparse.ArgumentParser(description="Train RandomForest on features for a selected pattern target.")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", default=None, help="Optional explicit validation CSV. If missing, uses GroupKFold on train.")
    ap.add_argument("--target", default="y_head_and_shoulders", help="Which binary target to train (column name)")
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--min_samples_leaf", type=int, default=1)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--out_dir", default="reports/runs/exp_rf")
    args = ap.parse_args()

    train_df = load_features(args.train_csv)
    X_train, y_train = pick_xy(train_df, args.target)

    # Basic sanity for class imbalance
    pos_rate = float(y_train.mean())
    print(f"Target={args.target} | train rows={len(train_df)} | positive rate={pos_rate:.3f}")

    # Model
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",      # handle imbalance without resampling
        n_jobs=-1,
        random_state=args.random_state
    )

    scores = {}
    feat_names = X_train.columns.tolist()

    if args.val_csv:
        # Train on train_csv, evaluate on val_csv (chronological split recommended)
        val_df = load_features(args.val_csv)
        X_val, y_val = pick_xy(val_df, args.target)

        rf.fit(X_train, y_train)
        p = rf.predict_proba(X_val)[:,1]
        scores["roc_auc"] = float(roc_auc_score(y_val, p))
        scores["pr_auc"]  = float(average_precision_score(y_val, p))

        # Choose threshold by maximizing F1 on validation
        ths = np.linspace(0.05, 0.95, 19)
        f1s = [f1_score(y_val, (p>=t).astype(int)) for t in ths]
        best_idx = int(np.argmax(f1s))
        best_th = float(ths[best_idx])
        scores["f1_at_best_th"] = float(f1s[best_idx])
        scores["best_threshold"] = best_th

        # Permutation importance (on val for honest estimate)
        try:
            perm = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=args.random_state, n_jobs=-1)
            imp = sorted(zip(feat_names, perm.importances_mean), key=lambda x: -x[1])[:30]
            feat_importance = [{"feature": n, "perm_importance": float(v)} for n,v in imp]
        except Exception as e:
            print(f"Permutation importance failed: {e}")
            feat_importance = []
    else:
        # GroupKFold by symbol to avoid leakage of adjacent windows of same asset
        groups = train_df["symbol"].astype(str)
        gkf = GroupKFold(n_splits=5)
        aucs, pr_aucs = [], []
        for k,(tr,va) in enumerate(gkf.split(X_train, y_train, groups)):
            rf_k = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                class_weight="balanced",
                n_jobs=-1,
                random_state=args.random_state + k
            )
            rf_k.fit(X_train.iloc[tr], y_train.iloc[tr])
            p = rf_k.predict_proba(X_train.iloc[va])[:,1]
            aucs.append(roc_auc_score(y_train.iloc[va], p))
            pr_aucs.append(average_precision_score(y_train.iloc[va], p))
        scores["cv5_roc_auc_mean"] = float(np.mean(aucs))
        scores["cv5_pr_auc_mean"]  = float(np.mean(pr_aucs))
        # Fit final on full train
        rf.fit(X_train, y_train)
        # Threshold fallback
        scores["best_threshold"] = 0.5
        feat_importance = [{"feature": n, "gini_importance": float(v)} for n,v in zip(feat_names, rf.feature_importances_)]

    # --- Save artifacts ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    dump(rf, model_path)

    manifest = {
        "created_utc": dt.datetime.utcnow().isoformat(),
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "target": args.target,
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state
        },
        "scores": scores,
        "best_threshold": scores.get("best_threshold", 0.5),
        "feature_names": feat_names,
        "feature_importance_top": feat_importance
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"✅ Saved model to {model_path}")
    print(f"✅ Train manifest: {out_dir/'train_manifest.json'}")
    if "roc_auc" in scores:
        print(f"Val ROC-AUC={scores['roc_auc']:.3f} PR-AUC={scores['pr_auc']:.3f} F1*={scores['f1_at_best_th']:.3f} @th={scores['best_threshold']:.2f}")
    else:
        print(f"CV5 ROC-AUC={scores['cv5_roc_auc_mean']:.3f} PR-AUC={scores['cv5_pr_auc_mean']:.3f}")
if __name__ == "__main__":
    main()

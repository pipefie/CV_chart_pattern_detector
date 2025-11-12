# scripts/eval_rf.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_fscore_support, confusion_matrix
)
from joblib import load
import datetime as dt

def load_features(path: str|Path) -> pd.DataFrame:
    return pd.read_csv(path)

def pick_xy(df: pd.DataFrame, target_col: str):
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
    ap = argparse.ArgumentParser(description="Evaluate a saved RF model on a test CSV.")
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--target", default="y_head_and_shoulders")
    ap.add_argument("--out_dir", default="reports/eval")
    ap.add_argument("--threshold", type=float, default=None, help="If not set, tries to read train_manifest.best_threshold")
    args = ap.parse_args()

    model = load(args.model_path)
    test_df = load_features(args.test_csv)
    X_test, y_test = pick_xy(test_df, args.target)

    # Threshold: pull from neighboring train manifest if not provided
    th = args.threshold
    if th is None:
        man_path = Path(args.model_path).with_name("train_manifest.json")
        if man_path.exists():
            try:
                man = json.loads(man_path.read_text(encoding="utf-8"))
                th = float(man.get("best_threshold", 0.5))
            except Exception:
                th = 0.5
        else:
            th = 0.5

    p = model.predict_proba(X_test)[:,1]
    y_hat = (p >= th).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, p)),
        "pr_auc": float(average_precision_score(y_test, p)),
        "f1_at_th": float(f1_score(y_test, y_hat)),
        "threshold": float(th)
    }
    pr, rc, f1, sup = precision_recall_fscore_support(y_test, y_hat, average=None, labels=[0,1])
    metrics["class_0"] = {"precision": float(pr[0]), "recall": float(rc[0]), "f1": float(f1[0]), "support": int(sup[0])}
    metrics["class_1"] = {"precision": float(pr[1]), "recall": float(rc[1]), "f1": float(f1[1]), "support": int(sup[1])}

    cm = confusion_matrix(y_test, y_hat, labels=[0,1])
    cm_df = pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"metrics_{stamp}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm_df.to_csv(out_dir / f"confusion_matrix_{stamp}.csv")

    print(f"✅ Eval metrics: {out_dir / f'metrics_{stamp}.json'}")
    print(f"ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  F1@{metrics['threshold']:.2f}={metrics['f1_at_th']:.3f}")

if __name__ == "__main__":
    main()

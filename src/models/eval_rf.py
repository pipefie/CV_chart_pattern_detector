from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
import datetime as dt

import pandas as pd
from joblib import load as joblib_load
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .train_rf import load_features, pick_xy


def evaluate_random_forest(
    test_csv: Path,
    model_path: Path,
    target: str,
    threshold: float | None,
    out_dir: Path,
) -> Dict:
    model = joblib_load(model_path)
    test_df = load_features(test_csv)
    X_test, y_test = pick_xy(test_df, target)

    if threshold is None:
        manifest_path = model_path.with_name("train_manifest.json")
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                threshold = float(manifest.get("best_threshold", 0.5))
            except Exception:
                threshold = 0.5
        else:
            threshold = 0.5

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    metrics: Dict[str, float | Dict] = {
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "f1_at_th": float(f1_score(y_test, preds)),
        "threshold": float(threshold),
    }
    pr, rc, f1, sup = precision_recall_fscore_support(y_test, preds, average=None, labels=[0, 1])
    metrics["class_0"] = {"precision": float(pr[0]), "recall": float(rc[0]), "f1": float(f1[0]), "support": int(sup[0])}
    metrics["class_1"] = {"precision": float(pr[1]), "recall": float(rc[1]), "f1": float(f1[1]), "support": int(sup[1])}

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    metrics_path = out_dir / f"metrics_{stamp}.json"
    cm_path = out_dir / f"confusion_matrix_{stamp}.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm_df.to_csv(cm_path)
    return {"metrics_path": metrics_path, "confusion_path": cm_path, "metrics": metrics}


__all__ = ["evaluate_random_forest"]

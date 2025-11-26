#!/usr/bin/env python
"""
Run RF inference on a feature CSV (built from a render manifest) using a saved baseline model.

Example:
    uv run python scripts/infer_rf.py \
      --features_csv data/features_infer/all_features.csv \
      --model_path reports/baselines/hs/model.joblib \
      --target y_head_and_shoulders \
      --threshold 0.60 \
      --out_csv reports/eval/hs_infer/predictions.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from src.models.train_rf import pick_xy


def main():
    ap = argparse.ArgumentParser(description="Inference for RF models on deterministic feature CSVs.")
    ap.add_argument("--features_csv", required=True, help="Feature CSV (e.g., data/features_infer/all_features.csv)")
    ap.add_argument("--model_path", required=True, help="Path to model.joblib")
    ap.add_argument("--target", required=True, help="Target column name (e.g., y_head_and_shoulders)")
    ap.add_argument("--threshold", type=float, default=None, help="Decision threshold; if omitted, load from train_manifest.json if present.")
    ap.add_argument("--out_csv", required=True, help="Where to write predictions (CSV with prob/pred appended)")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    model = joblib_load(model_path)
    df = pd.read_csv(args.features_csv)

    # load threshold from manifest if not provided
    th = args.threshold
    if th is None:
        manifest_path = model_path.with_name("train_manifest.json")
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                th = float(manifest.get("best_threshold", 0.5))
            except Exception:
                th = 0.5
        else:
            th = 0.5

    # align features the same way as training (drop_structural=True by default)
    X, _ = pick_xy(df, args.target, drop_structural=True)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= th).astype(int)

    out = df.copy()
    out[f"{args.target}_prob"] = probs
    out[f"{args.target}_pred"] = preds
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"✅ Wrote predictions to {args.out_csv} (threshold={th:.2f})")


if __name__ == "__main__":
    main()

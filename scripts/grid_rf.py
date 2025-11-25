#!/usr/bin/env python
# scripts/grid_rf.py
from pathlib import Path
import sys
import argparse
from itertools import product
import json
import yaml

# Ensure project root is on sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train_rf import train_random_forest
from src.models.eval_rf import evaluate_random_forest


def main():
    ap = argparse.ArgumentParser(description="Grid search for RandomForest hyperparameters (per target).")
    ap.add_argument("--train_csv", default="data/features/train_features.csv")
    ap.add_argument("--val_csv", default="data/features/val_features.csv")
    ap.add_argument("--test_csv", default="data/features/test_features.csv")
    ap.add_argument("--target", default="y_head_and_shoulders")
    ap.add_argument("--out_root", default="reports/runs/grid_rf")
    ap.add_argument("--pipeline_cfg", default="configs/pipeline.yaml")
    ap.add_argument("--min_samples_leaf", nargs="+", type=int, default=[3, 5, 8])
    ap.add_argument("--max_depth", nargs="+", type=int, default=[8, 12, 16])
    ap.add_argument("--n_estimators", nargs="+", type=int, default=[600, 800, 1000])
    ap.add_argument("--keep_structural", action="store_true", help="Keep structural features (default drops them).")
    args = ap.parse_args()

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    test_csv = Path(args.test_csv)
    target = args.target
    out_root = Path(args.out_root) / target
    cfg = yaml.safe_load(Path(args.pipeline_cfg).read_text()) or {}
    rf_cfg = (cfg.get("models") or {}).get("random_forest", {}) or {}

    grids = {
        "min_samples_leaf": args.min_samples_leaf,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
    }

    results = []
    for msl, md, ne in product(grids["min_samples_leaf"], grids["max_depth"], grids["n_estimators"]):
        params = {
            "min_samples_leaf": msl,
            "max_depth": md,
            "n_estimators": ne,
            "random_state": rf_cfg.get("random_state", 42),
            "class_weight": "balanced",
        }
        run_dir = out_root / f"msl{msl}_md{md}_ne{ne}"
        run_dir.mkdir(parents=True, exist_ok=True)

        res_train = train_random_forest(train_csv, target, params, run_dir, val_csv=val_csv, drop_structural=not args.keep_structural)
        res_eval = evaluate_random_forest(test_csv, res_train["model_path"], target, threshold=None, out_dir=run_dir)
        results.append({
            "params": params,
            "val_scores": res_train["manifest"]["scores"],
            "test_scores": res_eval["metrics"],
            "model_path": str(res_train["model_path"]),
        })

    (out_root / "grid_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Done. Wrote {len(results)} combos to {out_root/'grid_results.json'}")


if __name__ == "__main__":
    main()

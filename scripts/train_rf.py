# scripts/train_rf.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import yaml

from src.models.train_rf import train_random_forest


def main():
    ap = argparse.ArgumentParser(description="Train RandomForest on deterministic features.")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", default=None)
    ap.add_argument("--target", default="y_head_and_shoulders")
    ap.add_argument("--out_dir", default="reports/runs/exp_rf")
    ap.add_argument("--pipeline_cfg", default="configs/pipeline.yaml", help="Optional config to source RF defaults.")
    ap.add_argument("--n_estimators", type=int, default=None)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--min_samples_leaf", type=int, default=None)
    ap.add_argument("--random_state", type=int, default=None)
    ap.add_argument("--keep_structural", action="store_true", help="Keep structural pattern features (default drops hs_/dt_/db_/tri_ to reduce leakage).")
    args = ap.parse_args()

    defaults = {"n_estimators": 600, "max_depth": None, "min_samples_leaf": 1, "random_state": 42}
    if args.pipeline_cfg and Path(args.pipeline_cfg).exists():
        cfg = yaml.safe_load(Path(args.pipeline_cfg).read_text(encoding="utf-8")) or {}
        rf_cfg = ((cfg.get("models") or {}).get("random_forest")) or {}
        defaults.update({k: v for k, v in rf_cfg.items() if v is not None})

    params = {
        "n_estimators": args.n_estimators if args.n_estimators is not None else defaults["n_estimators"],
        "max_depth": args.max_depth if args.max_depth is not None else defaults["max_depth"],
        "min_samples_leaf": args.min_samples_leaf if args.min_samples_leaf is not None else defaults["min_samples_leaf"],
        "random_state": args.random_state if args.random_state is not None else defaults["random_state"],
        "class_weight": "balanced",
    }
    val_csv = Path(args.val_csv) if args.val_csv else None
    # Lightweight verbosity
    train_df = None
    try:
        import pandas as pd  # optional, just for logging
        train_df = pd.read_csv(args.train_csv)
        target_pos = int(train_df[args.target].sum()) if args.target in train_df else 0
        target_neg = len(train_df) - target_pos
        print(f"[train_rf] Rows={len(train_df)} target={args.target} pos={target_pos} neg={target_neg} keep_structural={args.keep_structural}")
    except Exception:
        pass
    result = train_random_forest(
        train_csv=Path(args.train_csv),
        target=args.target,
        params=params,
        out_dir=Path(args.out_dir),
        val_csv=val_csv,
        drop_structural=not args.keep_structural,
    )
    scores = result["manifest"]["scores"]
    print(f"✅ Saved model to {result['model_path']}")
    if "roc_auc" in scores:
        print(f"Val ROC-AUC={scores['roc_auc']:.3f} PR-AUC={scores['pr_auc']:.3f} F1*={scores.get('f1_at_best_th', 0):.3f}")
    else:
        print(f"CV5 ROC-AUC={scores.get('cv5_roc_auc_mean', 0):.3f} PR-AUC={scores.get('cv5_pr_auc_mean', 0):.3f}")


if __name__ == "__main__":
    main()

# scripts/eval_rf.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.models.eval_rf import evaluate_random_forest


def main():
    ap = argparse.ArgumentParser(description="Evaluate a saved RandomForest model.")
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--target", default="y_head_and_shoulders")
    ap.add_argument("--out_dir", default="reports/eval")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    result = evaluate_random_forest(
        test_csv=Path(args.test_csv),
        model_path=Path(args.model_path),
        target=args.target,
        threshold=args.threshold,
        out_dir=Path(args.out_dir),
    )
    metrics = result["metrics"]
    print(f"✅ Eval metrics: {result['metrics_path']}")
    print(f"ROC-AUC={metrics['roc_auc']:.3f} PR-AUC={metrics['pr_auc']:.3f} F1@{metrics['threshold']:.2f}={metrics['f1_at_th']:.3f}")


if __name__ == "__main__":
    main()

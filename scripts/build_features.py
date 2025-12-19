# scripts/build_features.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

from src.features.dataset import build_dataset


def main():
    ap = argparse.ArgumentParser(description="Build CV + OHLCV features and deterministic labels.")
    ap.add_argument("--pipeline_cfg", default="configs/pipeline.yaml", help="Pipeline configuration file.")
    ap.add_argument("--ohlcv_root", default="data/ohlcv", help="Root directory for OHLCV parquet files.")
    ap.add_argument("--images_root", default="data/images/rendered", help="Root folder of rendered images (used when labels_csv provided).")
    ap.add_argument("--labels_csv", default="reports/labels/weak_labels.csv", help="Optional CSV enumerating images + metadata. Only metadata columns are used.")
    ap.add_argument("--manifest", default=None, help="Optional render manifest JSON to enumerate samples (takes precedence over labels_csv).")
    ap.add_argument("--out_dir", default="data/features", help="Directory for output feature CSVs.")
    # legacy placeholders so existing scripts don't break
    ap.add_argument("--backtest_cfg", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--features_cfg", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--labels_csv_legacy", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    manifest = Path(args.manifest) if args.manifest else None
    labels_csv = Path(args.labels_csv) if args.labels_csv else None
    outputs = build_dataset(
        pipeline_cfg_path=Path(args.pipeline_cfg),
        ohlcv_root=Path(args.ohlcv_root),
        images_root=Path(args.images_root),
        out_dir=Path(args.out_dir),
        labels_csv=labels_csv if manifest is None else None,
        manifest_path=manifest,
    )
    for split, path in outputs.items():
        print(f"✅ {split}: {path}")


if __name__ == "__main__":
    main()

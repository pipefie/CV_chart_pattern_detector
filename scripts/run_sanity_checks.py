# scripts/run_sanity_checks.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import itertools
import logging
from pathlib import Path

from src.features.dataset import (
    load_ohlcv,
    load_yaml,
    samples_from_labels_csv,
    samples_from_manifest,
    slice_window,
)
from src.features.cv_hough import extract_hough_features
from src.labeling import PatternLabeler


def main():
    ap = argparse.ArgumentParser(description="Quick sanity pipeline for a handful of samples.")
    ap.add_argument("--pipeline_cfg", default="configs/pipeline.yaml")
    ap.add_argument("--ohlcv_root", default="data/ohlcv")
    ap.add_argument("--images_root", default="data/images/rendered")
    ap.add_argument("--labels_csv", default="reports/labels/weak_labels.csv")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_yaml(args.pipeline_cfg)
    labeler = PatternLabeler(cfg.get("labeling", {}))
    ohlcv_root = Path(args.ohlcv_root)

    if args.manifest:
        samples = samples_from_manifest(Path(args.manifest))
    else:
        samples = samples_from_labels_csv(Path(args.labels_csv), Path(args.images_root))

    for sample in itertools.islice(samples, args.limit):
        df_symbol = load_ohlcv(sample.symbol, ohlcv_root)
        win = slice_window(df_symbol, sample.start_ts, sample.end_ts)
        swings = labeler.detect_swings(win)
        labels, structural = labeler.label_window(win)
        cv_feats = extract_hough_features(sample.image_path, (cfg.get("cv_features") or {}).get("hough", {}))

        print("=" * 80)
        print(f"{sample.image} | split={sample.split} symbol={sample.symbol} bars={sample.bars}")
        print(f"Labels: {labels}")
        print(f"Structural features: {structural}")
        print(f"Swing highs={int(swings['swing_high'].sum())} lows={int(swings['swing_low'].sum())}")
        print(f"CV features: {cv_feats}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare in-house detectors vs TA ZigZag reference + inspect edge thresholds."""
from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import cv2 as cv

try:
    import pandas_ta as pta
    HAS_PANDAS_TA = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PANDAS_TA = False

from scripts.generate_weak_labels import (
    load_yaml,
    load_ohlcv,
    slice_window,
    compute_atr,
    extract_pivots_with_prominence,
    detect_double_top,
    detect_double_bottom,
    detect_head_shoulders,
    detect_inverse_head_shoulders,
    detect_triangle,
    resolve_config,
    deep_merge_dict,
)

SUPPORTED = [
    "head_and_shoulders",
    "inverse_head_and_shoulders",
    "double_top",
    "double_bottom",
    "descending_triangle",
]


def ta_zigzag_pivots(df: pd.DataFrame, percent: float) -> pd.Series | None:
    """Return ZigZag pivots using pandas_ta (if available)."""
    if not HAS_PANDAS_TA:
        return None
    try:
        piv = df.ta.zigzag(high="high", low="low", close="close", percent=percent)
    except Exception as exc:  # pragma: no cover - external lib behavior
        print(f"⚠️ pandas_ta zigzag failed ({exc}); skipping TA baseline.")
        return None
    if piv is None or piv.empty:
        return None
    # flatten to single series if DataFrame
    if isinstance(piv, pd.DataFrame):
        piv = piv.iloc[:, 0]
    return piv.dropna()


def ta_double_top_matches(pivots: pd.Series, tolerance_pct: float = 0.015) -> int:
    """Detect double tops from pivot series (high-low-high)."""
    if pivots is None or pivots.empty:
        return 0
    matches = 0
    piv_items = list(pivots.items())
    peaks = [(idx, val) for pos, (idx, val) in enumerate(piv_items) if pos % 2 == 0]  # heuristic parity
    for (i1, p1), (i2, p2) in zip(peaks, peaks[1:]):
        mid = 0.5 * (p1 + p2)
        if mid == 0:
            continue
        if abs(p1 - p2) / mid <= tolerance_pct:
            matches += 1
    return matches


def ta_double_bottom_matches(pivots: pd.Series, tolerance_pct: float = 0.015) -> int:
    if pivots is None or pivots.empty:
        return 0
    matches = 0
    piv_items = list(pivots.items())
    valleys = [(idx, val) for pos, (idx, val) in enumerate(piv_items) if pos % 2 == 1]
    for (i1, p1), (i2, p2) in zip(valleys, valleys[1:]):
        mid = 0.5 * (p1 + p2)
        if mid == 0:
            continue
        if abs(p1 - p2) / max(1e-9, mid) <= tolerance_pct:
            matches += 1
    return matches


def run_detectors(df: pd.DataFrame,
                  symbol: str,
                  timeframe: str,
                  eff_common: dict,
                  eff_patterns: dict) -> Dict[str, List[dict]]:
    atr_cfg = eff_common.get("atr", {}) or {}
    piv_cfg = eff_common.get("pivots", {}) or {}
    units_cfg = eff_common.get("units", {}) or {}
    labels_post = eff_common.get("labels_post", {}) or {}

    atr_period = int(atr_cfg.get("period", 14))
    atr_smoothing = atr_cfg.get("smoothing", "rma")
    min_prom_atr = float(piv_cfg.get("min_prominence_atr", 0.7))
    min_dist_bars = int(piv_cfg.get("min_distance_bars", 8))
    smooth_window = int(piv_cfg.get("smoothing_window_bars", 1))
    max_pivots = int(piv_cfg.get("max_pivots_per_window", 0))
    max_pivots = max_pivots if max_pivots > 0 else None
    adaptive_prom_pct = float(piv_cfg.get("adaptive_prominence_pct_of_range", 0.0))
    prefer_atr = bool(units_cfg.get("prefer_atr_over_percent", True))

    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    v = df["volume"].to_numpy() if "volume" in df.columns else None
    atr = compute_atr(h, l, c, period=atr_period, mode=atr_smoothing)

    piv_hi = extract_pivots_with_prominence(
        close=c,
        min_prom_atr=min_prom_atr,
        min_dist_bars=min_dist_bars,
        atr=atr,
        highs=True,
        max_pivots=max_pivots,
        smooth_window=smooth_window,
        adaptive_prom_pct_of_range=adaptive_prom_pct,
    )
    piv_lo = extract_pivots_with_prominence(
        close=c,
        min_prom_atr=min_prom_atr,
        min_dist_bars=min_dist_bars,
        atr=atr,
        highs=False,
        max_pivots=max_pivots,
        smooth_window=smooth_window,
        adaptive_prom_pct_of_range=adaptive_prom_pct,
    )

    det_map = {name: [] for name in SUPPORTED}
    if eff_patterns.get("head_and_shoulders", {}).get("enabled", True):
        det_map["head_and_shoulders"] = detect_head_shoulders(
            c, v, piv_hi, eff_patterns.get("head_and_shoulders", {}), np.nanmean(atr), prefer_atr
        )
    if eff_patterns.get("inverse_head_and_shoulders", {}).get("enabled", True):
        det_map["inverse_head_and_shoulders"] = detect_inverse_head_shoulders(
            c,
            v,
            piv_lo,
            eff_patterns.get("inverse_head_and_shoulders", eff_patterns.get("head_and_shoulders", {})),
            np.nanmean(atr),
            prefer_atr,
        )
    if eff_patterns.get("double_top", {}).get("enabled", True):
        det_map["double_top"] = detect_double_top(c, v, piv_hi, eff_patterns.get("double_top", {}), atr, prefer_atr, labels_post)
    if eff_patterns.get("double_bottom", {}).get("enabled", True):
        cfg_db = deep_merge_dict(eff_patterns.get("double_top", {}), eff_patterns.get("double_bottom", {}))
        det_map["double_bottom"] = detect_double_bottom(c, v, piv_lo, cfg_db, atr, prefer_atr, labels_post)
    if eff_patterns.get("descending_triangle", {}).get("enabled", True):
        tri_cfg = eff_patterns.get("descending_triangle", {})
        det_map["descending_triangle"] = detect_triangle(c, piv_hi, piv_lo, tri_cfg.get("geometry", {}))

    return det_map


def analyze_edges(image_path: Path, blur: int, thresholds: List[int]):
    img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} not readable")
    if blur >= 3:
        img = cv.GaussianBlur(img, (blur, blur), 0)
    stats = []
    for t in thresholds:
        edges = cv.Canny(img, t, t * 2)
        nonzero = np.count_nonzero(edges)
        stats.append({"t1": t, "t2": t * 2, "edge_pixels": int(nonzero)})
    return stats


def main():
    ap = argparse.ArgumentParser(description="Validate pattern detectors vs TA ZigZag baseline.")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start", required=True, help="ISO date inclusive")
    ap.add_argument("--end", required=True, help="ISO date inclusive")
    ap.add_argument("--window", type=int, default=160, help="Bars per sliding window")
    ap.add_argument("--stride", type=int, default=40, help="Stride between windows")
    ap.add_argument("--patterns_cfg", default="configs/patterns.yaml")
    ap.add_argument("--output", default="reports/validation/detector_audit.csv")
    ap.add_argument("--ta_zigzag_percent", type=float, default=1.2, help="Percent move for TA ZigZag baseline")
    ap.add_argument("--edge_debug", type=Path, help="Optional path to PNG for Canny threshold sweep")
    ap.add_argument("--edge_blur", type=int, default=5)
    ap.add_argument("--edge_thresholds", nargs="+", type=int, default=[20, 30, 40, 60, 80])
    args = ap.parse_args()

    if args.edge_debug:
        stats = analyze_edges(args.edge_debug, args.edge_blur, args.edge_thresholds)
        print(json.dumps(stats, indent=2))
        if not args.symbols:
            return

    cfg = load_yaml(Path(args.patterns_cfg)) or {}
    common_base = cfg.get("common", {}) or {}
    patterns_base = cfg.get("patterns", {}) or {}
    overrides = cfg.get("overrides", []) or []
    pattern_keys = set(patterns_base.keys()).union(SUPPORTED)

    rows = []
    for symbol in args.symbols:
        df_full = load_ohlcv(symbol)
        df_slice = df_full.loc[args.start:args.end]
        if df_slice.empty:
            print(f"⚠️ No data for {symbol} in range.")
            continue

        eff_common, eff_patterns = resolve_config(common_base, patterns_base, overrides, symbol, args.timeframe, pattern_keys)
        if not eff_patterns:
            continue

        idx = df_slice.index
        for start_idx in range(0, len(idx) - args.window + 1, args.stride):
            end_idx = start_idx + args.window
            sub = df_slice.iloc[start_idx:end_idx]
            if len(sub) < args.window:
                continue
            det_map = run_detectors(sub, symbol, args.timeframe, eff_common, eff_patterns)
            pivots = ta_zigzag_pivots(sub[["high", "low", "close"]], percent=args.ta_zigzag_percent)
            ta_dt = ta_double_top_matches(pivots)
            ta_db = ta_double_bottom_matches(pivots)

            rows.append({
                "symbol": symbol,
                "timeframe": args.timeframe,
                "window_start": sub.index[0].isoformat(),
                "window_end": sub.index[-1].isoformat(),
                "our_double_top": len(det_map["double_top"]),
                "our_double_top_confirmed": sum(1 for d in det_map["double_top"] if d.get("breakout_confirmed", True)),
                "ta_double_top": ta_dt,
                "our_double_bottom": len(det_map["double_bottom"]),
                "our_double_bottom_confirmed": sum(1 for d in det_map["double_bottom"] if d.get("breakout_confirmed", True)),
                "ta_double_bottom": ta_db,
                "our_desc_tri": len(det_map["descending_triangle"]),
                "our_head_shoulders": len(det_map["head_and_shoulders"]),
                "our_inverse_head_shoulders": len(det_map["inverse_head_and_shoulders"]),
                "ta_source": "pandas_ta" if HAS_PANDAS_TA else "fallback",
            })

    if not rows:
        print("⚠️ No windows processed.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"✅ Detector audit saved to {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

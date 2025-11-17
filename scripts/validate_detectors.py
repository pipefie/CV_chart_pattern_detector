#!/usr/bin/env python3
"""Compare in-house detectors vs TA ZigZag reference + inspect edge thresholds."""
from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import cv2 as cv

try:
    import pandas_ta as pta
    HAS_PANDAS_TA = True
except Exception:
    HAS_PANDAS_TA = False

from scripts.generate_weak_labels import (
    load_yaml,
    load_ohlcv,
    slice_window,
    compute_atr,
    extract_pivots_with_prominence,
    detect_double_top_visual,
    detect_double_bottom_visual,
    detect_head_shoulders_visual,
    detect_inverse_head_shoulders_visual,
    resolve_config,
    deep_merge_dict,
    init_debug_log,
    log_debug_message,
    write_debug_log,
)

SUPPORTED = [
    "head_and_shoulders",
    "inverse_head_and_shoulders",
    "double_top", 
    "double_bottom",
    "descending_triangle",
]

@dataclass
class ValidationStats:
    """Statistics for pattern validation"""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def ta_zigzag_pivots(df: pd.DataFrame, percent: float) -> Tuple[pd.Series | None, pd.Series | None]:
    """Return ZigZag pivots separated into highs and lows."""
    if not HAS_PANDAS_TA:
        return None, None
    
    try:
        piv = df.ta.zigzag(high="high", low="low", close="close", percent=percent)
    except Exception as exc:
        print(f"⚠️ pandas_ta zigzag failed ({exc}); skipping TA baseline.")
        return None, None
    
    if piv is None or piv.empty:
        return None, None
    
    # Convert to DataFrame if series
    if isinstance(piv, pd.Series):
        piv = piv.to_frame()
    
    # Extract highs and lows - simpler approach
    piv_values = piv.dropna()
    if piv_values.empty:
        return None, None
    
    # Create integer index positions for the pivots
    high_pivots = []
    low_pivots = []
    
    # Get the integer positions of pivots in the dataframe
    for idx, value in piv_values.iloc[:, 0].items():
        position = df.index.get_loc(idx)  # Get integer position
        if len(high_pivots) <= len(low_pivots):
            high_pivots.append((position, float(value)))
        else:
            low_pivots.append((position, float(value)))
    
    # Convert to pandas Series for compatibility (using integer positions as index)
    high_series = pd.Series({pos: price for pos, price in high_pivots})
    low_series = pd.Series({pos: price for pos, price in low_pivots})
    
    return high_series, low_series

def ta_double_top_matches(highs: pd.Series, tolerance_pct: float = 0.015, min_separation: int = 10) -> List[Tuple]:
    """Detect double tops from pivot highs with proper pattern structure."""
    if highs is None or highs.empty or len(highs) < 2:
        return []
    
    matches = []
    # Convert series to list of (position, price) tuples
    high_items = [(int(idx), float(price)) for idx, price in highs.items()]
    
    for i in range(len(high_items) - 1):
        pos1, price1 = high_items[i]
        pos2, price2 = high_items[i + 1]
        
        # Check separation (using integer positions)
        if abs(pos1 - pos2) < min_separation:
            continue
            
        # Check price similarity
        mid_price = 0.5 * (price1 + price2)
        if abs(price1 - price2) / mid_price <= tolerance_pct:
            matches.append((pos1, price1, pos2, price2))
    
    return matches

def ta_double_bottom_matches(lows: pd.Series, tolerance_pct: float = 0.015, min_separation: int = 10) -> List[Tuple]:
    """Detect double bottoms from pivot lows with proper pattern structure."""
    if lows is None or lows.empty or len(lows) < 2:
        return []
    
    matches = []
    low_items = [(int(idx), float(price)) for idx, price in lows.items()]
    
    for i in range(len(low_items) - 1):
        pos1, price1 = low_items[i]
        pos2, price2 = low_items[i + 1]
        
        # Check separation (using integer positions)
        if abs(pos1 - pos2) < min_separation:
            continue
            
        # Check price similarity
        mid_price = 0.5 * (price1 + price2)
        if abs(price1 - price2) / mid_price <= tolerance_pct:
            matches.append((pos1, price1, pos2, price2))
    
    return matches

def ta_head_shoulders_matches(highs: pd.Series, tolerance_pct: float = 0.25) -> List[Tuple]:
    """Detect head and shoulders patterns from pivot highs."""
    if highs is None or highs.empty or len(highs) < 3:
        return []
    
    matches = []
    high_items = [(int(idx), float(price)) for idx, price in highs.items()]
    
    for i in range(len(high_items) - 2):
        left_pos, left_price = high_items[i]
        head_pos, head_price = high_items[i + 1] 
        right_pos, right_price = high_items[i + 2]
        
        # Check shoulder symmetry
        shoulder_similarity = abs(left_price - right_price) / max(left_price, right_price)
        if shoulder_similarity > tolerance_pct:
            continue
            
        # Check head prominence (head should be higher than shoulders)
        shoulder_avg = 0.5 * (left_price + right_price)
        if head_price <= shoulder_avg:
            continue
            
        matches.append((left_pos, left_price, head_pos, head_price, right_pos, right_price))
    
    return matches

def ta_inverse_head_shoulders_matches(lows: pd.Series, tolerance_pct: float = 0.25) -> List[Tuple]:
    """Detect inverse head and shoulders patterns from pivot lows."""
    if lows is None or lows.empty or len(lows) < 3:
        return []
    
    matches = []
    low_items = [(int(idx), float(price)) for idx, price in lows.items()]
    
    for i in range(len(low_items) - 2):
        left_pos, left_price = low_items[i]
        head_pos, head_price = low_items[i + 1]
        right_pos, right_price = low_items[i + 2]
        
        # Check shoulder symmetry
        shoulder_similarity = abs(left_price - right_price) / max(left_price, right_price)
        if shoulder_similarity > tolerance_pct:
            continue
            
        # Check head depth (head should be lower than shoulders)
        shoulder_avg = 0.5 * (left_price + right_price)
        if head_price >= shoulder_avg:
            continue
            
        matches.append((left_pos, left_price, head_pos, head_price, right_pos, right_price))
    
    return matches

def calculate_pattern_metrics(our_detections: List, ta_detections: List, window_size: int) -> ValidationStats:
    """Calculate precision, recall, and F1 score between our detections and TA baseline."""
    stats = ValidationStats()
    
    if not ta_detections and not our_detections:
        return stats  # No patterns to compare
    
    # Simple temporal matching (within window/4 bars)
    match_threshold = window_size // 4
    
    matched_ours = set()
    matched_ta = set()
    
    # Match our detections to TA detections
    for i, our_det in enumerate(our_detections):
        our_center = our_det.get('icenter', our_det.get('iH', 0))
        
        for j, ta_det in enumerate(ta_detections):
            # TA detections are tuples, first element is the position
            ta_center = ta_det[0]  # First element is the position
            
            if abs(our_center - ta_center) <= match_threshold:
                matched_ours.add(i)
                matched_ta.add(j)
                break
    
    stats.true_positives = len(matched_ours)
    stats.false_positives = len(our_detections) - len(matched_ours)
    stats.false_negatives = len(ta_detections) - len(matched_ta)
    
    # Calculate metrics
    if stats.true_positives + stats.false_positives > 0:
        stats.precision = stats.true_positives / (stats.true_positives + stats.false_positives)
    
    if stats.true_positives + stats.false_negatives > 0:
        stats.recall = stats.true_positives / (stats.true_positives + stats.false_negatives)
    
    if stats.precision + stats.recall > 0:
        stats.f1_score = 2 * (stats.precision * stats.recall) / (stats.precision + stats.recall)
    
    return stats

def run_detectors(df: pd.DataFrame,
                  symbol: str,
                  timeframe: str,
                  eff_common: dict,
                  eff_patterns: dict,
                  debug_log: list = None) -> Dict[str, List[dict]]:
    """Run all pattern detectors with enhanced debugging."""
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
    atr = compute_atr(h, l, c, period=atr_period, mode=atr_smoothing)
    atr_mean = float(np.nanmean(atr))

    # Extract pivots with debug logging
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
    
    # Run detectors with individual debug logs
    if eff_patterns.get("head_and_shoulders", {}).get("enabled", True):
        try:
            det_map["head_and_shoulders"] = detect_head_shoulders_visual(
                c, piv_hi, piv_lo,
                eff_patterns.get("head_and_shoulders", {}),
                atr_mean, prefer_atr, debug_log
            )
        except Exception as e:
            print(f"⚠️ Head & Shoulders detection failed for {symbol}: {e}")
            det_map["head_and_shoulders"] = []
        
    if eff_patterns.get("inverse_head_and_shoulders", {}).get("enabled", True):
        try:
            det_map["inverse_head_and_shoulders"] = detect_inverse_head_shoulders_visual(
                c, piv_lo, piv_hi,
                eff_patterns.get("inverse_head_and_shoulders", eff_patterns.get("head_and_shoulders", {})),
                atr_mean, prefer_atr, debug_log
            )
        except Exception as e:
            print(f"⚠️ Inverse Head & Shoulders detection failed for {symbol}: {e}")
            det_map["inverse_head_and_shoulders"] = []
        
    if eff_patterns.get("double_top", {}).get("enabled", True):
        try:
            det_map["double_top"] = detect_double_top_visual(
                c, piv_hi, eff_patterns.get("double_top", {}), 
                atr, prefer_atr, labels_post, debug_log
            )
        except Exception as e:
            print(f"⚠️ Double Top detection failed for {symbol}: {e}")
            det_map["double_top"] = []
        
    if eff_patterns.get("double_bottom", {}).get("enabled", True):
        try:
            cfg_db = deep_merge_dict(eff_patterns.get("double_top", {}), eff_patterns.get("double_bottom", {}))
            det_map["double_bottom"] = detect_double_bottom_visual(
                c, piv_lo, cfg_db, atr, prefer_atr, labels_post, debug_log
            )
        except Exception as e:
            print(f"⚠️ Double Bottom detection failed for {symbol}: {e}")
            det_map["double_bottom"] = []

    # Note: Descending triangle temporarily removed until visual implementation is complete
    if "descending_triangle" in det_map:
        det_map["descending_triangle"] = []

    return det_map

def analyze_edges(image_path: Path, blur: int, thresholds: List[int]):
    """Enhanced edge analysis with visual pattern metrics."""
    img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} not readable")
    
    if blur >= 3:
        img = cv.GaussianBlur(img, (blur, blur), 0)
    
    stats = []
    for t in thresholds:
        edges = cv.Canny(img, t, t * 2)
        nonzero = np.count_nonzero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]
        
        # Calculate edge density and connectivity
        edge_density = nonzero / total_pixels
        
        # Find contours to measure pattern complexity
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)
        
        stats.append({
            "t1": t, 
            "t2": t * 2, 
            "edge_pixels": int(nonzero),
            "edge_density": float(edge_density),
            "contour_count": contour_count,
            "avg_contour_size": float(np.mean([len(c) for c in contours])) if contours else 0
        })
    
    return stats

def generate_validation_report(rows: List[dict], output_path: Path):
    """Generate comprehensive validation report with statistics."""
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("⚠️ No validation data to report.")
        return
    
    # Calculate overall statistics
    report = {
        "total_windows": len(df),
        "symbols_analyzed": df['symbol'].nunique(),
    }
    
    # Pattern-specific statistics
    patterns = ['double_top', 'double_bottom', 'head_shoulders', 'inverse_head_shoulders']
    
    for pattern in patterns:
        our_col = f'our_{pattern}'
        ta_col = f'ta_{pattern}'
        
        if our_col in df.columns and ta_col in df.columns:
            our_total = int(df[our_col].sum())
            ta_total = int(df[ta_col].sum())
            
            report[f'{pattern}_our_detections'] = our_total
            report[f'{pattern}_ta_detections'] = ta_total
            if ta_total > 0:
                report[f'{pattern}_detection_ratio'] = float(our_total / ta_total)
            else:
                report[f'{pattern}_detection_ratio'] = float('inf') if our_total > 0 else 0.0
    
    # Detection consistency
    consistency_metrics = {}
    for pattern in ['double_top', 'double_bottom', 'head_shoulders', 'inverse_head_shoulders']:
        our_col = f'our_{pattern}'
        ta_col = f'ta_{pattern}'
        
        if our_col in df.columns and ta_col in df.columns:
            # Windows where both detectors found patterns
            agreement = int(((df[our_col] > 0) & (df[ta_col] > 0)).sum())
            total_with_patterns = int(((df[our_col] > 0) | (df[ta_col] > 0)).sum())
            
            if total_with_patterns > 0:
                consistency = float(agreement / total_with_patterns)
                consistency_metrics[f'{pattern}_agreement'] = consistency
    
    report.update(consistency_metrics)
    
    # Convert all numpy types to native Python types
    report = convert_numpy_types(report)
    
    # Save detailed report
    report_path = output_path.with_suffix('.validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Validation report saved to {report_path}")
    print("\n📊 Validation Summary:")
    for key, value in report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

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
    ap.add_argument("--debug_log", action="store_true", help="Generate debug log for detection issues")
    ap.add_argument("--metrics", action="store_true", help="Calculate precision/recall metrics")
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
    debug_log = init_debug_log() if args.debug_log else None
    
    for symbol in args.symbols:
        print(f"🔍 Analyzing {symbol}...")
        try:
            df_full = load_ohlcv(symbol)
            df_slice = df_full.loc[args.start:args.end]
            if df_slice.empty:
                print(f"⚠️ No data for {symbol} in range {args.start} to {args.end}.")
                continue

            eff_common, eff_patterns = resolve_config(common_base, patterns_base, overrides, symbol, args.timeframe, pattern_keys)
            if not eff_patterns:
                print(f"⚠️ No effective patterns for {symbol}")
                continue

            idx = df_slice.index
            window_count = 0
            for start_idx in range(0, len(idx) - args.window + 1, args.stride):
                end_idx = start_idx + args.window
                sub = df_slice.iloc[start_idx:end_idx]
                if len(sub) < args.window:
                    continue
                    
                det_map = run_detectors(sub, symbol, args.timeframe, eff_common, eff_patterns, debug_log)
                
                # Get TA ZigZag pivots (separated into highs and lows)
                highs, lows = ta_zigzag_pivots(sub[["high", "low", "close"]], percent=args.ta_zigzag_percent)
                
                # Get TA pattern matches
                ta_dt_matches = ta_double_top_matches(highs) if highs is not None else []
                ta_db_matches = ta_double_bottom_matches(lows) if lows is not None else []
                ta_hs_matches = ta_head_shoulders_matches(highs) if highs is not None else []
                ta_ihs_matches = ta_inverse_head_shoulders_matches(lows) if lows is not None else []

                row = {
                    "symbol": symbol,
                    "timeframe": args.timeframe,
                    "window_start": sub.index[0].isoformat(),
                    "window_end": sub.index[-1].isoformat(),
                    "window_size": args.window,
                    "our_double_top": len(det_map["double_top"]),
                    "our_double_top_confirmed": sum(1 for d in det_map["double_top"] if d.get("breakout_confirmed", True)),
                    "ta_double_top": len(ta_dt_matches),
                    "our_double_bottom": len(det_map["double_bottom"]),
                    "our_double_bottom_confirmed": sum(1 for d in det_map["double_bottom"] if d.get("breakout_confirmed", True)),
                    "ta_double_bottom": len(ta_db_matches),
                    "our_head_shoulders": len(det_map["head_and_shoulders"]),
                    "ta_head_shoulders": len(ta_hs_matches),
                    "our_inverse_head_shoulders": len(det_map["inverse_head_and_shoulders"]),
                    "ta_inverse_head_shoulders": len(ta_ihs_matches),
                    "ta_source": "pandas_ta" if HAS_PANDAS_TA else "fallback",
                }

                # Calculate metrics if requested
                if args.metrics:
                    dt_metrics = calculate_pattern_metrics(det_map["double_top"], ta_dt_matches, args.window)
                    db_metrics = calculate_pattern_metrics(det_map["double_bottom"], ta_db_matches, args.window)
                    hs_metrics = calculate_pattern_metrics(det_map["head_and_shoulders"], ta_hs_matches, args.window)
                    ihs_metrics = calculate_pattern_metrics(det_map["inverse_head_and_shoulders"], ta_ihs_matches, args.window)
                    
                    row.update({
                        "dt_precision": float(dt_metrics.precision),
                        "dt_recall": float(dt_metrics.recall),
                        "dt_f1": float(dt_metrics.f1_score),
                        "db_precision": float(db_metrics.precision),
                        "db_recall": float(db_metrics.recall),
                        "db_f1": float(db_metrics.f1_score),
                        "hs_precision": float(hs_metrics.precision),
                        "hs_recall": float(hs_metrics.recall),
                        "hs_f1": float(hs_metrics.f1_score),
                        "ihs_precision": float(ihs_metrics.precision),
                        "ihs_recall": float(ihs_metrics.recall),
                        "ihs_f1": float(ihs_metrics.f1_score),
                    })

                rows.append(row)
                window_count += 1
            
            print(f"✅ Processed {window_count} windows for {symbol}")
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            continue

    if not rows:
        print("⚠️ No windows processed.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    df_results = pd.DataFrame(rows)
    df_results.to_csv(out_path, index=False)
    print(f"✅ Detector audit saved to {out_path} ({len(rows)} rows)")
    
    # Generate validation report
    generate_validation_report(rows, out_path)
    
    # Save debug log if enabled
    if debug_log and args.debug_log:
        debug_path = out_path.with_suffix('.debug.jsonl')
        write_debug_log(debug_log, debug_path)
        print(f"✅ Debug log saved to {debug_path}")

if __name__ == "__main__":
    main()
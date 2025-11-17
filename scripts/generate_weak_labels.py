# scripts/generate_weak_labels.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from copy import deepcopy
from fnmatch import fnmatch
import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks
import math
import cv2 as cv


# ---------- Helpers ----------
def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def load_meta(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))

def load_ohlcv(symbol: str) -> pd.DataFrame:
    root = Path("data/ohlcv")
    if (root / "equities_etf" / f"{symbol}.parquet").exists():
        p = root / "equities_etf" / f"{symbol}.parquet"
    else:
        p = root / "crypto" / f"{symbol}.parquet"
    df = pd.read_parquet(p)
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df

def init_debug_log() -> list:
    """Initialize an empty debug log"""
    return []

def log_debug_message(debug_log: list, pattern_type: str, reason: str, 
                     details: dict = None, indices: tuple = None):
    """Add a debug message to the log"""
    if debug_log is not None:
        entry = {
            "pattern": pattern_type,
            "reason": reason,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        if details:
            entry.update(details)
        if indices:
            entry["indices"] = indices
        debug_log.append(entry)

def write_debug_log(debug_log: list, output_path: Path):
    """Write debug log to JSONL file"""
    if debug_log:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in debug_log:
                f.write(json.dumps(entry) + "\n")

def slice_window(df: pd.DataFrame, start_iso: str, end_iso: str) -> pd.DataFrame:
    s = pd.Timestamp(start_iso); e = pd.Timestamp(end_iso)
    if s.tzinfo is None: s = s.tz_localize("UTC")
    else: s = s.tz_convert("UTC")
    if e.tzinfo is None: e = e.tz_localize("UTC")
    else: e = e.tz_convert("UTC")
    return df.loc[(df.index >= s) & (df.index <= e)]

def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")

def deep_merge_dict(base: dict | None, override: dict | None) -> dict:
    if base is None:
        base = {}
    if override is None:
        return deepcopy(base)
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge_dict(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result

def override_matches(match_cfg: dict | None, symbol: str, timeframe: str) -> bool:
    if not match_cfg:
        return False
    tf = match_cfg.get("timeframe")
    if tf and tf != timeframe:
        return False
    symbols = match_cfg.get("symbols")
    if symbols:
        if not any(fnmatch(symbol, pat) for pat in symbols):
            return False
    return True

COMMON_SECTION_KEYS = {
    "atr",
    "units",
    "pivots",
    "lines",
    "windows",
    "nms",
    "labels_post",
}

def resolve_config(common_base: dict,
                   patterns_base: dict,
                   overrides: list[dict],
                   symbol: str,
                   timeframe: str,
                   pattern_keys: set[str]) -> tuple[dict, dict]:
    common = deepcopy(common_base or {})
    patterns = deepcopy(patterns_base or {})
    for ov in overrides or []:
        if not override_matches(ov.get("match"), symbol, timeframe):
            continue
        for key, val in ov.items():
            if key == "match":
                continue
            if key in pattern_keys:
                patterns[key] = deep_merge_dict(patterns.get(key, {}), val)
            elif key in COMMON_SECTION_KEYS or key in (common_base or {}):
                common[key] = deep_merge_dict(common.get(key, {}), val)
            else:
                # fall back to treating unknown keys as common tweaks
                common[key] = deep_merge_dict(common.get(key, {}), val)
    return common, patterns

# Pixel mapping (used for YOLO bboxes)
def bar_to_x(i: int, N: int, W: int) -> int:
    if N <= 1: return (W - 1) // 2
    return int(round((i / (N - 1)) * (W - 1)))

def price_to_y(p: float, y_min: float, y_max: float, H: int) -> int:
    y_rel = (p - y_min) / max(1e-9, (y_max - y_min))
    return int(round((1.0 - y_rel) * (H - 1)))

def write_yolo(path: Path, boxes: list[tuple[int, float, float, float, float]]):
    with open(path, "w", encoding="utf-8") as f:
        for cid, cx, cy, w, h in boxes:
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# --- ATR & smoothing ---
def compute_atr(high, low, close, period=14, mode="rma"):
    # TR
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    # smoothing
    if mode == "rma":
        alpha = 1.0 / period
        out = np.empty_like(tr, dtype=float)
        out[0] = tr[0]
        for i in range(1, len(tr)):
            out[i] = alpha * tr[i] + (1 - alpha) * out[i-1]
        return out
    elif mode == "ema":
        alpha = 2.0 / (period + 1)
        out = np.empty_like(tr, dtype=float)
        out[0] = tr[0]
        for i in range(1, len(tr)):
            out[i] = alpha * tr[i] + (1 - alpha) * out[i-1]
        return out
    else:  # sma
        k = int(period)
        if k <= 1: return tr
        return np.convolve(tr, np.ones(k, dtype=float)/k, mode="same")

# --- Pivot extraction via scipy.signal.find_peaks with ATR-scaled prominence ---
def extract_pivots_with_prominence(close: np.ndarray,
                                   min_prom_atr: float,
                                   min_dist_bars: int,
                                   atr: np.ndarray,
                                   highs: bool = True,
                                   max_pivots: int | None = None,
                                   smooth_window: int = 1,
                                   adaptive_prom_pct_of_range: float = 0.0) -> list[tuple[int, float]]:
    """find_peaks on close (for highs) or on -close (for lows), with prominence in ATR units."""
    # convert ATR-threshold to data units (≈ average ATR over window)
    atr_mean = float(np.nanmean(atr)) if atr is not None and len(atr) else 0.0
    prom_abs = max(1e-9, min_prom_atr * atr_mean)
    if adaptive_prom_pct_of_range > 0.0:
        price_range = float(np.nanmax(close) - np.nanmin(close))
        prom_abs = max(prom_abs, adaptive_prom_pct_of_range * max(price_range, 1e-9))

    series = close if highs else (-close)
    series = _smooth_series(series, smooth_window)
    peaks, props = find_peaks(series, prominence=prom_abs, distance=max(1, int(min_dist_bars)))
    if max_pivots and len(peaks) > max_pivots:
        prominences = props.get("prominences")
        if prominences is not None and len(prominences) == len(peaks):
            order = np.argsort(prominences)[::-1]
        else:
            order = np.argsort(series[peaks])[::-1]
        keep = np.sort(peaks[order[:max_pivots]])
        peaks = keep
    piv = []
    for idx in sorted(peaks):
        price = close[idx]  # note: for lows we used -close to find 'peaks', but keep true price
        piv.append((int(idx), float(price)))
    return piv

# --- Threshold chooser (ATR or %) ---
def _choose_abs_threshold(value_atr, value_pct, atr_mean, mid_price, prefer_atr=True) -> float:
    if prefer_atr and value_atr is not None:
        return float(value_atr) * atr_mean
    if value_pct is not None:
        return float(value_pct) * max(1e-9, mid_price)
    return 0.003 * max(1e-9, mid_price)  # tiny fallback

# --- Breakout confirmation relative to a level ---
def _level_breakout_confirm(close_future: np.ndarray, level: float, side: str, thr_abs: float, within_bars: int | None) -> bool:
    rng = range(0, min(len(close_future), within_bars)) if within_bars else range(0, len(close_future))
    if side == "down":
        return any(close_future[k] <= level - thr_abs for k in rng)
    else:
        return any(close_future[k] >= level + thr_abs for k in rng)

def _nms_time(dets: list[dict], dedup_bars: int) -> list[dict]:
    """Simple 1D NMS over time: keep best score, drop overlapping (|i_center diff| < dedup_bars)."""
    dets = sorted(dets, key=lambda d: -d.get("score", 0.0))
    kept = []
    centers = []
    for d in dets:
        ic = d.get("icenter", None)
        if ic is None:
            kept.append(d); continue
        if any(abs(ic - kc) < dedup_bars for kc in centers):
            continue
        centers.append(ic)
        kept.append(d)
    return kept

def _apply_dynamic_floor(base_abs: float,
                         dyn_cfg: dict | None,
                         pct_key: str,
                         pattern_height: float,
                         price_ref: float,
                         atr_recent: float) -> float:
    val = base_abs
    if not dyn_cfg:
        return val
    pct = float(dyn_cfg.get(pct_key, 0.0)) / 100.0
    if pct > 0.0:
        val = max(val, pct * max(pattern_height, 1e-9))
    atr_mult = float(dyn_cfg.get("atr_floor_mult", 0.0))
    if atr_mult > 0.0:
        val = max(val, atr_mult * max(atr_recent, 1e-9))
    return val


# --- Robust Double Bottom ---

def detect_double_bottom_visual(close: np.ndarray,
                               piv_lo: list,
                               cfg_db: dict,
                               atr: np.ndarray,
                               prefer_atr: bool,
                               labels_post: dict = None,
                               debug_log: list = None) -> list[dict]:
    """Enhanced double bottom detection focusing on visual 'W' shape"""
    out = []
    atr_mean = float(np.nanmean(atr)) if len(atr) else 0.0
    
    geom = cfg_db.get("geometry", {})
    visual_cfg = cfg_db.get("visual", {})
    
    dur = geom.get("duration", {})
    min_span = int(dur.get("min_bars", 22))
    max_span = int(dur.get("max_bars", 95))
    
    sim_tol = float(geom.get("peak_height_similarity_pct", 6)) / 100.0
    peak_sep = int(geom.get("valley_min_bars_from_peaks", 2))
    max_pairs = max(1, int(geom.get("max_pairs_per_peak", 5)))
    
    # Allow wider separations so long "W" patterns are not discarded
    min_trough_sep = visual_cfg.get("min_trough_separation_bars", 8)
    max_trough_sep = visual_cfg.get("max_trough_separation_bars", 100)
    
    br = cfg_db.get("breakout", {})
    brc = br.get("confirm", {})
    br_thr_atr = brc.get("threshold_atr")
    br_thr_pct = brc.get("threshold_percent")
    br_within = brc.get("within_bars")
    require_confirmation = bool(br.get("require_confirmation", True))
    allow_pre_breakout = bool(cfg_db.get("label_allow_pre_breakout", False))
    
    dyn = geom.get("dynamic_thresholds", {})
    dyn_lookback = int(dyn.get("vol_lookback_bars", 36))

    def _recent_atr():
        if len(atr) == 0: return atr_mean
        return float(np.nanmean(atr[-dyn_lookback:])) if dyn_lookback > 0 else atr_mean

    piv_sorted = sorted(piv_lo, key=lambda t: t[0])
    total = len(piv_sorted)
    
    for idx1, (i1, p1) in enumerate(piv_sorted):
        limit = min(total, idx1 + 1 + max_pairs)
        for idx2 in range(idx1 + 1, limit):
            i2, p2 = piv_sorted[idx2]
            
            # Visual trough separation
            trough_sep = i2 - i1
            if trough_sep < min_trough_sep or trough_sep > max_trough_sep:
                log_debug_message(debug_log, "double_bottom", "invalid_trough_separation",
                                {"separation": trough_sep, "min_sep": min_trough_sep, "max_sep": max_trough_sep},
                                (i1, i2))
                continue
                
            if i2 <= i1 + peak_sep: 
                log_debug_message(debug_log, "double_bottom", "troughs_too_close",
                                {"peak_sep_required": peak_sep},
                                (i1, i2))
                continue
            
            span = i2 - i1
            if span < min_span or span > max_span:
                log_debug_message(debug_log, "double_bottom", "invalid_span",
                                {"span": span, "min_span": min_span, "max_span": max_span},
                                (i1, i2))
                continue
            
            # Visual trough alignment
            mid_price = 0.5 * (p1 + p2)
            height_diff_pct = abs(p1 - p2) / max(1e-9, mid_price)
            if height_diff_pct > sim_tol:
                log_debug_message(debug_log, "double_bottom", "trough_height_mismatch",
                                {"height_diff_pct": height_diff_pct * 100, "max_allowed_pct": sim_tol * 100},
                                (i1, i2))
                continue

            # Find peak for "W" shape
            j0, j1 = i1 + peak_sep, i2 - peak_sep
            if j1 <= j0: 
                log_debug_message(debug_log, "double_bottom", "invalid_peak_range",
                                {"j0": j0, "j1": j1},
                                (i1, i2))
                continue
            
            local = close[j0:j1]
            if len(local) == 0: 
                log_debug_message(debug_log, "double_bottom", "no_peak_data",
                                {"j0": j0, "j1": j1},
                                (i1, i2))
                continue
            
            peak_idx = int(np.argmax(local) + j0)
            peak_price = float(close[peak_idx])
            
            # Visual peak height check
            peak_ratio = (peak_price - mid_price) / mid_price
            min_peak_ratio = visual_cfg.get("min_peak_to_trough_ratio", 0.008)
            if peak_ratio < min_peak_ratio:
                log_debug_message(debug_log, "double_bottom", "insufficient_peak_height",
                                {"peak_ratio": peak_ratio, "min_required": min_peak_ratio},
                                (i1, i2))
                continue
            
            # Visual pattern height
            pattern_height = peak_price - min(p1, p2)
            min_pattern_height = geom.get("min_pattern_height_atr", 1.0) * atr_mean
            if pattern_height < min_pattern_height:
                log_debug_message(debug_log, "double_bottom", "insufficient_pattern_height",
                                {"pattern_height": pattern_height, "min_required": min_pattern_height},
                                (i1, i2))
                continue
            
            # Validate "W" shape visually
            w_shape_valid = False
            if visual_cfg.get("require_w_shape", True):
                w_shape_valid = _validate_w_shape(close, i1, i2, peak_idx, p1, p2, peak_price)
            
            if not w_shape_valid:
                log_debug_message(debug_log, "double_bottom", "invalid_w_shape", {}, (i1, i2))
                continue
            
            # Breakout confirmation
            breakout_ok = True
            if brc:
                base_thr = _choose_abs_threshold(br_thr_atr, br_thr_pct, atr_mean, mid_price, prefer_atr)
                recent_atr_val = _recent_atr()
                thr_abs = _apply_dynamic_floor(
                    base_thr, dyn, "breakout_height_pct",
                    pattern_height, mid_price, recent_atr_val
                )
                
                if br_within:
                    future = close[i2+1 : i2+1 + br_within]
                else:
                    future = close[i2+1:]
                
                if len(future) == 0:
                    log_debug_message(debug_log, "double_bottom", "no_future_data", {}, (i1, i2))
                    continue
                
                breakout_ok = _level_breakout_confirm(future, level=peak_price, side="up", 
                                                     thr_abs=thr_abs, within_bars=br_within)
            
            if not breakout_ok and require_confirmation and not allow_pre_breakout:
                log_debug_message(debug_log, "double_bottom", "no_breakout_confirmation",
                                {"threshold": thr_abs, "within_bars": br_within},
                                (i1, i2))
                continue

            # Visual scoring
            score = _score_double_bottom_visual(p1, p2, peak_price, i1, i2, span, visual_cfg)
            
            out.append({
                "type": "double_bottom", "i1": i1, "p1": p1, "i2": i2, "p2": p2,
                "ipeak": peak_idx, "ppeak": peak_price, "span": span, "score": score,
                "icenter": int(0.5*(i1+i2)), "breakout_confirmed": bool(breakout_ok)
            })
    
    dedup = int((labels_post or {}).get("dedup_time_overlap_bars", 10))
    return _nms_time(out, dedup)

def _validate_w_shape(close: np.ndarray, i1: int, i2: int, peak_idx: int, p1: float, p2: float, peak_price: float) -> bool:
    """Validate visual 'W' shape characteristics"""
    # Check that prices ascend into peak and descend out
    left_ascend_count = 0
    right_descend_count = 0
    
    # Check left side (i1 to peak_idx)
    if peak_idx - i1 > 1:
        for i in range(i1, peak_idx-1):
            if close[i] <= close[i+1]:
                left_ascend_count += 1
    
    # Check right side (peak_idx to i2)  
    if i2 - peak_idx > 1:
        for i in range(peak_idx, i2-1):
            if close[i] >= close[i+1]:
                right_descend_count += 1
    
    left_ratio = left_ascend_count / max(1, (peak_idx - i1 - 1))
    right_ratio = right_descend_count / max(1, (i2 - peak_idx - 1))
    
    # Slightly relax monotonicity to keep visually plausible but noisy moves
    return left_ratio > 0.5 and right_ratio > 0.5

def _score_double_bottom_visual(p1: float, p2: float, peak_price: float, i1: int, i2: int, span: int, visual_cfg: dict) -> float:
    """Score double bottom based on visual quality"""
    score = 0.0
    
    # Trough alignment score
    trough_similarity = 1.0 - (abs(p1 - p2) / max(p1, p2))
    score += trough_similarity * visual_cfg.get("weight_trough_alignment", 1.2)
    
    # Peak rise score
    mid_price = 0.5 * (p1 + p2)
    peak_rise = (peak_price - mid_price) / mid_price
    score += min(peak_rise * 25, 1.0) * visual_cfg.get("weight_peak_rise", 0.6)
    
    # Pattern height score
    pattern_height = peak_price - min(p1, p2)
    height_score = min(pattern_height / (mid_price * 0.08), 1.0)
    score += height_score * visual_cfg.get("weight_pattern_height", 0.8)
    
    return score / 3.0

# --- VISUAL Head & Shoulders Detection ---
# --- COMPLETE VISUAL Head & Shoulders Detection ---
def detect_head_shoulders_visual(closes: np.ndarray,
                                piv_hi: list,
                                piv_lo: list,
                                cfg: dict,
                                atr_mean: float,
                                prefer_atr: bool,
                                debug_log: list = None) -> list[dict]:
    """Enhanced H&S detection focusing on visual characteristics"""
    out = []
    geom = cfg.get("geometry", {})
    visual_cfg = cfg.get("visual", {})
    
    # Visual constraints
    shoulder_sim = float(geom.get("shoulder_height_similarity_pct", 35)) / 100.0
    timing_sim = float(geom.get("shoulder_timing_similarity_pct", 70)) / 100.0
    # Align default with config (allows flatter heads)
    min_head_ratio = float(visual_cfg.get("min_head_to_shoulder_ratio", 1.02))
    
    dur = geom.get("duration", {})
    min_span = int(dur.get("min_bars", 30))
    max_span = int(dur.get("max_bars", 200))
    
    piv_sorted = sorted(piv_hi, key=lambda t: t[0])
    
    for iL, pL in piv_sorted:
        for iH, pH in piv_sorted:
            if iH <= iL + 2: 
                continue
            for iR, pR in piv_sorted:
                if iR <= iH + 2: 
                    continue
                
                span = iR - iL
                if span < min_span or span > max_span:
                    log_debug_message(debug_log, "head_and_shoulders", "invalid_span", 
                                    {"span": span, "min_span": min_span, "max_span": max_span},
                                    (iL, iH, iR))
                    continue
                
                # Visual symmetry check
                shoulder_avg = 0.5 * (pL + pR)
                if abs(pL - pR) / max(1e-9, shoulder_avg) > shoulder_sim:
                    log_debug_message(debug_log, "head_and_shoulders", "shoulder_height_asymmetry",
                                    {"height_diff_pct": abs(pL - pR) / shoulder_avg * 100},
                                    (iL, iH, iR))
                    continue
                
                # Visual head prominence
                head_need = _choose_abs_threshold(
                    geom.get("head_above_shoulders_min_atr"),
                    geom.get("head_above_shoulders_min_pct"),
                    atr_mean, shoulder_avg, prefer_atr
                )
                
                # Enhanced visual check: head must be clearly taller
                head_to_shoulder_ratio = pH / shoulder_avg
                head_prominence = pH - shoulder_avg
                
                if (head_prominence < head_need) or (head_to_shoulder_ratio < min_head_ratio):
                    log_debug_message(debug_log, "head_and_shoulders", "insufficient_head_prominence",
                                    {"head_prominence": head_prominence, "required": head_need,
                                     "head_ratio": head_to_shoulder_ratio, "min_ratio": min_head_ratio},
                                    (iL, iH, iR))
                    continue
                
                # Visual timing symmetry
                ideal_right_pos = iL + 2 * (iH - iL) / 3  # Right shoulder at 2/3 position
                tol_bars = max(3, int(timing_sim * span))
                if abs(iR - ideal_right_pos) > tol_bars:
                    log_debug_message(debug_log, "head_and_shoulders", "timing_asymmetry",
                                    {"actual_pos": iR, "ideal_pos": ideal_right_pos, "tolerance": tol_bars},
                                    (iL, iH, iR))
                    continue
                
                # Find neckline points (troughs between shoulders)
                neckline_points = []
                # Left trough between left shoulder and head
                if iL + 1 < iH:
                    left_trough_range = closes[iL+1:iH]
                    if len(left_trough_range) > 0:
                        left_trough_idx = iL + 1 + np.argmin(left_trough_range)
                        neckline_points.append((left_trough_idx, closes[left_trough_idx]))
                
                # Right trough between head and right shoulder  
                if iH + 1 < iR:
                    right_trough_range = closes[iH+1:iR]
                    if len(right_trough_range) > 0:
                        right_trough_idx = iH + 1 + np.argmin(right_trough_range)
                        neckline_points.append((right_trough_idx, closes[right_trough_idx]))
                
                # Validate neckline visually
                neckline_valid = False
                if len(neckline_points) >= 2:
                    neckline_valid = _validate_neckline_visual(neckline_points, geom.get("neckline", {}))
                
                if not neckline_valid:
                    log_debug_message(debug_log, "head_and_shoulders", "invalid_neckline",
                                    {"neckline_points": len(neckline_points)},
                                    (iL, iH, iR))
                    continue
                
                score = _score_hs_visual(pL, pH, pR, iL, iH, iR, span, visual_cfg)
                
                out.append({
                    "type": "head_and_shoulders", 
                    "iL": iL, "pL": pL, "iH": iH, "pH": pH, "iR": iR, "pR": pR,
                    "score": score, "icenter": iH,
                    "neckline_points": neckline_points
                })
    
    return out

def _validate_neckline_visual(neckline_points: list, neckline_cfg: dict) -> bool:
    """Validate neckline visual characteristics"""
    if len(neckline_points) < 2:
        return False
    
    indices = np.array([p[0] for p in neckline_points])
    prices = np.array([p[1] for p in neckline_points])
    
    # Fit line
    if len(indices) >= 2:
        slope, intercept = np.polyfit(indices, prices, 1)
        predicted = slope * indices + intercept
        residuals = np.abs(prices - predicted)
        
        # Check line fit quality
        max_rmsd = neckline_cfg.get("max_rmsd_pct_of_height", 0.35)
        price_range = np.max(prices) - np.min(prices)
        if price_range > 0 and np.mean(residuals) / price_range > max_rmsd:
            return False
        
        # Check slope
        max_slope_deg = neckline_cfg.get("max_slope_deg", 10)
        # Calculate approximate slope in degrees (simplified)
        if len(indices) > 1:
            x_range = indices[-1] - indices[0]
            if x_range > 0:
                slope_deg = math.degrees(math.atan(slope * x_range / max(price_range, 1e-9)))
                if abs(slope_deg) > max_slope_deg:
                    return False
    
    return True

def _score_hs_visual(pL: float, pH: float, pR: float, iL: int, iH: int, iR: int, span: int, visual_cfg: dict) -> float:
    """Score H&S pattern based on visual quality"""
    score = 0.0
    
    # Symmetry score
    shoulder_symmetry = 1.0 - (abs(pL - pR) / max(pL, pR))
    score += shoulder_symmetry * visual_cfg.get("weight_symmetry", 0.8)
    
    # Head prominence score
    shoulder_avg = 0.5 * (pL + pR)
    head_prominence = (pH - shoulder_avg) / shoulder_avg
    score += min(head_prominence * 10, 1.0) * visual_cfg.get("weight_head_prominence", 1.2)
    
    # Timing symmetry score
    ideal_right_pos = iL + 2 * (iH - iL) / 3
    timing_symmetry = 1.0 - min(abs(iR - ideal_right_pos) / (span * 0.5), 1.0)
    score += timing_symmetry * visual_cfg.get("weight_symmetry", 0.8)
    
    return score / 3.0  # Normalize

# --- VISUAL Double Top Detection ---
# --- COMPLETE VISUAL Double Top Detection ---
def detect_double_top_visual(close: np.ndarray,
                            piv_hi: list,
                            cfg_dt: dict,
                            atr: np.ndarray,
                            prefer_atr: bool,
                            labels_post: dict = None,
                            debug_log: list = None) -> list[dict]:
    """Enhanced double top detection focusing on visual 'M' shape"""
    out = []
    atr_mean = float(np.nanmean(atr)) if len(atr) else 0.0
    
    geom = cfg_dt.get("geometry", {})
    visual_cfg = cfg_dt.get("visual", {})
    
    dur = geom.get("duration", {})
    min_span = int(dur.get("min_bars", 20))
    max_span = int(dur.get("max_bars", 90))
    
    sim_tol = float(geom.get("peak_height_similarity_pct", 5)) / 100.0
    valley_sep = int(geom.get("valley_min_bars_from_peaks", 2))
    max_pairs = max(1, int(geom.get("max_pairs_per_peak", 4)))
    
    min_peak_sep = geom.get("min_peak_separation_bars", 10)
    max_peak_sep = geom.get("max_peak_separation_bars", 60)
    
    br = cfg_dt.get("breakout", {})
    brc = br.get("confirm", {})
    br_thr_atr = brc.get("threshold_atr")
    br_thr_pct = brc.get("threshold_percent")
    br_within = brc.get("within_bars")
    require_confirmation = bool(br.get("require_confirmation", True))
    allow_pre_breakout = bool(cfg_dt.get("label_allow_pre_breakout", False))
    
    dyn = geom.get("dynamic_thresholds", {})
    dyn_lookback = int(dyn.get("vol_lookback_bars", 36))

    def _recent_atr():
        if len(atr) == 0: return atr_mean
        return float(np.nanmean(atr[-dyn_lookback:])) if dyn_lookback > 0 else atr_mean

    piv_sorted = sorted(piv_hi, key=lambda t: t[0])
    total = len(piv_sorted)
    
    for idx1, (i1, p1) in enumerate(piv_sorted):
        limit = min(total, idx1 + 1 + max_pairs)
        for idx2 in range(idx1 + 1, limit):
            i2, p2 = piv_sorted[idx2]
            
            # Visual peak separation
            peak_sep = i2 - i1
            if peak_sep < min_peak_sep or peak_sep > max_peak_sep:
                log_debug_message(debug_log, "double_top", "invalid_peak_separation",
                                {"separation": peak_sep, "min_sep": min_peak_sep, "max_sep": max_peak_sep},
                                (i1, i2))
                continue
                
            if i2 <= i1 + valley_sep: 
                log_debug_message(debug_log, "double_top", "peaks_too_close", 
                                {"valley_sep_required": valley_sep},
                                (i1, i2))
                continue
            
            span = i2 - i1
            if span < min_span or span > max_span:
                log_debug_message(debug_log, "double_top", "invalid_span",
                                {"span": span, "min_span": min_span, "max_span": max_span},
                                (i1, i2))
                continue
            
            # Visual peak alignment
            mid_price = 0.5 * (p1 + p2)
            height_diff_pct = abs(p1 - p2) / max(1e-9, mid_price)
            if height_diff_pct > sim_tol:
                log_debug_message(debug_log, "double_top", "peak_height_mismatch",
                                {"height_diff_pct": height_diff_pct * 100, "max_allowed_pct": sim_tol * 100},
                                (i1, i2))
                continue

            # Find valley for "M" shape
            j0, j1 = i1 + valley_sep, i2 - valley_sep
            if j1 <= j0: 
                log_debug_message(debug_log, "double_top", "invalid_valley_range",
                                {"j0": j0, "j1": j1},
                                (i1, i2))
                continue
            
            local = close[j0:j1]
            if len(local) == 0: 
                log_debug_message(debug_log, "double_top", "no_valley_data",
                                {"j0": j0, "j1": j1},
                                (i1, i2))
                continue
            
            v_idx = int(np.argmin(local) + j0)
            v_price = float(close[v_idx])
            
            # Visual valley depth check
            valley_ratio = (mid_price - v_price) / mid_price
            # Use shallow default (6%) to match config; previously 0.7 starved detections
            min_valley_ratio = visual_cfg.get("max_valley_to_peak_ratio", 0.06)
            if valley_ratio < min_valley_ratio:
                log_debug_message(debug_log, "double_top", "insufficient_valley_depth",
                                {"valley_ratio": valley_ratio, "min_required": min_valley_ratio},
                                (i1, i2))
                continue
            
            # Visual pattern height
            pattern_height = max(p1, p2) - v_price
            min_pattern_height = geom.get("min_pattern_height_atr", 1.2) * atr_mean
            if pattern_height < min_pattern_height:
                log_debug_message(debug_log, "double_top", "insufficient_pattern_height",
                                {"pattern_height": pattern_height, "min_required": min_pattern_height},
                                (i1, i2))
                continue
            
            # Validate "M" shape visually
            m_shape_valid = False
            if visual_cfg.get("require_m_shape", True):
                m_shape_valid = _validate_m_shape(close, i1, i2, v_idx, p1, p2, v_price)
            
            if not m_shape_valid:
                log_debug_message(debug_log, "double_top", "invalid_m_shape", {}, (i1, i2))
                continue
            
            # Breakout confirmation
            breakout_ok = True
            if brc:
                base_thr = _choose_abs_threshold(br_thr_atr, br_thr_pct, atr_mean, mid_price, prefer_atr)
                recent_atr_val = _recent_atr()
                thr_abs = _apply_dynamic_floor(
                    base_thr, dyn, "breakout_height_pct",
                    pattern_height, mid_price, recent_atr_val
                )
                
                if br_within:
                    future = close[i2+1 : i2+1 + br_within]
                else:
                    future = close[i2+1:]
                
                if len(future) == 0:
                    log_debug_message(debug_log, "double_top", "no_future_data", {}, (i1, i2))
                    continue
                
                breakout_ok = _level_breakout_confirm(future, level=v_price, side="down", 
                                                     thr_abs=thr_abs, within_bars=br_within)
            
            if not breakout_ok and require_confirmation and not allow_pre_breakout:
                log_debug_message(debug_log, "double_top", "no_breakout_confirmation",
                                {"threshold": thr_abs, "within_bars": br_within},
                                (i1, i2))
                continue

            # Visual scoring
            score = _score_double_top_visual(p1, p2, v_price, i1, i2, span, visual_cfg)
            
            out.append({
                "type": "double_top", "i1": i1, "p1": p1, "i2": i2, "p2": p2,
                "ivalley": v_idx, "pvalley": v_price, "span": span, "score": score,
                "icenter": int(0.5*(i1+i2)), "breakout_confirmed": bool(breakout_ok)
            })
    
    dedup = int((labels_post or {}).get("dedup_time_overlap_bars", 10))
    return _nms_time(out, dedup)

def _validate_m_shape(close: np.ndarray, i1: int, i2: int, v_idx: int, p1: float, p2: float, v_price: float) -> bool:
    """Validate visual 'M' shape characteristics"""
    # Check that prices descend into valley and ascend out
    left_descend_count = 0
    right_ascend_count = 0
    
    # Check left side (i1 to v_idx)
    if v_idx - i1 > 1:
        for i in range(i1, v_idx-1):
            if close[i] >= close[i+1]:
                left_descend_count += 1
    
    # Check right side (v_idx to i2)  
    if i2 - v_idx > 1:
        for i in range(v_idx, i2-1):
            if close[i] <= close[i+1]:
                right_ascend_count += 1
    
    left_ratio = left_descend_count / max(1, (v_idx - i1 - 1))
    right_ratio = right_ascend_count / max(1, (i2 - v_idx - 1))
    
    return left_ratio > 0.6 and right_ratio > 0.6

def _score_double_top_visual(p1: float, p2: float, v_price: float, i1: int, i2: int, span: int, visual_cfg: dict) -> float:
    """Score double top based on visual quality"""
    score = 0.0
    
    # Peak alignment score
    peak_similarity = 1.0 - (abs(p1 - p2) / max(p1, p2))
    score += peak_similarity * visual_cfg.get("weight_peak_alignment", 1.3)
    
    # Valley depth score
    mid_price = 0.5 * (p1 + p2)
    valley_depth = (mid_price - v_price) / mid_price
    score += min(valley_depth * 20, 1.0) * visual_cfg.get("weight_valley_depth", 0.7)
    
    # Pattern height score
    pattern_height = max(p1, p2) - v_price
    height_score = min(pattern_height / (mid_price * 0.1), 1.0)
    score += height_score * visual_cfg.get("weight_pattern_height", 0.9)
    
    return score / 3.0

# --- VISUAL Inverse Head & Shoulders Detection ---
# --- COMPLETE VISUAL Inverse Head & Shoulders Detection ---
def detect_inverse_head_shoulders_visual(closes: np.ndarray,
                                        piv_lo: list,
                                        piv_hi: list,
                                        cfg: dict,
                                        atr_mean: float,
                                        prefer_atr: bool,
                                        debug_log: list = None) -> list[dict]:
    """Enhanced inverse H&S detection focusing on visual 'W' shape characteristics"""
    out = []
    geom = cfg.get("geometry", {})
    visual_cfg = cfg.get("visual", {})
    
    # Visual constraints (different from regular H&S)
    shoulder_sim = float(geom.get("shoulder_height_similarity_pct", 40)) / 100.0
    timing_sim = float(geom.get("shoulder_timing_similarity_pct", 75)) / 100.0
    # Align default with config (allows flatter inverse heads)
    min_head_ratio = float(visual_cfg.get("min_head_depth_ratio", 0.92))
    
    dur = geom.get("duration", {})
    min_span = int(dur.get("min_bars", 35))
    max_span = int(dur.get("max_bars", 220))
    
    piv_sorted = sorted(piv_lo, key=lambda t: t[0])
    
    for iL, pL in piv_sorted:
        for iH, pH in piv_sorted:
            if iH <= iL + 3: 
                continue
            for iR, pR in piv_sorted:
                if iR <= iH + 3: 
                    continue
                
                span = iR - iL
                if span < min_span or span > max_span:
                    log_debug_message(debug_log, "inverse_head_and_shoulders", "invalid_span",
                                    {"span": span, "min_span": min_span, "max_span": max_span},
                                    (iL, iH, iR))
                    continue
                
                # Visual symmetry check
                shoulder_avg = 0.5 * (pL + pR)
                if abs(pL - pR) / max(1e-9, shoulder_avg) > shoulder_sim:
                    log_debug_message(debug_log, "inverse_head_and_shoulders", "shoulder_height_asymmetry",
                                    {"height_diff_pct": abs(pL - pR) / shoulder_avg * 100},
                                    (iL, iH, iR))
                    continue
                
                # Visual head depth (head should be clearly lower)
                head_need = _choose_abs_threshold(
                    geom.get("head_above_shoulders_min_atr"),
                    geom.get("head_above_shoulders_min_pct"), 
                    atr_mean, shoulder_avg, prefer_atr
                )
                
                # For inverse, head should be BELOW shoulders
                head_depth = shoulder_avg - pH
                head_to_shoulder_ratio = pH / shoulder_avg
                
                if head_depth < head_need or head_to_shoulder_ratio > min_head_ratio:
                    log_debug_message(debug_log, "inverse_head_and_shoulders", "insufficient_head_depth",
                                    {"head_depth": head_depth, "required": head_need,
                                     "head_ratio": head_to_shoulder_ratio, "max_ratio": min_head_ratio},
                                    (iL, iH, iR))
                    continue
                
                # Visual timing symmetry
                ideal_right_pos = iL + (iH - iL) * 0.6
                tol_bars = max(4, int(timing_sim * span))
                if abs(iR - ideal_right_pos) > tol_bars:
                    log_debug_message(debug_log, "inverse_head_and_shoulders", "timing_asymmetry",
                                    {"actual_pos": iR, "ideal_pos": ideal_right_pos, "tolerance": tol_bars},
                                    (iL, iH, iR))
                    continue
                
                # Find neckline points (highs between shoulders)
                neckline_points = []
                # Left peak between left shoulder and head
                if iL + 1 < iH:
                    left_peak_range = closes[iL+1:iH]
                    if len(left_peak_range) > 0:
                        left_peak_idx = iL + 1 + np.argmax(left_peak_range)
                        neckline_points.append((left_peak_idx, closes[left_peak_idx]))
                
                # Right peak between head and right shoulder  
                if iH + 1 < iR:
                    right_peak_range = closes[iH+1:iR]
                    if len(right_peak_range) > 0:
                        right_peak_idx = iH + 1 + np.argmax(right_peak_range)
                        neckline_points.append((right_peak_idx, closes[right_peak_idx]))
                
                # Validate neckline visually
                neckline_valid = False
                if len(neckline_points) >= 2:
                    neckline_valid = _validate_neckline_visual(neckline_points, geom.get("neckline", {}))
                
                if not neckline_valid:
                    log_debug_message(debug_log, "inverse_head_and_shoulders", "invalid_neckline",
                                    {"neckline_points": len(neckline_points)},
                                    (iL, iH, iR))
                    continue
                
                score = _score_inverse_hs_visual(pL, pH, pR, iL, iH, iR, span, visual_cfg)
                
                out.append({
                    "type": "inverse_head_and_shoulders", 
                    "iL": iL, "pL": pL, "iH": iH, "pH": pH, "iR": iR, "pR": pR,
                    "score": score, "icenter": iH,
                    "neckline_points": neckline_points
                })
    
    return out

def _score_inverse_hs_visual(pL: float, pH: float, pR: float, iL: int, iH: int, iR: int, span: int, visual_cfg: dict) -> float:
    """Score inverse H&S pattern based on visual quality"""
    score = 0.0
    
    # Symmetry score (more relaxed)
    shoulder_symmetry = 1.0 - (abs(pL - pR) / max(pL, pR))
    score += shoulder_symmetry * visual_cfg.get("weight_symmetry", 0.7)
    
    # Head depth score
    shoulder_avg = 0.5 * (pL + pR)
    head_depth = (shoulder_avg - pH) / shoulder_avg
    score += min(head_depth * 15, 1.0) * visual_cfg.get("weight_head_prominence", 1.1)
    
    # Timing symmetry score
    ideal_right_pos = iL + (iH - iL) * 0.6
    timing_symmetry = 1.0 - min(abs(iR - ideal_right_pos) / (span * 0.6), 1.0)
    score += timing_symmetry * visual_cfg.get("weight_symmetry", 0.7)
    
    # Base formation score
    base_score = min(span / 80.0, 1.0)
    score += base_score * 0.5
    
    return score / 3.5

# --- VISUAL Descending Triangle Detection ---
def detect_descending_triangle_visual(close: np.ndarray,
                                     piv_hi: list,
                                     piv_lo: list,
                                     cfg_tri: dict,
                                     atr_mean: float,
                                     img_path: Path = None,
                                     debug_log: list = None) -> list[dict]:
    """Enhanced descending triangle detection with visual convergence validation"""
    out = []
    geom = cfg_tri.get("geometry", {})
    visual_cfg = cfg_tri.get("visual", {})
    hough_cfg = cfg_tri.get("hough", {})
    
    dur = geom.get("duration", {})
    min_span = int(dur.get("min_bars", 25))
    max_span = int(dur.get("max_bars", 150))
    
    upper_cfg = geom.get("upper_trendline", {})
    lower_cfg = geom.get("lower_boundary", {})
    
    min_upper_touches = upper_cfg.get("min_touches", 3)
    min_lower_touches = lower_cfg.get("min_touches", 2)
    max_gap_bars = upper_cfg.get("max_gap_bars", 10)
    
    # Visual convergence requirements
    min_convergence = visual_cfg.get("min_convergence_ratio", 0.3)
    min_contraction = visual_cfg.get("min_contraction_pct", 25) / 100.0
    
    if len(piv_hi) < min_upper_touches or len(piv_lo) < min_lower_touches:
        return out
    
    # Sort pivots by time
    hi_sorted = sorted(piv_hi, key=lambda x: x[0])
    lo_sorted = sorted(piv_lo, key=lambda x: x[0])
    
    # Look for descending upper line (lower highs)
    for i in range(len(hi_sorted) - min_upper_touches + 1):
        upper_points = hi_sorted[i:i + min_upper_touches]
        
        # Check if highs are generally descending
        upper_indices = [p[0] for p in upper_points]
        upper_prices = [p[1] for p in upper_points]
        
        # Fit upper trendline
        if len(upper_indices) >= 2:
            upper_slope, upper_intercept = np.polyfit(upper_indices, upper_prices, 1)
            
            # Upper line must be descending (visually clear)
            upper_slope_deg = math.degrees(math.atan(upper_slope))
            max_upper_slope = upper_cfg.get("max_slope_deg", -3)  # Must be negative
            min_upper_slope = upper_cfg.get("min_slope_deg", -15)
            
            if upper_slope_deg > max_upper_slope or upper_slope_deg < min_upper_slope:
                continue
            
            # Check upper line fit quality
            upper_predicted = upper_slope * np.array(upper_indices) + upper_intercept
            upper_residuals = np.abs(upper_prices - upper_predicted)
            upper_rmsd = np.sqrt(np.mean(upper_residuals**2))
            upper_price_range = max(upper_prices) - min(upper_prices)
            
            if upper_price_range > 0 and upper_rmsd / upper_price_range > upper_cfg.get("max_rmsd_pct_of_height", 0.25):
                continue
            
            # Find matching lower horizontal support
            start_idx = min(upper_indices)
            end_idx = max(upper_indices)
            span = end_idx - start_idx
            
            if span < min_span or span > max_span:
                continue
            
            # Get lower pivots within this timeframe
            lower_in_range = [(idx, price) for idx, price in lo_sorted 
                             if start_idx <= idx <= end_idx]
            
            if len(lower_in_range) < min_lower_touches:
                continue
            
            # Check if lower points form a horizontal support
            lower_indices = [p[0] for p in lower_in_range]
            lower_prices = [p[1] for p in lower_in_range]
            
            # Fit horizontal line to lower points
            lower_avg = np.mean(lower_prices)
            lower_residuals = np.abs(lower_prices - lower_avg)
            lower_rmsd = np.sqrt(np.mean(lower_residuals**2))
            lower_price_range = max(lower_prices) - min(lower_prices) if len(lower_prices) > 1 else 0
            
            max_lower_rmsd = lower_cfg.get("max_rmsd_pct_of_height", 0.25)
            if lower_price_range > 0 and lower_rmsd / lower_price_range > max_lower_rmsd:
                continue
            
            # Visual convergence check
            start_height = (upper_slope * start_idx + upper_intercept) - lower_avg
            end_height = (upper_slope * end_idx + upper_intercept) - lower_avg
            
            if start_height <= 0:  # Invalid triangle
                continue
                
            convergence_ratio = end_height / start_height
            if convergence_ratio > min_convergence:  # Not converging enough
                continue
            
            # Visual contraction check
            price_range_start = max(upper_prices) - min(lower_prices)
            price_range_end = (upper_slope * end_idx + upper_intercept) - lower_avg
            contraction = (price_range_start - price_range_end) / price_range_start
            
            if contraction < min_contraction:
                continue
            
            # Hough line validation (if image available)
            hough_valid = True
            if img_path and hough_cfg.get("enabled", True):
                hough_valid = hough_desc_triangle_ok(img_path, cfg_tri)
            
            if not hough_valid:
                continue
            
            # Calculate visual score
            score = _score_triangle_visual(upper_slope_deg, convergence_ratio, contraction, 
                                         upper_rmsd/upper_price_range if upper_price_range > 0 else 0,
                                         lower_rmsd/lower_price_range if lower_price_range > 0 else 0,
                                         visual_cfg)
            
            out.append({
                "type": "descending_triangle",
                "start": start_idx,
                "end": end_idx, 
                "upper_slope": upper_slope,
                "upper_intercept": upper_intercept,
                "support_level": lower_avg,
                "convergence_ratio": convergence_ratio,
                "contraction_pct": contraction * 100,
                "score": score,
                "icenter": int((start_idx + end_idx) / 2)
            })
    
    return out

def _score_triangle_visual(upper_slope_deg: float, convergence_ratio: float, contraction: float,
                          upper_fit_quality: float, lower_fit_quality: float, visual_cfg: dict) -> float:
    """Score triangle pattern based on visual quality"""
    score = 0.0
    
    # Upper slope score (steeper negative is better)
    ideal_slope = -8.0  # Ideal descending slope
    slope_score = 1.0 - min(abs(upper_slope_deg - ideal_slope) / 15.0, 1.0)
    score += slope_score * visual_cfg.get("weight_slope", 1.0)
    
    # Convergence score (more convergence is better)
    convergence_score = 1.0 - convergence_ratio  # Lower ratio = better convergence
    score += convergence_score * visual_cfg.get("weight_convergence", 1.2)
    
    # Contraction score
    contraction_score = min(contraction / 0.5, 1.0)  # Normalize
    score += contraction_score * visual_cfg.get("weight_contraction", 0.8)
    
    # Line fit quality
    upper_quality_score = 1.0 - min(upper_fit_quality / 0.3, 1.0)
    lower_quality_score = 1.0 - min(lower_fit_quality / 0.3, 1.0)
    score += (upper_quality_score + lower_quality_score) * 0.5 * visual_cfg.get("weight_line_quality", 0.9)
    
    return score / 4.0

# --- Triangle Hough gate (image-based confirmation for descending triangle) ---
def _deg_from_rise_run(dy: float, dx: float) -> float:
    if dx == 0: return 90.0 * (1 if dy > 0 else -1)
    return math.degrees(math.atan2(dy, dx))

def hough_desc_triangle_ok(png_path: Path, tri_cfg: dict) -> bool:
    hc = (tri_cfg.get("hough") or {})
    if not hc.get("enabled", True):
        return True
    can = hc.get("canny", {}) or {}
    hp  = hc.get("houghp", {}) or {}
    blur_ksize = int(can.get("blur_ksize", 3))
    t1, t2 = int(can.get("t1", 50)), int(can.get("t2", 150))

    rho = float(hp.get("rho", 1.0))
    theta = math.radians(float(hp.get("theta_deg", 1.0)))
    thresh = int(hp.get("thresh", 45))
    min_len = int(hp.get("min_line_len", 40))
    max_gap = int(hp.get("max_line_gap", 10))

    im = cv.imread(str(png_path), cv.IMREAD_GRAYSCALE)
    if im is None: 
        return False
    if blur_ksize >= 3:
        im = cv.GaussianBlur(im, (blur_ksize, blur_ksize), 0)
    edges = cv.Canny(im, t1, t2)

    lines = cv.HoughLinesP(edges, rho, theta, threshold=thresh, minLineLength=min_len, maxLineGap=max_gap)
    if lines is None or len(lines) == 0:
        return False

    # classify lines by slope (deg)
    upper_max_deg = float(hc.get("upper_max_deg", -5.0))            # should be <= this (negative)
    lower_abs_max = float(hc.get("lower_abs_max_deg", 4.0))         # near-flat
    min_support   = int(hc.get("min_support_lines", 3))

    uppers, lowers = 0, 0
    for ln in lines[:,0,:]:
        x1,y1,x2,y2 = ln
        dx, dy = (x2-x1), (y2-y1)
        deg = _deg_from_rise_run(-dy, dx)  # image y-down => invert dy for math up
        if deg <= upper_max_deg:
            uppers += 1
        if abs(deg) <= lower_abs_max:
            lowers += 1

    return (uppers >= min_support) and (lowers >= min_support)
# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Generate weak labels (YOLO + CSV presence flags) for rendered images.")
    ap.add_argument("--patterns_cfg", default="configs/patterns.yaml")
    ap.add_argument("--images_root", default="data/images/rendered")
    ap.add_argument("--labels_root", default="data/labels/rendered")
    ap.add_argument("--labels_csv", default="reports/labels/weak_labels.csv")
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    args = ap.parse_args()

    cfg = load_yaml(Path(args.patterns_cfg)) or {}
    common_base = cfg.get("common", {}) or {}
    patterns_base = cfg.get("patterns", {}) or {}
    overrides = cfg.get("overrides", []) or []
    labels_map = cfg.get("labels", {}) or {}
    # Initialize debug log
    debug_log = init_debug_log()

    supported = {
        "head_and_shoulders",
        "inverse_head_and_shoulders",
        "double_top",
        "double_bottom",
        "descending_triangle",
    }
    pattern_keys = set(patterns_base.keys()).union(supported)
    default_active = sorted([name for name in supported if patterns_base.get(name, {}).get("enabled", True)])
    if not default_active:
        print("⚠️ Base config has no supported patterns enabled; relying on overrides.")

    cfg_cache: dict[tuple[str,str], tuple[dict, dict, list[str]]] = {}
    rows = []

    for split in args.splits:
        img_dir = Path(args.images_root) / split
        out_yolo = Path(args.labels_root) / split
        out_json = Path(args.labels_root) / f"{split}_json"
        out_yolo.mkdir(parents=True, exist_ok=True)
        out_json.mkdir(parents=True, exist_ok=True)

        pngs = sorted(img_dir.glob("*.png"))
        for png in pngs:
            # --- load metadata + OHLCV window ---
            meta = load_meta(png.with_suffix(".json"))
            symbol = meta["symbol"]
            N = meta["bars"]; W = meta["img_w"]; H = meta["img_h"]
            y_min, y_max = meta["axes_ylim"]
            timeframe = meta.get("timeframe", "")

            cache_key = (symbol, timeframe)
            if cache_key not in cfg_cache:
                eff_common, eff_patterns = resolve_config(common_base, patterns_base, overrides, symbol, timeframe, pattern_keys)
                eff_active = sorted([name for name in supported if eff_patterns.get(name, {}).get("enabled", True)])
                cfg_cache[cache_key] = (eff_common, eff_patterns, eff_active)
            eff_common, eff_patterns, active = cfg_cache[cache_key]
            if not active:
                continue

            atr_cfg   = eff_common.get("atr", {}) or {}
            piv_cfg   = eff_common.get("pivots", {}) or {}
            units_cfg = eff_common.get("units", {}) or {}
            labels_post = eff_common.get("labels_post", {}) or {}
            prefer_atr = bool(units_cfg.get("prefer_atr_over_percent", True))

            atr_period   = int(atr_cfg.get("period", 14))
            atr_smoothing = atr_cfg.get("smoothing", "rma")
            min_prom_atr = float(piv_cfg.get("min_prominence_atr", 0.7))
            min_dist_bars = int(piv_cfg.get("min_distance_bars", 8))
            smooth_window = int(piv_cfg.get("smoothing_window_bars", 1))
            max_pivots = int(piv_cfg.get("max_pivots_per_window", 0))
            max_pivots = max_pivots if max_pivots > 0 else None
            adaptive_prom_pct = float(piv_cfg.get("adaptive_prominence_pct_of_range", 0.0))

            df_full = load_ohlcv(symbol)
            win = slice_window(df_full, meta["start_ts"], meta["end_ts"])
            if len(win) != N:
                win = win.iloc[-N:]

            h = win["high"].to_numpy()
            l = win["low"].to_numpy()
            c = win["close"].to_numpy()

            # --- ATR + robust pivots via SciPy find_peaks (ATR-prominence) ---
            atr = compute_atr(h, l, c, period=atr_period, mode=atr_smoothing)
            atr_mean = float(np.nanmean(atr)) if len(atr) else 0.0

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

            # --- pattern detections ---
            det_map = {name: [] for name in supported}

            if "head_and_shoulders" in active:
                det_map["head_and_shoulders"] = detect_head_shoulders_visual(
                    c, piv_hi, piv_lo,  # Added piv_lo for neckline detection
                    eff_patterns.get("head_and_shoulders", {}),
                    atr_mean, prefer_atr, debug_log
                )

            if "inverse_head_and_shoulders" in active:
                det_map["inverse_head_and_shoulders"] = detect_inverse_head_shoulders_visual(
                    c, piv_lo, piv_hi,
                    eff_patterns.get("inverse_head_and_shoulders", {}),
                    atr_mean, prefer_atr, debug_log
                )

            # Robust Double Top / Bottom (ATR thresholds, duration, spacing, breakout, NMS)
            if "double_top" in active:
                det_map["double_top"] = detect_double_top_visual(
                    close=c, piv_hi=piv_hi, cfg_dt=eff_patterns.get("double_top", {}),
                    atr=atr, prefer_atr=prefer_atr, labels_post=labels_post, debug_log=debug_log
                )

            if "double_bottom" in active:
                cfg_db = deep_merge_dict(eff_patterns.get("double_top", {}), eff_patterns.get("double_bottom", {}))
                det_map["double_bottom"] = detect_double_bottom_visual(
                    close=c, piv_lo=piv_lo, cfg_db=cfg_db,
                    atr=atr, prefer_atr=prefer_atr, labels_post=labels_post, debug_log=debug_log
                )

            if "descending_triangle" in active:
                det_map["descending_triangle"] = detect_descending_triangle_visual(
                    c, piv_hi, piv_lo,
                    eff_patterns.get("descending_triangle", {}),
                    atr_mean, png, debug_log  # Pass image path for Hough validation
                )

            # --- presence flags and counts ---
            counts = {name+"_cnt": len(det_map[name]) for name in active}
            confirmed_counts = {
                name+"_confirmed_cnt": sum(1 for d in det_map[name] if d.get("breakout_confirmed", True))
                for name in active
            }
            # Presence flag can ignore breakout when label_allow_pre_breakout is enabled
            flags = {}
            for name in active:
                allow_pre = bool(eff_patterns.get(name, {}).get("label_allow_pre_breakout", False))
                if allow_pre:
                    flags[name] = 1 if counts[name+"_cnt"] > 0 else 0
                else:
                    flags[name] = 1 if confirmed_counts[name+"_confirmed_cnt"] > 0 else 0

            # --- YOLO + rich JSON sidecars ---
            cid_lookup = {v: int(k) for k, v in labels_map.items() if v in supported}
            yolo_boxes = []
            rich_all = {"image": png.name, "detections": []}

            for name in active:
                for d in det_map[name]:
                    xs, ys = [], []
                    if name in ("head_and_shoulders","inverse_head_shoulders","inverse_head_and_shoulders"):
                        for (i,p) in [(d["iL"],d["pL"]), (d["iH"],d["pH"]), (d["iR"],d["pR"])]:
                            xs.append(bar_to_x(i, N, W)); ys.append(price_to_y(p, y_min, y_max, H))
                    elif name == "double_top":
                        for (i,p) in [(d["i1"],d["p1"]), (d["i2"],d["p2"]), (d["ivalley"],d["pvalley"])]:
                            xs.append(bar_to_x(i, N, W)); ys.append(price_to_y(p, y_min, y_max, H))
                    elif name == "double_bottom":
                        for (i,p) in [(d["i1"],d["p1"]), (d["i2"],d["p2"]), (d["ipeak"],d["ppeak"])]:
                            xs.append(bar_to_x(i, N, W)); ys.append(price_to_y(p, y_min, y_max, H))
                    elif name == "descending_triangle":
                        iL, iR = d["start"], d["end"]
                        p_topL = d["a_hi"]*iL + d["b_hi"]; p_topR = d["a_hi"]*iR + d["b_hi"]
                        p_botL = d["a_lo"]*iL + d["b_lo"]; p_botR = d["a_lo"]*iR + d["b_lo"]
                        for (i,pv) in [(iL,p_topL),(iR,p_topR),(iL,p_botL),(iR,p_botR)]:
                            xs.append(bar_to_x(i, N, W)); ys.append(price_to_y(pv, y_min, y_max, H))
                    else:
                        continue

                    xmin, xmax = max(0, min(xs)), min(W-1, max(xs))
                    ymin, ymax = max(0, min(ys)), min(H-1, max(ys))
                    pad = int(eff_patterns.get(name, {}).get("bbox_padding_px", 6))
                    xmin = max(0, xmin - pad); xmax = min(W-1, xmax + pad)
                    ymin = max(0, ymin - pad); ymax = min(H-1, ymax + pad)

                    cid = cid_lookup.get(name, 0)
                    cx = (xmin + xmax) / 2 / W
                    cy = (ymin + ymax) / 2 / H
                    bw = (xmax - xmin) / W
                    bh = (ymax - ymin) / H
                    yolo_boxes.append((cid, cx, cy, bw, bh))

                    rich_all["detections"].append({
                        "class": name,
                        "bbox_px": [int(xmin), int(ymin), int(xmax), int(ymax)],
                        "raw": d
                    })

            if yolo_boxes:
                write_yolo(out_yolo / (png.stem + ".txt"), yolo_boxes)
                (out_json / (png.stem + ".json")).write_text(json.dumps(rich_all, indent=2), encoding="utf-8")

            # --- Presence CSV row (for RF) ---
            row = {
                "image": png.name,
                "split": split,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_ts": meta["start_ts"],
                "end_ts": meta["end_ts"],
                "bars": int(N),
                "img_w": int(W),
                "img_h": int(H),
            }
            for name in ["head_and_shoulders","inverse_head_and_shoulders","double_top","double_bottom","descending_triangle"]:
                if name in active:
                    row[f"y_{name}"] = int(flags[name])
                    row[f"{name}_cnt"] = int(counts[name+"_cnt"])
                    row[f"{name}_confirmed_cnt"] = int(confirmed_counts[name+"_confirmed_cnt"])
                else:
                    row[f"y_{name}"] = 0
                    row[f"{name}_cnt"] = 0
                    row[f"{name}_confirmed_cnt"] = 0

            rows.append(row)

        print(f"✅ Split {split}: computed labels for {len(pngs)} images.")

    out_csv = Path(args.labels_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(by=["split","symbol","end_ts","image"])
    df.to_csv(out_csv, index=False)
    print(f"✅ Presence-labels CSV written: {out_csv} (rows={len(df)})")


if __name__ == "__main__":
    main()

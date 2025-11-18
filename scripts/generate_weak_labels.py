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

def _has_trend(closes: np.ndarray, idx: int, lookback: int, direction: str) -> bool:
    """Rough trend check using start-end slope over lookback bars prior to idx."""
    if lookback <= 1 or idx <= 1:
        return True
    start = max(0, idx - lookback)
    if start >= idx:
        return True
    c0 = float(closes[start]); c1 = float(closes[idx])
    if direction == "up":
        return c1 > c0
    else:
        return c1 < c0

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

# --- Simple labeling-function ensembles for DT/DB ---
def lf_votes_double_top(cands: list[dict], atr_mean_val: float) -> tuple[int,int,float]:
    """LF ensemble for double_top: strict/loose depth vs ATR, symmetry, breakout; top 2 candidates with penalties/negatives."""
    if not cands:
        return 0, 0, 0.0
    if len(cands) > 12:
        return 0, 1, 0.0
    votes_pos = votes_tot = 0
    atrm = atr_mean_val if atr_mean_val else 0.0
    for d in sorted(cands, key=lambda x: x.get("score", 0.0), reverse=True)[:2]:
        p1, p2, v = d.get("p1"), d.get("p2"), d.get("pvalley")
        if p1 is None or p2 is None or v is None:
            continue
        mid = 0.5*(p1+p2)
        height = max(p1,p2) - v
        sym = abs(p1 - p2)/max(1e-9, mid)
        breakout = d.get("breakout_confirmed", False)
        # strict depth
        votes_tot += 1; votes_pos += 1 if (atrm > 0 and height/atrm >= 1.2) else 0
        # loose depth
        votes_tot += 1; votes_pos += 1 if (atrm > 0 and height/atrm >= 0.8) else 0
        # symmetry votes at two levels; add a penalty if poor symmetry
        votes_tot += 2
        votes_pos += 1 if sym <= 0.05 else 0
        votes_pos += 1 if sym <= 0.09 else 0
        if sym > 0.12:
            votes_tot += 1  # explicit negative weight
        # breakout vote
        votes_tot += 1; votes_pos += 1 if breakout else 0
        # quality: very shallow gets a penalty
        votes_tot += 1; votes_pos += 1 if (atrm > 0 and height/atrm >= 0.5) else 0
    p = votes_pos / votes_tot if votes_tot else 0.0
    return votes_pos, votes_tot, p

def lf_votes_double_bottom(cands: list[dict], atr_mean_val: float) -> tuple[int,int,float]:
    """LF ensemble for double_bottom: strict/loose rise vs ATR, symmetry, breakout; top 2 candidates with penalties/negatives."""
    if not cands:
        return 0, 0, 0.0
    if len(cands) > 12:
        return 0, 1, 0.0
    votes_pos = votes_tot = 0
    atrm = atr_mean_val if atr_mean_val else 0.0
    for d in sorted(cands, key=lambda x: x.get("score", 0.0), reverse=True)[:2]:
        p1, p2, pk = d.get("p1"), d.get("p2"), d.get("ppeak")
        if p1 is None or p2 is None or pk is None:
            continue
        mid = 0.5*(p1+p2)
        height = pk - min(p1,p2)
        sym = abs(p1 - p2)/max(1e-9, mid)
        breakout = d.get("breakout_confirmed", False)
        votes_tot += 1; votes_pos += 1 if (atrm > 0 and height/atrm >= 1.2) else 0
        votes_tot += 1; votes_pos += 1 if (atrm > 0 and height/atrm >= 0.8) else 0
        votes_tot += 2
        votes_pos += 1 if sym <= 0.05 else 0
        votes_pos += 1 if sym <= 0.09 else 0
        if sym > 0.12:
            votes_tot += 1
        votes_tot += 1; votes_pos += 1 if breakout else 0
        votes_tot += 1; votes_pos += 1 if (atrm > 0 and height/atrm >= 0.5) else 0
    p = votes_pos / votes_tot if votes_tot else 0.0
    return votes_pos, votes_tot, p

# --- Robust Double Top ---
def detect_double_top(close: np.ndarray,
                      volume: np.ndarray | None,
                      piv_hi: list[tuple[int,float]],
                      cfg_dt: dict,
                      atr: np.ndarray,
                      prefer_atr: bool,
                      labels_post: dict | None = None,
                      debug_log: list | None = None) -> list[dict]:
    out = []
    atr_mean = float(np.nanmean(atr)) if len(atr) else 0.0
    geom = (cfg_dt.get("geometry") or {})
    dur  = geom.get("duration", {})
    min_span = int(dur.get("min_bars", 16)); max_span = int(dur.get("max_bars", 120))
    sim_tol  = float(geom.get("peak_height_similarity_pct", 15)) / 100.0
    valley_sep = int(geom.get("valley_min_bars_from_peaks", 2))
    max_pairs = max(1, int(geom.get("max_pairs_per_peak", 5)))

    br = (cfg_dt.get("breakout") or {})
    br_side = br.get("side", "down")
    brc = (br.get("confirm") or {})
    br_thr_atr = brc.get("threshold_atr", None)
    br_thr_pct = brc.get("threshold_percent", None)
    br_within  = brc.get("within_bars", None)
    br_vol_mult = brc.get("volume_mult", None)
    br_vol_mult = brc.get("volume_mult", None)
    require_confirmation = bool(br.get("require_confirmation", True))
    allow_pre_breakout = bool(cfg_dt.get("label_allow_pre_breakout", False))

    sc = (cfg_dt.get("scoring") or {})
    w_depth = float(sc.get("weight_valley_depth_atr", 1.0))
    w_span  = float(sc.get("weight_span", 0.2))

    dyn = (geom.get("dynamic_thresholds") or {})
    dyn_lookback = int(dyn.get("vol_lookback_bars", 36))
    vol_lookback = max(dyn_lookback, 60)
    vol_lookback = max(dyn_lookback, 50)
    vol_lookback = max(dyn_lookback, 50)

    min_height_atr = float(geom.get("min_height_atr", 0.0))

    def _recent_atr():
        if len(atr) == 0:
            return atr_mean
        return float(np.nanmean(atr[-dyn_lookback:])) if dyn_lookback > 0 else atr_mean

    piv_sorted = sorted(piv_hi, key=lambda t: t[0])
    total = len(piv_sorted)
    for idx1, (i1, p1) in enumerate(piv_sorted):
        limit = min(total, idx1 + 1 + max_pairs)
        for idx2 in range(idx1 + 1, limit):
            i2, p2 = piv_sorted[idx2]
            if i2 <= i1 + valley_sep:
                continue
            span = i2 - i1
            if span < min_span or span > max_span: continue
            mid = 0.5*(p1+p2)
            if abs(p1 - p2)/max(1e-9, mid) > sim_tol:
                continue
            j0, j1 = i1 + valley_sep, i2 - valley_sep
            if j1 <= j0: continue
            local = close[j0:j1]
            if len(local) == 0:
                continue
            v_idx = int(np.argmin(local) + j0)
            v = float(close[v_idx])

            base_need = _choose_abs_threshold(
                geom.get("valley_drop_min_atr"),
                geom.get("valley_drop_min_pct"),
                atr_mean, mid, prefer_atr
            )
            pattern_height = max(p1, p2) - v
            recent_atr = _recent_atr()
            dyn_need = _apply_dynamic_floor(
                base_need,
                dyn,
                pct_key="valley_drop_min_height_pct",
                pattern_height=pattern_height,
                price_ref=mid,
                atr_recent=recent_atr
            )
            if (mid - v) < dyn_need:
                if debug_log is not None:
                    debug_log.append({"pattern":"double_top","reason":"valley_depth","mid":mid,"v":v,"need":dyn_need,"i1":i1,"i2":i2})
                continue

            # pattern height floor vs ATR
            if min_height_atr > 0 and (max(p1, p2) - v) < (min_height_atr * max(1e-9, atr_mean)):
                if debug_log is not None:
                    debug_log.append({"pattern":"double_top","reason":"height_floor","height":max(p1,p2)-v,"need":min_height_atr*atr_mean,"i1":i1,"i2":i2})
                continue

            breakout_ok = True
            if brc:
                base_thr = _choose_abs_threshold(br_thr_atr, br_thr_pct, atr_mean, mid, prefer_atr)
                thr_abs = _apply_dynamic_floor(
                    base_thr,
                    dyn,
                    pct_key="breakout_height_pct",
                    pattern_height=pattern_height,
                    price_ref=mid,
                    atr_recent=recent_atr
                )
                if br_within:
                    future = close[i2+1 : i2+1 + br_within]
                else:
                    future = close[i2+1 :]
                if len(future) == 0:
                    continue
                breakout_ok = _level_breakout_confirm(future, level=v, side=br_side, thr_abs=thr_abs, within_bars=br_within)

                # volume spike on breakout relative to recent average (soften to 1.1x)
                if breakout_ok and volume is not None and len(volume) == len(close):
                    mult = float(br_vol_mult) if br_vol_mult is not None else 1.1
                    recent_vol = float(np.nanmean(volume[max(0, i2 - vol_lookback):i2])) if vol_lookback > 0 else float(np.nanmean(volume))
                    brk_window = br_within or 1
                    brk_vol = float(np.nanmean(volume[i2+1 : i2+1 + brk_window]))
                    if recent_vol > 0 and brk_vol < mult * recent_vol:
                        breakout_ok = False
                        if debug_log is not None:
                            debug_log.append({"pattern":"double_top","reason":"low_breakout_volume","brk_vol":brk_vol,"recent_vol":recent_vol,"i2":i2,"needed_mult":mult})
            if not breakout_ok and require_confirmation and not allow_pre_breakout:
                if debug_log is not None:
                    debug_log.append({"pattern":"double_top","reason":"no_breakout","level":v,"thr_abs":thr_abs,"i2":i2})
                continue

            ic = int(0.5*(i1+i2))
            depth = (mid - v)/max(1e-9, atr_mean)
            span_score = span / max(1.0, float(max_span))
            score = w_depth * depth + w_span * span_score
            out.append({"type":"double_top","i1":i1,"p1":p1,"i2":i2,"p2":p2,
                        "ivalley":v_idx,"pvalley":v,"span":span,"score":float(score),
                        "icenter":ic,"breakout_confirmed": bool(breakout_ok)})
    dedup = int((labels_post or {}).get("dedup_time_overlap_bars", 10))
    return _nms_time(out, dedup)

# --- Robust Double Bottom ---
def detect_double_bottom(close: np.ndarray,
                         volume: np.ndarray | None,
                         piv_lo: list[tuple[int,float]],
                         cfg_db: dict,
                         atr: np.ndarray,
                         prefer_atr: bool,
                         labels_post: dict | None = None,
                         debug_log: list | None = None) -> list[dict]:
    out = []
    atr_mean = float(np.nanmean(atr)) if len(atr) else 0.0
    geom = (cfg_db.get("geometry") or {})
    dur  = geom.get("duration", {})
    min_span = int(dur.get("min_bars", 16)); max_span = int(dur.get("max_bars", 120))
    sim_tol  = float(geom.get("peak_height_similarity_pct", 15)) / 100.0
    peak_sep = int(geom.get("valley_min_bars_from_peaks", 2))  # reuse name symmetrically
    max_pairs = max(1, int(geom.get("max_pairs_per_peak", 5)))

    br = (cfg_db.get("breakout") or {})
    br_side = br.get("side", "up")
    brc = (br.get("confirm") or {})
    br_thr_atr = brc.get("threshold_atr", None)
    br_thr_pct = brc.get("threshold_percent", None)
    br_within  = brc.get("within_bars", None)
    br_vol_mult = brc.get("volume_mult", None)
    require_confirmation = bool(br.get("require_confirmation", True))
    allow_pre_breakout = bool(cfg_db.get("label_allow_pre_breakout", False))

    sc = (cfg_db.get("scoring") or {})
    w_rise = float(sc.get("weight_valley_depth_atr", 1.0))
    w_span = float(sc.get("weight_span", 0.2))

    dyn = (geom.get("dynamic_thresholds") or {})
    dyn_lookback = int(dyn.get("vol_lookback_bars", 36))
    vol_lookback = max(dyn_lookback, 60)

    min_height_atr = float(geom.get("min_height_atr", 0.0))

    def _recent_atr():
        if len(atr) == 0:
            return atr_mean
        return float(np.nanmean(atr[-dyn_lookback:])) if dyn_lookback > 0 else atr_mean

    piv_sorted = sorted(piv_lo, key=lambda t: t[0])
    total = len(piv_sorted)
    for idx1, (i1, p1) in enumerate(piv_sorted):
        limit = min(total, idx1 + 1 + max_pairs)
        for idx2 in range(idx1 + 1, limit):
            i2, p2 = piv_sorted[idx2]
            if i2 <= i1 + peak_sep:
                continue
            span = i2 - i1
            if span < min_span or span > max_span: continue
            mid = 0.5*(p1+p2)
            if abs(p1 - p2)/max(1e-9, mid) > sim_tol:
                continue
            j0, j1 = i1 + peak_sep, i2 - peak_sep
            if j1 <= j0: continue
            local = close[j0:j1]
            if len(local) == 0:
                continue
            peak_idx = int(np.argmax(local) + j0)
            pk = float(close[peak_idx])

            base_need = _choose_abs_threshold(
                geom.get("peak_rise_min_atr"),
                geom.get("peak_rise_min_pct"),
                atr_mean, mid, prefer_atr
            )
            pattern_height = pk - min(p1, p2)
            recent_atr = _recent_atr()
            dyn_need = _apply_dynamic_floor(
                base_need,
                dyn,
                pct_key="valley_drop_min_height_pct",
                pattern_height=pattern_height,
                price_ref=mid,
                atr_recent=recent_atr
            )
            if (pk - mid) < dyn_need:
                if debug_log is not None:
                    debug_log.append({"pattern":"double_bottom","reason":"peak_rise","mid":mid,"pk":pk,"need":dyn_need,"i1":i1,"i2":i2})
                continue

            if min_height_atr > 0 and (pk - min(p1, p2)) < (min_height_atr * max(1e-9, atr_mean)):
                if debug_log is not None:
                    debug_log.append({"pattern":"double_bottom","reason":"height_floor","height":pk-min(p1,p2),"need":min_height_atr*atr_mean,"i1":i1,"i2":i2})
                continue

            breakout_ok = True
            if brc:
                base_thr = _choose_abs_threshold(br_thr_atr, br_thr_pct, atr_mean, mid, prefer_atr)
                thr_abs = _apply_dynamic_floor(
                    base_thr,
                    dyn,
                    pct_key="breakout_height_pct",
                    pattern_height=pattern_height,
                    price_ref=mid,
                    atr_recent=recent_atr
                )
                if br_within:
                    future = close[i2+1 : i2+1 + br_within]
                else:
                    future = close[i2+1 :]
                if len(future) == 0:
                    continue
                breakout_ok = _level_breakout_confirm(future, level=pk, side=br_side, thr_abs=thr_abs, within_bars=br_within)

                if breakout_ok and volume is not None and len(volume) == len(close):
                    mult = float(br_vol_mult) if br_vol_mult is not None else 1.1
                    recent_vol = float(np.nanmean(volume[max(0, i2 - vol_lookback):i2])) if vol_lookback > 0 else float(np.nanmean(volume))
                    brk_window = br_within or 1
                    brk_vol = float(np.nanmean(volume[i2+1 : i2+1 + brk_window]))
                    if recent_vol > 0 and brk_vol < mult * recent_vol:
                        breakout_ok = False
                        if debug_log is not None:
                            debug_log.append({"pattern":"double_bottom","reason":"low_breakout_volume","brk_vol":brk_vol,"recent_vol":recent_vol,"i2":i2,"needed_mult":mult})
            if not breakout_ok and require_confirmation and not allow_pre_breakout:
                if debug_log is not None:
                    debug_log.append({"pattern":"double_bottom","reason":"no_breakout","level":pk,"thr_abs":thr_abs,"i2":i2})
                continue

            ic = int(0.5*(i1+i2))
            rise = (pk - mid)/max(1e-9, atr_mean)
            span_score = span / max(1.0, float(max_span))
            score = w_rise * rise + w_span * span_score
            out.append({"type":"double_bottom","i1":i1,"p1":p1,"i2":i2,"p2":p2,
                        "ipeak":peak_idx,"ppeak":pk,"span":span,"score":float(score),
                        "icenter":ic,"breakout_confirmed": bool(breakout_ok)})
    dedup = int((labels_post or {}).get("dedup_time_overlap_bars", 10))
    return _nms_time(out, dedup)

def detect_head_shoulders(closes: np.ndarray,
                          volume: np.ndarray | None,
                          piv_hi: list[tuple[int,float]],
                          cfg: dict,
                          atr_mean: float,
                          prefer_atr: bool,
                          debug_log: list | None = None) -> list[dict]:
    """Classic H&S: three highs with middle > shoulders + rough time symmetry."""
    out = []
    geom = cfg.get("geometry", {})
    sym_pct = float(geom.get("shoulder_timing_similarity_pct", 40)) / 100.0
    dur     = geom.get("duration", {})
    min_span = int(dur.get("min_bars", 20)); max_span = int(dur.get("max_bars", 120))
    shoulder_sim = float(geom.get("shoulder_height_similarity_pct", 25)) / 100.0
    min_height_atr = float(geom.get("min_height_atr", 0.0))
    trend_lookback = int(geom.get("trend_lookback_bars", 0))
    br = cfg.get("breakout", {}) or {}
    brc = br.get("confirm", {}) or {}
    br_side = br.get("side", "down")
    br_thr_atr = brc.get("threshold_atr", None)
    br_thr_pct = brc.get("threshold_percent", None)
    br_within = brc.get("within_bars", None)
    piv_sorted = sorted(piv_hi, key=lambda t: t[0])
    for iL, pL in piv_sorted:
        for iH, pH in piv_sorted:
            if iH <= iL + 2: continue
            for iR, pR in piv_sorted:
                if iR <= iH + 2: continue
                span = iR - iL
                if span < min_span or span > max_span: continue
                sh_avg = 0.5*(pL+pR)
                if abs(pL - pR)/max(1e-9, sh_avg) > shoulder_sim:
                    if debug_log is not None:
                        debug_log.append({"pattern":"head_shoulders","reason":"shoulder_height","iL":iL,"iR":iR})
                    continue
                head_need = _choose_abs_threshold(
                    geom.get("head_above_shoulders_min_atr"),
                    geom.get("head_above_shoulders_min_pct"),
                    atr_mean,
                    sh_avg,
                    prefer_atr
                )
                if (pH - sh_avg) < max(head_need, 0.01 * sh_avg):
                    if debug_log is not None:
                        debug_log.append({"pattern":"head_shoulders","reason":"head_height","iH":iH,"need":head_need})
                    continue
                if min_height_atr > 0 and (pH - min(pL, pR)) < (min_height_atr * max(1e-9, atr_mean)):
                    if debug_log is not None:
                        debug_log.append({"pattern":"head_shoulders","reason":"height_floor","height":pH-min(pL,pR),"need":min_height_atr*atr_mean})
                    continue
                ideal = 0.5*(iL+iH)
                tol_bars = max(1, int(sym_pct * span))
                if abs(iR - ideal) > tol_bars:
                    if debug_log is not None:
                        debug_log.append({"pattern":"head_shoulders","reason":"timing_symmetry","iR":iR})
                    continue
                if trend_lookback > 0 and not _has_trend(closes, iL, trend_lookback, direction="up"):
                    if debug_log is not None:
                        debug_log.append({"pattern":"head_shoulders","reason":"no_uptrend","iL":iL})
                    continue
                # volume ordering: head < LS, RS < head (hard gate with mild 0.95 ratio)
                if volume is not None and len(volume) == len(closes):
                    vol_L = float(volume[iL]) if iL < len(volume) else 0.0
                    vol_H = float(volume[iH]) if iH < len(volume) else 0.0
                    vol_R = float(volume[iR]) if iR < len(volume) else 0.0
                    if not (vol_H < 0.95 * vol_L and vol_R < 0.95 * vol_H):
                        if debug_log is not None:
                            debug_log.append({"pattern":"head_shoulders","reason":"volume_order","vol_L":vol_L,"vol_H":vol_H,"vol_R":vol_R})
                        continue

                # breakout confirmation: compute neckline between two troughs
                if brc:
                    t1_idx = iL + int(np.argmin(closes[iL:iH+1]))
                    t2_idx = iH + int(np.argmin(closes[iH:iR+1]))
                    neckline = 0.5 * (closes[t1_idx] + closes[t2_idx])
                    thr_abs = _choose_abs_threshold(br_thr_atr, br_thr_pct, atr_mean, neckline, prefer_atr)
                    if thr_abs > 0:  # gate label on confirmed break
                        confirmed = False
                        future = closes[iR+1 : iR+1 + (br_within or 0)] if br_within else closes[iR+1 :]
                        confirmed = len(future) > 0 and _level_breakout_confirm(future, level=neckline, side=br_side, thr_abs=thr_abs, within_bars=br_within)
                        if not confirmed:
                            continue  # do not emit unconfirmed H&S
                else:
                    # if no breakout block is provided, we keep the detection
                    pass
                out.append({"type":"head_shoulders","iL":iL,"pL":pL,"iH":iH,"pH":pH,"iR":iR,"pR":pR,"breakout_confirmed": True})
    return out

def detect_inverse_head_shoulders(closes: np.ndarray,
                                  volume: np.ndarray | None,
                                  piv_lo: list[tuple[int,float]],
                                  cfg: dict,
                                  atr_mean: float,
                                  prefer_atr: bool,
                                  debug_log: list | None = None) -> list[dict]:
    """Inverse H&S: three lows with middle (head) lower than shoulders + rough time symmetry."""
    out = []
    geom = cfg.get("geometry", {})
    sym_pct = float(geom.get("shoulder_timing_similarity_pct", 40)) / 100.0
    dur     = geom.get("duration", {})
    min_span = int(dur.get("min_bars", 20)); max_span = int(dur.get("max_bars", 120))
    shoulder_sim = float(geom.get("shoulder_height_similarity_pct", 25)) / 100.0
    min_height_atr = float(geom.get("min_height_atr", 0.0))
    trend_lookback = int(geom.get("trend_lookback_bars", 0))
    piv_sorted = sorted(piv_lo, key=lambda t: t[0])
    for iL, pL in piv_sorted:
        for iH, pH in piv_sorted:
            if iH <= iL + 2: continue
            for iR, pR in piv_sorted:
                if iR <= iH + 2: continue
                span = iR - iL
                if span < min_span or span > max_span: continue
                sh_avg = 0.5*(pL+pR)
                if abs(pL - pR)/max(1e-9, sh_avg) > shoulder_sim:
                    if debug_log is not None:
                        debug_log.append({"pattern":"inverse_head_shoulders","reason":"shoulder_height","iL":iL,"iR":iR})
                    continue
                head_need = _choose_abs_threshold(
                    geom.get("head_above_shoulders_min_atr"),
                    geom.get("head_above_shoulders_min_pct"),
                    atr_mean,
                    sh_avg,
                    prefer_atr
                )
                if (sh_avg - pH) < max(head_need, 0.01 * sh_avg):
                    if debug_log is not None:
                        debug_log.append({"pattern":"inverse_head_shoulders","reason":"head_height","iH":iH,"need":head_need})
                    continue
                if min_height_atr > 0 and (max(pL, pR) - pH) < (min_height_atr * max(1e-9, atr_mean)):
                    if debug_log is not None:
                        debug_log.append({"pattern":"inverse_head_shoulders","reason":"height_floor","height":max(pL,pR)-pH,"need":min_height_atr*atr_mean})
                    continue
                ideal = 0.5*(iL+iH)
                tol_bars = max(1, int(sym_pct * span))
                if abs(iR - ideal) > tol_bars:
                    if debug_log is not None:
                        debug_log.append({"pattern":"inverse_head_shoulders","reason":"timing_symmetry","iR":iR})
                    continue
                if trend_lookback > 0 and not _has_trend(closes, iL, trend_lookback, direction="down"):
                    if debug_log is not None:
                        debug_log.append({"pattern":"inverse_head_shoulders","reason":"no_downtrend","iL":iL})
                    continue
                if volume is not None and len(volume) == len(closes):
                    vol_L = float(volume[iL]) if iL < len(volume) else 0.0
                    vol_H = float(volume[iH]) if iH < len(volume) else 0.0
                    vol_R = float(volume[iR]) if iR < len(volume) else 0.0
                    if not (vol_L < 0.95 * vol_H and vol_R < 0.95 * vol_L):
                        if debug_log is not None:
                            debug_log.append({"pattern":"inverse_head_shoulders","reason":"volume_order","vol_L":vol_L,"vol_H":vol_H,"vol_R":vol_R})
                        continue
                confirmed = True
                if cfg.get("breakout", {}).get("confirm"):
                    br = cfg.get("breakout", {}) or {}
                    brc = br.get("confirm", {}) or {}
                    br_side = br.get("side", "up")
                    br_thr_atr = brc.get("threshold_atr", None)
                    br_thr_pct = brc.get("threshold_percent", None)
                    br_within = brc.get("within_bars", None)
                    t1_idx = iL + int(np.argmax(closes[iL:iH+1]))
                    t2_idx = iH + int(np.argmax(closes[iH:iR+1]))
                    neckline = 0.5 * (closes[t1_idx] + closes[t2_idx])
                    thr_abs = _choose_abs_threshold(br_thr_atr, br_thr_pct, atr_mean, neckline, prefer_atr)
                    future = closes[iR+1 : iR+1 + (br_within or 0)] if br_within else closes[iR+1 :]
                    confirmed = len(future) > 0 and _level_breakout_confirm(future, level=neckline, side=br_side, thr_abs=thr_abs, within_bars=br_within)
                    if not confirmed:
                        continue
                out.append({"type":"inverse_head_shoulders","iL":iL,"pL":pL,"iH":iH,"pH":pH,"iR":iR,"pR":pR,"breakout_confirmed": True})
    return out

def detect_triangle(closes: np.ndarray, piv_hi: list[tuple[int,float]], piv_lo: list[tuple[int,float]], cfg_geom: dict) -> list[dict]:
    """Generic converging upper/lower envelopes."""
    out = []
    min_tu = int(cfg_geom.get("upper_trendline", {}).get("min_touches", 3))
    min_tl = int(cfg_geom.get("lower_boundary", {}).get("min_touches", 3))
    tol_rel = float(cfg_geom.get("envelope_tol_rel", 0.004))
    conv_min = float(cfg_geom.get("convergence_min_rel", 0.01))
    dur = cfg_geom.get("duration", {})
    min_span = int(dur.get("min_bars", 30)); max_span = int(dur.get("max_bars", 160))
    if len(piv_hi) < min_tu or len(piv_lo) < min_tl: return out
    for s in range(0, len(closes) - min_span):
        e = min(len(closes)-1, s + max_span)
        if e - s < min_span: continue
        hi = [(i,p) for (i,p) in piv_hi if s <= i <= e]
        lo = [(i,p) for (i,p) in piv_lo if s <= i <= e]
        if len(hi) < min_tu or len(lo) < min_tl: continue
        xi_hi = np.array([i for (i,_) in hi]); yi_hi = np.array([p for (_,p) in hi])
        xi_lo = np.array([i for (i,_) in lo]); yi_lo = np.array([p for (_,p) in lo])
        a_hi, b_hi = np.polyfit(xi_hi, yi_hi, 1)
        a_lo, b_lo = np.polyfit(xi_lo, yi_lo, 1)
        gap_s = (a_hi*s + b_hi) - (a_lo*s + b_lo)
        gap_e = (a_hi*e + b_hi) - (a_lo*e + b_lo)
        mid = 0.5*((a_hi*s+b_hi) + (a_lo*s+b_lo))
        if mid == 0: continue
        if (gap_s - gap_e)/abs(mid) < conv_min: continue
        ok_hi = np.mean(np.abs(yi_hi - (a_hi*xi_hi+b_hi))/np.maximum(1e-9, np.abs(yi_hi))) < tol_rel
        ok_lo = np.mean(np.abs(yi_lo - (a_lo*xi_lo+b_lo))/np.maximum(1e-9, np.abs(yi_lo))) < tol_rel
        if not (ok_hi and ok_lo): continue
        out.append({"type":"triangle","start":s,"end":e,"a_hi":float(a_hi),"b_hi":float(b_hi),"a_lo":float(a_lo),"b_lo":float(b_lo)})
    return out

def is_descending_triangle(det: dict, closes: np.ndarray, flat_tol_rel: float = 0.002) -> bool:
    """Upper slope negative, lower nearly flat (normalized by mid-price)."""
    a_hi = det["a_hi"]; a_lo = det["a_lo"]
    s = det["start"]; e = det["end"]; mid = float(closes[s:e+1].mean())
    if mid == 0: return False
    cond_upper_down = (a_hi < 0.0)
    cond_lower_flat = abs(a_lo)/abs(mid) < flat_tol_rel
    return cond_upper_down and cond_lower_flat

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
    debug_log = []

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
            vol = win["volume"].to_numpy() if "volume" in win.columns else None

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
                det_map["head_and_shoulders"] = detect_head_shoulders(
                    c,
                    vol,
                    piv_hi,
                    eff_patterns.get("head_and_shoulders", {}),
                    atr_mean,
                    prefer_atr,
                    debug_log
                )

            if "inverse_head_and_shoulders" in active:
                det_map["inverse_head_and_shoulders"] = detect_inverse_head_shoulders(
                    c,
                    vol,
                    piv_lo,
                    eff_patterns.get("inverse_head_and_shoulders", eff_patterns.get("head_and_shoulders", {})),
                    atr_mean,
                    prefer_atr,
                    debug_log
                )

            # Robust Double Top / Bottom (ATR thresholds, duration, spacing, breakout, NMS)
            if "double_top" in active:
                det_map["double_top"] = detect_double_top(
                    close=c,
                    volume=vol,
                    piv_hi=piv_hi,
                    cfg_dt=eff_patterns.get("double_top", {}),
                    atr=atr,
                    prefer_atr=prefer_atr,
                    labels_post=labels_post,
                    debug_log=debug_log
                )

            if "double_bottom" in active:
                cfg_db = deep_merge_dict(eff_patterns.get("double_top", {}), eff_patterns.get("double_bottom", {}))
                det_map["double_bottom"] = detect_double_bottom(
                    close=c,
                    volume=vol,
                    piv_lo=piv_lo,
                    cfg_db=cfg_db,
                    atr=atr,
                    prefer_atr=prefer_atr,
                    labels_post=labels_post,
                    debug_log=debug_log
                )

            # Descending triangle: geometry + image Hough gate
            if "descending_triangle" in active:
                tri_cfg = eff_patterns.get("descending_triangle", {}) or {}
                tri_geom = tri_cfg.get("geometry", {}) or {}
                tri_all = detect_triangle(c, piv_hi, piv_lo, tri_geom)
                # image-based confirmation
                tri_ok = []
                if tri_all:
                    if hough_desc_triangle_ok(png, tri_cfg):
                        tri_ok = tri_all
                if tri_ok:
                    centers = [{"icenter": int(0.5*(d["start"]+d["end"])), **d} for d in tri_ok]
                    tri_ok = _nms_time(centers, int(labels_post.get("dedup_time_overlap_bars", 10)))
                det_map["descending_triangle"] = tri_ok

            # --- presence flags and counts ---
            counts = {name+"_cnt": len(det_map[name]) for name in active}
            confirmed_counts = {
                name+"_confirmed_cnt": sum(1 for d in det_map[name] if d.get("breakout_confirmed", True))
                for name in active
            }

            flags = {name: (1 if confirmed_counts[name+"_confirmed_cnt"] > 0 else 0) for name in active}

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
                    raw = counts.get(name+"_cnt", 0)
                    conf = confirmed_counts.get(name+"_confirmed_cnt", 0)
                    if name == "double_top":
                        pos_votes, total_votes, p = lf_votes_double_top(det_map[name], atr_mean)
                        row[f"p_{name}"] = round(p,4)
                        row[f"lf_votes_{name}"] = pos_votes
                        row[f"lf_total_{name}"] = total_votes
                        row[f"y_{name}"] = 1 if p >= 0.9 else 0 if p <= 0.1 else -1
                    elif name == "double_bottom":
                        pos_votes, total_votes, p = lf_votes_double_bottom(det_map[name], atr_mean)
                        row[f"p_{name}"] = round(p,4)
                        row[f"lf_votes_{name}"] = pos_votes
                        row[f"lf_total_{name}"] = total_votes
                        row[f"y_{name}"] = 1 if p >= 0.9 else 0 if p <= 0.1 else -1
                    else:
                        row[f"y_{name}"] = int(flags[name])
                        row[f"p_{name}"] = ""
                        row[f"lf_votes_{name}"] = ""
                        row[f"lf_total_{name}"] = ""
                    row[f"{name}_cnt"] = int(raw)
                    row[f"{name}_confirmed_cnt"] = int(conf)
                else:
                    row[f"y_{name}"] = 0
                    row[f"{name}_cnt"] = 0
                    row[f"{name}_confirmed_cnt"] = 0
                    row[f"p_{name}"] = ""
                    row[f"lf_votes_{name}"] = ""
                    row[f"lf_total_{name}"] = ""

            rows.append(row)

        print(f"✅ Split {split}: computed labels for {len(pngs)} images.")

    out_csv = Path(args.labels_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(by=["split","symbol","end_ts","image"])
    df.to_csv(out_csv, index=False)
    print(f"✅ Presence-labels CSV written: {out_csv} (rows={len(df)})")
    if debug_log:
        dbg_path = out_csv.with_suffix(".debug.jsonl")
        with dbg_path.open("w", encoding="utf-8") as f:
            for item in debug_log:
                f.write(json.dumps(item) + "\n")
        print(f"ℹ️ Debug log written: {dbg_path} (entries={len(debug_log)})")


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .swing_points import compute_atr


def _choose_abs_threshold(value_atr, value_pct, atr_mean, mid_price, prefer_atr=True) -> float:
    if prefer_atr and value_atr is not None:
        return float(value_atr) * atr_mean
    if value_pct is not None:
        return float(value_pct) * max(1e-9, mid_price)
    return 0.003 * max(1e-9, mid_price)


def _level_breakout_confirm(close_future: np.ndarray, level: float, side: str, thr_abs: float, within_bars: int | None) -> bool:
    rng = range(0, min(len(close_future), within_bars)) if within_bars else range(0, len(close_future))
    if side == "down":
        return any(close_future[k] <= level - thr_abs for k in rng)
    return any(close_future[k] >= level + thr_abs for k in rng)


def _nms_time(dets: List[Dict], dedup_bars: int) -> List[Dict]:
    dets = sorted(dets, key=lambda d: -d.get("score", 0.0))
    kept: List[Dict] = []
    centers: List[int] = []
    for d in dets:
        ic = d.get("icenter")
        if ic is None:
            kept.append(d)
            continue
        if any(abs(ic - kc) < dedup_bars for kc in centers):
            continue
        centers.append(ic)
        kept.append(d)
    return kept


def _apply_dynamic_floor(
    base_abs: float,
    dyn_cfg: Dict | None,
    pct_key: str,
    pattern_height: float,
    price_ref: float,
    atr_recent: float,
) -> float:
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


def _pivots_from_swings(df: pd.DataFrame, swings: pd.DataFrame, kind: str) -> List[Tuple[int, float]]:
    flag_col = "swing_high" if kind == "high" else "swing_low"
    price_col = "swing_high_price" if kind == "high" else "swing_low_price"
    fallback = "high" if kind == "high" else "low"
    pivots: List[Tuple[int, float]] = []
    if swings is None or flag_col not in swings:
        return pivots
    for pos, (_, row) in enumerate(swings.iterrows()):
        if not bool(row.get(flag_col, False)):
            continue
        price = row.get(price_col)
        if price is None or (isinstance(price, float) and np.isnan(price)):
            price = float(df[fallback].iloc[pos])
        pivots.append((pos, float(price)))
    return pivots


def _volume_array(df: pd.DataFrame) -> np.ndarray | None:
    if "volume" not in df.columns:
        return None
    vol = df["volume"].to_numpy(dtype=float)
    if len(vol) == 0:
        return None
    return vol


def detect_double_top(
    close: np.ndarray,
    volume: np.ndarray | None,
    piv_hi: List[Tuple[int, float]],
    cfg_dt: Dict,
    atr: np.ndarray,
    prefer_atr: bool,
    labels_post: Dict | None = None,
) -> List[Dict]:
    out: List[Dict] = []
    atr_mean = float(np.nanmean(atr)) if len(atr) else 0.0
    geom = cfg_dt.get("geometry", {}) or {}
    dur = geom.get("duration", {}) or {}
    min_span = int(dur.get("min_bars", 16))
    max_span = int(dur.get("max_bars", 120))
    sim_tol = float(geom.get("peak_height_similarity_pct", 15)) / 100.0
    valley_sep = int(geom.get("valley_min_bars_from_peaks", 2))
    max_pairs = max(1, int(geom.get("max_pairs_per_peak", 5)))

    br = cfg_dt.get("breakout", {}) or {}
    br_side = br.get("side", "down")
    brc = br.get("confirm", {}) or {}
    br_thr_atr = brc.get("threshold_atr")
    br_thr_pct = brc.get("threshold_percent")
    br_within = brc.get("within_bars")
    br_vol_mult = brc.get("volume_mult")
    require_confirmation = bool(br.get("require_confirmation", True))
    allow_pre_breakout = bool(cfg_dt.get("label_allow_pre_breakout", False))

    sc = cfg_dt.get("scoring", {}) or {}
    w_depth = float(sc.get("weight_valley_depth_atr", 1.0))
    w_span = float(sc.get("weight_span", 0.2))

    dyn = geom.get("dynamic_thresholds", {}) or {}
    dyn_lookback = int(dyn.get("vol_lookback_bars", 36))
    vol_lookback = max(dyn_lookback, 50)

    min_height_atr = float(geom.get("min_height_atr", 0.0))

    def _recent_atr() -> float:
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
            if span < min_span or span > max_span:
                continue
            mid = 0.5 * (p1 + p2)
            if abs(p1 - p2) / max(1e-9, mid) > sim_tol:
                continue
            j0, j1 = i1 + valley_sep, i2 - valley_sep
            if j1 <= j0:
                continue
            local = close[j0:j1]
            if len(local) == 0:
                continue
            v_idx = int(np.argmin(local) + j0)
            v = float(close[v_idx])

            base_need = _choose_abs_threshold(
                geom.get("valley_drop_min_atr"),
                geom.get("valley_drop_min_pct"),
                atr_mean,
                mid,
                prefer_atr,
            )
            pattern_height = max(p1, p2) - v
            recent_atr = _recent_atr()
            dyn_need = _apply_dynamic_floor(
                base_need,
                dyn,
                pct_key="valley_drop_min_height_pct",
                pattern_height=pattern_height,
                price_ref=mid,
                atr_recent=recent_atr,
            )
            if (mid - v) < dyn_need:
                continue
            if min_height_atr > 0 and (max(p1, p2) - v) < (min_height_atr * max(1e-9, atr_mean)):
                continue

            breakout_ok = True
            thr_abs = 0.0
            if brc:
                base_thr = _choose_abs_threshold(br_thr_atr, br_thr_pct, atr_mean, mid, prefer_atr)
                thr_abs = _apply_dynamic_floor(
                    base_thr,
                    dyn,
                    pct_key="breakout_height_pct",
                    pattern_height=pattern_height,
                    price_ref=mid,
                    atr_recent=recent_atr,
                )
                future = close[i2 + 1 : i2 + 1 + br_within] if br_within else close[i2 + 1 :]
                if len(future) == 0:
                    continue
                breakout_ok = _level_breakout_confirm(future, level=v, side=br_side, thr_abs=thr_abs, within_bars=br_within)
                if breakout_ok and volume is not None and len(volume) == len(close):
                    mult = float(br_vol_mult) if br_vol_mult is not None else 1.1
                    recent_vol = float(np.nanmean(volume[max(0, i2 - vol_lookback) : i2])) if vol_lookback > 0 else float(np.nanmean(volume))
                    brk_window = br_within or 1
                    brk_vol = float(np.nanmean(volume[i2 + 1 : i2 + 1 + brk_window]))
                    if recent_vol > 0 and brk_vol < mult * recent_vol:
                        breakout_ok = False
            if not breakout_ok and require_confirmation and not allow_pre_breakout:
                continue

            ic = int(0.5 * (i1 + i2))
            depth = (mid - v) / max(1e-9, atr_mean)
            span_score = span / max(1.0, float(max_span))
            score = w_depth * depth + w_span * span_score
            out.append(
                {
                    "type": "double_top",
                    "i1": i1,
                    "p1": p1,
                    "i2": i2,
                    "p2": p2,
                    "ivalley": v_idx,
                    "pvalley": v,
                    "span": span,
                    "score": float(score),
                    "icenter": ic,
                    "breakout_confirmed": bool(breakout_ok),
                }
            )
    dedup = int((labels_post or {}).get("dedup_time_overlap_bars", 10))
    return _nms_time(out, dedup)


def detect_double_bottom(
    close: np.ndarray,
    volume: np.ndarray | None,
    piv_lo: List[Tuple[int, float]],
    cfg_db: Dict,
    atr: np.ndarray,
    prefer_atr: bool,
    labels_post: Dict | None = None,
) -> List[Dict]:
    out: List[Dict] = []
    atr_mean = float(np.nanmean(atr)) if len(atr) else 0.0
    geom = cfg_db.get("geometry", {}) or {}
    dur = geom.get("duration", {}) or {}
    min_span = int(dur.get("min_bars", 16))
    max_span = int(dur.get("max_bars", 120))
    sim_tol = float(geom.get("peak_height_similarity_pct", 15)) / 100.0
    peak_sep = int(geom.get("valley_min_bars_from_peaks", 2))
    max_pairs = max(1, int(geom.get("max_pairs_per_peak", 5)))

    br = cfg_db.get("breakout", {}) or {}
    br_side = br.get("side", "up")
    brc = br.get("confirm", {}) or {}
    br_thr_atr = brc.get("threshold_atr")
    br_thr_pct = brc.get("threshold_percent")
    br_within = brc.get("within_bars")
    br_vol_mult = brc.get("volume_mult")
    require_confirmation = bool(br.get("require_confirmation", True))
    allow_pre_breakout = bool(cfg_db.get("label_allow_pre_breakout", False))

    sc = cfg_db.get("scoring", {}) or {}
    w_rise = float(sc.get("weight_valley_depth_atr", 1.0))
    w_span = float(sc.get("weight_span", 0.2))

    dyn = geom.get("dynamic_thresholds", {}) or {}
    dyn_lookback = int(dyn.get("vol_lookback_bars", 36))
    vol_lookback = max(dyn_lookback, 60)

    min_height_atr = float(geom.get("min_height_atr", 0.0))

    def _recent_atr() -> float:
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
            if span < min_span or span > max_span:
                continue
            mid = 0.5 * (p1 + p2)
            if abs(p1 - p2) / max(1e-9, mid) > sim_tol:
                continue
            j0, j1 = i1 + peak_sep, i2 - peak_sep
            if j1 <= j0:
                continue
            local = close[j0:j1]
            if len(local) == 0:
                continue
            peak_idx = int(np.argmax(local) + j0)
            pk = float(close[peak_idx])

            base_need = _choose_abs_threshold(
                geom.get("peak_rise_min_atr"),
                geom.get("peak_rise_min_pct"),
                atr_mean,
                mid,
                prefer_atr,
            )
            pattern_height = pk - min(p1, p2)
            recent_atr = _recent_atr()
            dyn_need = _apply_dynamic_floor(
                base_need,
                dyn,
                pct_key="valley_drop_min_height_pct",
                pattern_height=pattern_height,
                price_ref=mid,
                atr_recent=recent_atr,
            )
            if (pk - mid) < dyn_need:
                continue
            if min_height_atr > 0 and (pk - min(p1, p2)) < (min_height_atr * max(1e-9, atr_mean)):
                continue

            breakout_ok = True
            thr_abs = 0.0
            if brc:
                base_thr = _choose_abs_threshold(br_thr_atr, br_thr_pct, atr_mean, mid, prefer_atr)
                thr_abs = _apply_dynamic_floor(
                    base_thr,
                    dyn,
                    pct_key="breakout_height_pct",
                    pattern_height=pattern_height,
                    price_ref=mid,
                    atr_recent=recent_atr,
                )
                future = close[i2 + 1 : i2 + 1 + br_within] if br_within else close[i2 + 1 :]
                if len(future) == 0:
                    continue
                breakout_ok = _level_breakout_confirm(future, level=pk, side=br_side, thr_abs=thr_abs, within_bars=br_within)
                if breakout_ok and volume is not None and len(volume) == len(close):
                    mult = float(br_vol_mult) if br_vol_mult is not None else 1.1
                    recent_vol = float(np.nanmean(volume[max(0, i2 - vol_lookback) : i2])) if vol_lookback > 0 else float(np.nanmean(volume))
                    brk_window = br_within or 1
                    brk_vol = float(np.nanmean(volume[i2 + 1 : i2 + 1 + brk_window]))
                    if recent_vol > 0 and brk_vol < mult * recent_vol:
                        breakout_ok = False
            if not breakout_ok and require_confirmation and not allow_pre_breakout:
                continue

            ic = int(0.5 * (i1 + i2))
            rise = (pk - mid) / max(1e-9, atr_mean)
            span_score = span / max(1.0, float(max_span))
            score = w_rise * rise + w_span * span_score
            out.append(
                {
                    "type": "double_bottom",
                    "i1": i1,
                    "p1": p1,
                    "i2": i2,
                    "p2": p2,
                    "ipeak": peak_idx,
                    "ppeak": pk,
                    "span": span,
                    "score": float(score),
                    "icenter": ic,
                    "breakout_confirmed": bool(breakout_ok),
                }
            )
    dedup = int((labels_post or {}).get("dedup_time_overlap_bars", 10))
    return _nms_time(out, dedup)


def label_double_top_window(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    config: Dict | None,
    units_cfg: Dict | None = None,
    labels_post: Dict | None = None,
) -> Tuple[int, Dict[str, float]]:
    config = config or {}
    prefer_atr = bool((units_cfg or {}).get("prefer_atr_over_percent", True))
    atr_period = int(config.get("atr_period", 14))
    atr = compute_atr(df, atr_period=atr_period).to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    volume = _volume_array(df)
    piv_hi = _pivots_from_swings(df, swings, "high")
    detections = detect_double_top(close, volume, piv_hi, config, atr, prefer_atr, labels_post)
    if not detections:
        return 0, {
            "dt_peak_similarity": 0.0,
            "dt_valley_depth": 0.0,
            "dt_span_bars": 0.0,
            "dt_breakout_confirmed": 0.0,
            "dt_score": 0.0,
        }
    best = max(detections, key=lambda d: d.get("score", 0.0))
    mid = 0.5 * (best["p1"] + best["p2"])
    valley = best.get("pvalley", mid)
    sim = 1.0 - abs(best["p1"] - best["p2"]) / max(mid, 1e-9)
    depth = (mid - valley) / max(mid, 1e-9)
    feats = {
        "dt_peak_similarity": float(sim),
        "dt_valley_depth": float(depth),
        "dt_span_bars": float(best.get("span", 0.0)),
        "dt_breakout_confirmed": 1.0 if best.get("breakout_confirmed") else 0.0,
        "dt_score": float(best.get("score", 0.0)),
    }
    return 1, feats


def label_double_bottom_window(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    config: Dict | None,
    units_cfg: Dict | None = None,
    labels_post: Dict | None = None,
) -> Tuple[int, Dict[str, float]]:
    config = config or {}
    prefer_atr = bool((units_cfg or {}).get("prefer_atr_over_percent", True))
    atr_period = int(config.get("atr_period", 14))
    atr = compute_atr(df, atr_period=atr_period).to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    volume = _volume_array(df)
    piv_lo = _pivots_from_swings(df, swings, "low")
    detections = detect_double_bottom(close, volume, piv_lo, config, atr, prefer_atr, labels_post)
    if not detections:
        return 0, {
            "db_trough_similarity": 0.0,
            "db_peak_height": 0.0,
            "db_span_bars": 0.0,
            "db_breakout_confirmed": 0.0,
            "db_score": 0.0,
        }
    best = max(detections, key=lambda d: d.get("score", 0.0))
    mid = 0.5 * (best["p1"] + best["p2"])
    peak = best.get("ppeak", mid)
    sim = 1.0 - abs(best["p1"] - best["p2"]) / max(mid, 1e-9)
    rise = (peak - mid) / max(mid, 1e-9)
    feats = {
        "db_trough_similarity": float(sim),
        "db_peak_height": float(rise),
        "db_span_bars": float(best.get("span", 0.0)),
        "db_breakout_confirmed": 1.0 if best.get("breakout_confirmed") else 0.0,
        "db_score": float(best.get("score", 0.0)),
    }
    return 1, feats


__all__ = [
    "detect_double_top",
    "detect_double_bottom",
    "label_double_top_window",
    "label_double_bottom_window",
]

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .swing_points import compute_atr

LOG = logging.getLogger(__name__)


def _safe_mean(values: List[float]) -> float:
    arr = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(arr)) if arr else 0.0


def _threshold_abs(value_atr: float | None, value_pct: float | None, atr_val: float, price_ref: float) -> float:
    if value_atr is not None:
        return float(value_atr) * max(atr_val, 1e-9)
    if value_pct is not None:
        return float(value_pct) * max(price_ref, 1e-9)
    return 0.0


def _neckline_value(v1_idx: int, v1_price: float, v2_idx: int, v2_price: float, idx: int) -> float:
    if v2_idx == v1_idx:
        return v1_price
    slope = (v2_price - v1_price) / (v2_idx - v1_idx)
    return v1_price + slope * (idx - v1_idx)


def _neckline_slope_deg(v1_idx: int, v1_price: float, v2_idx: int, v2_price: float) -> float:
    if v2_idx == v1_idx:
        return 0.0
    slope = (v2_price - v1_price) / (v2_idx - v1_idx)
    return float(np.degrees(np.arctan(slope)))


def detect_hs_pattern(df: pd.DataFrame, swings: pd.DataFrame, config: Dict | None) -> List[Dict]:
    """
    Scan swing points and identify head & shoulders candidates.

    Returns a list of dictionaries describing each detection.
    """
    if config is None:
        config = {}
    geom = config.get("geometry", {})
    duration_cfg = geom.get("duration", {})
    breakout_cfg = (config.get("breakout") or {}).get("confirm", {})

    min_bars = int(duration_cfg.get("min_bars", 36))
    max_bars = int(duration_cfg.get("max_bars", 200))
    max_neckline_deg = float(geom.get("neckline", {}).get("max_slope_deg", 12))
    shoulder_sim_pct = float(geom.get("shoulder_height_similarity_pct", 25)) / 100.0
    shoulder_time_pct = float(geom.get("shoulder_timing_similarity_pct", 60)) / 100.0
    head_min_atr = float(geom.get("head_above_shoulders_min_atr", 0.5))
    head_min_pct = float(geom.get("head_above_shoulders_min_pct", 0.006))

    breakout_side = (config.get("breakout") or {}).get("side", "down")
    breakout_within = breakout_cfg.get("within_bars")
    breakout_thr_atr = breakout_cfg.get("threshold_atr")
    breakout_thr_pct = breakout_cfg.get("threshold_percent")

    atr = swings["atr"] if "atr" in swings else compute_atr(df)
    atr_vals = atr.to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    index_positions = {idx: pos for pos, idx in enumerate(df.index)}

    events: List[Tuple[int, str, float]] = []
    for pos, (ts, row) in enumerate(swings.iterrows()):
        if row.get("swing_high"):
            val = row.get("swing_high_price")
            price = float(val) if val == val else float(df["high"].iloc[pos])
            events.append((pos, "high", price))
        if row.get("swing_low"):
            val = row.get("swing_low_price")
            price = float(val) if val == val else float(df["low"].iloc[pos])
            events.append((pos, "low", price))

    detections: List[Dict] = []
    for i in range(len(events) - 4):
        seq = events[i : i + 5]
        pattern = [typ for _, typ, _ in seq]
        if pattern != ["high", "low", "high", "low", "high"]:
            continue
        (p1_idx, _, p1_price), (v1_idx, _, v1_price), (p2_idx, _, p2_price), (v2_idx, _, v2_price), (p3_idx, _, p3_price) = seq
        if not (min_bars <= (p3_idx - p1_idx) <= max_bars):
            continue
        head = p2_price
        shoulders = [p1_price, p3_price]
        shoulder_mid = _safe_mean(shoulders)
        atr_slice = atr_vals[p1_idx : p3_idx + 1]
        atr_mean = float(np.nanmean(atr_slice)) if atr_slice.size else float(np.nanmean(atr_vals))
        atr_mean = max(atr_mean, 1e-6)

        # head prominence
        if (head - shoulder_mid) < max(
            head_min_atr * atr_mean,
            head_min_pct * max(shoulder_mid, 1e-6),
        ):
            continue

        shoulder_sim = abs(shoulders[0] - shoulders[1]) / max(shoulder_mid, 1e-6)
        if shoulder_sim > shoulder_sim_pct:
            continue

        # temporal symmetry: compare spacing left vs right shoulder
        ls_width = v1_idx - p1_idx
        rs_width = p3_idx - v2_idx
        total = max((p3_idx - p1_idx), 1)
        time_sym = 1.0 - abs(ls_width - rs_width) / total
        if time_sym < shoulder_time_pct:
            continue

        # neckline slope constraint
        neck_deg = abs(_neckline_slope_deg(v1_idx, v1_price, v2_idx, v2_price))
        if neck_deg > max_neckline_deg:
            continue

        # breakout
        breakout_ok = False
        breakout_bar_delta = None
        thr_abs = _threshold_abs(breakout_thr_atr, breakout_thr_pct, atr_mean, shoulder_mid)
        if breakout_within is None:
            breakout_within = int(max_bars / 4)
        end_lookup = min(len(closes), p3_idx + breakout_within + 1)
        for future_idx in range(p3_idx + 1, end_lookup):
            level = _neckline_value(v1_idx, v1_price, v2_idx, v2_price, future_idx)
            price = closes[future_idx]
            if breakout_side == "down":
                if price <= level - thr_abs:
                    breakout_ok = True
                    breakout_bar_delta = future_idx - p3_idx
                    break
            else:
                if price >= level + thr_abs:
                    breakout_ok = True
                    breakout_bar_delta = future_idx - p3_idx
                    break

        detections.append(
            {
                "pattern": "head_and_shoulders",
                "start_idx": p1_idx,
                "end_idx": p3_idx,
                "p1_idx": p1_idx,
                "p2_idx": p2_idx,
                "p3_idx": p3_idx,
                "v1_idx": v1_idx,
                "v2_idx": v2_idx,
                "p1_price": p1_price,
                "p2_price": p2_price,
                "p3_price": p3_price,
                "v1_price": v1_price,
                "v2_price": v2_price,
                "head_to_shoulder_ratio": float((head - shoulder_mid) / max(atr_mean, 1e-6)),
                "shoulder_similarity": float(1.0 - shoulder_sim),
                "temporal_symmetry": float(time_sym),
                "neckline_slope_deg": float(neck_deg),
                "breakout_confirmed": bool(breakout_ok),
                "breakout_bars": breakout_bar_delta if breakout_bar_delta is not None else -1,
                "span_bars": int(p3_idx - p1_idx),
            }
        )

    LOG.debug("H&S detections=%d", len(detections))
    return detections


def label_hs_window(df: pd.DataFrame, swings: pd.DataFrame, config: Dict | None) -> Tuple[int, Dict]:
    """
    Compute a binary H&S label plus structural features.

    Returns
    -------
    (y, features_dict)
    """
    detections = detect_hs_pattern(df, swings, config)
    if not detections:
        return 0, {
            "hs_neckline_slope_deg": 0.0,
            "hs_head_to_shoulder_ratio": 0.0,
            "hs_shoulder_similarity": 0.0,
            "hs_temporal_symmetry": 0.0,
            "hs_breakout_confirmed": 0.0,
            "hs_span_bars": 0.0,
        }
    best = max(detections, key=lambda d: d["head_to_shoulder_ratio"])
    feats = {
        "hs_neckline_slope_deg": best["neckline_slope_deg"],
        "hs_head_to_shoulder_ratio": best["head_to_shoulder_ratio"],
        "hs_shoulder_similarity": best["shoulder_similarity"],
        "hs_temporal_symmetry": best["temporal_symmetry"],
        "hs_breakout_confirmed": 1.0 if best["breakout_confirmed"] else 0.0,
        "hs_span_bars": float(best["span_bars"]),
    }
    return 1, feats


__all__ = ["detect_hs_pattern", "label_hs_window"]

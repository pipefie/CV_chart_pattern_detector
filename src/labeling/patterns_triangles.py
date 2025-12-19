from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .swing_points import compute_atr

LOG = logging.getLogger(__name__)


def _fit_line(points: List[Tuple[int, float]]) -> Tuple[float, float]:
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if len(points) < 2:
        return 0.0, float(ys[0] if len(ys) else 0.0)
    a, b = np.polyfit(xs, ys, 1)
    return float(a), float(b)


def _slope_deg(slope: float) -> float:
    return float(np.degrees(np.arctan(slope)))


def detect_ascending_triangle(df: pd.DataFrame, swings: pd.DataFrame, config: Dict | None) -> List[Dict]:
    """
    Detect ascending triangle candidates using swing highs/lows.
    """
    if config is None:
        config = {}
    geom = config.get("geometry", {})
    duration_cfg = geom.get("duration", {})
    breakout_cfg = config.get("breakout", {})

    min_bars = int(duration_cfg.get("min_bars", 30))
    max_bars = int(duration_cfg.get("max_bars", 160))
    max_peak_diff_pct = float(geom.get("max_peak_diff_pct", 0.01))
    max_res_slope = float(geom.get("max_resistance_slope_deg", 6))
    min_sup_slope = float(geom.get("min_support_slope_deg", 8))
    min_lows = int(geom.get("min_lows", 2))

    breakout_within_pct = float(breakout_cfg.get("max_time_to_break_pct", 0.75))
    br_side = breakout_cfg.get("side", "up")
    confirm = breakout_cfg.get("confirm", {})
    br_thr_atr = confirm.get("threshold_atr")
    br_thr_pct = confirm.get("threshold_percent")

    atr = swings["atr"] if "atr" in swings else compute_atr(df)
    atr_vals = atr.to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    swing_rows = list(swings.itertuples())

    def _pivot_price(row, attr: str, fallback_series: pd.Series, idx: int) -> float:
        val = getattr(row, attr, np.nan)
        if val is None or np.isnan(val):
            return float(fallback_series.iloc[idx])
        return float(val)

    peaks = [
        (idx, _pivot_price(row, "swing_high_price", df["high"], idx))
        for idx, row in enumerate(swing_rows)
        if getattr(row, "swing_high", False)
    ]
    troughs = [
        (idx, _pivot_price(row, "swing_low_price", df["low"], idx))
        for idx, row in enumerate(swing_rows)
        if getattr(row, "swing_low", False)
    ]

    detections: List[Dict] = []
    if len(peaks) < 2 or len(troughs) < min_lows:
        return detections

    for i in range(len(peaks) - 1):
        for j in range(i + 1, len(peaks)):
            p1 = peaks[i]
            p2 = peaks[j]
            span = p2[0] - p1[0]
            if not (min_bars <= span <= max_bars):
                continue
            peak_mid = 0.5 * (p1[1] + p2[1])
            if abs(p1[1] - p2[1]) / max(peak_mid, 1e-6) > max_peak_diff_pct:
                continue

            # gather lows between the two peaks
            lows_window = [pt for pt in troughs if p1[0] <= pt[0] <= p2[0]]
            if len(lows_window) < min_lows:
                continue

            res_slope, res_intercept = _fit_line([p1, p2])
            res_slope_deg = abs(_slope_deg(res_slope))
            if res_slope_deg > max_res_slope:
                continue

            sup_slope, sup_intercept = _fit_line(lows_window)
            sup_slope_deg = _slope_deg(sup_slope)
            if sup_slope_deg < min_sup_slope:
                continue

            # apex (intersection)
            if np.isclose(res_slope, sup_slope):
                continue
            apex_x = (sup_intercept - res_intercept) / (res_slope - sup_slope)
            apex_y = res_slope * apex_x + res_intercept
            if apex_x <= p2[0]:
                continue

            atr_slice = atr_vals[p1[0] : int(min(len(atr_vals), apex_x))]
            atr_mean = float(np.nanmean(atr_slice)) if atr_slice.size else float(np.nanmean(atr_vals))
            atr_mean = max(atr_mean, 1e-6)
            thr_abs = 0.0
            if br_thr_atr is not None:
                thr_abs = max(thr_abs, float(br_thr_atr) * atr_mean)
            if br_thr_pct is not None:
                thr_abs = max(thr_abs, float(br_thr_pct) * max(peak_mid, 1e-6))

            apex_limit = p1[0] + (apex_x - p1[0]) * breakout_within_pct
            apex_limit = min(apex_limit, len(df) - 1)
            breakout_ok = False
            breakout_bars = None
            for idx in range(p2[0] + 1, int(apex_limit) + 1):
                resistance_level = res_slope * idx + res_intercept
                price = closes[idx]
                if br_side == "up":
                    if price >= resistance_level + thr_abs:
                        breakout_ok = True
                        breakout_bars = idx - p2[0]
                        break
                else:
                    if price <= resistance_level - thr_abs:
                        breakout_ok = True
                        breakout_bars = idx - p2[0]
                        break

            gap_start = (res_slope * p1[0] + res_intercept) - (sup_slope * p1[0] + sup_intercept)
            gap_end = (res_slope * p2[0] + res_intercept) - (sup_slope * p2[0] + sup_intercept)
            convergence_ratio = (gap_start - gap_end) / max(gap_start, 1e-6)

            detections.append(
                {
                    "pattern": "ascending_triangle",
                    "start_idx": p1[0],
                    "end_idx": p2[0],
                    "peak_prices": (p1[1], p2[1]),
                    "support_slope_deg": sup_slope_deg,
                    "resistance_slope_deg": res_slope_deg,
                    "convergence_ratio": float(convergence_ratio),
                    "apex_x": float(apex_x),
                    "apex_y": float(apex_y),
                    "breakout_confirmed": bool(breakout_ok),
                    "breakout_bars": breakout_bars if breakout_bars is not None else -1,
                    "span_bars": int(span),
                }
            )

    LOG.debug("Ascending triangle detections=%d", len(detections))
    return detections


def label_ascending_triangle_window(df: pd.DataFrame, swings: pd.DataFrame, config: Dict | None) -> Tuple[int, Dict]:
    detections = detect_ascending_triangle(df, swings, config)
    if not detections:
        return 0, {
            "tri_support_slope_deg": 0.0,
            "tri_resistance_slope_deg": 0.0,
            "tri_convergence_ratio": 0.0,
            "tri_breakout_confirmed": 0.0,
            "tri_span_bars": 0.0,
        }
    best = max(detections, key=lambda d: d["convergence_ratio"])
    feats = {
        "tri_support_slope_deg": best["support_slope_deg"],
        "tri_resistance_slope_deg": best["resistance_slope_deg"],
        "tri_convergence_ratio": best["convergence_ratio"],
        "tri_breakout_confirmed": 1.0 if best["breakout_confirmed"] else 0.0,
        "tri_span_bars": float(best["span_bars"]),
    }
    return 1, feats


__all__ = ["detect_ascending_triangle", "label_ascending_triangle_window"]

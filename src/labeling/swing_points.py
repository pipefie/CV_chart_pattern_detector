from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


@dataclass
class SwingPointConfig:
    """Configuration for swing detection."""

    window: int = 3
    min_distance_bars: int = 4
    prominence_atr: float = 0.8
    atr_period: int = 14

    @classmethod
    def from_dict(cls, data: Dict | None) -> "SwingPointConfig":
        data = data or {}
        return cls(
            window=int(data.get("window", 3)),
            min_distance_bars=int(data.get("min_distance_bars", 4)),
            prominence_atr=float(data.get("prominence_atr", 0.8)),
            atr_period=int(data.get("atr_period", 14)),
        )


def compute_atr(df: pd.DataFrame, atr_period: int = 14) -> pd.Series:
    """
    Wilder-style ATR computed on OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV window with 'high', 'low', 'close'.
    atr_period : int
        Lookback for the smoothing.
    """
    if len(df) == 0:
        return pd.Series(dtype=float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    period = int(atr_period)
    atr = tr.ewm(alpha=1 / max(1, period), adjust=False).mean()
    return atr


def detect_swing_points(
    df: pd.DataFrame,
    window: int,
    prominence_atr: float,
    min_distance_bars: int,
    atr_period: int,
) -> pd.DataFrame:
    """
    Flag swing highs/lows inside ``df`` using a symmetric rolling window.

    A bar is a swing high if its high is the maximum inside +/- ``window`` bars
    and its prominence exceeds ``prominence_atr`` × ATR. Swing lows are mirrored.

    Returns a DataFrame aligned to ``df`` with boolean columns ``swing_high`` and
    ``swing_low`` plus helper metadata (prices, ATR).
    """
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "swing_high",
                "swing_low",
                "swing_high_price",
                "swing_low_price",
                "atr",
            ]
        )

    window = max(1, int(window))
    min_distance_bars = max(1, int(min_distance_bars))
    atr = compute_atr(df, atr_period=atr_period)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    swings = pd.DataFrame(index=df.index)
    swings["swing_high"] = False
    swings["swing_low"] = False
    swings["swing_high_price"] = np.nan
    swings["swing_low_price"] = np.nan
    swings["atr"] = atr

    last_high_idx = -10**9
    last_low_idx = -10**9
    atr_array = atr.to_numpy(dtype=float)
    atr_mean = float(np.nanmean(atr_array)) if atr_array.size else 0.0

    def _local_prominence(kind: str, idx: int) -> float:
        left = max(0, idx - window)
        right = min(len(df), idx + window + 1)
        if kind == "high":
            left_low = np.min(lows[left : idx + 1]) if idx + 1 > left else lows[idx]
            right_low = np.min(lows[idx:right]) if right > idx else lows[idx]
            base = max(left_low, right_low)
            return highs[idx] - base
        else:
            left_high = np.max(highs[left : idx + 1]) if idx + 1 > left else highs[idx]
            right_high = np.max(highs[idx:right]) if right > idx else highs[idx]
            base = min(left_high, right_high)
            return base - lows[idx]

    for idx in range(window, len(df) - window):
        atr_here = atr_array[idx] if idx < len(atr_array) and not np.isnan(atr_array[idx]) else atr_mean
        atr_here = max(atr_here, 1e-6)
        prom_need = prominence_atr * atr_here

        segment = slice(idx - window, idx + window + 1)
        if (highs[idx] >= np.max(highs[segment])) and (idx - last_high_idx >= min_distance_bars):
            prom = _local_prominence("high", idx)
            if prom >= prom_need:
                swings.iloc[idx, swings.columns.get_loc("swing_high")] = True
                swings.iloc[idx, swings.columns.get_loc("swing_high_price")] = highs[idx]
                last_high_idx = idx

        if (lows[idx] <= np.min(lows[segment])) and (idx - last_low_idx >= min_distance_bars):
            prom = _local_prominence("low", idx)
            if prom >= prom_need:
                swings.iloc[idx, swings.columns.get_loc("swing_low")] = True
                swings.iloc[idx, swings.columns.get_loc("swing_low_price")] = lows[idx]
                last_low_idx = idx

    LOG.debug(
        "Detected %d swing highs / %d swing lows",
        int(swings["swing_high"].sum()),
        int(swings["swing_low"].sum()),
    )
    return swings


def detect_swing_points_from_config(df: pd.DataFrame, config: Dict | None) -> pd.DataFrame:
    """Convenience wrapper that accepts a config dict."""
    cfg = SwingPointConfig.from_dict(config)
    return detect_swing_points(
        df=df,
        window=cfg.window,
        prominence_atr=cfg.prominence_atr,
        min_distance_bars=cfg.min_distance_bars,
        atr_period=cfg.atr_period,
    )


__all__ = [
    "SwingPointConfig",
    "compute_atr",
    "detect_swing_points",
    "detect_swing_points_from_config",
]

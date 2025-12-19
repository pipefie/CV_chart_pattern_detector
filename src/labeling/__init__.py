"""Labeling utilities for OHLCV-derived pattern detection."""

from .swing_points import detect_swing_points, compute_atr
from .patterns_hs import detect_hs_pattern, label_hs_window
from .patterns_triangles import detect_ascending_triangle, label_ascending_triangle_window
from .patterns_double import (
    detect_double_top,
    detect_double_bottom,
    label_double_top_window,
    label_double_bottom_window,
)
from .labeler import PatternLabeler

__all__ = [
    "detect_swing_points",
    "compute_atr",
    "detect_hs_pattern",
    "label_hs_window",
    "detect_ascending_triangle",
    "label_ascending_triangle_window",
    "detect_double_top",
    "detect_double_bottom",
    "label_double_top_window",
    "label_double_bottom_window",
    "PatternLabeler",
]

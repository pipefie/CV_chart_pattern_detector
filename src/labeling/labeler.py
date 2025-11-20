from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from .swing_points import detect_swing_points_from_config, SwingPointConfig
from .patterns_hs import label_hs_window
from .patterns_triangles import label_ascending_triangle_window
from .patterns_double import label_double_top_window, label_double_bottom_window

LOG = logging.getLogger(__name__)


@dataclass
class PatternLabeler:
    """High-level orchestrator for OHLCV-based labeling."""

    config: Dict

    def __post_init__(self):
        self.swing_cfg = SwingPointConfig.from_dict(self.config.get("swing_points"))
        self.pattern_cfg = self.config.get("patterns", {})
        self.units_cfg = self.config.get("units", {}) or {}
        self.labels_post = self.config.get("labels_post", {}) or {}

    def detect_swings(self, df_window: pd.DataFrame) -> pd.DataFrame:
        cfg = self.swing_cfg
        return detect_swing_points_from_config(
            df_window,
            {
                "window": cfg.window,
                "min_distance_bars": cfg.min_distance_bars,
                "prominence_atr": cfg.prominence_atr,
                "atr_period": cfg.atr_period,
            },
        )

    def label_window(self, df_window: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Return label dict and structural feature dict for a window.
        """
        swings = self.detect_swings(df_window)
        labels: Dict[str, int] = {}
        features: Dict[str, float] = {}

        hs_cfg = self.pattern_cfg.get("head_and_shoulders")
        if hs_cfg:
            y_hs, feats_hs = label_hs_window(df_window, swings, hs_cfg)
            labels["y_head_and_shoulders"] = y_hs
            features.update(feats_hs)

        tri_cfg = self.pattern_cfg.get("ascending_triangle")
        if tri_cfg:
            y_tri, feats_tri = label_ascending_triangle_window(df_window, swings, tri_cfg)
            labels["y_ascending_triangle"] = y_tri
            features.update(feats_tri)

        dt_cfg = self.pattern_cfg.get("double_top")
        if dt_cfg:
            y_dt, feats_dt = label_double_top_window(
                df_window,
                swings,
                dt_cfg,
                units_cfg=self.units_cfg,
                labels_post=self.labels_post,
            )
            labels["y_double_top"] = y_dt
            features.update(feats_dt)

        db_cfg = self.pattern_cfg.get("double_bottom")
        if db_cfg:
            y_db, feats_db = label_double_bottom_window(
                df_window,
                swings,
                db_cfg,
                units_cfg=self.units_cfg,
                labels_post=self.labels_post,
            )
            labels["y_double_bottom"] = y_db
            features.update(feats_db)

        return labels, features


__all__ = ["PatternLabeler"]

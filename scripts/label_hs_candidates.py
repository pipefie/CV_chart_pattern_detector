#!/usr/bin/env python
"""
label_hs_candidates.py

Use the existing H&S labeler in src/labeling/patterns_hs.py to assign
a binary label + structural features to each candidate window produced
by scan_hs_regimes.py.

Input:
    data/labels/hs_candidates.csv

Output:
    data/labels/hs_labeled.csv

Each output row keeps the original candidate metadata and adds:
    y_hs          (0/1)
    hs_neckline_slope_deg
    hs_head_to_shoulder_ratio
    hs_shoulder_similarity
    hs_temporal_symmetry
    hs_breakout_confirmed
    hs_span_bars
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Tuple

import logging
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.dataset import load_yaml
from src.labeling.patterns_hs import label_hs_window
from src.labeling.swing_points import detect_swing_points_from_config


COMMON_PATTERN_KEYS = {
    "head_and_shoulders",
    "inverse_head_and_shoulders",
    "double_top",
    "double_bottom",
    "ascending_triangle",
    "descending_triangle",
}

DIAG_KEYS = [
    "swing_highs",
    "swing_lows",
    "swing_events",
    "candidate_sequences",
    "reject_duration",
    "reject_head_prominence",
    "reject_shoulder_similarity",
    "reject_time_symmetry",
    "reject_neckline_slope",
    "detections",
    "breakout_confirmed",
]

LOG = logging.getLogger("label_hs_candidates")


def deep_merge_dict(base: Dict | None, override: Dict | None) -> Dict:
    """
    Recursively merge dictionaries (override wins) without mutating inputs.
    """
    if base is None:
        base = {}
    if override is None:
        return deepcopy(base)
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def override_matches(match_cfg: Dict | None, symbol: str, timeframe: str) -> bool:
    """Return True if an override should apply to (symbol, timeframe)."""
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


class PatternConfigResolver:
    """
    Load configs/patterns.yaml (or a compatible file) and resolve per-symbol overrides.
    """

    def __init__(self, cfg_path: Path):
        cfg = load_yaml(cfg_path)
        if not cfg:
            raise ValueError(f"Empty config at {cfg_path}")
        self.cfg_path = cfg_path
        # Patterns.yaml style
        if "patterns" in cfg and ("common" in cfg or "overrides" in cfg):
            self.mode = "patterns"
            self.patterns_base = cfg.get("patterns") or {}
            self.overrides = cfg.get("overrides") or []
            if not self.patterns_base:
                raise ValueError(f"No patterns specified in {cfg_path}")
            self.pattern_keys = set(self.patterns_base.keys()) or COMMON_PATTERN_KEYS
        else:
            # Fallback: assume pipeline.yaml style (labeling.patterns.*)
            labeling = cfg.get("labeling") or {}
            self.mode = "pipeline"
            self.patterns_base = (labeling.get("patterns") or cfg.get("patterns") or {})
            self.overrides = []
            self.pattern_keys = set(self.patterns_base.keys())
        self._cache: Dict[Tuple[str, str], Dict] = {}

    def resolve_patterns(self, symbol: str, timeframe: str) -> Dict:
        key = (symbol, timeframe)
        if key in self._cache:
            return self._cache[key]
        if self.mode == "patterns":
            patterns = deepcopy(self.patterns_base)
            for override in self.overrides:
                if not override_matches(override.get("match"), symbol, timeframe):
                    continue
                for section, values in override.items():
                    if section == "match":
                        continue
                    if section in self.pattern_keys:
                        patterns[section] = deep_merge_dict(patterns.get(section, {}), values)
            self._cache[key] = patterns
        else:
            self._cache[key] = deepcopy(self.patterns_base)
        return self._cache[key]

    def get_hs_config(self, symbol: str, timeframe: str) -> Dict:
        patterns = self.resolve_patterns(symbol, timeframe)
        hs_cfg = patterns.get("head_and_shoulders")
        if hs_cfg is None:
            return {}
        return deepcopy(hs_cfg)


def load_ohlcv(data_root: str | Path, symbol: str) -> pd.DataFrame:
    """
    Load OHLCV for a symbol from Parquet, assuming:
        data/ohlcv/crypto/{symbol}.parquet
        data/ohlcv/equities_etf/{symbol}.parquet
    with timestamp as DatetimeIndex.
    """
    data_root = Path(data_root)

    candidate_paths = []
    for sub in ("crypto", "equities_etf"):
        p = data_root / sub
        if p.exists():
            candidate = p / f"{symbol}.parquet"
            if candidate.exists():
                candidate_paths.append(candidate)

    if not candidate_paths:
        raise FileNotFoundError(
            f"No parquet file found for symbol={symbol} under {data_root}"
        )
    if len(candidate_paths) > 1:
        opts = "\n  - ".join(str(p) for p in candidate_paths)
        raise ValueError(
            f"Multiple parquet files found for {symbol}. "
            f"Disambiguate manually.\n{opts}"
        )

    parquet_path = candidate_paths[0]
    print(f"[label_hs] Using {parquet_path} for {symbol}")
    df = pd.read_parquet(parquet_path)

    # Normalize timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        # assume DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"{parquet_path} index is not DatetimeIndex and no 'timestamp' column."
            )
        df = df.copy()
        df["timestamp"] = df.index
        df = df.reset_index(drop=True)

    return df


# --- CLI & main ---


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Label H&S on candidate windows from hs_candidates.csv"
    )
    p.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root folder for OHLCV Parquets (e.g., data/ohlcv)",
    )
    p.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to hs_candidates.csv produced by scan_hs_regimes.py",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output CSV path for hs_labeled.csv",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/patterns.yaml",
        help="Path to the main patterns configuration YAML.",
    )
    p.add_argument(
        "--swing-window",
        type=int,
        default=4,
        help="Half-window (bars on each side) for swing detection.",
    )
    p.add_argument(
        "--swing-min-distance",
        type=int,
        default=4,
        help="Minimum bars between consecutive swing highs/lows.",
    )
    p.add_argument(
        "--swing-prominence-atr",
        type=float,
        default=0.9,
        help="Minimum prominence in ATR units for a swing to qualify.",
    )
    p.add_argument(
        "--swing-atr-period",
        type=int,
        default=14,
        help="ATR period for swing detection.",
    )
    p.add_argument(
        "--diag-out",
        type=str,
        default=None,
        help="Optional path for aggregated diagnostics CSV (defaults to <out>_diagnostics.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_root = Path(args.data_root)
    candidates_path = Path(args.candidates)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_cand = pd.read_csv(candidates_path)
    print(f"[label_hs] Loaded {len(df_cand)} candidate windows from {candidates_path}")

    # Cache OHLCV per symbol so we don't reload on every row
    ohlcv_cache: Dict[str, pd.DataFrame] = {}
    cfg_resolver = PatternConfigResolver(Path(args.config))
    swing_cfg = {
        "window": args.swing_window,
        "min_distance_bars": args.swing_min_distance,
        "prominence_atr": args.swing_prominence_atr,
        "atr_period": args.swing_atr_period,
    }
    print(f"[label_hs] Using swing config: {swing_cfg}")
    print(f"[label_hs] Patterns config: {args.config}")

    labeled_rows = []

    for i, row in df_cand.iterrows():
        symbol = row["symbol"]
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])
        timeframe = row.get("timeframe", "")

        if symbol not in ohlcv_cache:
            ohlcv_cache[symbol] = load_ohlcv(data_root, symbol)
        df_full = ohlcv_cache[symbol]

        if end_idx > len(df_full):
            # Safety check
            print(
                f"[label_hs] Warning: window ({start_idx}, {end_idx}) "
                f"out of bounds for {symbol} ({len(df_full)} bars). Skipping."
            )
            continue

        window_df = df_full.iloc[start_idx:end_idx].copy()

        # Compute ATR-aware swing points
        swings = detect_swing_points_from_config(window_df, swing_cfg)

        hs_cfg = cfg_resolver.get_hs_config(symbol, timeframe)
        diagnostics: Dict[str, float] = {}

        # Use your existing labeler from patterns_hs.py while capturing diagnostics
        y, feats = label_hs_window(window_df, swings, hs_cfg, diagnostics=diagnostics)

        out_row = row.to_dict()
        out_row["y_hs"] = int(y)
        out_row.update(feats)
        for key in DIAG_KEYS:
            out_row[f"hs_diag_{key}"] = diagnostics.get(key, 0)
        labeled_rows.append(out_row)

    df_out = pd.DataFrame(labeled_rows)
    df_out.to_csv(out_path, index=False)
    print(
        f"[label_hs] Wrote {len(df_out)} labeled windows to {out_path} "
        f"(positives: {df_out['y_hs'].sum()})"
    )
    diag_cols = [f"hs_diag_{key}" for key in DIAG_KEYS]
    diag_cols = [c for c in diag_cols if c in df_out.columns]
    if diag_cols:
        summary = df_out.groupby("symbol")[["y_hs"] + diag_cols].sum()
        LOG.info("Diagnostics per symbol:\n%s", summary)
        diag_path = Path(args.diag_out) if args.diag_out else out_path.with_name(f"{out_path.stem}_diagnostics.csv")
        summary.reset_index().to_csv(diag_path, index=False)
        LOG.info("Saved diagnostics summary to %s", diag_path)


if __name__ == "__main__":
    main()


#!/usr/bin/env python
"""
scan_hs_regimes.py

Scan OHLCV data for windows where the market regime is favorable
for Head & Shoulders (H&S) or Inverse H&S patterns.

This script does NOT detect the pattern itself.
It only finds windows with:
    - prior trend (up for H&S top, down for inverse H&S)
    - enough volatility

Output: a CSV of candidate windows to feed into your H&S labeler
and rendering pipeline.

Example:

    python scan_hs_regimes.py \
        --data-root data/ohlcv \
        --symbols BTC-USD ETH-USD AAPL TSLA \
        --timeframes 1h 4h \
        --window 100 \
        --step 10 \
        --hs-type both \
        --out data/hs_candidates.csv

You can later join these candidates with your geometric H&S labeler.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Tuple

import numpy as np
import pandas as pd
import json


HsType = Literal["top", "inverse", "both"]


@dataclass
class RegimeConfig:
    window: int = 100
    min_trend_strength: float = 0.001  # avg log-return per bar magnitude
    min_volatility: float = 0.005      # std of log-returns in window
    hs_type: HsType = "top"            # "top" | "inverse" | "both"


def is_regime_favorable_for_hs(
    prices: np.ndarray,
    config: RegimeConfig,
    hs_type: HsType,
) -> bool:
    """
    Check if the current price window has the 'energy' to form an H&S.

    Conditions:
      - There is a prior trend (up for H&S top, down for inverse H&S)
      - There is enough volatility (not a dead, flat market)

    Parameters
    ----------
    prices : np.ndarray
        1D array of prices (close or HLC3) of length `config.window`.
    config : RegimeConfig
        Thresholds and window length.
    hs_type : "top" | "inverse" | "both"
        What kind of H&S we are interested in.

    Returns
    -------
    bool
        True if it is worth running an H&S detector on this window.
    """
    W = config.window
    if prices.shape[0] != W:
        # Safety: should not happen if caller slices correctly
        return False

    # --- 1) Trend check on first half of the window ---
    half = W // 2
    first_half = prices[:half]

    # Normalize prices: use log-price so slope is in "log-return per bar"
    y = np.log(first_half)
    x = np.arange(len(first_half))
    slope, _ = np.polyfit(x, y, 1)  # slope ≈ avg log-return per bar

    trend_strength = abs(slope)

    # Directional condition
    if hs_type in ("top", "both"):
        # For a classic H&S top we want a prior UP trend
        if slope <= 0:
            # Not an uptrend; not good for H&S top
            if hs_type == "top":
                return False
            # If "both", we still might accept as inverse H&S
        else:
            # Uptrend is present; good for H&S top
            pass

    if hs_type in ("inverse", "both"):
        # For inverse H&S (reversal up) we want a prior DOWN trend
        if slope >= 0:
            # Not a downtrend; not good for inverse H&S
            if hs_type == "inverse":
                return False
            # If "both", maybe good for top H&S
        else:
            # Downtrend is present; good for inverse H&S
            pass

    # If "both", we only require that |slope| is big enough:
    # direction-specific check above prevents obvious nonsense.

    # Require minimum trend strength (magnitude)
    if trend_strength < config.min_trend_strength:
        return False

    # --- 2) Volatility check on whole window ---
    log_rets = np.diff(np.log(prices))
    volatility = np.std(log_rets)

    if volatility < config.min_volatility:
        return False

    return True


def load_ohlcv(data_root: str | Path, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load OHLCV directly from Parquet, detecting timestamp whether it is:
      - in a column called 'timestamp'
      - in a column called 'timestamp_ms'
      - OR stored as the DataFrame index
    """
    data_root = Path(data_root)

    # --- 1) Find parquet file ---
    candidate_paths = []
    for sub in ("crypto", "equities_etf"):
        p = data_root / sub
        if p.exists():
            candidate_paths.extend((p / f"{symbol}.parquet",))
            candidate_paths.extend(p.glob(f"{symbol}_*.parquet"))

    candidate_paths = [p for p in candidate_paths if p.exists()]

    if not candidate_paths:
        raise FileNotFoundError(
            f"No parquet file found for symbol={symbol} under {data_root}"
        )

    if len(candidate_paths) > 1:
        # If you want to select one with timeframe explicitly:
        tf_match = [p for p in candidate_paths if timeframe in p.name]
        if len(tf_match) == 1:
            parquet_path = tf_match[0]
        else:
            opts = "\n  - ".join(str(p) for p in candidate_paths)
            raise ValueError(
                f"Ambiguous parquet files for {symbol}. "
                f"Please keep only one. Candidates:\n  - {opts}"
            )
    else:
        parquet_path = candidate_paths[0]

    print(f"[load_ohlcv] Using {parquet_path} for {symbol} {timeframe}")

    df = pd.read_parquet(parquet_path)

    # --- 2) Detect timestamp ---
    if "timestamp" in df.columns:
        # Case A: timestamp column
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

    elif "timestamp_ms" in df.columns:
        # Case B: timestamp_ms column
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

    else:
        # Case C: timestamp is the index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df["timestamp"] = df.index
            df = df.reset_index(drop=True)
        else:
            raise ValueError(
                f"{parquet_path} has no timestamp column and index is not DatetimeIndex.\n"
                f"Columns: {list(df.columns)}\nIndex type: {type(df.index)}"
            )

    # --- 3) Validate OHLCV columns ---
    expected = {"open", "high", "low", "close", "volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"{parquet_path} missing expected OHLCV columns: {missing}"
        )

    return df

def rolling_windows(
    n_rows: int,
    window: int,
    step: int,
) -> Iterable[Tuple[int, int]]:
    """
    Generate (start, end) indices for rolling windows of length `window`,
    stepping by `step` bars.

    end index is exclusive: [start, end)
    """
    start = 0
    while start + window <= n_rows:
        yield start, start + window
        start += step


def scan_symbol_timeframe(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: RegimeConfig,
    step: int,
) -> List[dict]:
    """
    Scan a single symbol/timeframe OHLCV DataFrame for favorable H&S regimes.

    Returns a list of candidate window dicts ready to be turned into a DataFrame.
    """
    n = len(df)
    candidates: List[dict] = []

    if n < config.window:
        return candidates

    # Use close price as main series; you could also use HLC3
    prices = df["close"].to_numpy(dtype=float)

    for start_idx, end_idx in rolling_windows(n_rows=n, window=config.window, step=step):
        window_prices = prices[start_idx:end_idx]
        ts_start = df["timestamp"].iloc[start_idx]
        ts_end = df["timestamp"].iloc[end_idx - 1]

        favorable = is_regime_favorable_for_hs(
            window_prices,
            config=config,
            hs_type=config.hs_type,
        )

        if not favorable:
            continue

        candidates.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_ts": ts_start,
                "end_ts": ts_end,
                "window": config.window,
                "hs_type": config.hs_type,
                # These flags are for *regime only*.
                # A later stage will compute y_hs, y_inverse_hs, etc.
                "regime_favorable_for_hs": True,
            }
        )

    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan OHLCV data for windows with favorable regime for Head & Shoulders patterns."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root folder where OHLCV files live (e.g., data/ohlcv).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        required=True,
        help="List of symbols to scan, e.g. BTC-USD ETH-USD AAPL TSLA.",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        nargs="+",
        required=True,
        help="List of timeframes, e.g. 1h 4h 1D.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Window length in bars used for regime detection.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Stride in bars between windows.",
    )
    parser.add_argument(
        "--hs-type",
        type=str,
        choices=["top", "inverse", "both"],
        default="top",
        help="Which type of H&S regime to look for.",
    )
    parser.add_argument(
        "--min-trend-strength",
        type=float,
        default=0.001,
        help="Minimum |slope| of log-price per bar in first half of window.",
    )
    parser.add_argument(
        "--min-volatility",
        type=float,
        default=0.005,
        help="Minimum std of log-returns in the whole window.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output CSV path for candidate windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = RegimeConfig(
        window=args.window,
        min_trend_strength=args.min_trend_strength,
        min_volatility=args.min_volatility,
        hs_type=args.hs_type,  # type: ignore[arg-type]
    )

    all_candidates: List[dict] = []

    for symbol in args.symbols:
        for timeframe in args.timeframes:
            print(f"[scan_hs_regimes] Loading {symbol} {timeframe}...")
            try:
                df = load_ohlcv(data_root=data_root, symbol=symbol, timeframe=timeframe)
            except FileNotFoundError as e:
                print(f"  ! Skipping {symbol} {timeframe}: {e}")
                continue

            print(
                f"  Scanning {len(df)} bars with window={config.window}, "
                f"step={args.step}, hs_type={config.hs_type}..."
            )
            candidates = scan_symbol_timeframe(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                config=config,
                step=args.step,
            )
            print(f"  Found {len(candidates)} favorable windows.")
            all_candidates.extend(candidates)

    if not all_candidates:
        print("[scan_hs_regimes] No favorable windows found. Nothing to write.")
        return

    df_out = pd.DataFrame(all_candidates)
    df_out.to_csv(out_path, index=False)
    print(f"[scan_hs_regimes] Wrote {len(df_out)} candidate windows to {out_path}")


if __name__ == "__main__":
    main()

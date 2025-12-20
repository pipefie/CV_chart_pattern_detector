# scripts/make_images.py
from __future__ import annotations
import argparse, json, random, gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from refactored renderer
from src.rendering.renderer import render_window_png, load_yaml

# ---- Main orchestration ----
def main():
    ap = argparse.ArgumentParser(description="Render candlestick chart images with jitter from Parquet OHLCV.")
    ap.add_argument("--backtest_cfg", default="configs/backtest.yaml")
    ap.add_argument("--render_cfg",  default="configs/render.yaml")
    ap.add_argument("--window_bars", type=int, default=160)
    ap.add_argument("--stride_bars", type=int, default=40)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--out_root",    default="data/images/rendered")
    ap.add_argument("--windows_csv", default=None, help="Optional CSV listing explicit windows (e.g., hs_labeled.csv).")
    ap.add_argument("--windows_filter_col", default=None, help="Column to filter on when rendering from CSV.")
    ap.add_argument("--windows_filter_value", default=None, help="Value required in --windows_filter_col.")
    ap.add_argument("--windows_split_col", default=None, help="Column holding split name (train/val/test).")
    ap.add_argument("--windows_default_split", default="train", help="Split to use when CSV has no split column.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    backtest   = load_yaml(args.backtest_cfg)
    render_cfg = load_yaml(args.render_cfg)

    src_root = Path("data/ohlcv")
    out_root = Path(args.out_root)

    universes = {
        "crypto": backtest["universes"]["crypto"],
        "equities_etf": backtest["universes"]["equities_etf"],
    }

    def load_symbol_df(symbol: str) -> pd.DataFrame | None:
        for sub in ("equities_etf", "crypto"):
            f = src_root / sub / f"{symbol}.parquet"
            if f.exists():
                df = pd.read_parquet(f)
                df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
                return df
        print(f"⚠️ Missing OHLCV for {symbol}")
        return None

    def render_job(symbol: str, timeframe: str, win: pd.DataFrame, split: str, counter: int) -> int:
        ts_end = win.index[-1]
        schema = render_cfg["export"]["naming"]["schema"]
        start_epoch = int(win.index[0].timestamp())
        end_epoch = int(win.index[-1].timestamp())
        out_name = schema.format(symbol=symbol, tf=timeframe, startts=start_epoch, endts=end_epoch, seed=args.seed)
        out_png = out_root / split / out_name
        meta_out = out_png.with_suffix(".json")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        render_window_png(win, out_png, meta_out, render_cfg, symbol=symbol, timeframe=timeframe, rng=rng)
        counter += 1
        if (counter % 50) == 0:
            plt.close("all")
            gc.collect()
        print(f"   ↳ rendered {symbol} {timeframe} @ {ts_end} → {split}")
        return counter

    if args.windows_csv:
        jobs = pd.read_csv(args.windows_csv)
        if args.windows_filter_col and args.windows_filter_col in jobs.columns and args.windows_filter_value is not None:
            col = args.windows_filter_col
            val = args.windows_filter_value
            series = jobs[col]
            # Try to coerce filter value to column dtype (handles numeric labels like y_hs=1)
            try:
                if pd.api.types.is_numeric_dtype(series):
                    val_coerced = float(val)
                    jobs = jobs[series.astype(float) == val_coerced]
                else:
                    jobs = jobs[series.astype(str) == str(val)]
            except Exception:
                jobs = jobs[series.astype(str) == str(val)]
        if jobs.empty:
            print("⚠️ No windows to render after filtering.")
            return
        cache: dict[str, pd.DataFrame] = {}
        counter = 0
        for row in jobs.itertuples(index=False):
            symbol = getattr(row, "symbol", None)
            timeframe = getattr(row, "timeframe", None)
            start_idx = getattr(row, "start_idx", None)
            end_idx = getattr(row, "end_idx", None)
            if symbol is None or start_idx is None or end_idx is None:
                continue
            if timeframe is None:
                print(f"⚠️ Missing timeframe for {symbol}; skipping row.")
                continue
            if symbol not in cache:
                df_sym = load_symbol_df(symbol)
                if df_sym is None:
                    continue
                cache[symbol] = df_sym
            df_sym = cache[symbol]
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            if end_idx > len(df_sym):
                print(f"⚠️ window ({start_idx}, {end_idx}) out of range for {symbol}")
                continue
            win = df_sym.iloc[start_idx:end_idx].copy()
            if win.empty:
                continue
            split = args.windows_default_split
            if args.windows_split_col and args.windows_split_col in jobs.columns:
                split = getattr(row, args.windows_split_col, split) or split
            counter = render_job(symbol, timeframe, win, split, counter)
        print("✅ Candidate windows rendered.")
        return

    counter = 0
    for group, ucfg in universes.items():
        timeframe = ucfg["timeframe"]
        folder = "crypto" if group == "crypto" else "equities_etf"
        for sym in ucfg["symbols"]:
            f = src_root / folder / f"{sym}.parquet"
            if not f.exists():
                print(f"⚠️ Missing OHLCV: {f}")
                continue

            df = pd.read_parquet(f)
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            if len(df) < args.window_bars:
                print(f"⚠️ Not enough bars for {sym}")
                continue

            for i in range(args.window_bars, len(df) + 1, args.stride_bars):
                win = df.iloc[i - args.window_bars : i]
                ts_end = win.index[-1]
                bucket = split_tag(ts_end, backtest["walk_forward"])
                if bucket == "ignore":
                    continue

                out_root.mkdir(parents=True, exist_ok=True)
                counter = render_job(sym, timeframe, win, bucket, counter)

            print(f"✅ Rendered {sym} ({group})")

    print("✅ All renders complete.")

if __name__ == "__main__":
    main()

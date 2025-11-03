# scripts/fetch_ohlcv.py
from __future__ import annotations
import argparse, json, time, hashlib
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import ccxt

# ---------- Helpers: loading config ----------

def load_backtest_config(path: str = "configs/backtest.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------- Core: one-symbol fetch with retries ----------

def yf_to_ccxt_symbol(yf_symbol: str) -> str:
    # Map "BTC-USD" -> "BTC/USDT" (Binance uses USDT quote)
    if yf_symbol.endswith("-USD"):
        base = yf_symbol.replace("-USD", "")
        return f"{base}/USDT"
    return yf_symbol  # fall back

def fetch_binance_ohlcv(
    yf_symbol: str,
    start: str,
    end: str,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Fetch OHLCV from Binance via CCXT (1h only), return UTC tz-aware df with
    columns: ['open','high','low','close','volume'] and DatetimeIndex.
    """
    assert interval == "1h", "This fetcher is scoped to 1h."
    symbol = yf_to_ccxt_symbol(yf_symbol)

    ex = ccxt.binance({"enableRateLimit": True})
    tf = "1h"
    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    all_rows = []
    limit = 1000
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        # CCXT returns [timestamp_ms, open, high, low, close, volume]
        last_ts = batch[-1][0]
        # advance by one candle to avoid duplicates
        next_since = last_ts + ex.parse_timeframe(tf) * 1000
        if next_since >= end_ms or next_since == since:
            break
        since = next_since

    if not all_rows:
        raise ValueError(f"No data returned from Binance for {yf_symbol} [{start} → {end}]")

    df = pd.DataFrame(
        all_rows, columns=["timestamp_ms","open","high","low","close","volume"]
    )
    # Timestamps as UTC tz-aware DatetimeIndex
    idx = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])
    df.index = idx

    # Drop last partial hour if present
    now_utc = pd.Timestamp.now(tz="UTC")
    if len(df) and (df.index[-1] > (now_utc - pd.Timedelta("59min"))):
        df = df.iloc[:-1]

    # Clean & checks
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in ["open","high","low","close"]:
        if (df[c] <= 0).any():
            raise ValueError(f"Non-positive {c} in {yf_symbol}")
    return df[["open","high","low","close","volume"]]

def fetch_yahoo_ohlcv(
    symbol: str,
    start: str | None,
    end: str | None,
    interval: str = "1h",
    max_retries: int = 3,
    retry_sleep_s: float = 2.0,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Yahoo via yfinance, return a UTC tz-aware, 1h-aligned, clean DataFrame
    with columns: ['open','high','low','close','volume'] and DatetimeIndex.
    """
    assert interval in {"1h"}, "This project fixes 1h; extend here if you add others."

    # IMPORTANT: Use start/end OR period, not both. We'll use start/end for reproducibility.

    print(f"Fetching {symbol} from {start} to {end} with interval {interval}")

    # --- Clamp for Yahoo intraday history (hard limit ~730 days) ---

    if interval == "1h" and start and end:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts   = pd.Timestamp(end, tz="UTC")
        max_span = pd.Timedelta(days=720)  # a little under 730 to be safe
        if (end_ts - start_ts) > max_span:
            clamped_start_ts = end_ts - max_span
            print(f"[Yahoo clamp] {symbol} 1h range {start_ts.date()}→{end_ts.date()} "
                f"exceeds limit; using {clamped_start_ts.date()}→{end_ts.date()} instead.")
            start = clamped_start_ts.strftime("%Y-%m-%d")

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,   # keep raw OHLC intraday
                progress=False,
                threads=True,
                group_by='column',
            )
            break
        except Exception as e:
            last_err = e
            time.sleep(retry_sleep_s * attempt)
    else:
        raise RuntimeError(f"yfinance failed after {max_retries} retries: {last_err}")

    # --- Flatten columns from yfinance robustly ---
    if isinstance(df.columns, pd.MultiIndex):
        # Typical case: levels ['Price','Ticker'] like ('Open','AAPL')
        names = list(df.columns.names or [])
        if "Ticker" in names:
            try:
                # Select the single ticker level cleanly
                df = df.xs(symbol, axis=1, level="Ticker", drop_level=True)
            except KeyError:
                # Sometimes the ticker formatting differs; fall back to "the only one"
                tickers = df.columns.get_level_values("Ticker").unique()
                if len(tickers) == 1:
                    df = df.xs(tickers[0], axis=1, level="Ticker", drop_level=True)
                else:
                    raise ValueError(f"Unexpected multi-ticker columns for {symbol}: {tickers}")
    else:
        # If there isn't a 'Ticker' level, but it's still MultiIndex, drop the
        # single extra level if possible; otherwise join tuples into strings.
        levels_n = [len(l) for l in df.columns.levels]
        if any(n == 1 for n in levels_n):
            # drop first single-cardinality level
            for lvl in range(df.columns.nlevels):
                if len(df.columns.levels[lvl]) == 1:
                    df.columns = df.columns.droplevel(lvl)
                    break
        else:
            df.columns = df.columns.map(lambda t: "_".join(map(str, t)))



    if df is None or df.empty:
        # yfinance returns empty for some symbols/ranges; surface early
        raise ValueError(f"No data returned for {symbol} [{start} → {end}]")

    # Standardize schema
    # yfinance returns columns like ['Open','High','Low','Close','Adj Close','Volume']
    # Drop 'Adj Close' for intraday
    cols = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols)
    if "adj_close" in df.columns:
        df = df.drop(columns=["adj_close"])

    # Ensure tz-aware UTC index. yfinance sometimes gives tz-aware; be defensive:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # 1) Drop duplicate timestamps; 2) Sort; 3) Enforce monotonic increasing
    df = df[~df.index.duplicated(keep="last")].sort_index()


    # Drop the last still-forming bar (if the end time is exactly now)
    now_utc = pd.Timestamp.now(tz="UTC") 
    if len(df) and (df.index[-1] > (now_utc - pd.Timedelta("59min"))):
        df = df.iloc[:-1]

    # Basic sanity checks
    #  - no negative or zero prices
    #  - strictly increasing index
    #  - all required columns present
    required_cols = ["open", "high", "low", "close", "volume"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {symbol}")
    if (df[["open","high","low","close"]] <= 0).any().any():
        raise ValueError(f"Non-positive OHLC detected for {symbol}")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Index not monotonic for {symbol}")

    # Final: keep only the required columns in canonical order
    df = df[required_cols]
    return df

# ---------- Session filters ----------

def filter_rth_newyork(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only Regular Trading Hours for NY equities/ETFs: 09:30–16:00 America/New_York.
    For 1h bars, this typically corresponds to bars starting at 09:30,10:30,...,15:30 local.
    """
    if df.empty:
        return df
    local = df.copy()
    local.index = local.index.tz_convert("America/New_York")

    # Keep bars whose *start time* minute:hour match NY RTH starts (09:30..15:30)
    valid = []
    for ts in local.index:
        hr, mn = ts.hour, ts.minute
        # Bars starting at :30 between 9:30 and 15:30 inclusive
        if (mn == 30) and (9 <= hr <= 15):
            valid.append(True)
        else:
            valid.append(False)
    keep = pd.Series(valid, index=local.index)

    out = df.loc[keep.values]
    # Ensure back to UTC
    out.index = out.index.tz_convert("UTC")
    return out

# ---------- Save utilities ----------

def to_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

def write_manifest(path: Path, *, provider: str, symbol: str, start: str, end: str, df_hash: str):
    meta = {
        "provider": provider,
        "symbol": symbol,
        "start": start,
        "end": end,
        "rows": int(len(pd.read_parquet(path))),
        "sha256": df_hash,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
    }
    manpath = path.with_suffix(".manifest.json")
    manpath.write_text(json.dumps(meta, indent=2))

def sha256_df(df: pd.DataFrame) -> str:
    # stable hash of content + index to track reproducibility
    b = df.reset_index().to_csv(index=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

# ---------- Orchestrate by universe from backtest.yaml ----------

def run_from_config(cfg: dict, start: str, end: str):
    root = Path("data/ohlcv")
    # --- Crypto via Binance/CCXT (24/7) ---
    u_crypto   = cfg["universes"]["crypto"]
    for sym in u_crypto["symbols"]:
        # Use Binance for deep, consistent intraday
        df = fetch_binance_ohlcv(sym, start=start, end=end, interval=u_crypto["timeframe"])
        out = root / "crypto" / f"{sym}.parquet"
        dfhash = sha256_df(df)
        to_parquet(df, out)
        write_manifest(out, provider="binance-ccxt", symbol=sym, start=start, end=end, df_hash=dfhash)

    # --- Equities/ETFs via Yahoo (RTH) ---
    u_eq = cfg["universes"]["equities_etf"]
    for sym in u_eq["symbols"]:
        df = fetch_yahoo_ohlcv(sym, start=start, end=end, interval=u_eq["timeframe"])
        df = filter_rth_newyork(df) if u_eq["session"].get("use_regular_hours_only", True) else df
        out = root / "equities_etf" / f"{sym}.parquet"
        dfhash = sha256_df(df)
        to_parquet(df, out)
        write_manifest(out, provider="yahoo", symbol=sym, start=start, end=end, df_hash=dfhash)

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Fetch OHLCV (1h) for configured universes using yfinance.")
    parser.add_argument("--start", type=str, required=True, help="UTC start date, e.g. 2019-01-01")
    parser.add_argument("--end",   type=str, required=True, help="UTC end date, e.g. 2025-10-31")
    parser.add_argument("--config", type=str, default="configs/backtest.yaml")
    args = parser.parse_args()

    cfg = load_backtest_config(args.config)
    run_from_config(cfg, start=args.start, end=args.end)
    print("✅ OHLCV fetch completed.")

if __name__ == "__main__":
    main()

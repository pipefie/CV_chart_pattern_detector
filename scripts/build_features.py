# scripts/build_features.py
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

try:
    import cv2 as cv
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    warnings.warn("OpenCV (cv2) not available. Image edge/Hough features will be skipped.")

# ---------------- I/O helpers ----------------
def load_yaml(path: str|Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_meta(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))

def load_ohlcv(symbol: str) -> pd.DataFrame:
    root = Path("data/ohlcv")
    if (root / "equities_etf" / f"{symbol}.parquet").exists():
        p = root / "equities_etf" / f"{symbol}.parquet"
    else:
        p = root / "crypto" / f"{symbol}.parquet"
    df = pd.read_parquet(p)
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df

def slice_window(df: pd.DataFrame, start_iso: str, end_iso: str) -> pd.DataFrame:
    s = pd.Timestamp(start_iso); e = pd.Timestamp(end_iso)
    if s.tzinfo is None: s = s.tz_localize("UTC")
    else: s = s.tz_convert("UTC")
    if e.tzinfo is None: e = e.tz_localize("UTC")
    else: e = e.tz_convert("UTC")
    return df.loc[(df.index >= s) & (df.index <= e)]

# ---------------- OHLCV utilities ----------------
def _safe_div(a, b, eps=1e-9):
    return a / (b if abs(b) > eps else np.sign(b) * eps if b != 0 else eps)

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    # RMA (Wilder) smoothing: ATR_t = ATR_{t-1}*(n-1)/n + TR_t/n
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def pivots(series: np.ndarray, L: int, R: int, kind: str, min_sep: int) -> list[tuple[int,float]]:
    N = len(series)
    out = []
    last = -10**9
    for i in range(L, N - R):
        v = series[i]
        left = series[i-L:i]
        right = series[i+1:i+1+R]
        if kind == "high":
            if v >= left.max() and v >= right.max() and (i - last) >= min_sep:
                out.append((i, float(v))); last = i
        else:
            if v <= left.min() and v <= right.min() and (i - last) >= min_sep:
                out.append((i, float(v))); last = i
    return out

# ---------------- Feature blocks (OHLCV) ----------------
def features_ohlcv_stats(win: pd.DataFrame, params: dict) -> dict:
    close = win["close"].astype(float)
    ret = close.pct_change().dropna()
    atr = compute_atr(win, params.get("atr_period",14))

    feats = {}
    feats["ret_mean"] = float(ret.mean())
    feats["ret_std"]  = float(ret.std(ddof=0))
    feats["abs_ret_mean"] = float(ret.abs().mean())
    feats["atr_mean"] = float(atr.mean())
    feats["atr_std"]  = float(atr.std(ddof=0))

    # ROC and moving-average relativity
    for p in params.get("roc_periods", [5,10,20]):
        feats[f"roc_{p}"] = float(_safe_div(close.iloc[-1], close.iloc[max(0,len(close)-p)])) - 1.0
    for p in params.get("ma_periods", [10,20,50]):
        ma = close.rolling(p).mean()
        feats[f"rel_close_ma{p}"] = float(_safe_div(close.iloc[-1], ma.iloc[-1])) - 1.0 if not np.isnan(ma.iloc[-1]) else 0.0

    return feats

def features_candle_shape(win: pd.DataFrame) -> dict:
    o,h,l,c = [win[k].astype(float).to_numpy() for k in ["open","high","low","close"]]
    rng = np.maximum(1e-9, h - l)
    body = np.abs(c - o)
    up_wick = h - np.maximum(c, o)
    dn_wick = np.minimum(c, o) - l
    feats = {
        "body_pct_mean": float(np.mean(_safe_div(body, rng))),
        "body_pct_std":  float(np.std (_safe_div(body, rng))),
        "up_wick_pct_mean": float(np.mean(_safe_div(up_wick, rng))),
        "dn_wick_pct_mean": float(np.mean(_safe_div(dn_wick, rng))),
        "bull_frac": float(np.mean((c - o) > 0)),
        "bear_frac": float(np.mean((c - o) < 0)),
    }
    return feats

def features_swings(win: pd.DataFrame, params: dict) -> dict:
    close = win["close"].astype(float).to_numpy()
    L = int(params.get("pivot_left",3)); R = int(params.get("pivot_right",3)); sep = int(params.get("pivot_min_sep",4))
    piv_hi = pivots(close, L,R,"high",sep); piv_lo = pivots(close, L,R,"low",sep)

    # spacing between successive highs/lows (bars)
    def spacing(iv):
        if len(iv) < 2: return np.nan
        diffs = np.diff([i for i,_ in iv])
        return float(np.mean(diffs))
    # prominence proxy: diff to nearest neighbor around pivot (simple, ATR-agnostic baseline)
    def prominence(iv):
        if len(iv) < 2: return np.nan
        vals = [p for _,p in iv]
        return float(np.std(vals))  # cheap variance proxy

    feats = {
        "n_piv_hi": int(len(piv_hi)),
        "n_piv_lo": int(len(piv_lo)),
        "piv_hi_spacing_mean": spacing(piv_hi),
        "piv_lo_spacing_mean": spacing(piv_lo),
        "piv_hi_val_std": prominence(piv_hi),
        "piv_lo_val_std": prominence(piv_lo),
    }
    return feats

def features_pattern_proxies(win: pd.DataFrame) -> dict:
    """
    Interpretable proxies aligned with H&S / DT / DB / Triangle definitions,
    computed directly on OHLCV to keep them robust across themes.
    """
    close = win["close"].astype(float).to_numpy()
    N = len(close)

    # Peak/trough similarity (double top/bottom proxy) using last ~1/2 window
    half = max(8, N//2)
    segment = close[-half:]
    # very simple: find top-2 local maxima/minima
    def two_extrema(arr, is_max=True):
        idx = []
        for i in range(1, len(arr)-1):
            if is_max and arr[i] >= arr[i-1] and arr[i] >= arr[i+1]:
                idx.append(i)
            if not is_max and arr[i] <= arr[i-1] and arr[i] <= arr[i+1]:
                idx.append(i)
        if len(idx) < 2: return None
        idx = sorted(idx, key=lambda i: arr[i], reverse=is_max)[:2]
        i1,i2 = sorted(idx)
        return (i1, arr[i1], i2, arr[i2])

    dt = two_extrema(segment, True)
    db = two_extrema(segment, False)

    feats = {}
    if dt:
        i1,p1,i2,p2 = dt
        mid = 0.5*(p1+p2)
        feats["dt_peak_similarity"] = float(1.0 - abs(p1-p2)/max(1e-9, mid))
        if i2 - i1 >= 2:
            valley = float(segment[i1+1:i2].min())
            feats["dt_valley_depth"] = float((mid - valley)/max(1e-9, mid))
        else:
            feats["dt_valley_depth"] = 0.0
        feats["dt_span_bars"] = float(i2-i1)
    else:
        feats["dt_peak_similarity"] = 0.0
        feats["dt_valley_depth"] = 0.0
        feats["dt_span_bars"] = 0.0

    if db:
        i1,p1,i2,p2 = db
        mid = 0.5*(p1+p2)
        feats["db_trough_similarity"] = float(1.0 - abs(p1-p2)/max(1e-9, mid))
        if i2 - i1 >= 2:
            peak = float(segment[i1+1:i2].max())
            feats["db_peak_height"] = float((peak - mid)/max(1e-9, mid))
        else:
            feats["db_peak_height"] = 0.0
        feats["db_span_bars"] = float(i2-i1)
    else:
        feats["db_trough_similarity"] = 0.0
        feats["db_peak_height"] = 0.0
        feats["db_span_bars"] = 0.0

    # Head/Shoulders proxy: best triple (L,H,R) in last half with middle highest and some symmetry
    best_hs = 0.0
    best_sym = 0.0
    for i in range(2, len(segment)-2):
        L = segment[i-1]; H = segment[i]; R = segment[i+1]
        if H <= max(L,R): continue
        sh_avg = 0.5*(L+R)
        rel = (H - sh_avg)/max(1e-9, sh_avg)
        best_hs = max(best_hs, float(rel))
        # symmetry proxy: compare (H-L) vs (R-H)
        sym = 1.0 - abs((i - (i-1)) - ((i+1) - i))/max(1, (i - (i-1)) + ((i+1) - i))
        best_sym = max(best_sym, float(sym))
    feats["hs_head_rel"] = best_hs
    feats["hs_time_sym"] = best_sym

    # Triangle proxy: linear fit to upper/lower envelopes on last half; convergence rate
    x = np.arange(len(segment))
    a = np.polyfit(x, segment, 1)  # overall slope (sanity)
    feats["seg_slope"] = float(a[0])

    # crude envelopes using rolling maxima/minima
    roll = max(3, len(segment)//10)
    upper = pd.Series(segment).rolling(roll, center=True).max().fillna(method="bfill").fillna(method="ffill")
    lower = pd.Series(segment).rolling(roll, center=True).min().fillna(method="bfill").fillna(method="ffill")
    xu = np.arange(len(upper)); xl = np.arange(len(lower))
    au,bu = np.polyfit(xu, upper, 1); al,bl = np.polyfit(xl, lower, 1)
    gap_s = (au*0+bu) - (al*0+bl); gap_e = (au*(len(x)-1)+bu) - (al*(len(x)-1)+bl)
    mid = float(np.mean(segment))
    feats["tri_conv_rate"] = float(_safe_div(gap_s - gap_e, abs(mid)))

    return feats

# ---------------- Feature blocks (image; optional) ----------------
def features_image_edges_hough(png_path: Path, params: dict) -> dict:
    feats = {}
    if not _HAS_CV2:
        return feats

    img = cv.imread(str(png_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        return feats

    low = int(params.get("canny_low",50)); high = int(params.get("canny_high",150))
    edges = cv.Canny(img, low, high)

    rho = float(params.get("hough_rho",1))
    theta = np.deg2rad(float(params.get("hough_theta_deg",1)))
    thresh = int(params.get("hough_thresh",60))
    lines = cv.HoughLines(edges, rho, theta, thresh)

    feats["edge_density"] = float(edges.mean())  # 0..255 → density proxy

    # collect top-k line slopes (in degrees) and residual scores
    K = int(params.get("top_hough_lines",4))
    if lines is not None:
        # lines: Nx1x2 (rho, theta)
        angles = []
        for rtheta in lines[:K]:
            rho_i, theta_i = rtheta[0]
            # Convert to slope in degrees (vertical lines ~ inf; represent as 90 deg)
            deg = float(np.rad2deg(theta_i))
            # normalize to [-90, 90)
            if deg >= 90: deg -= 180
            angles.append(deg)
        angles += [0.0] * (K - len(angles))  # pad
    else:
        angles = [0.0]*K

    for j,ang in enumerate(angles, start=1):
        feats[f"hough_angle_{j}"] = float(ang)

    return feats

# ---------------- Main orchestration ----------------
def main():
    ap = argparse.ArgumentParser(description="Build tabular features per image and join with weak labels.")
    ap.add_argument("--backtest_cfg", default="configs/backtest.yaml")
    ap.add_argument("--features_cfg", default="configs/features.yaml")
    ap.add_argument("--labels_csv",   default="reports/labels/weak_labels.csv")
    ap.add_argument("--images_root",  default="data/images/rendered")
    ap.add_argument("--out_dir",      default="data/features")
    args = ap.parse_args()

    feat_cfg = load_yaml(args.features_cfg)
    toggles  = (feat_cfg.get("features") or {})
    params   = (feat_cfg.get("params") or {})

    labels = pd.read_csv(args.labels_csv)

    rows = []
    for idx, row in labels.iterrows():
        image = row["image"]; split = row["split"]; symbol = row["symbol"]
        start_ts = row["start_ts"]; end_ts = row["end_ts"]
        png_path = Path(args.images_root) / split / image

        # Load window OHLCV
        df_full = load_ohlcv(symbol)
        win = slice_window(df_full, start_ts, end_ts)
        N = int(row["bars"])
        if len(win) != N:
            win = win.iloc[-N:]

        feats = {
            "image": image,
            "split": split,
            "symbol": symbol,
            "start_ts": start_ts,
            "end_ts": end_ts,
        }

        # OHLCV feature blocks
        if toggles.get("ohlcv_stats", True):
            feats.update(features_ohlcv_stats(win, params))
        if toggles.get("candle_shape", True):
            feats.update(features_candle_shape(win))
        if toggles.get("swings", True):
            feats.update(features_swings(win, params))
        if toggles.get("pattern_proxies", True):
            feats.update(features_pattern_proxies(win))

        # Image feature blocks (optional)
        if toggles.get("image_edges_hough", False):
            feats.update(features_image_edges_hough(png_path, params))

        # Targets (copy from labels row)
        for col in labels.columns:
            if col.startswith("y_") or col.endswith("_cnt"):
                feats[col] = row[col]

        rows.append(feats)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    # Write per-split CSVs for convenience
    for sp in sorted(df["split"].unique()):
        df_sp = df[df["split"] == sp].copy()
        out_p = out_dir / f"{sp}_features.csv"
        df_sp.to_csv(out_p, index=False)
        print(f"✅ Wrote {out_p} (rows={len(df_sp)})")

    # Also write an 'all' CSV
    all_p = out_dir / "all_features.csv"
    df.to_csv(all_p, index=False)
    print(f"✅ Wrote {all_p} (rows={len(df)})")

if __name__ == "__main__":
    main()

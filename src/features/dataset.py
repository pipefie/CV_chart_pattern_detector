from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.labeling import PatternLabeler
from src.labeling.swing_points import compute_atr as labeling_atr

from .cv_hough import extract_hough_features

LOG = logging.getLogger(__name__)


# ---------------------------- I/O helpers ---------------------------- #
def load_yaml(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_meta(json_path: Path) -> Dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def load_ohlcv(symbol: str, root: Path) -> pd.DataFrame:
    equities = root / "equities_etf" / f"{symbol}.parquet"
    crypto = root / "crypto" / f"{symbol}.parquet"
    if equities.exists():
        df = pd.read_parquet(equities)
    else:
        df = pd.read_parquet(crypto)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def slice_window(df: pd.DataFrame, start_iso: str, end_iso: str) -> pd.DataFrame:
    s = pd.Timestamp(start_iso)
    e = pd.Timestamp(end_iso)
    if s.tzinfo is None:
        s = s.tz_localize("UTC")
    else:
        s = s.tz_convert("UTC")
    if e.tzinfo is None:
        e = e.tz_localize("UTC")
    else:
        e = e.tz_convert("UTC")
    return df.loc[(df.index >= s) & (df.index <= e)]


# ---------------------------- Feature helpers ---------------------------- #
def _safe_div(a, b, eps: float = 1e-9):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    denom = np.where(np.abs(b_arr) > eps, b_arr, np.where(b_arr >= 0, eps, -eps))
    result = a_arr / denom
    if np.ndim(result) == 0:
        return float(result)
    return result


def features_ohlcv_stats(win: pd.DataFrame, params: Dict) -> Dict[str, float]:
    close = win["close"].astype(float)
    ret = close.pct_change().dropna()
    atr = labeling_atr(win, params.get("atr_period", 14))
    feats: Dict[str, float] = {
        "ret_mean": float(ret.mean()) if len(ret) else 0.0,
        "ret_std": float(ret.std(ddof=0)) if len(ret) else 0.0,
        "abs_ret_mean": float(ret.abs().mean()) if len(ret) else 0.0,
        "atr_mean": float(atr.mean()) if len(atr) else 0.0,
        "atr_std": float(atr.std(ddof=0)) if len(atr) else 0.0,
    }

    for p in params.get("roc_periods", [5, 10, 20]):
        idx = max(0, len(close) - p)
        feats[f"roc_{p}"] = float(_safe_div(close.iloc[-1], close.iloc[idx])) - 1.0 if len(close) > idx else 0.0
    for p in params.get("ma_periods", [10, 20, 50]):
        ma = close.rolling(p).mean()
        feats[f"rel_close_ma{p}"] = float(_safe_div(close.iloc[-1], ma.iloc[-1])) - 1.0 if not np.isnan(ma.iloc[-1]) else 0.0
    return feats


def features_candle_shape(win: pd.DataFrame) -> Dict[str, float]:
    o, h, l, c = [win[k].astype(float).to_numpy() for k in ("open", "high", "low", "close")]
    rng = np.maximum(1e-9, h - l)
    body = np.abs(c - o)
    up_wick = h - np.maximum(c, o)
    dn_wick = np.minimum(c, o) - l
    feats = {
        "body_pct_mean": float(np.mean(_safe_div(body, rng))),
        "body_pct_std": float(np.std(_safe_div(body, rng))),
        "up_wick_pct_mean": float(np.mean(_safe_div(up_wick, rng))),
        "dn_wick_pct_mean": float(np.mean(_safe_div(dn_wick, rng))),
        "bull_frac": float(np.mean((c - o) > 0)),
        "bear_frac": float(np.mean((c - o) < 0)),
    }
    return feats


def _pivots(series: np.ndarray, L: int, R: int, kind: str, min_sep: int) -> List[tuple]:
    N = len(series)
    out = []
    last = -10**9
    for i in range(L, N - R):
        v = series[i]
        left = series[i - L : i]
        right = series[i + 1 : i + 1 + R]
        if kind == "high":
            cond = v >= left.max() and v >= right.max()
        else:
            cond = v <= left.min() and v <= right.min()
        if cond and (i - last) >= min_sep:
            out.append((i, float(v)))
            last = i
    return out


def features_swings(win: pd.DataFrame, params: Dict) -> Dict[str, float]:
    close = win["close"].astype(float).to_numpy()
    L = int(params.get("pivot_left", 3))
    R = int(params.get("pivot_right", 3))
    sep = int(params.get("pivot_min_sep", 4))
    piv_hi = _pivots(close, L, R, "high", sep)
    piv_lo = _pivots(close, L, R, "low", sep)

    def spacing(points: List[tuple]) -> float:
        if len(points) < 2:
            return 0.0
        diffs = np.diff([i for i, _ in points])
        return float(np.mean(diffs))

    def prominence(points: List[tuple]) -> float:
        if len(points) < 2:
            return 0.0
        vals = [p for _, p in points]
        return float(np.std(vals))

    feats = {
        "n_piv_hi": float(len(piv_hi)),
        "n_piv_lo": float(len(piv_lo)),
        "piv_hi_spacing_mean": spacing(piv_hi),
        "piv_lo_spacing_mean": spacing(piv_lo),
        "piv_hi_val_std": prominence(piv_hi),
        "piv_lo_val_std": prominence(piv_lo),
    }
    return feats


def features_pattern_proxies(win: pd.DataFrame) -> Dict[str, float]:
    close = win["close"].astype(float).to_numpy()
    N = len(close)
    half = max(8, N // 2)
    segment = close[-half:]

    def two_extrema(arr, is_max=True):
        idx = []
        for i in range(1, len(arr) - 1):
            if is_max and arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
                idx.append(i)
            if not is_max and arr[i] <= arr[i - 1] and arr[i] <= arr[i + 1]:
                idx.append(i)
        if len(idx) < 2:
            return None
        idx = sorted(idx, key=lambda i: arr[i], reverse=is_max)[:2]
        i1, i2 = sorted(idx)
        return i1, arr[i1], i2, arr[i2]

    feats: Dict[str, float] = {}
    dt = two_extrema(segment, True)
    db = two_extrema(segment, False)
    if dt:
        i1, p1, i2, p2 = dt
        mid = 0.5 * (p1 + p2)
        feats["dt_peak_similarity"] = float(1.0 - abs(p1 - p2) / max(mid, 1e-9))
        if i2 - i1 >= 2:
            valley = float(segment[i1 + 1 : i2].min())
            feats["dt_valley_depth"] = float((mid - valley) / max(mid, 1e-9))
        else:
            feats["dt_valley_depth"] = 0.0
        feats["dt_span_bars"] = float(i2 - i1)
    else:
        feats["dt_peak_similarity"] = 0.0
        feats["dt_valley_depth"] = 0.0
        feats["dt_span_bars"] = 0.0
    if db:
        i1, p1, i2, p2 = db
        mid = 0.5 * (p1 + p2)
        feats["db_trough_similarity"] = float(1.0 - abs(p1 - p2) / max(mid, 1e-9))
        if i2 - i1 >= 2:
            peak = float(segment[i1 + 1 : i2].max())
            feats["db_peak_height"] = float((peak - mid) / max(mid, 1e-9))
        else:
            feats["db_peak_height"] = 0.0
        feats["db_span_bars"] = float(i2 - i1)
    else:
        feats["db_trough_similarity"] = 0.0
        feats["db_peak_height"] = 0.0
        feats["db_span_bars"] = 0.0

    best_hs = 0.0
    best_sym = 0.0
    for i in range(2, len(segment) - 2):
        L = segment[i - 1]
        H = segment[i]
        R = segment[i + 1]
        if H <= max(L, R):
            continue
        sh_avg = 0.5 * (L + R)
        rel = (H - sh_avg) / max(sh_avg, 1e-9)
        best_hs = max(best_hs, float(rel))
        sym = 1.0 - abs(((i) - (i - 1)) - ((i + 1) - i)) / max((i + 1) - (i - 1), 1)
        best_sym = max(best_sym, float(sym))
    feats["hs_head_rel"] = best_hs
    feats["hs_time_sym"] = best_sym

    x = np.arange(len(segment))
    trend = np.polyfit(x, segment, 1)
    feats["seg_slope"] = float(trend[0])
    roll = max(3, len(segment) // 10)
    upper = pd.Series(segment).rolling(roll, center=True).max().bfill().ffill()
    lower = pd.Series(segment).rolling(roll, center=True).min().bfill().ffill()
    xu = np.arange(len(upper))
    xl = np.arange(len(lower))
    au, bu = np.polyfit(xu, upper, 1)
    al, bl = np.polyfit(xl, lower, 1)
    gap_s = (au * 0 + bu) - (al * 0 + bl)
    gap_e = (au * (len(x) - 1) + bu) - (al * (len(x) - 1) + bl)
    mid = float(np.mean(segment))
    feats["tri_conv_rate"] = float(_safe_div(gap_s - gap_e, abs(mid)))
    return feats


# ---------------------------- Sample index helpers ---------------------------- #
@dataclass
class Sample:
    image: str
    split: str
    symbol: str
    timeframe: str
    start_ts: str
    end_ts: str
    bars: int
    image_path: Path


def samples_from_labels_csv(labels_csv: Path, images_root: Path) -> Iterator[Sample]:
    df = pd.read_csv(labels_csv)
    required = {"image", "split", "symbol", "timeframe", "start_ts", "end_ts", "bars"}
    if not required.issubset(df.columns):
        raise ValueError(f"labels CSV missing required columns: {required - set(df.columns)}")
    for row in df.itertuples(index=False):
        img_path = images_root / getattr(row, "split") / getattr(row, "image")
        yield Sample(
            image=getattr(row, "image"),
            split=getattr(row, "split"),
            symbol=getattr(row, "symbol"),
            timeframe=getattr(row, "timeframe"),
            start_ts=getattr(row, "start_ts"),
            end_ts=getattr(row, "end_ts"),
            bars=int(getattr(row, "bars")),
            image_path=img_path,
        )


def samples_from_manifest(manifest_path: Path) -> Iterator[Sample]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_root = Path(manifest["outputs"]["out_root"])
    if not out_root.is_absolute():
        candidate = (manifest_path.parent / out_root).resolve()
        if candidate.exists():
            out_root = candidate
        else:
            out_root = out_root.resolve()
    for split in ("train", "val", "test"):
        split_dir = out_root / split
        if not split_dir.exists():
            continue
        for png in sorted(split_dir.glob("*.png")):
            meta = load_meta(png.with_suffix(".json"))
            yield Sample(
                image=png.name,
                split=split,
                symbol=meta["symbol"],
                timeframe=meta.get("timeframe", ""),
                start_ts=meta["start_ts"],
                end_ts=meta["end_ts"],
                bars=int(meta["bars"]),
                image_path=png,
            )


# ---------------------------- Dataset builder ---------------------------- #
class DatasetBuilder:
    def __init__(
        self,
        pipeline_cfg: Dict,
        ohlcv_root: Path,
        images_root: Path,
        manifest_path: Optional[Path] = None,
        labels_csv: Optional[Path] = None,
    ):
        self.pipeline_cfg = pipeline_cfg
        self.ohlcv_root = ohlcv_root
        self.images_root = images_root
        self.manifest_path = manifest_path
        self.labels_csv = labels_csv
        self.pattern_labeler = PatternLabeler(pipeline_cfg.get("labeling", {}))
        self.cv_cfg = (pipeline_cfg.get("cv_features") or {}).get("hough", {})
        self.feature_cfg = pipeline_cfg.get("features", {})
        self._ohlcv_cache: Dict[str, pd.DataFrame] = {}

    def _iter_samples(self) -> Iterator[Sample]:
        if self.manifest_path:
            return samples_from_manifest(self.manifest_path)
        if self.labels_csv:
            return samples_from_labels_csv(self.labels_csv, self.images_root)
        raise ValueError("Either labels_csv or manifest_path must be provided.")

    def _get_symbol_df(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._ohlcv_cache:
            self._ohlcv_cache[symbol] = load_ohlcv(symbol, self.ohlcv_root)
        return self._ohlcv_cache[symbol]

    def _compute_features(self, win: pd.DataFrame, png_path: Path) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        ta_cfg = self.feature_cfg.get("ta", {})
        if ta_cfg:
            feats.update(features_ohlcv_stats(win, ta_cfg))
        if (self.feature_cfg.get("candles") or {}).get("enabled", True):
            feats.update(features_candle_shape(win))
        if (self.feature_cfg.get("swings") or {}).get("enabled", True):
            feats.update(features_swings(win, self.feature_cfg.get("swings", {})))
        if (self.feature_cfg.get("pattern_proxies") or {}).get("enabled", True):
            feats.update(features_pattern_proxies(win))
        if self.cv_cfg:
            feats.update(extract_hough_features(png_path, self.cv_cfg))
        return feats

    def build(self) -> pd.DataFrame:
        rows: List[Dict] = []
        for sample in self._iter_samples():
            symbol_df = self._get_symbol_df(sample.symbol)
            win = slice_window(symbol_df, sample.start_ts, sample.end_ts)
            if len(win) == 0:
                LOG.warning("Empty OHLCV window for %s (%s)", sample.image, sample.symbol)
                continue
            if len(win) != sample.bars:
                win = win.iloc[-sample.bars :]
            labels, structural = self.pattern_labeler.label_window(win)
            if not labels:
                labels = {}
            feats = self._compute_features(win, sample.image_path)
            feats.update(structural)

            row = {
                "image": sample.image,
                "image_path": str(sample.image_path),
                "split": sample.split,
                "symbol": sample.symbol,
                "timeframe": sample.timeframe,
                "start_ts": sample.start_ts,
                "end_ts": sample.end_ts,
                "bars": sample.bars,
            }
            row.update(feats)
            for key, value in labels.items():
                row[key] = int(value)
            rows.append(row)
        if not rows:
            raise RuntimeError("No rows built; check inputs.")
        df = pd.DataFrame(rows)
        return df


def build_dataset(
    pipeline_cfg_path: Path,
    ohlcv_root: Path,
    images_root: Path,
    out_dir: Path,
    labels_csv: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Path]:
    cfg = load_yaml(pipeline_cfg_path)
    builder = DatasetBuilder(
        pipeline_cfg=cfg,
        ohlcv_root=ohlcv_root,
        images_root=images_root,
        manifest_path=manifest_path,
        labels_csv=labels_csv,
    )
    df = builder.build()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for split in sorted(df["split"].unique()):
        df_sp = df[df["split"] == split]
        path = out_dir / f"{split}_features.csv"
        df_sp.to_csv(path, index=False)
        outputs[split] = path
        LOG.info("Wrote %s rows=%d", path, len(df_sp))
    path_all = out_dir / "all_features.csv"
    df.to_csv(path_all, index=False)
    outputs["all"] = path_all
    LOG.info("Wrote %s rows=%d", path_all, len(df))
    return outputs


__all__ = [
    "build_dataset",
    "DatasetBuilder",
    "samples_from_manifest",
    "samples_from_labels_csv",
    "load_ohlcv",
    "slice_window",
    "load_yaml",
]

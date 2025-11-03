from __future__ import annotations
import argparse, json, random
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

def load_meta(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))

def load_ohlcv(symbol: str) -> pd.DataFrame:
    root = Path("data/ohlcv")
    # infer asset folder from what you used during fetch
    if (root / "equities_etf" / f"{symbol}.parquet").exists():
        p = root / "equities_etf" / f"{symbol}.parquet"
    else:
        p = root / "crypto" / f"{symbol}.parquet"
    df = pd.read_parquet(p)
    # ensure UTC tz-aware index
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df

def slice_ohlcv(df: pd.DataFrame, start_iso: str, end_iso: str) -> pd.DataFrame:
    s = pd.Timestamp(start_iso).tz_convert("UTC") if pd.Timestamp(start_iso).tzinfo else pd.Timestamp(start_iso, tz="UTC")
    e = pd.Timestamp(end_iso).tz_convert("UTC") if pd.Timestamp(end_iso).tzinfo else pd.Timestamp(end_iso, tz="UTC")
    out = df.loc[(df.index >= s) & (df.index <= e)]
    # drop last partial bar if present (should already be clean)
    return out

def price_to_y(px: float, img_h: int, y_min: float, y_max: float) -> int:
    # invert y (pixel 0 = top). Clamp for safety.
    y_rel = (px - y_min) / max(1e-9, (y_max - y_min))
    y = int(round((1.0 - y_rel) * (img_h - 1)))
    return max(0, min(img_h - 1, y))

def index_to_x(i: int, n: int, img_w: int) -> int:
    # spread bars across full width [0, W-1]
    if n <= 1:
        return (img_w - 1) // 2
    x = int(round((i / (n - 1)) * (img_w - 1)))
    return max(0, min(img_w - 1, x))

def overlay_polyline(png_path: Path, json_path: Path, out_dir: Path):
    meta = load_meta(json_path)
    symbol   = meta["symbol"]
    start_ts = meta["start_ts"]
    end_ts   = meta["end_ts"]
    img_w    = int(meta["img_w"])
    img_h    = int(meta["img_h"])
    y_min, y_max = float(meta["axes_ylim"][0]), float(meta["axes_ylim"][1])

    # Load OHLCV slice
    df_full = load_ohlcv(symbol)
    df_win  = slice_ohlcv(df_full, start_ts, end_ts)
    if df_win.empty:
        print(f"⚠️ No OHLCV slice for {symbol} {start_ts}→{end_ts}")
        return

    closes = df_win["close"].to_numpy()
    n = len(closes)

    img = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw small points along the expected close path
    pts = []
    for i, c in enumerate(closes):
        x = index_to_x(i, n, img_w)
        y = price_to_y(float(c), img_h, y_min, y_max)
        pts.append((x, y))
        # a tiny 3x3 marker
        draw.ellipse((x-1, y-1, x+1, y+1), fill=(0, 200, 0))

    # Optionally connect with thin lines
    for i in range(1, len(pts)):
        draw.line([pts[i-1], pts[i]], fill=(0, 200, 0), width=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / png_path.name
    img.save(out)

def main():
    ap = argparse.ArgumentParser(description="Overlay expected close polyline (from Parquet) onto rendered PNGs.")
    ap.add_argument("--split", choices=["train","val","test"], default="val")
    ap.add_argument("--num", type=int, default=8, help="number of samples to validate")
    args = ap.parse_args()

    src = Path("data/images/rendered") / args.split
    if not src.exists():
        print(f"Folder not found: {src}")
        return

    # pick pngs that have a matching .json sidecar
    pngs = [p for p in src.glob("*.png") if p.with_suffix(".json").exists()]
    if not pngs:
        print("No PNG+JSON pairs found.")
        return

    rng = random.Random(42)
    sel = rng.sample(pngs, min(args.num, len(pngs)))
    out_dir = Path("data/images/_debug_overlays") / args.split

    for p in sel:
        overlay_polyline(p, p.with_suffix(".json"), out_dir)
        print(f"✅ Overlay -> {out_dir / p.name}")

    print("✅ Validation done.")

if __name__ == "__main__":
    main()

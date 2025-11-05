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
    if (root / "equities_etf" / f"{symbol}.parquet").exists():
        p = root / "equities_etf" / f"{symbol}.parquet"
    else:
        p = root / "crypto" / f"{symbol}.parquet"
    df = pd.read_parquet(p)
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df

def slice_ohlcv(df: pd.DataFrame, start_iso: str, end_iso: str) -> pd.DataFrame:
    s = pd.Timestamp(start_iso)
    e = pd.Timestamp(end_iso)
    s = s.tz_localize("UTC") if s.tzinfo is None else s.tz_convert("UTC")
    e = e.tz_localize("UTC") if e.tzinfo is None else e.tz_convert("UTC")
    return df.loc[(df.index >= s) & (df.index <= e)]

def price_to_y(px: float, axes_top: int, axes_height: int, y_min: float, y_max: float) -> int:
    rel = (px - y_min) / max(1e-9, (y_max - y_min))
    y_in_axes = int(round((1.0 - rel) * (axes_height - 1)))   # 0 at top
    return axes_top + y_in_axes

def index_to_x(i: int, n: int, axes_left: int, axes_width: int) -> int:
    if n <= 1:
        return axes_left + (axes_width - 1) // 2
    rel = i / (n - 1)
    return axes_left + int(round(rel * (axes_width - 1)))

def overlay_polyline(png_path: Path, json_path: Path, out_dir: Path):
    meta = load_meta(json_path)
    symbol   = meta["symbol"]
    start_ts = meta["start_ts"]
    end_ts   = meta["end_ts"]
    y_min, y_max = float(meta["axes_ylim"][0]), float(meta["axes_ylim"][1])

    # NEW: axes bbox in image pixels (top-left origin)
    axes_left, axes_top, axes_width, axes_height = map(int, meta["axes_bbox_px"])

    df_full = load_ohlcv(symbol)
    df_win  = slice_ohlcv(df_full, start_ts, end_ts)
    if df_win.empty:
        print(f"⚠️ No OHLCV slice for {symbol} {start_ts}→{end_ts}")
        return

    closes = df_win["close"].to_numpy()
    n = len(closes)

    img = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    pts = []
    for i, c in enumerate(closes):
        x = index_to_x(i, n, axes_left, axes_width)
        y = price_to_y(float(c), axes_top, axes_height, y_min, y_max)
        pts.append((x, y))
        draw.ellipse((x-1, y-1, x+1, y+1), fill=(0, 200, 0))
    for i in range(1, len(pts)):
        draw.line([pts[i-1], pts[i]], fill=(0, 200, 0), width=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / png_path.name)

def main():
    ap = argparse.ArgumentParser(description="Overlay expected close polyline onto rendered PNGs.")
    ap.add_argument("--split", choices=["train","val","test"], default="val")
    ap.add_argument("--num", type=int, default=8)
    args = ap.parse_args()

    src = Path("data/images/rendered") / args.split
    if not src.exists():
        print(f"Folder not found: {src}")
        return

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

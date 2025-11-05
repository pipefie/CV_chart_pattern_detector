# scripts/make_images.py
from __future__ import annotations
import argparse, json, random, gc
from pathlib import Path
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import yaml

# ---- Load configs ----
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---- Utility: choose jitter parameter from [min,max] or list ----
def _pick(val, rng: random.Random):
    if isinstance(val, list) and len(val) == 2 and all(isinstance(x, (int,float)) for x in val):
        a, b = val
        return rng.uniform(a, b)
    if isinstance(val, list):
        return rng.choice(val) if hasattr(rng, "choice") else random.choice(val)
    return val

# ---- Split by dates into train/val/test ----
def split_tag(ts_end: pd.Timestamp, wf_cfg: dict) -> str:
    train_end    = pd.Timestamp(wf_cfg["train_end"]).tz_localize("UTC")
    validate_end = pd.Timestamp(wf_cfg["validate_end"]).tz_localize("UTC")
    test_end     = pd.Timestamp(wf_cfg["test_end"]).tz_localize("UTC")
    if ts_end <= train_end:        return "train"
    elif ts_end <= validate_end:   return "val"
    elif ts_end <= test_end:       return "test"
    else:                          return "ignore"

# ---- Render one window to PNG + metadata ----
def render_window_png(
    df_win: pd.DataFrame,
    out_png: Path,
    meta_out: Path,
    render_cfg: dict,
    symbol: str,
    timeframe: str,
    rng: random.Random,
):
    # Canvas & style jitter
    W   = render_cfg["canvas"]["width"]
    H   = render_cfg["canvas"]["height"]
    dpi = render_cfg["canvas"]["dpi"]
    bg  = render_cfg["canvas"]["bg_color"]

    bull = _pick(render_cfg["chart"]["bull_color"], rng)
    bear = _pick(render_cfg["chart"]["bear_color"], rng)
    wick = int(round(_pick(render_cfg["chart"]["wick_thickness_px"], rng)))
    body = int(round(_pick(render_cfg["chart"]["body_thickness_px"], rng)))
    grid_on    = _pick(render_cfg["chart"]["grid"]["enabled"], rng)
    grid_alpha = render_cfg["chart"]["grid"]["alpha"] if grid_on else 0.0
    font       = _pick(render_cfg["axes"]["font_family"], rng)
    font_size  = int(round(_pick(render_cfg["axes"]["font_size"], rng)))
    show_ticks = _pick(render_cfg["axes"]["show_ticks"], rng)

    matplotlib.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.facecolor": bg,
        "figure.facecolor": bg,
        "font.family": font,
        "font.size": font_size,
    })

    style = mpf.make_mpf_style(
        base_mpl_style="seaborn-v0_8",
        gridstyle="-" if grid_on else "",
        gridcolor="#000000",
        facecolor=bg,
        edgecolor=bg,
        marketcolors=mpf.make_marketcolors(
            up=bull, down=bear,
            wick={"up": bull, "down": bear},
            edge={"up": bull, "down": bear},
            volume="in",
        ),
    )

    plot_kwargs = dict(
        type="candle",
        style=style,
        figsize=(W / dpi, H / dpi),
        tight_layout=True,
        xrotation=0,
        volume=False,
        returnfig=True,
    )
    uwc = {"candle_linewidth": body}

    fig = None
    # Defaults for metadata in case of early error
    x_min = x_max = y_min = y_max = 0.0
    axes_left = axes_top = axes_width = axes_height = 0
    fig_w_px = W; fig_h_px = H

    try:
        # Try wick width variants (mplfinance versions differ)
        try:
            fig, axes = mpf.plot(df_win, **plot_kwargs,
                                 update_width_config={**uwc, "wick_linewidth": wick})
        except Exception:
            try:
                fig, axes = mpf.plot(df_win, **plot_kwargs,
                                     update_width_config={**uwc, "wick_width": wick})
            except Exception:
                fig, axes = mpf.plot(df_win, **plot_kwargs, update_width_config=uwc)

        ax_main = axes[0] if isinstance(axes, (list, tuple, np.ndarray)) else axes

        if not show_ticks:
            ax_main.set_xticks([]); ax_main.set_yticks([])

        # Layout must be drawn before reading pixel geometry
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Axis limits for price mapping
        x_min, x_max = ax_main.get_xlim()
        y_min, y_max = ax_main.get_ylim()

        # Axes bbox in display pixels (origin: lower-left), then convert to image top-left
        bbox_disp = ax_main.get_window_extent(renderer=renderer)
        x0, y0, w, h = bbox_disp.x0, bbox_disp.y0, bbox_disp.width, bbox_disp.height

        fig_w_px = int(round(fig.get_size_inches()[0] * fig.dpi))
        fig_h_px = int(round(fig.get_size_inches()[1] * fig.dpi))

        axes_left   = int(round(x0))
        axes_top    = int(round(fig_h_px - (y0 + h)))  # flip Y
        axes_width  = int(round(w))
        axes_height = int(round(h))

        # Grid alpha
        if grid_on:
            axes_iter = axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
            for ax in axes_iter:
                try:
                    for line in ax.get_xgridlines() + ax.get_ygridlines():
                        line.set_alpha(grid_alpha)
                except Exception:
                    pass

        out_png.parent.mkdir(parents=True, exist_ok=True)
        # IMPORTANT: no bbox_inches="tight" (keeps geometry stable)
        fig.savefig(out_png)

    finally:
        if fig is not None:
            plt.close(fig)

    # Post-process (blur/noise) with deterministic RNG
    blur_sigma = _pick(render_cfg["style_jitter"]["blur_sigma"], rng)
    noise_std  = _pick(render_cfg["style_jitter"]["noise_std"], rng)

    im = None
    try:
        im = Image.open(out_png)
        if blur_sigma and float(blur_sigma) > 0:
            im = im.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
        if noise_std and float(noise_std) > 0:
            np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
            arr = np.asarray(im).astype(np.float32)
            noise = np_rng.normal(0.0, float(noise_std), size=arr.shape)
            arr = np.clip(arr + noise * 255.0, 0, 255).astype(np.uint8)
            im = Image.fromarray(arr)
        im.save(out_png)
    finally:
        if im is not None:
            im.close()

    # Metadata (sidecar JSON)
    meta = {
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": int(len(df_win)),
        "start_ts": df_win.index[0].isoformat(),
        "end_ts": df_win.index[-1].isoformat(),
        "img_w": W, "img_h": H, "dpi": dpi,
        "axes_xlim": [float(x_min), float(x_max)],
        "axes_ylim": [float(y_min), float(y_max)],
        "axes_bbox_px": [axes_left, axes_top, axes_width, axes_height],  # <-- NEW
        "fig_size_px": [fig_w_px, fig_h_px],                              # <-- NEW
        "bull_color": bull, "bear_color": bear,
        "wick_px": wick, "body_px": body,
        "grid": bool(grid_on), "font": font, "font_size": font_size,
    }
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

# ---- Main orchestration ----
def main():
    ap = argparse.ArgumentParser(description="Render candlestick chart images with jitter from Parquet OHLCV.")
    ap.add_argument("--backtest_cfg", default="configs/backtest.yaml")
    ap.add_argument("--render_cfg",  default="configs/render.yaml")
    ap.add_argument("--window_bars", type=int, default=160)
    ap.add_argument("--stride_bars", type=int, default=40)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    backtest   = load_yaml(args.backtest_cfg)
    render_cfg = load_yaml(args.render_cfg)

    src_root = Path("data/ohlcv")
    out_root = Path("data/images/rendered")

    universes = {
        "crypto": backtest["universes"]["crypto"],
        "equities_etf": backtest["universes"]["equities_etf"],
    }

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

                schema   = render_cfg["export"]["naming"]["schema"]
                out_name = schema.format(
                    symbol=sym,
                    tf=timeframe,
                    startts=int(win.index[0].timestamp()),
                    endts=int(win.index[-1].timestamp()),
                    seed=args.seed,
                )
                out_png  = out_root / bucket / out_name
                meta_out = out_png.with_suffix(".json")

                render_window_png(win, out_png, meta_out, render_cfg, symbol=sym, timeframe=timeframe, rng=rng)

                counter += 1
                if (counter % 50) == 0:
                    plt.close("all"); gc.collect()

            print(f"✅ Rendered {sym} ({group})")

    print("✅ All renders complete.")

if __name__ == "__main__":
    main()

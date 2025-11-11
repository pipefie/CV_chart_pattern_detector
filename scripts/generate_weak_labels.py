# scripts/generate_weak_labels.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# ---------- Helpers ----------
def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))

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

# Pixel mapping
def bar_to_x(i: int, N: int, W: int) -> int:
    if N <= 1: return (W - 1) // 2
    return int(round((i / (N - 1)) * (W - 1)))

def price_to_y(p: float, y_min: float, y_max: float, H: int) -> int:
    y_rel = (p - y_min) / max(1e-9, (y_max - y_min))
    return int(round((1.0 - y_rel) * (H - 1)))

# YOLO writer
def write_yolo(path: Path, boxes: list[tuple[int, float, float, float, float]]):
    with open(path, "w", encoding="utf-8") as f:
        for cid, cx, cy, w, h in boxes:
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# ---------- Pivot detection ----------
def pivots(series: np.ndarray, L: int, R: int, kind: str, min_sep: int) -> list[tuple[int, float]]:
    N = len(series)
    idx = []
    last_i = -10**9
    for i in range(L, N - R):
        window_left  = series[i - L: i]
        window_right = series[i + 1: i + 1 + R]
        v = series[i]
        if kind == "high":
            if v >= window_left.max() and v >= window_right.max():
                if i - last_i >= min_sep:
                    idx.append((i, float(v))); last_i = i
        else:
            if v <= window_left.min() and v <= window_right.min():
                if i - last_i >= min_sep:
                    idx.append((i, float(v))); last_i = i
    return idx

# ---------- Pattern rules (baseline) ----------
def detect_double_top(closes: np.ndarray, piv_hi: list[tuple[int,float]], cfg: dict) -> list[dict]:
    out = []
    tol_rel  = cfg["peak_tolerance_rel"]
    drop_rel = cfg["valley_drop_min_rel"]
    min_span = cfg["min_span_bars"]; max_span = cfg["max_span_bars"]
    for i1, p1 in piv_hi:
        for i2, p2 in piv_hi:
            if i2 <= i1 + 2: continue
            span = i2 - i1
            if span < min_span or span > max_span: continue
            mid = (p1 + p2) * 0.5
            if abs(p1 - p2) / max(1e-9, mid) > tol_rel: continue
            j0, j1 = i1 + 1, i2
            if j1 <= j0 + 1: continue
            valley = float(closes[j0:j1].min())
            if (mid - valley) / max(1e-9, mid) < drop_rel: continue
            out.append({"type":"double_top","i1":i1,"p1":p1,"i2":i2,"p2":p2,"ivalley":int(np.argmin(closes[j0:j1])+j0),"pvalley":valley})
    return out

def detect_double_bottom(closes: np.ndarray, piv_lo: list[tuple[int,float]], cfg: dict) -> list[dict]:
    out = []
    tol_rel  = cfg["trough_tolerance_rel"]
    rise_rel = cfg["peak_rise_min_rel"]
    min_span = cfg["min_span_bars"]; max_span = cfg["max_span_bars"]
    for i1, p1 in piv_lo:
        for i2, p2 in piv_lo:
            if i2 <= i1 + 2: continue
            span = i2 - i1
            if span < min_span or span > max_span: continue
            mid = (p1 + p2) * 0.5
            if abs(p1 - p2) / max(1e-9, mid) > tol_rel: continue
            j0, j1 = i1 + 1, i2
            if j1 <= j0 + 1: continue
            peak = float(closes[j0:j1].max())
            if (peak - mid) / max(1e-9, mid) < rise_rel: continue
            out.append({"type":"double_bottom","i1":i1,"p1":p1,"i2":i2,"p2":p2,"ipeak":int(np.argmax(closes[j0:j1])+j0),"ppeak":peak})
    return out

def detect_head_shoulders(closes: np.ndarray, piv_hi: list[tuple[int,float]], cfg: dict) -> list[dict]:
    out = []
    min_rel = cfg["min_rel_height_head_vs_shoulders"]
    sym_tol = cfg["shoulder_symmetry_tolerance_bars"]
    min_span = cfg["min_span_bars"]; max_span = cfg["max_span_bars"]
    for iL, pL in piv_hi:
        for iH, pH in piv_hi:
            if iH <= iL + 2: continue
            for iR, pR in piv_hi:
                if iR <= iH + 2: continue
                span = iR - iL
                if span < min_span or span > max_span: continue
                sh_avg = 0.5*(pL+pR)
                rel = (pH - sh_avg)/max(1e-9, sh_avg)
                if rel < min_rel: continue
                mid = 0.5*(iL + iH)
                if abs(iR - mid) > sym_tol: continue
                out.append({"type":"head_shoulders","iL":iL,"pL":pL,"iH":iH,"pH":pH,"iR":iR,"pR":pR})
    return out

def detect_triangle(closes: np.ndarray, piv_hi: list[tuple[int,float]], piv_lo: list[tuple[int,float]], cfg: dict) -> list[dict]:
    out = []
    min_tu = cfg["min_touches_upper"]; min_tl = cfg["min_touches_lower"]
    tol_rel = cfg["envelope_tol_rel"]; conv_min = cfg["convergence_min_rel"]
    min_span = cfg["min_span_bars"]; max_span = cfg["max_span_bars"]
    if len(piv_hi) < min_tu or len(piv_lo) < min_tl: return out
    for s in range(0, len(closes) - min_span):
        e = min(len(closes)-1, s + max_span)
        if e - s < min_span: continue
        hi = [(i,p) for (i,p) in piv_hi if s <= i <= e]
        lo = [(i,p) for (i,p) in piv_lo if s <= i <= e]
        if len(hi) < min_tu or len(lo) < min_tl: continue
        xi_hi = np.array([i for (i,_) in hi]); yi_hi = np.array([p for (_,p) in hi])
        xi_lo = np.array([i for (i,_) in lo]); yi_lo = np.array([p for (_,p) in lo])
        a_hi, b_hi = np.polyfit(xi_hi, yi_hi, 1)
        a_lo, b_lo = np.polyfit(xi_lo, yi_lo, 1)
        gap_s = (a_hi*s + b_hi) - (a_lo*s + b_lo)
        gap_e = (a_hi*e + b_hi) - (a_lo*e + b_lo)
        mid = 0.5*((a_hi*s+b_hi) + (a_lo*s+b_lo))
        if mid == 0: continue
        if (gap_s - gap_e)/abs(mid) < conv_min: continue
        ok_hi = np.mean(np.abs(yi_hi - (a_hi*xi_hi+b_hi))/np.maximum(1e-9, np.abs(yi_hi))) < tol_rel
        ok_lo = np.mean(np.abs(yi_lo - (a_lo*xi_lo+b_lo))/np.maximum(1e-9, np.abs(yi_lo))) < tol_rel
        if not (ok_hi and ok_lo): continue
        out.append({"type":"triangle","start":s,"end":e,"a_hi":float(a_hi),"b_hi":float(b_hi),"a_lo":float(a_lo),"b_lo":float(b_lo)})
    return out

# ---------- Main ----------
CLASS_IDS = {"head_shoulders":0, "double_top":1, "double_bottom":2, "triangle":3}

def main():
    ap = argparse.ArgumentParser(description="Generate weak labels (YOLO + CSV presence flags) for rendered images.")
    ap.add_argument("--patterns_cfg", default="configs/patterns.yaml")
    ap.add_argument("--images_root", default="data/images/rendered")
    ap.add_argument("--labels_root", default="data/labels/rendered")
    ap.add_argument("--labels_csv", default="reports/labels/weak_labels.csv",
                    help="CSV with one row per image: presence flags and metadata")
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    args = ap.parse_args()

    cfg = load_yaml(Path(args.patterns_cfg))
    labels_root = Path(args.labels_root); labels_root.mkdir(parents=True, exist_ok=True)

    # accumulate CSV rows here
    rows = []

    for split in args.splits:
        img_dir = Path(args.images_root) / split
        out_yolo = labels_root / split; out_json = labels_root / f"{split}_json"
        out_yolo.mkdir(parents=True, exist_ok=True); out_json.mkdir(parents=True, exist_ok=True)

        pngs = sorted(img_dir.glob("*.png"))
        for png in pngs:
            meta = load_meta(png.with_suffix(".json"))
            symbol = meta["symbol"]
            N = meta["bars"]; W = meta["img_w"]; H = meta["img_h"]
            y_min, y_max = meta["axes_ylim"]
            timeframe = meta.get("timeframe", "")  # stored by your renderer
            df_full = load_ohlcv(symbol)
            win = slice_window(df_full, meta["start_ts"], meta["end_ts"])
            if len(win) != N:
                win = win.iloc[-N:]

            closes = win["close"].to_numpy()

            # pivots
            L = cfg["swing"]["lookback_left"]; R = cfg["swing"]["lookback_right"]; sep = cfg["swing"]["min_separation"]
            piv_hi = pivots(closes, L, R, "high", sep)
            piv_lo = pivots(closes, L, R, "low",  sep)

            # detections
            hs  = detect_head_shoulders(closes, piv_hi, cfg["head_shoulders"])
            dt  = detect_double_top(closes, piv_hi, cfg["double_top"])
            db  = detect_double_bottom(closes, piv_lo, cfg["double_bottom"])
            tri = detect_triangle(closes, piv_hi, piv_lo, cfg["triangle"])

            # presence flags and counts
            y_hs  = 1 if len(hs)  > 0 else 0
            y_dt  = 1 if len(dt)  > 0 else 0
            y_db  = 1 if len(db)  > 0 else 0
            y_tri = 1 if len(tri) > 0 else 0

            hs_cnt, dt_cnt, db_cnt, tri_cnt = len(hs), len(dt), len(db), len(tri)

            # ---- Optional: still write YOLO bboxes & rich JSON (unchanged) ----
            yolo_boxes = []
            rich = {"image": png.name, "detections": []}

            for det_list, cls_name in [(hs,"head_shoulders"), (dt,"double_top"), (db,"double_bottom"), (tri,"triangle")]:
                for d in det_list:
                    # Build bbox from defining vertices
                    xs, ys = [], []
                    if cls_name == "head_shoulders":
                        for (i,p) in [(d["iL"],d["pL"]), (d["iH"],d["pH"]), (d["iR"],d["pR"])]: 
                            xs.append(bar_to_x(i, N, W)); ys.append(price_to_y(p, y_min, y_max, H))
                    elif cls_name == "double_top":
                        for (i,p) in [(d["i1"],d["p1"]), (d["i2"],d["p2"]), (d["ivalley"],d["pvalley"])]:
                            xs.append(bar_to_x(i, N, W)); ys.append(price_to_y(p, y_min, y_max, H))
                    elif cls_name == "double_bottom":
                        for (i,p) in [(d["i1"],d["p1"]), (d["i2"],d["p2"]), (d["ipeak"],d["ppeak"])]:
                            xs.append(bar_to_x(i, N, W)); ys.append(price_to_y(p, y_min, y_max, H))
                    elif cls_name == "triangle":
                        iL, iR = d["start"], d["end"]
                        p_topL = d["a_hi"]*iL + d["b_hi"]; p_topR = d["a_hi"]*iR + d["b_hi"]
                        p_botL = d["a_lo"]*iL + d["b_lo"]; p_botR = d["a_lo"]*iR + d["b_lo"]
                        for (i,p) in [(iL,p_topL),(iR,p_topR),(iL,p_botL),(iR,p_botR)]:
                            xs.append(bar_to_x(i, N, W)); ys.append(price_to_y(p, y_min, y_max, H))

                    xmin, xmax = max(0, min(xs)), min(W-1, max(xs))
                    ymin, ymax = max(0, min(ys)), min(H-1, max(ys))
                    pad = cfg[cls_name].get("bbox_padding_px", 4)
                    xmin = max(0, xmin - pad); xmax = min(W-1, xmax + pad)
                    ymin = max(0, ymin - pad); ymax = min(H-1, ymax + pad)

                    cid = {"head_shoulders":0,"double_top":1,"double_bottom":2,"triangle":3}[cls_name]
                    cx = (xmin + xmax) / 2 / W
                    cy = (ymin + ymax) / 2 / H
                    bw = (xmax - xmin) / W
                    bh = (ymax - ymin) / H
                    yolo_boxes.append((cid, cx, cy, bw, bh))

                    rich["detections"].append({"class": cls_name, "bbox_px": [int(xmin),int(ymin),int(xmax),int(ymax)], "raw": d})

            # write YOLO (only if any detection)
            if yolo_boxes:
                (labels_root / split).mkdir(parents=True, exist_ok=True)
                write_yolo((labels_root / split / (png.stem + ".txt")), yolo_boxes)
                # rich json (audit trail)
                (labels_root / f"{split}_json").mkdir(parents=True, exist_ok=True)
                (labels_root / f"{split}_json" / (png.stem + ".json")).write_text(json.dumps(rich, indent=2), encoding="utf-8")

            # ---- Append CSV row (for RF) ----
            rows.append({
                "image": png.name,
                "split": split,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_ts": meta["start_ts"],
                "end_ts": meta["end_ts"],
                "bars": int(N),
                "img_w": int(W),
                "img_h": int(H),
                "y_hs": y_hs, "y_dt": y_dt, "y_db": y_db, "y_tri": y_tri,
                "hs_cnt": hs_cnt, "dt_cnt": dt_cnt, "db_cnt": db_cnt, "tri_cnt": tri_cnt
            })

        print(f"✅ Split {split}: YOLO & presence flags computed.")

    # ---- Write consolidated CSV ----
    out_csv = Path(args.labels_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(by=["split","symbol","end_ts","image"])
    df.to_csv(out_csv, index=False)
    print(f"✅ Presence-labels CSV: {out_csv} (rows={len(df)})")

if __name__ == "__main__":
    main()

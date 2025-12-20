from __future__ import annotations
import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rendering.renderer import render_window_png, load_yaml
from src.standardize.standardize_images import standardize_one
from src.features.dataset import build_dataset

LOG = logging.getLogger("predict")


def setup_logging(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )


def load_model(model_path: Path) -> Tuple[any, float, bool, List[str]]:
    """
    Load a model and its metadata (threshold, dropped columns).
    Returns: (model, threshold, drop_structural, dropped_cols)
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # helper validation
    manifest_path = model_path.parent / "train_manifest.json"
    threshold = 0.5
    drop_structural = True
    dropped_cols = []
    
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        threshold = meta.get("best_threshold", 0.5)
        # Check if we can deduce drop_structural from meta or if it's explicit
        # The user said: "Training drops structural features by default... Inference must respect the same feature selection."
        # We'll assume True unless we see evidence otherwise, or if keys are present
        if "params" in meta:
            drop_structural = meta["params"].get("drop_structural", True)
        
        # Some manifests might store the kept/dropped columns
        if "features" in meta and "dropped" in meta["features"]:
            dropped_cols = meta["features"]["dropped"]
            
    return model, threshold, drop_structural, dropped_cols


def predict_process(
    ohlcv_path: Path,
    out_dir: Path,
    pipeline_cfg_path: Path,
    models_root: Path,
    timeframe_tag: str,
    window_bars: int,
    stride_bars: int,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    cleanup: bool = False
):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load OHLCV
    LOG.info(f"Loading OHLCV from {ohlcv_path}")
    df = pd.read_parquet(ohlcv_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
        
    symbol = ohlcv_path.stem
    
    # Filter by time if requested
    if start_ts:
        s = pd.Timestamp(start_ts).tz_localize("UTC")
        df = df[df.index >= s]
    if end_ts:
        e = pd.Timestamp(end_ts).tz_localize("UTC")
        df = df[df.index <= e]
        
    if len(df) < window_bars:
        LOG.warning(f"Not enough data for {symbol} (rows={len(df)}, window={window_bars})")
        return

    # 2. Generate Windows & Render
    render_root = out_dir / "rendered"
    render_root.mkdir(parents=True, exist_ok=True)
    # We create a split subfolder 'test' because make_images/renderer logic often assumes a split structure
    render_split_dir = render_root / "test"
    render_split_dir.mkdir(parents=True, exist_ok=True)
    
    std_root = out_dir / "standardized"
    std_root.mkdir(parents=True, exist_ok=True)
    std_split_dir = std_root / "test"
    std_split_dir.mkdir(parents=True, exist_ok=True)
    
    render_cfg_path = Path("configs/render.yaml") # Default assumption, or pass via args if needed. 
    # For now we assume configs/render.yaml exists since it's in the repo.
    if not render_cfg_path.exists():
        # Fallback or error? The user command example implies pipeline_cfg but strictly speaking render config is separate.
        # implementation plan didn't specify render_cfg arg, so we use default.
        pass
    
    render_cfg = load_yaml(render_cfg_path)
    
    rng = random.Random(42)
    
    windows_meta = []
    
    LOG.info("Rendering windows...")
    
    # Sliding window
    count = 0
    # Create a temporary OHLCV root for build_features to use later, 
    # since build_features expects a specific hierarchy data/ohlcv/{crypto|equities_etf}/{symbol}.parquet
    # We will simulate this by passing the actual ohlcv root if possible, or mocking it.
    # Actually build_features takes --ohlcv_root. We can point it to a temp dir where we symlink this file?
    # Or simplified: generic build_features looks for {root}/equities_etf/{sym}.parquet or {root}/crypto/{sym}.parquet.
    # We need to structure a temp ohlcv dir.
    temp_ohlcv_dir = out_dir / "temp_ohlcv"
    temp_ohlcv_sub = temp_ohlcv_dir / "equities_etf" # default to equities for simplicity or try to guess
    temp_ohlcv_sub.mkdir(parents=True, exist_ok=True)
    # Copy or symlink source parquet to the expected location
    # If the source is already in a structure, we might be able to use logic, but safer to just link it
    dest_parquet = temp_ohlcv_sub / f"{symbol}.parquet"
    if not dest_parquet.exists():
        shutil.copy(ohlcv_path, dest_parquet)
        
    for i in range(window_bars, len(df) + 1, stride_bars):
        win = df.iloc[i - window_bars : i]
        ts_end = win.index[-1]
        
        # Name format: {symbol}_{tf}_{start}_{end}_{seed}.png
        start_epoch = int(win.index[0].timestamp())
        end_epoch = int(win.index[-1].timestamp())
        
        fname = f"{symbol}_{timeframe_tag}_{start_epoch}_{end_epoch}_42.png"
        out_png = render_split_dir / fname
        meta_out = out_png.with_suffix(".json")
        
        # Render
        meta = render_window_png(win, out_png, meta_out, render_cfg, symbol, timeframe_tag, rng)
        
        # Standardize
        # standardize_one writes to out_dir / (img_path.stem + ".png")
        # We want it in std_split_dir
        standardize_one(out_png, std_split_dir, W=960, H=640, inset=0.02)
        
        # Track for manifest
        # keys needed by build_features manifest: "outputs": {"out_root": ...}, "runs": ...
        # Actually build_features uses samples_from_manifest which iterates glob patterns or specified files.
        # But samples_from_manifest reads meta json from standardized PNGs too? 
        # Wait, standardize_one doesn't propagate the original metadata JSON (ohlcv context) to the standardized folder? 
        # Standardize ONE creates a .transform.json. 
        # dataset.py: samples_from_manifest iterates `*.png` in split dir, and loads `png.with_suffix(".json")`.
        # So we MUST copy the metadata JSON from rendered to standardized folder!
        
        std_meta_src = meta_out
        std_meta_dst = std_split_dir / meta_out.name
        shutil.copy(std_meta_src, std_meta_dst)
        
        count += 1
        if count % 10 == 0:
            LOG.info(f"Processed {count} windows...")

    LOG.info(f"Total windows processed: {count}")
    
    if count == 0:
        LOG.warning("No windows generated. Exiting.")
        return

    # 3. Create Manifest
    manifest_data = {
        "outputs": {
            "out_root": str(std_root.absolute())
        }
    }
    manifest_path = out_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2))
    
    # 3b. Create Temp Pipeline Config (Keep CV features enabled, but we rely on safer implementation)
    cfg = load_yaml(pipeline_cfg_path)
    # Reverting disable logic to respect user request for CV
    
    temp_cfg_path = out_dir / "pipeline_infer.yaml"
    with open(temp_cfg_path, "w") as f:
        yaml.dump(cfg, f)
    with open(temp_cfg_path, "w") as f:
        yaml.dump(cfg, f)
    
    # 4. Build Features
    LOG.info("Building features...")
    # We use our temp ohlcv root
    feature_outputs = build_dataset(
        pipeline_cfg_path=temp_cfg_path,
        ohlcv_root=temp_ohlcv_dir,
        images_root=std_root,
        out_dir=out_dir,
        manifest_path=manifest_path
    )
    
    # Feature extraction completed

    
    # 5. Inference
    LOG.info("Running inference...")
    # Load features
    features_csv = feature_outputs.get("test")
    if not features_csv:
        # fallback
        features_csv = out_dir / "test_features.csv"
        
    if not features_csv.exists():
        LOG.error("Features CSV not found.")
        return
        
    df_feats = pd.read_csv(features_csv)
    
    results = []
    
    # Patterns to detect
    patterns = ["head_and_shoulders", "double_top", "double_bottom", "ascending_triangle"]
    # map to directory codes if different? 
    # reports/baselines/ has {hs, dt, db, tri}
    pat_map = {
        "head_and_shoulders": "hs",
        "double_top": "dt",
        "double_bottom": "db",
        "ascending_triangle": "tri"
    }
    
    prediction_cols = []
    
    for pat in patterns:
        code = pat_map[pat]
        model_dir = models_root / code
        model_path = model_dir / "model.joblib"
        
        target_col = f"y_{pat}"
        prob_col = f"p_{pat}"
        pred_col = f"y_pred_{pat}"
        
        if not model_path.exists():
            LOG.warning(f"Model for {pat} not found at {model_path}, skipping.")
            continue
            
        model, threshold, drop_structural, dropped_cols = load_model(model_path)
        
        # Prepare X
        # Filter columns
        # We need to replicate logic from train_rf: drop identifiers, target columns, and maybe structural
        # Identifiers: image, image_path, split, symbol, timeframe, start_ts, end_ts, bars
        # And any other y_* columns
        
        drop_cols_always = [
            "image", "image_path", "split", "symbol", "timeframe", "start_ts", "end_ts", "bars"
        ]
        # Also drop all y_* targets present in df
        y_cols = [c for c in df_feats.columns if c.startswith("y_")]
        
        # Structural features to drop?
        structural_feats = [
            "hs_neckline_slope_deg", "hs_head_to_shoulder_ratio", "hs_shoulder_similarity",
            "hs_temporal_symmetry", "hs_breakout_confirmed", "hs_span_bars",
            "dt_peak_similarity", "dt_valley_depth", "dt_span_bars",
            "db_trough_similarity", "db_peak_height", "db_span_bars",
            "tri_conv_rate", "tri_support_slope_deg",
            # add others if they exist in valid set
        ]
        
        # Configure X
        X = df_feats.drop(columns=drop_cols_always + y_cols, errors="ignore")
        if drop_structural:
            X = X.drop(columns=structural_feats, errors="ignore")
            
        # Also drop explicitly dropped cols from manifest
        if dropped_cols:
             X = X.drop(columns=dropped_cols, errors="ignore")
             
        # Align columns with model
        # model.n_features_in_ might match X.shape[1]
        # Ideally we check feature names if model supports it
        try:
             required_feats = model.feature_names_in_
             missing_cols = [c for c in required_feats if c not in X.columns]
             if missing_cols:
                 LOG.warning(f"Missing {len(missing_cols)} features for model {pat} (e.g. {missing_cols[:3]}), filling with 0.0")
                 for c in missing_cols:
                     X[c] = 0.0
             
             X = X[required_feats]
        except AttributeError:
            pass # older sklearn or pipeline
            
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        df_feats[prob_col] = probs
        df_feats[pred_col] = preds
        
        prediction_cols.append((pat, prob_col, pred_col, threshold))
        
    # 6. Final Decision Logic
    # 1. "None" if all y_pred == 0
    # 2. Single hit: return it
    # 3. Multi hit: highest (prob - thresh) margin? 
    
    final_labels = []
    final_confs = []
    multi_hits = []
    
    for _, row in df_feats.iterrows():
        hits = []
        best_pat = "none"
        best_margin = -1.0
        # If no hits, confidence = 1 - max(probs) ? Or something else.
        max_prob = 0.0
        
        for pat, prob_key, pred_key, thresh in prediction_cols:
            p = row[prob_key]
            max_prob = max(max_prob, p)
            if row[pred_key] == 1:
                margin = p - thresh
                hits.append((pat, margin, p))
                
        if not hits:
            final_labels.append("none")
            final_confs.append(1.0 - max_prob) # Confidence that it's NOT any of them
            multi_hits.append(False)
        elif len(hits) == 1:
            pat, margin, p = hits[0]
            final_labels.append(pat)
            final_confs.append(p)
            multi_hits.append(False)
        else:
            # Multi hit
            hits.sort(key=lambda x: x[1], reverse=True) # sort by margin
            winner = hits[0][0]
            winner_p = hits[0][2]
            final_labels.append(winner)
            final_confs.append(winner_p)
            multi_hits.append(True)
            
    df_feats["final_label"] = final_labels
    df_feats["final_confidence"] = final_confs
    df_feats["is_multi_hit"] = multi_hits
    
    # Save output
    out_csv = out_dir / "predictions.csv"
    output_cols = [
        "symbol", "timeframe", "start_ts", "end_ts", "bars",
        "final_label", "final_confidence", "is_multi_hit"
    ]
    # Add per-pattern columns
    for pat, prob_key, pred_key, _ in prediction_cols:
         output_cols.extend([prob_key, pred_key])
         
    df_feats[output_cols].to_csv(out_csv, index=False)
    LOG.info(f"Saved predictions to {out_csv}")
    
    # Cleanup temp files if requested (skip for now to aid debugging)
    if cleanup:
        shutil.rmtree(temp_ohlcv_dir, ignore_errors=True)
        # Maybe keep rendered/standardized for inspection? User manual step.

    return out_csv


def predict_image_process(
    image_path: Path,
    out_dir: Path,
    pipeline_cfg_path: Path,
    models_root: Path
):
    """
    Run inference on a single image file, bypassing OHLCV/Rendering/TA generation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info(f"Processing single image: {image_path}")
    
    # 1. Standardize Image (Resize/Pad)
    # We need to replicate standardization logic or assume input is somewhat ready.
    # Ideally reuse standardize_one but we don't have metadata.
    # We will just resize to standard dimensions used in training (likely 960x640 or similar from pipeline)
    # For now, we invoke CV extraction directly on the image as-is, relying on its internal resizing if needed?
    # Actually standardize_one creates the 960x640 letterboxed version. We should probably do that.
    
    std_root = out_dir / "standardized"
    std_root.mkdir(parents=True, exist_ok=True)
    
    # We use a dummy standardize implementation that just copies/resizes
    # But standardize_one requires 'ohlcv' logic? 
    # Let's import standardize_one from src? 
    # Problem: standardize_one takes a rendered image path.
    
    # Let's do a simple resize + pad logic here to be safe
    # Or just use the original extraction logic which resizes internally for HOG/Template?
    # cv_hough.py's extract_cv_features reads the image.
    
    # 2. Extract CV Features
    pipeline_cfg = load_yaml(pipeline_cfg_path)
    cv_cfg = pipeline_cfg.get("cv_features", {}).get("hough", {})
    
    # Check if we need to auto-disable for safety based on previous logic (we removed disable, but added density check)
    # The density check is inside extract_cv_features now.
    
    LOG.info("Extracting CV features...")
    # cv_hough expects a string or Path
    from src.features.cv_hough import extract_cv_features
    
    cv_feats = extract_cv_features(image_path, cv_cfg)
    
    # 3. Build Feature Row
    # We need a DataFrame compatible with the model (X)
    # We will start with CV features and fill zeros for everything else.
    
    # We need to know ALL possible columns to fill zeros correctly? 
    # Actually we just create a DF with CV cols, and let the model loading logic 
    # (which we added robustness to) fill the missing ones with 0.0.
    
    row = {}
    row.update(cv_feats)
    
    # Add dummy identifiers
    row["image"] = image_path.name
    row["symbol"] = "IMAGE_INPUT"
    row["timeframe"] = "UNK"
    row["split"] = "test"
    
    df_feats = pd.DataFrame([row])
    
    # 4. Inference
    LOG.info("Running inference on image features...")
    results = []
    patterns = ["head_and_shoulders", "double_top", "double_bottom", "ascending_triangle"]
    pat_map = {
        "head_and_shoulders": "hs",
        "double_top": "dt",
        "double_bottom": "db",
        "ascending_triangle": "tri"
    }
    
    prediction_cols = []
    
    for pat in patterns:
        code = pat_map[pat]
        model_dir = models_root / code
        model_path = model_dir / "model.joblib"
        
        prob_col = f"p_{pat}"
        pred_col = f"y_pred_{pat}"
        
        thresh_col = f"thresh_{pat}"
        
        if not model_path.exists():
            LOG.warning(f"Model for {pat} not found as {model_path}, skipping.")
            continue
            
        model, threshold, drop_structural, dropped_cols = load_model(model_path)
        
        # Prepare X
        # Re-use logic: drop non-features
        drop_cols_always = [
            "image", "image_path", "split", "symbol", "timeframe", "start_ts", "end_ts", "bars"
        ]
        
        # Also drop all y_* targets present in df
        y_cols_in_df = [c for c in df_feats.columns if c.startswith("y_")]
        X = df_feats.drop(columns=drop_cols_always + y_cols_in_df, errors="ignore")
        if drop_structural:
            structural_feats = [
                "hs_neckline_slope_deg", "hs_head_to_shoulder_ratio", "hs_shoulder_similarity",
                "hs_temporal_symmetry", "hs_breakout_confirmed", "hs_span_bars",
                "dt_peak_similarity", "dt_valley_depth", "dt_span_bars",
                "db_trough_similarity", "db_peak_height", "db_span_bars",
                "tri_conv_rate", "tri_support_slope_deg",
            ]
            X = X.drop(columns=structural_feats, errors="ignore")
            
        # Also drop explicitly dropped cols from manifest
        if dropped_cols:
             X = X.drop(columns=dropped_cols, errors="ignore")

        # Align columns (Fill zeros for missing TA features)
        try:
             required_feats = model.feature_names_in_
             missing_cols = [c for c in required_feats if c not in X.columns]
             if missing_cols:
                 LOG.debug(f"Filling {len(missing_cols)} non-CV features with 0.0 for model {pat}")
                 for c in missing_cols:
                     X[c] = 0.0
             
             X = X[required_feats]
        except AttributeError:
            pass # older sklearn
            
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        df_feats[prob_col] = probs
        df_feats[pred_col] = preds
        df_feats[thresh_col] = threshold
        
        prediction_cols.append((pat, prob_col, pred_col, threshold))
        
    # 5. Decision Logic (Same as before)
    final_labels = []
    final_confs = []
    multi_hits = []
    
    for _, r in df_feats.iterrows():
        hits = []
        max_prob = 0.0
        for pat, prob_key, pred_key, thresh in prediction_cols:
            p = r[prob_key]
            max_prob = max(max_prob, p)
            if r[pred_key] == 1:
                hits.append((pat, p - thresh, p))
                
        if not hits:
            final_labels.append("none")
            final_confs.append(1.0 - max_prob)
            multi_hits.append(False)
        elif len(hits) == 1:
            final_labels.append(hits[0][0])
            final_confs.append(hits[0][2])
            multi_hits.append(False)
        else:
            hits.sort(key=lambda x: x[1], reverse=True)
            final_labels.append(hits[0][0])
            final_confs.append(hits[0][2])
            multi_hits.append(True)
            
    df_feats["final_label"] = final_labels
    df_feats["final_confidence"] = final_confs
    df_feats["is_multi_hit"] = multi_hits
    
    # Save output
    out_csv = out_dir / "predictions.csv"
    
    # Simplified output cols
    cols_to_save = ["image", "final_label", "final_confidence"] + \
                   [c for c in df_feats.columns if c.startswith("p_") or c.startswith("y_pred_") or c.startswith("thresh_")]
                   
    df_feats[cols_to_save].to_csv(out_csv, index=False)
    LOG.info(f"Saved image prediction to {out_csv}")
    print(df_feats[cols_to_save].to_string()) # Print to stdout for immediate feedback
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Single-entrypoint inference for chart patterns.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ohlcv_path", type=Path, help="Path to OHLCV parquet file (Data+Vision mode)")
    group.add_argument("--image_path", type=Path, help="Path to Chart Image file (Vision-Only mode)")
    
    parser.add_argument("--pipeline_cfg", default="configs/pipeline.yaml", type=Path)
    parser.add_argument("--models_root", default="reports/baselines", type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--timeframe_tag", default="1h")
    parser.add_argument("--window_bars", type=int, default=160)
    parser.add_argument("--stride_bars", type=int, default=40)
    parser.add_argument("--start_ts", default=None)
    parser.add_argument("--end_ts", default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)
    
    if args.image_path:
        predict_image_process(
            image_path=args.image_path,
            out_dir=args.out_dir,
            pipeline_cfg_path=args.pipeline_cfg,
            models_root=args.models_root
        )
    else:
        predict_process(
            ohlcv_path=args.ohlcv_path,
            out_dir=args.out_dir,
            pipeline_cfg_path=args.pipeline_cfg,
            models_root=args.models_root,
            timeframe_tag=args.timeframe_tag,
            window_bars=args.window_bars,
            stride_bars=args.stride_bars,
            start_ts=args.start_ts,
            end_ts=args.end_ts
        )


if __name__ == "__main__":
    main()

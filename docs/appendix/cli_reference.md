## Core data & rendering
- `scripts/fetch_ohlcv.py --start YYYY-MM-DD --end YYYY-MM-DD --config configs/backtest.yaml`  
  Fetch Yahoo (equities/RTH) and Binance (crypto) 1h OHLCV into `data/ohlcv/<universe>/symbol.parquet`; clamps Yahoo intraday span to ~730 days; drops forming bar; writes manifest JSON.
- `scripts/make_images.py --backtest_cfg configs/backtest.yaml --render_cfg configs/render.yaml --window_bars 160 --stride_bars 40 --seed 42 --out_root data/images/rendered`  
  Render walk-forward windows to PNG + sidecars, split by end timestamp. Use `--windows_csv data/labels/hs_labeled_with_split.csv --windows_filter_col y_hs --windows_filter_value 1 --windows_split_col split` to render curated windows without walk-forward.
- `scripts/write_render_manifest.py --out_root data/images/rendered --backtest configs/backtest.yaml --render configs/render.yaml --patterns configs/patterns.yaml --window_bars 160 --stride_bars 40 --seed 42`  
  Record run metadata (hashes, counts, env versions) at `reports/runs/<run_id>/render_manifest.json`.
- `src/standardize/standardize_images.py --inp data/images/rendered --out data/images/standardized --width 960 --height 640 --inset 0.02`  
  Rectify any images (renders, screenshots, photos) to canonical canvas; emits transform JSON sidecars.

## Labeling & candidates
- `scripts/scan_hs_regimes.py --data-root data/ohlcv --symbols BTC-USD ETH-USD AAPL TSLA --timeframes 1h --window 100 --step 10 --hs-type both --out data/labels/hs_candidates.csv`  
  Regime pre-filter based on trend/volatility; outputs candidate windows.
- `scripts/label_hs_candidates.py --data-root data/ohlcv --candidates data/labels/hs_candidates.csv --out data/labels/hs_labeled.csv --config configs/patterns.yaml --swing-window 4 --swing-min-distance 4 --swing-prominence-atr 0.9 --swing-atr-period 14`  
  Runs deterministic H&S labeler on candidates; adds structural metrics + diagnostics; optional `--diag-out` for per-symbol summary.
- `scripts/validate_detectors.py` / `scripts/validate_renders.py` (not required in main flow) audit weak labels vs ZigZag TA baseline and render integrity respectively; see script docstrings for flags.
- `scripts/run_sanity_checks.py --pipeline_cfg configs/pipeline.yaml --ohlcv_root data/ohlcv --manifest reports/runs/<run_id>/render_manifest.json --limit 3`  
  Print swings, labels, structural features, and CV stats for a few samples.

## Features & CV vocab
- `scripts/build_features.py --pipeline_cfg configs/pipeline.yaml --manifest reports/runs/<run_id>/render_manifest.json --ohlcv_root data/ohlcv --images_root data/images/standardized --out_dir data/features`  
  Builds split CSVs with deterministic labels, TA, structural, CV (Hough/HOG/grad/contours, optional BoVW). Use `--labels_csv reports/labels/weak_labels.csv` if you must bypass manifest.
- `scripts/build_bovw_vocab.py --pipeline_cfg configs/pipeline.yaml --manifest reports/runs/<run_id>/render_manifest.json --images_root data/images/standardized --out_dir reports/cv_vocab`  
  Fits MiniBatchKMeans vocab from ORB descriptors; saves joblib + JSON manifest. Ensure `cv_features.bovw.n_clusters/max_keypoints/...` set in pipeline config.

## Modeling
- `scripts/train_rf.py --train_csv data/features/train_features.csv --val_csv data/features/val_features.csv --target y_double_top --out_dir reports/runs/rf_dt --pipeline_cfg configs/pipeline.yaml --n_estimators 800 --max_depth 16 --min_samples_leaf 3 --keep_structural`  
  Trains RandomForest (class_weight=balanced); reads RF defaults from pipeline unless overridden; drops structural features unless `--keep_structural`. If `--val_csv` missing, uses 5-fold GroupKFold by symbol.
- `scripts/eval_rf.py --test_csv data/features/test_features.csv --model_path reports/runs/rf_dt/model.joblib --target y_double_top --threshold 0.30 --out_dir reports/eval/dt`  
  Evaluates saved RF; threshold defaults to model manifest’s best_threshold if not provided.
- `scripts/grid_rf.py` / `scripts/grid_rf.py` (pattern-specific sections) illustrate tiny sweeps over `min_samples_leaf`, `max_depth`, `n_estimators`; adapt paths/targets as needed.
- `scripts/infer_rf.py` (thin helper) loads a model + CSV and writes probabilities; see in-script help for options.

Read this in 10 minutes
- Deterministic labels come from OHLCV geometry (ATR-scaled swings → H&S, double top/bottom, ascending triangles) and are mapped to pixels via render sidecars; no deep learning is used.
- Images are standardized (Canny → closing → quad → homography → inset → LAB L equalization) so downstream Hough/HOG/ORB features see a rectified chart.
- Features = TA + candle stats + swing summaries + rule-derived structure + classical CV (Hough lines, HOG/gradient histograms, contours, optional ORB+BoVW).
- Models = RandomForest with class_weight=balanced; structural features dropped by default to avoid learning the labeler.
- Baselines (test, non-BoVW): H&S ROC-AUC 0.916 / PR-AUC 0.670 / F1 0.615 @0.60; DT 0.841 / 0.889 / 0.822 @0.30; DB 0.820 / 0.843 / 0.807 @0.30; Asc Triangle 0.831 / 0.738 / 0.634 @0.35. BoVW is a modest lift for DT/DB/H&S, negative for triangles.

## Abstract
We build a classical-computer-vision chart pattern detector that never relies on neural nets. Weak labels are generated deterministically from OHLCV geometry (ATR/HLC3-aware swings, rule-based pattern tests, breakout confirmation), then mapped into pixel coordinates using render sidecars. Images are standardized through Canny edges, morphological closing, convex-quad detection, homography rectification, and LAB L-channel equalization before computing Hough-line and gradient descriptors; ORB+Bag-of-Visual-Words is optional. RandomForests trained on TA + CV features deliver reproducible baselines: test PR-AUC/F1 of 0.889/0.822 for Double Top, 0.843/0.807 for Double Bottom, 0.670/0.615 for Head & Shoulders, and 0.738/0.634 for Ascending Triangles, with small BoVW gains except on triangles. Manifests, sidecars, and YAML configs make the pipeline reproducible end to end.

## 1. Introduction
We aim to detect chart patterns in candlestick images using only deterministic labelers and classical CV. Motivation: finance practitioners value transparency and reproducibility, and standardized chart images allow CV features to capture geometry that raw OHLCV misses (e.g., visual trendlines, wick intersections). Scope: rule-based labeling, homography-based standardization, Hough/HOG/ORB descriptors, RandomForest models. Non-goals: any deep learning or YOLO-style detectors.

## 2. Background & Glossary (selected)
- Finance: OHLCV (open/high/low/close/volume); HLC3 = (H+L+C)/3; TRₜ = max(H−L, |H−Cₜ₋₁|, |L−Cₜ₋₁|); ATR = Wilder’s RMA of TR (period 14 by default). ATR expresses volatility in “typical move per bar,” so thresholds scale across assets.
- Patterns: Head & Shoulders (peak–valley–peak–valley–peak with neckline break), Double Top/Bottom (two similar extrema with a confirming break), Ascending Triangle (flat-ish resistance + rising support; upside break).
- CV terms: Canny edges (gradient magnitude + non-maximum suppression + dual thresholds); morphological closing (dilation→erosion to seal gaps); homography (3×3 projective map aligning a quadrilateral to a rectangle); Hough transform (votes in ρ–θ space to recover lines); HOG/gradient histograms (edge orientation distributions); ORB (FAST keypoints + oriented BRIEF descriptors); BoVW (KMeans codebook, histograms of visual words).
- Metrics: ROC-AUC, PR-AUC (better for rare positives), F1 at tuned threshold; GroupKFold by symbol when val split absent.
See `docs/appendix/glossary.md` for full definitions.

## 3. System Overview
Mermaid pipeline:
```mermaid
flowchart LR
    A[OHLCV Parquet (Yahoo/Binance)] --> B[Regime Scan (scan_hs_regimes)]
    B --> C[Deterministic Labelers (swing + geometry + breakout)]
    C --> D[Window Selection]
    D --> E[Rendering (make_images.py) + Sidecars]
    E --> F[Standardization (homography, LAB)]
    F --> G[CV Features (Hough/HOG/grad/contours/ORB BoVW)]
    C --> H[Structural TA Features]
    G --> I[Feature Builder (build_features.py)]
    H --> I
    I --> J[RandomForest Train/Eval]
    J --> K[Metrics/Manifests (reports/)]
```
Data sources: Yahoo Finance for equities/ETFs (1h, clamped to ~730-day intraday limit), Binance via CCXT for crypto (1h, UTC). Walk-forward splits from `configs/backtest.yaml`. Timezones normalized to UTC; NY equities optionally filtered to RTH. Reproducibility: render manifest (`scripts/write_render_manifest.py`) records hashes, counts, params; DVC tracks data/ohlcv and images.

## 4. Data & Rendering
- Fetching (`scripts/fetch_ohlcv.py`): Yahoo clamp to 720–730 days for 1h bars; Binance paginator for deep intraday; tz_localize/convert to UTC; drop forming bar; optional RTH filter for NY symbols.
- Rendering (`scripts/make_images.py`): 960×640 @120 dpi, candlesticks with jittered colors/widths/grid/fonts; HLC axes captured in sidecar JSON (`axes_xlim/ylim`, `axes_bbox_px`, `fig_size_px`, bar count, timestamps). Split assigned by walk-forward dates or via `--windows_csv` with per-row split.
- Style jitter: mild blur/noise, optional grid alpha. Wick/body widths randomly picked from config ranges; avoids leaking gridlines as labels.
- Known mplfinance quirks handled: wick_linewidth vs wick_width fallbacks; no bbox_inches tight; periodic plt.close to avoid “More than 20 figures”.

## 5. Image Standardization (theory → implementation)
Pipeline (`src/standardize/standardize_images.py`):
- Canny edges (50/150) find intensity gradients; non-maximum suppression and dual thresholds keep coherent edges.
- Morphological closing (5×5 kernel, 2 iters) seals small gaps so contours form a single chart frame.
- Largest convex 4-point contour (Douglas–Peucker at 2% perimeter) chosen as the chart rectangle; requires ≥10% of image area. If absent, Laplacian-energy mask picks the high-texture box (glare/axes fallback).
- Corner ordering (sum/diff heuristic) → homography: 3×3 projective transform warps the quadrilateral to a canonical 960×640 rectangle with 2% inset to trim axes.
- LAB equalization: histogram-equalize L channel only, normalizing illumination without altering bull/bear colors.
Why it matters: rectification aligns axes and candles, stabilizing Hough angles, HOG bins, and ORB keypoints across renders, screenshots, and phone photos. Failure modes: very faint frames (tune Canny/closing), extreme tilt (quad mis-order), or flat charts (Laplacian fallback box used).

## 6. Deterministic Labeling from OHLCV Geometry
- Swings (`src/labeling/swing_points.py`): symmetric window (default 4 bars), min_distance=4, prominence ≥0.9×ATR (period 14). ATR provides volatility normalization so peaks stand out equally on BTC and SPY.
- H&S (`src/labeling/patterns_hs.py`, config `labeling.patterns.head_and_shoulders`): sequence high–low–high–low–high; span 24–180 bars; head ≥0.45 ATR / 0.45% above shoulder mid; shoulder height sim ≤35%, timing sim ≥75%; neckline slope ≤18°; breakout side=down within 22 bars by 0.08 ATR / 5%. Structural outputs: neckline slope, head/shoulder ratio, symmetry, breakout flag, span.
- Ascending triangle (`patterns_triangles.py`): two similar peaks (≤1% diff), resistance slope ≤8°, support slope ≥10°, ≥2 lows, span 30–150 bars, breakout up within 75% of apex with ≥0.1 ATR or 5% headroom. Structural: support/resistance slopes, convergence ratio, breakout flag, span.
- Double Top/Bottom (`patterns_double.py`): two extrema with similarity ≤8%, span 24–110 bars, dynamic valley/peak depth using ATR and % thresholds; breakout confirmation (down for DT, up for DB) with ATR/% and volume gates; optional pre-breakout labels disabled. Structural: peak/trough similarity, valley/peak depth, span, breakout flag, score.
- Post-processing: deduplicate overlapping detections by time (`labels_post.dedup_time_overlap_bars`).
- Mapping to pixels: render sidecar (`axes_bbox_px`, `axes_xlim/ylim`, image size) translates swing/bar indices to coordinates when validating labels on images.

## 7. Classical CV Features (theory + project use)
- Hough lines (`src/features/cv_hough.py`): Probabilistic Hough over Canny edges (ρ=1px, θ=1° bins, threshold=70). Features count lines, mean/std θ (deg ∈ [−90,90)), accumulator max, angle fractions (horizontal ±10°, upward >10°, downward <−10°), mean pairwise angle diffs. Captures gridlines/trendlines and directional bias of price action post-rectification.
- HOG / gradient histograms: HOGDescriptor over 64×64 resize (mean/std/max); Sobel-based magnitude-weighted orientation histograms (9 bins, 0–180°). Encodes local edge texture (consolidation vs thrust) invariant to small lighting changes.
- Contours & entropy: external contour count/areas, edge density, edge entropy as texture/noise proxies (e.g., heavy overlays vs clean charts).
- Template correlation: 1D column-mean profile correlated with synthetic shoulder–head–shoulder pattern as a cheap H&S cue.
- ORB + BoVW (`src/features/cv_local.py`): FAST keypoints + oriented BRIEF descriptors; MiniBatchKMeans vocabulary (default K=64). Per-image L2-normalized histogram `cv_bovw_*` summarizes local corner/ intersection structures (wicks, MA crossings). Enabled via `cv_features.bovw.enabled`; vocab built with `scripts/build_bovw_vocab.py`.
- Design choice: pure classical CV—no CNNs—so features are interpretable, light, and stable across domains; homography ensures geometry comparability.

## 8. Feature Builder & Datasets
`scripts/build_features.py` → `src/features/dataset.py`:
- Enumerates samples from render manifest (preferred) or labels CSV; loads OHLCV (UTC) with fallback tz_localize; trims to expected bar count.
- Computes TA stats (returns, ATR, ROC, MA relativity), candle composition, swing summaries, pattern proxies, structural outputs from labeler, CV features (Hough/HOG/grad/contours/optional BoVW).
- Outputs split CSVs + all_features with identity columns and targets `y_head_and_shoulders`, `y_double_top`, `y_double_bottom`, `y_ascending_triangle`.
- Dataset stats (non-BoVW & BoVW identical): train 1012 rows (H&S 109, DT 551, DB 533, Tri 274); val 284 (31/135/166/97); test 170 (18/103/98/55); all 1466 (158/789/797/426). See `docs/assets/dataset_stats.md`.

## 9. Models, Training & Evaluation
- Model: RandomForestClassifier (scikit-learn), class_weight=balanced, n_estimators default 600, min_samples_leaf=2, max_depth=None unless overridden. Structural features (`hs_/dt_/db_/tri_`) dropped by default to avoid leakage; pass `--keep_structural` to retain.
- Training (`scripts/train_rf.py`): accepts train/val CSV; if val is omitted, uses 5-fold GroupKFold by symbol. Saves `model.joblib` + `train_manifest.json` with params, feature names, top permutation/Gini importances, best threshold (val sweep 0.05–0.95).
- Evaluation (`scripts/eval_rf.py`): loads model + best threshold (or `--threshold`), writes metrics JSON and confusion matrix CSV with timestamp.
- Grid search example (`scripts/grid_rf.py`) demonstrates small sweeps; baselines recorded in `reports/baselines/*`.

## 10. Experiments & Results (current repo)
Baselines (drop structural features):
- Head & Shoulders: test ROC-AUC 0.916, PR-AUC 0.670, F1 0.615 @0.60 (`reports/baselines/hs/*`).
- Double Top: 0.841 / 0.889 / 0.822 @0.30 (`reports/baselines/dt/*`).
- Double Bottom: 0.820 / 0.843 / 0.807 @0.30 (`reports/baselines/db/*`).
- Ascending Triangle: 0.831 / 0.738 / 0.634 @0.35 (`reports/baselines/tri/*`).
BoVW comparisons:
- H&S: small PR-AUC gain to ~0.681 with similar F1 (~0.615); ROC ~0.909. Modest, optional.
- Double Top: PR-AUC ~0.898, ROC ~0.845–0.849, F1 ~0.81–0.83 → slight improvement.
- Double Bottom: PR-AUC ~0.854, ROC ~0.833–0.835, F1 ~0.81 → slight improvement.
- Ascending Triangle: BoVW hurts (PR-AUC ~0.66 vs 0.738); keep non-BoVW.
Domain shift: features trained on rendered charts remain stable on standardized screenshots/photos thanks to homography + LAB equalization; expect degradation if frames are missing or glare saturates edges—standardize first.
See `docs/assets/results_summary.md` for compact tables.

## 11. Troubleshooting (what we hit & fixed)
- Yahoo 1h limit: clamp start date when span > ~730 days; stitch with CCXT for crypto.
- Timezones: enforce tz_localize vs tz_convert to UTC; RTH filter for NY equities.
- mplfinance kwargs: fallback between `wick_linewidth`/`wick_width`; avoid bbox_inches tight; close figs to prevent leaks.
- Rendering artifacts: grid alpha randomization prevents gridlines leaking into labels; disable volume overlays to keep frame detection clean.
- Detector drift: if positives vanish, relax head prominence/confirmation windows in `configs/pipeline.yaml`; if flooding, tighten geometry or breakout gates.
- BoVW sparsity: low-contrast renders can yield few ORB keypoints; increase max_keypoints or skip BoVW for triangles.

## 12. Limitations & Future Work
- Label noise vs coverage: deterministic rules can miss valid but atypical patterns; per-symbol overrides help but are manual.
- Ambiguity near confirmations: current binary labels drop abstention except for WIP LFs; adding uncertainty bands could help.
- Frame finding: Hough-based frame finder could replace contour fallback for extreme photos.
- Axes/OCR: adding axis OCR would validate price-to-pixel mapping on real screenshots.
- DL baselines deliberately excluded; could be future optional comparison.

## 13. Reproducibility Guide (CLI)
1) Fetch data: `uv run python scripts/fetch_ohlcv.py --start 2023-01-01 --end 2025-10-31`.
2) Render charts: `uv run python scripts/make_images.py --backtest_cfg configs/backtest.yaml --render_cfg configs/render.yaml --window_bars 160 --stride_bars 40 --seed 42 --out_root data/images/rendered`.
3) Standardize: `uv run python src/standardize/standardize_images.py --inp data/images/rendered --out data/images/standardized`.
4) Manifest: `uv run python scripts/write_render_manifest.py --out_root data/images/rendered --backtest configs/backtest.yaml --render configs/render.yaml --patterns configs/patterns.yaml --window_bars 160 --stride_bars 40 --seed 42`.
5) Features + labels: `uv run python scripts/build_features.py --pipeline_cfg configs/pipeline.yaml --manifest reports/runs/<run_id>/render_manifest.json --ohlcv_root data/ohlcv --images_root data/images/standardized --out_dir data/features`.
6) Train RF: `uv run python scripts/train_rf.py --train_csv data/features/train_features.csv --val_csv data/features/val_features.csv --target y_double_top --out_dir reports/runs/rf_dt --pipeline_cfg configs/pipeline.yaml`.
7) Evaluate: `uv run python scripts/eval_rf.py --test_csv data/features/test_features.csv --model_path reports/runs/rf_dt/model.joblib --target y_double_top --out_dir reports/eval/dt`.
Artifacts: renders/sidecars under `data/images/rendered`; standardized PNGs under `data/images/standardized`; manifests under `reports/runs/<run_id>/`; features under `data/features`; models/manifests under `reports/runs/*` or `reports/baselines/*`; metrics/confusion matrices under `reports/eval/*`.

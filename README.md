## WeaK labeling on the rendered set 

We use the data we already have (the parquet OHLCV) to detect patterns of the price series (not the pixels) then convert the detected pattern’s geometry to pixel coordinates using each image’s sidecar metadata. That gives us labels without manual annotation 


## How to run the standardization script for any type of image

### Screeenshots

uv run python src/standardize/standardize_images.py \
  --inp data/images/real/screenshots \
  --out data/images/standardized/screenshots

### Phone Photos

uv run python src/standardize/standardize_images.py \
  --inp data/images/real/photos \
  --out data/images/standardized/photos


## Rendered Images

uv run python src/standardize/standardize_images.py \
  --inp data/images/rendered/val \
  --out data/images/standardized/rendered_val


## Glosary

### ATR = average true range
It’s a volatility measure: “how much price typically moves per bar.” Using ATR lets us express thresholds in volatility units so the same rule scales between quiet markets (SPY) and wild ones (BTC).

### true Rnage (TR)
For bar t with High Hₜ, Low Lₜ, Close Cₜ and previous close Cₜ₋₁:

TRₜ = max(
  Hₜ - Lₜ,
  |Hₜ - Cₜ₋₁|,
  |Lₜ - Cₜ₋₁|
)
This captures gaps: if price gaps up or down, TR reflects that, not just high–low.

#### ATR computation 

ATR is a moving average of TR. In patterns.yaml we set:

  - period: 14 → average over the last 14 bars

  - smoothing: "rma" → RMA (a.k.a. Wilder’s smoothing), commonly used in TA:

ATRₜ = ATRₜ₋₁ + (TRₜ - ATRₜ₋₁)/period

#### Why we use ATR in rules

  - “Peak must stick out by ≥ 0.7×ATR” scales to volatility.

  - “Breakout must exceed neckline by 0.35×ATR” means we need a move big enough to matter for this market.

If we used raw points or percentages everywhere, the same thresholds would be too strict for some assets and too loose for others.

### HLC3

HLC3 = (High + Low + Close) / 3.

It’s a “typical price” per bar. Using HLC3 to compute indicators (like ATR or moving averages) reduces noise vs just Close:

  - If a bar has a big wick but closes back near the open, close-only indicators may miss that volatility.

  - HLC3 averages intra-bar information a tiny bit without requiring tick data.

So when we calculate ATR (or any indicator that uses a “price series”), we first transform OHLCV into a single “price per bar” = HLC3. (If you prefer, you can set "close" or "hl2" = (High+Low)/2.)

## Heuristic Labeling Journey (Weak Labels)

We tightened the scripted labeler step by step to make it usable for CV training while staying close to TA intuition:

- **Initial flood:** Double-top was ~90% positive; H&S/IHS/triangles were mostly zero. Causes: loose geometry, no per-symbol overrides, no breakout gating.
- **Geometry/vol filters:** Raised pivot prominence, valley/peak depth, added ATR/percent hybrids, per-symbol overrides (equities vs crypto), dynamic thresholds, deduplication, and separated raw `*_cnt` from `*_confirmed_cnt`.
- **ATR/volume/trend guards:** Added pattern-height floors, volume ordering on shoulders/head, breakout volume spikes, and prior-trend checks. This reduced noise but temporarily over-pruned H&S/IHS.
- **Breakout gating for H&S/IHS:** Now require a neckline break to emit; height floors and volume ordering (0.95 ratios) remain. Without confirmation H&S over-fired; with very strict gates it vanished—timing/height/volume/breakout must be balanced.
- **Per-symbol confirmation tuning:** Equities use short confirmation windows and softer breakout volume; BTC/ETH use stricter volume. DT/DB geometry tightened (height_pct up) to reduce over-firing; short confirm windows keep signals timely.
- **Ensembles/abstention (in progress):** We started emitting per-pattern probabilities and LF vote counts (DT/DB) and allowing abstain (`y=-1`) when votes are ambiguous. Next steps: expand LFs per pattern and combine multiple signals instead of a single hard decision.
- **Alignment checks:** `scripts/validate_detectors.py` compares our detections to a TA ZigZag baseline and emits CSV + JSON (`reports/validation/detector_audit.*`) with only-ours/only-TA counts per pattern/symbol. Use agreement rates and confirm rates each run to spot over/under-fire and guide tuning.

When adjusting:
- Geometry: height_pct, min_height_atr, symmetry.
- Confirmation: percent/ATR break, within_bars, volume multiplier (per symbol).
- If a class floods, tighten geometry/confirmation; if it vanishes, ease timing/height or allow pre-breakout flags while training on confirmed.
- For ensembles: tune LF weights, thresholds for p (e.g., p ≥ 0.9 → positive, p ≤ 0.1 → negative, else abstain), and monitor LF agreement in the validator report.

## Deterministic Labeling & Feature Pipeline (2025 refresh)

We refactored the weak-label heuristics into a modular, reproducible pipeline that works directly from rendered-image metadata + OHLCV windows. **`scripts/generate_weak_labels.py` is now deprecated** (kept only for historical reference); please use the new modules below.

### New modules & responsibilities
- `src/labeling/swing_points.py`: ATR-aware swing-high/low detector with configurable windows, prominence, and ATR periods.
- `src/labeling/patterns_hs.py`: Head & Shoulders detector that enforces peak/valley ordering, ATR-normalised head prominence, neckline slope caps, and breakout confirmation.
- `src/labeling/patterns_triangles.py`: Ascending-triangle detector that fits resistance/support lines via swings, enforces slope limits, computes apex/convergence, and validates breakouts.
- `src/labeling/labeler.py`: High-level orchestrator that runs swing detection once per window and emits labels + structural features for each supported pattern.
- `src/features/cv_hough.py`: Classical CV block (Canny → Hough) that summarises line counts, mean/std angles, and directional fractions for every standardised chart image.
- `src/features/dataset.py`: Dataset builder that enumerates samples (via render manifest or labels CSV), slices the OHLCV windows, calls the labeler, computes TA/CV feature blocks, and writes per-split CSVs.
- `configs/pipeline.yaml`: Central config describing swing/pattern parameters, feature toggles, CV params, and RandomForest defaults.
- `src/models/train_rf.py` / `src/models/eval_rf.py`: Reusable training/eval utilities shared by the CLI scripts (`scripts/train_rf.py`, `scripts/eval_rf.py`).
- `scripts/run_sanity_checks.py`: Smoke-test helper that takes a few samples, prints swing counts, labels, structural features, and CV vectors for quick inspection.

### End-to-end usage
1. **(Optional) Sanity check a few samples**
   ```bash
   uv run python scripts/run_sanity_checks.py \
     --pipeline_cfg configs/pipeline.yaml \
     --labels_csv reports/labels/weak_labels.csv \
     --images_root data/images/rendered \
     --ohlcv_root data/ohlcv \
     --limit 3
   ```
   Use `--manifest path/to/render_manifest.json` instead of `--labels_csv` to iterate over a specific render run.

2. **Build deterministic labels + features**
   ```bash
   # labels CSV route (only uses metadata columns)
   uv run python scripts/build_features.py \
     --pipeline_cfg configs/pipeline.yaml \
     --labels_csv reports/labels/weak_labels.csv \
     --images_root data/images/rendered \
     --ohlcv_root data/ohlcv \
     --out_dir data/features

   # OR manifest route (preferred for new renders)
   uv run python scripts/build_features.py \
     --pipeline_cfg configs/pipeline.yaml \
     --manifest reports/runs/<run_id>/render_manifest.json \
     --images_root data/images/rendered \
     --ohlcv_root data/ohlcv \
     --out_dir data/features
   ```
   Output: `{split}_features.csv` plus `all_features.csv` containing TA, CV, and structural pattern stats alongside deterministic labels (`y_head_and_shoulders`, `y_ascending_triangle`, etc.).

3. **Train RandomForest (per pattern target)**
   ```bash
   uv run python scripts/train_rf.py \
     --train_csv data/features/train_features.csv \
     --val_csv data/features/val_features.csv \
     --target y_head_and_shoulders \
     --pipeline_cfg configs/pipeline.yaml \
     --out_dir reports/runs/exp_rf_hs
   ```
   - Omitting `--val_csv` triggers 5-fold GroupKFold (grouped by symbol).
   - Override hyperparameters inline (`--n_estimators 800`, etc.) or edit `configs/pipeline.yaml -> models.random_forest`.

4. **Evaluate on held-out split**
   ```bash
   uv run python scripts/eval_rf.py \
     --test_csv data/features/test_features.csv \
     --model_path reports/runs/exp_rf_hs/model.joblib \
     --target y_head_and_shoulders \
     --out_dir reports/eval/hs
   ```
   Metrics JSON + confusion matrix CSV are timestamped inside `reports/eval/hs/`.

### Migration notes
- The pipeline only depends on standardised renders + OHLCV parquet files; no YOLO/YOLOv5 artifacts or manual annotations are needed.
- Legacy weak-label probabilities/votes are ignored; all labels now come from deterministic pattern logic documented above.
- To add new patterns (e.g., inverse H&S, descending triangles), create a sibling module in `src/labeling/`, plug it into `PatternLabeler`, and extend `configs/pipeline.yaml`.
- For CV experimentation (different bins, additional stats), update `src/features/cv_hough.py` and expose new feature flags in the pipeline config.

### First sanity-check snapshot
Running `scripts/run_sanity_checks.py` on the latest render manifest is the quickest way to validate configs. Example:

```
uv run python scripts/run_sanity_checks.py \
  --pipeline_cfg configs/pipeline.yaml \
  --manifest reports/runs/20251111-1113_seed42/render_manifest.json \
  --images_root data/images/rendered \
  --ohlcv_root data/ohlcv \
  --limit 3
```

Sample output (AAPL, 1h bars):
```
================================================================================
AAPL_1h_1700058600_1703003400_42.png | split=train symbol=AAPL bars=160
Labels: {'y_head_and_shoulders': 0, 'y_ascending_triangle': 0}
Structural features: {'hs_neckline_slope_deg': 0.0, 'hs_head_to_shoulder_ratio': 0.0, ...}
Swing highs=15 lows=14
CV features: {'cv_has_lines': 1.0, 'cv_num_lines': 493.0, 'cv_theta_mean': 17.18, ...}
```

Interpretation:
- `y_* = 0` with zeroed structural metrics means no candidate satisfied the ATR/symmetry/breakout rules—expected for neutral trends.
- Swing counts (~15 highs/lows per 160-bar window) show the detector is catching reasonable pivots.
- CV stats confirm the image edge extractor is working (hundreds of lines, balanced angle distribution). Watch `cv_frac_horizontal / upward / downward` to ensure the renderer produces clean, rectified charts (extreme skews hint at preprocessing issues).

As you tune configs, rerun the script to spot windows that flip positive and inspect their structural attributes (e.g., non-zero `hs_head_to_shoulder_ratio`, `tri_support_slope_deg`). This gives a fast feedback loop before rebuilding the full dataset.

## How deterministic label building works (plain-language guide)
When you run:
```
uv run python scripts/build_features.py \
  --pipeline_cfg configs/pipeline.yaml \
  --manifest reports/runs/<run_id>/render_manifest.json \
  --images_root data/images/rendered \
  --ohlcv_root data/ohlcv \
  --out_dir data/features
```
the script does the following for every rendered chart:

1. **Look up the window**  
   The manifest tells us which image belongs to which symbol, time range, and split. We read the matching OHLCV candles (Open, High, Low, Close, Volume) from the parquet files so we have the exact price series that created the image.

2. **Find “swing” points**  
   Using `src/labeling/swing_points.py`, we scan for local highs and lows that stand out by at least a configurable number of ATRs (Average True Range = “typical price movement per bar”). ATR keeps the detector consistent across quiet vs volatile markets.

3. **Apply geometric rules**  
   - `patterns_hs.py` checks whether any 5-point sequence of swings matches the Head & Shoulders blueprint (left shoulder – head – right shoulder) and confirms that price later breaks the neckline.  
   - `patterns_triangles.py` fits resistance/support lines through swing highs/lows and validates ascending triangles plus their breakout.  
   Each rule outputs:
     * A binary label (`y_head_and_shoulders`, `y_ascending_triangle`) saying whether that pattern exists in the window.
     * Structural measurements (neckline slope, head-to-shoulder ratio, triangle support slope, convergence ratio, etc.) that describe how strong/clean the geometry is.

4. **Compute classical statistics (“features”)**  
   We add interpretable columns so that later models can learn correlations:
   - **Meta columns:** `image`, `split`, `symbol`, `start_ts`, `bars`, etc.  
   - **TA/return stats:** `ret_mean`, `ret_std`, `abs_ret_mean`, `atr_mean`, `roc_*` (rate of change over 5/10/20 bars), `rel_close_ma*` (position vs moving averages). These just capture how price has been moving.  
   - **Candle-shape stats:** `body_pct_mean`, `bull_frac`, `bear_frac`, etc. explain the mix of bullish/bearish candles.  
   - **Swing summary:** counts of swing highs/lows and spacing/variation metrics (`n_piv_hi`, `piv_hi_spacing_mean`, …) so we roughly know how choppy the period was.  
   - **Pattern proxies:** quick heuristics like `dt_peak_similarity` or `hs_head_rel`; they’re simple signals that don’t enforce the full rules but help the model.  
   - **CV (computer vision) stats:** `cv_has_lines`, `cv_num_lines`, `cv_theta_mean`, `cv_frac_horizontal`, etc. come from Canny edge detection + Hough transform on the standardized PNG; they summarize the dominant line directions/strengths in the chart image.  
   - **Structural features:** direct output from the rule-based labeler (`hs_head_to_shoulder_ratio`, `hs_span_bars`, `tri_support_slope_deg`, …). Even if the final label is 0, these features stay informative (e.g., a near-miss might have a non-zero head-to-shoulder ratio).

5. **Write split CSVs**  
   The script saves `{split}_features.csv` plus `all_features.csv`. Each row now has:

| Column category         | Examples                                           | Plain meaning                                                                 |
|-------------------------|---------------------------------------------------|--------------------------------------------------------------------------------|
| Identity                | `image`, `split`, `symbol`, `start_ts`, `bars`     | Which chart window we’re talking about.                                        |
| Returns & volatility    | `ret_mean`, `ret_std`, `atr_mean`, `roc_5`         | How much price moved, on average. ATR is “typical move per bar.”                |
| Candle composition      | `bull_frac`, `bear_frac`, `body_pct_mean`          | Share of bullish vs bearish candles and their body/wick sizes.                 |
| Swing metrics           | `n_piv_hi`, `piv_lo_spacing_mean`, `piv_hi_val_std`| How many local highs/lows and how regularly they appear.                       |
| Pattern proxies         | `dt_peak_similarity`, `hs_head_rel`, `tri_conv_rate`| Quick indicators for double tops, head prominence, triangle squeeze speed.    |
| CV / image stats        | `cv_num_lines`, `cv_theta_mean`, `cv_frac_upward`  | Counts/orientation of lines detected in the rendered PNG (rough shape cues).  |
| Structural measurements | `hs_head_to_shoulder_ratio`, `tri_support_slope_deg`, `hs_breakout_confirmed` | Measurements directly from the deterministic labeler; 0 when no candidate passes. |
| Targets                 | `y_head_and_shoulders`, `y_ascending_triangle`     | 1 if the window satisfies the rule-based definition, else 0.                   |

This process is deterministic: the same manifest + config will always yield the same labels/features. If you need more positives (e.g., only 2 H&S positives in `train`), tweak `configs/pipeline.yaml -> labeling` or render windows focusing on periods where those patterns occur more frequently before rerunning the pipeline.

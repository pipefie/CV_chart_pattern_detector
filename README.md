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


## Head & Shoulders Labeling Pipeline (Regime Scan → Heuristic Labeler → Rendering)

This project deliberately avoids Deep Learning. All pattern detection is based on:

  1. Heuristic rules on OHLCV (swings, geometry, breakouts), and

  2. Classical Computer Vision on rendered charts (edges, Hough lines, etc.).

Because of that, the quality of our labels is absolutely critical: if the labeler is weak or noisy, every model we build on top of it will be garbage.

Originally, the labeling logic was embedded directly in the rendering / image pipeline and/or in weak supervision rules. That led to several problems:

  - Labels were often collapsed (almost all 0 or almost all 1 for some patterns).

  - It was hard to debug why a specific window was considered a pattern or not.

  - The same heuristics were being re-implemented in different places.

To fix this, we introduced a two-stage pipeline for Head & Shoulders (H&S):

  1. Regime scan on OHLCV → “where could an H&S even exist?”

  2. Heuristic H&S labeler → “does this specific window satisfy the H&S geometry?”

Only after those two steps do we render images and do any CV-based feature extraction.

### 1. OHLCV storage

All raw price data lives as Parquet files under data/ohlcv:

data/ohlcv/
  crypto/
    BTC-USD.parquet
    ETH-USD.parquet
  equities_etf/
    AAPL.parquet
    AMZN.parquet
    SPY.parquet
    QQQ.parquet
    TSLA.parquet
    NVDA.parquet


Each Parquet file contains 1-hour bars (for now) with:

  - a DatetimeIndex (timestamps),

  - columns: open, high, low, close, volume.

This is the single source of truth for price data. All later steps slice these Parquets by row index.

### 2. Regime scanner: scripts/scan_hs_regimes.py

Goal: do not try to detect patterns on the whole history. Instead, first find windows where the market regime is compatible with an H&S.

H&S is a reversal pattern:

  - It needs a prior trend (for classic H&S, a prior uptrend).

  - It needs enough volatility to be visually recognizable.

scan_hs_regimes.py:

  - Loads OHLCV per symbol from Parquet.

  - Slides a window of fixed length (e.g. 100 bars) over the series:
    [start_idx, end_idx) with step bars between windows.

  - For each window, it computes:

    - a simple trend proxy (slope of log-prices in the first half of the window),

    - a simple volatility proxy (std of log returns in the whole window).

If both trend and volatility are above thresholds, we flag this window as “regime favorable for H&S”. This does not mean there is a pattern there; it just means it’s worth running the real H&S labeler.

The output is:

data/labels/hs_candidates.csv


with columns like:

  - symbol

  - timeframe (currently "1h" as a tag)

  - start_idx, end_idx (row indices into the Parquet)

  - start_ts, end_ts (timestamps for logging / sanity)

  - window (bars per window)

  - hs_type (top / inverse / both)

  - regime_favorable_for_hs (bool flag)

This step spreads our compute budget intelligently: instead of scanning millions of bars blindly, we focus H&S detection on a few thousand candidate windows that already have the right “energy” (trend + vol).

### 3. Geometric H&S labeler: src/labeling/patterns_hs.py + scripts/label_hs_candidates.py

Once we have hs_candidates.csv, we still need to answer:

  “In this specific window, is there actually a Head & Shoulders according to our rules?”

The H&S logic itself lives in:

  src/labeling/patterns_hs.py

with two key functions:

  - detect_hs_pattern(df, swings, config)
    Scans swing points looking for sequences of 5 events:

    high → low → high → low → high


    that fit the H&S structure (P1–V1–P2–V2–P3) and respect the YAML-defined geometry.

  - label_hs_window(df, swings, config)
    Wraps detect_hs_pattern:

      - returns a binary label y (0/1),

      - plus a dictionary of structural features such as:

        neckline slope in degrees,

        head-to-shoulder height ratio,

        shoulder similarity,

        temporal symmetry between shoulders,

        whether a breakout was confirmed and within how many bars,

        total span in bars.

The configuration (config) is not hard-coded. It is loaded from the YAML config (e.g. configs/patterns.yaml):

min_bars / max_bars for the pattern span,

max allowed neckline slope,

max allowed difference between shoulder heights,

min required head prominence above the shoulders (in ATR units and/or %),

breakout rules (direction, within N bars, threshold in ATR / %).

The new script:

scripts/label_hs_candidates.py

does the wiring:

Reads hs_candidates.csv.

For each row:

Loads the full OHLCV for that symbol from Parquet.

Extracts the window [start_idx:end_idx] into a small DataFrame.

Computes swing points for that window (using the project’s swing logic, not a random ad-hoc detector).

Calls label_hs_window(window_df, swings, config_from_yaml_for_HS).

Writes a new CSV:

data/labels/hs_labeled.csv


containing:

all original candidate fields, plus

y_hs (0/1),

hs_neckline_slope_deg,

hs_head_to_shoulder_ratio,

hs_shoulder_similarity,

hs_temporal_symmetry,

hs_breakout_confirmed,

hs_span_bars.

This hs_labeled.csv is the canonical ground truth for H&S used later by the rendering + CV + ML pipeline.

4. Why we’re doing it this way (and what we’re struggling with)

We’re doing this because earlier attempts at labeling patterns had serious issues:

The labelers were tied to specific scripts or rendering flows, so they were hard to reuse.

Some weak supervision heuristics produced degenerate labels (e.g. almost all ones or almost all zeros).

Debugging “why this window was labeled 1/0” was painful.

This new pipeline separates concerns clearly:

Regime scan (cheap, broad, approximate) → “where might patterns live?”

Geometric labeler (expensive, strict, config-driven) → “is this really an H&S?”

The main things we’re still struggling / iterating on:

Swing detection vs. strict geometry
If swing detection is too crude or too sparse, or if the YAML thresholds are too strict, it’s easy to end up with no detections at all (all y_hs = 0 even in promising windows).
This is a sign that we need to:

refine swing points (maybe using swing_points.py more carefully), and/or

relax some H&S thresholds in the YAML (neckline slope, shoulder similarity, head prominence) until we have a reasonable number of positive labels.

Balance between purity and dataset size
For the ML part (Random Forest on geometric + CV features) we need:

enough positive H&S examples,

plus a balanced set of negative windows from similar regimes.
Being too strict yields a “pure but empty” dataset; being too lax yields noisy labels. We’re iterating to find a middle ground that gives:

non-trivial number of H&S labels, and

reasonably interpretable patterns.

Aligning configuration
The H&S heuristics and thresholds live in a central YAML.
All scripts (scanner, labeler, renderer, feature builder) must read from this config instead of hard-coding magic numbers. A lot of the current work is about wiring everything to the same YAML so the behavior is consistent and changes are traceable.

5. How this feeds into the rest of the project

Once hs_labeled.csv is in a good place (non-empty, config-driven, debugged), the next steps are:

Select windows to render:

all positives (y_hs == 1),

a controlled sample of negatives (y_hs == 0) per symbol/timeframe.

Render charts for those windows using the existing rendering pipeline (candles, fixed DPI, consistent styling) and track them in DVC.

Standardize images with src/standardize/standardize_images.py:

Canny → edges,

morphology → clean shapes,

Hough transform → detect dominant lines.

Build features:

CV features from Hough (line counts, angle distribution, strength of dominant lines),

plus structural features from the labeler (neckline slope, head/shoulder ratio, etc.),

plus optional indicators (RSI, volume stats).

Train & evaluate models (e.g. Random Forest) on these features to predict y_hs.

The entire point of this pipeline is to have a transparent, reproducible, explainable path from:

raw OHLCV → regime-filtered windows → heuristic H&S labels → rendered images → CV features → classical ML.

No deep learning, no black boxes: every step is inspectable and driven by the same configuration file.

## 2025-11 H&S candidate tuning recap

Recent work focused on turning the `scan_hs_regimes.py → label_hs_candidates.py` pipeline into a practical way to harvest clean Head & Shoulders labels:

1. **Why two scripts?**
   - `scan_hs_regimes.py` carves out windows that already exhibit trend + volatility, cutting the search space from millions of bars to ~1k promising slices.
   - `label_hs_candidates.py` reuses the canonical swing detector and YAML heuristics so the same rulebook drives both this intermediate labeler and the final dataset builder. It also caches OHLCV per symbol and emits structural features + diagnostics.

2. **What was broken?**
   - Early runs yielded 0 positives because we were feeding the H&S detector naive swings and `config=None`. Crypto overrides were also out-of-sync, demanding ≥1 ATR of head prominence.
   - Lack of diagnostics made it impossible to tell *why* a candidate failed (span too short? shoulders unequal? breakout missing?).

3. **Fixes that brought it back to life:**
   - Wired the labeler to `src/labeling/swing_points.py` and `configs/patterns.yaml`, including per-symbol overrides.
   - Added instrumentation (`hs_diag_*` columns) and a summary CSV so every rejection reason is traceable.
   - Relaxed duration and prominence thresholds in line with classic TA (min span 28 bars ≈ 1.2 trading days on 1h charts, head prominence ≈ 0.5–0.6 ATR depending on asset).

4. **Where we landed:**
   - `data/labels/hs_labeled.csv` now contains 24 positives across AMZN, NVDA, ETH, QQQ, and TSLA out of 1,239 regime-filtered windows.
   - `data/labels/hs_labeled_diagnostics.csv` captures per-symbol totals (candidate sequences, rejection counts, detections, breakout confirmations) so future tweaks remain data-driven.
   - These CSVs feed directly into the render/standardize/feature pipeline described above; you can filter on `y_hs=1` to enumerate the exact windows to render or standardize.

Anyone—regardless of finance background—can follow this flow: find regimes with energy, apply the same geometric rules everywhere, log why each window passed or failed, and only then render and featurize the winners.

## Latest deterministic build (Nov 2025)
We folded the legacy double-top/bottom detectors into the deterministic pipeline, relaxed H&S heuristics to recover coverage, and aligned rendering/standardization with the manifest-driven feature builder.

What changed:
- Added `double_top` / `double_bottom` detectors to `src/labeling/` and `PatternLabeler`; `configs/pipeline.yaml` now carries their geometry/breakout rules, so `build_features.py` emits `y_double_top` / `y_double_bottom` alongside H&S and triangles.
- Relaxed H&S in `configs/pipeline.yaml` (min span 24 bars, head prominence ≈0.45 ATR / 0.0045 pct, breakout 0.08 ATR / 22 bars) to surface more positives while staying TA-plausible. The deterministic build now has ~150 H&S positives spread across train/val/test.
- Made `scripts/make_images.py` accept `--windows_csv` so you can render specific windows (e.g., `hs_labeled_with_split.csv` with `y_hs=1`) into the appropriate split folders; existing PNGs are preserved.
- Manifests remain metadata only: `scripts/write_render_manifest.py` records hashes/params and points to the render root; it does not render or delete images.

How to go from labels to features (step-by-step):
1. **Render (or append) windows**  
   - To render specific positives with splits (e.g., H&S):  
     ```bash
     uv run python scripts/make_images.py \
       --windows_csv data/labels/hs_labeled_with_split.csv \
       --windows_filter_col y_hs --windows_filter_value 1 \
       --windows_split_col split \
       --out_root data/images/rendered
     ```
   - Otherwise, render the full walk-forward grid via `scripts/make_images.py` with `--backtest_cfg/--render_cfg`.

2. **Standardize images** (required before feature extraction):  
   ```bash
   uv run python src/standardize/standardize_images.py \
     --inp data/images/rendered --out data/images/standardized
   ```
   (You can target per-split folders if preferred.)

3. **Write a manifest** pointing at the render root:  
   ```bash
   uv run python scripts/write_render_manifest.py \
     --out_root data/images/rendered \
     --backtest configs/backtest.yaml \
     --render configs/render.yaml \
     --patterns configs/patterns.yaml \
     --window_bars 160 --stride_bars 40 --seed 42
   ```
   This creates `reports/runs/<run_id>/render_manifest.json` covering everything under `out_root`.

4. **Build features + deterministic labels** using standardized images and the manifest:  
   ```bash
   uv run python scripts/build_features.py \
     --pipeline_cfg configs/pipeline.yaml \
     --manifest reports/runs/<run_id>/render_manifest.json \
     --ohlcv_root data/ohlcv \
     --images_root data/images/standardized \
     --out_dir data/features
   ```
   The builder instantiates `PatternLabeler` (H&S, ascending triangles, double top/bottom), computes structural TA features, and merges CV Hough stats. Outputs: `train/val/test/all_features.csv`.

5. **Train/evaluate models** (e.g., Random Forest) using the feature CSVs; account for class imbalance (DT/DB are dense, H&S/triangles sparser).

If you need to inject curated labels instead of the deterministic ones, you can pass `--labels_csv data/labels/hs_labeled_with_split.csv` to `build_features.py`, but the recommended path is to tune the YAML so the deterministic pass reflects your TA rules.

## Suggested next steps
1. Run `build_features.py` with the relaxed H&S config (already done) and inspect label counts per split; they should show H&S coverage across train/val/test (~150 positives total).
2. If DT/DB density is too high for your model, tighten their YAML (height/breakout thresholds) or downsample during training.
3. Train/evaluate per-pattern models (`scripts/train_rf.py` or notebooks), using class weights or sampling to address imbalance.
4. Keep diagnostics on during future H&S tuning; only adjust YAML thresholds, not hard-coded values, so changes stay traceable.

## CV feature expansion (Hough + handcrafted)
To keep the project “classical CV” while adding useful image cues, we expanded the Hough-based extractor:
- **Edges/entropy:** Canny edge density and pixel entropy to gauge texture/noise.
- **Hough lines:** Counts, angle stats, accumulator strength, angle differences, and fractions of horizontal/up/down lines.
- **HOG summaries:** Mean/std/max of HOG descriptors (windowed) for shape texture.
- **Gradient histograms:** Orientation histograms (0–180°) weighted by gradient magnitude.
- **Contours:** Count and area stats of external contours.
- **Simple H&S template match:** 1D column profile correlation against a synthetic shoulder–head–shoulder pattern.

All of these live in `src/features/cv_hough.py` via `extract_cv_features` (aliased to `extract_hough_features` for compatibility). The pipeline reads CV params from `configs/pipeline.yaml -> cv_features.hough` (e.g., Canny thresholds, Hough rho/theta/threshold, HOG window, grad bins, template width) and merges these into the feature CSVs. You just need `cv2` installed; no deep models are used. After updating, rerun `build_features.py` to populate the new `cv_*` columns.***

## Training scripts vs modules
- `src/models/train_rf.py` / `src/models/eval_rf.py` hold the core logic (load CSVs, drop non-feature columns, fit/evaluate scikit RFs, save models/manifests/metrics).
- `scripts/train_rf.py` / `scripts/eval_rf.py` are thin CLI wrappers that parse args (including defaults from `configs/pipeline.yaml`) and call the module functions. Use the scripts on the CLI; import the modules in notebooks.

### Making RF training more informative
The RF fit itself is fast and silent, but you still get:
- A printed ROC-AUC/PR-AUC/F1 when using a val split.
- Saved `train_manifest.json` alongside `model.joblib` with params, scores, best threshold, and top feature importance.
- Evaluation scripts write `metrics_*.json` and confusion matrices in `reports/eval/<target>/`.

If you want more visibility during training, consider:
- Adding a brief log of class balance and feature count at the start of `scripts/train_rf.py`.
- Printing permutation importances (already attempted; saved in the manifest when val split is used).
- For deeper inspection, run a notebook to plot PR curves/feature importances using the saved manifest and model.

## Avoiding overfitting (H&S) and hyperparameter sweeps
We hit “perfect” metrics on H&S when training on structural features (`hs_*`, `dt_*`, `db_*`, `tri_*`) because the RF simply memorized the deterministic labeler. To force learning from generic TA/CV cues:
- Structural features are dropped by default in `src/models/train_rf.py` (pass `--keep_structural` to override).
- `scripts/train_rf.py` now logs row counts, class balance, and whether structural features are kept.

For controlled tuning without leakage, use the helper grid approach (see `scripts/grid_rf.py` example):
- Loop over small grids of `min_samples_leaf`, `max_depth`, `n_estimators` with `class_weight=balanced` and `drop_structural=True`.
- Call `train_random_forest` and `evaluate_random_forest` directly to capture val/test metrics per combo, and save a `grid_results.json` summary under `reports/runs/grid_rf_hs/`.
- Pick the best hyperparams from the grid and re-run `scripts/train_rf.py` (still dropping structural features) to produce the final model/manifest. Record your choices and results here for future readers.

### H&S RF baseline (locked)
- Params: `min_samples_leaf=3`, `max_depth=16`, `n_estimators=800`, `class_weight=balanced`, `drop_structural=True`. Best threshold ≈ 0.60 (from val).
- Val metrics: ROC-AUC 0.844, PR-AUC 0.467, F1 0.410.
- Test metrics: ROC-AUC 0.916, PR-AUC 0.670, F1 0.615 @ threshold 0.60.
- Artifacts (copied to a stable location): `reports/baselines/hs/model.joblib`, `reports/baselines/hs/train_manifest.json`, latest test metrics `reports/baselines/hs/metrics_20251125T101016Z.json`.

### Double Top RF baseline (locked)
- Grid sweep showed a tight cluster of high performers; we picked a conservative, slightly shallow option.
- Params: `min_samples_leaf=3`, `max_depth=8`, `n_estimators=600`, `class_weight=balanced`, `drop_structural=True`. Best threshold ≈ 0.30.
- Val metrics: ROC-AUC 0.808, PR-AUC 0.783, F1 0.752.
- Test metrics: ROC-AUC 0.841, PR-AUC 0.889, F1 0.822 @ threshold 0.30.
- Artifacts (stable): `reports/baselines/dt/model.joblib`, `reports/baselines/dt/train_manifest.json`, test metrics `reports/baselines/dt/metrics_20251125T113004Z.json`.

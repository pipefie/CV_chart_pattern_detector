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
- **Alignment checks:** `scripts/validate_detectors.py` compares our detections to a TA ZigZag baseline and emits CSV + JSON (`reports/validation/detector_audit.*`) with only-ours/only-TA counts per pattern/symbol. Use agreement rates and confirm rates each run to spot over/under-fire and guide tuning.

When adjusting:
- Geometry: height_pct, min_height_atr, symmetry.
- Confirmation: percent/ATR break, within_bars, volume multiplier (per symbol).
- If a class floods, tighten geometry/confirmation; if it vanishes, ease timing/height or allow pre-breakout flags while training on confirmed.


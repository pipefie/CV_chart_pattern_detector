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


## Finance terms
- **OHLCV**: Open, High, Low, Close, Volume per bar.
- **HLC3 (typical price)**: `(High + Low + Close) / 3`; smoother than Close-only for indicators.
- **True Range (TRₜ)**: `max( Highₜ − Lowₜ, |Highₜ − Closeₜ₋₁|, |Lowₜ − Closeₜ₋₁| )`; captures gaps.
- **Average True Range (ATRₜ)**: Wilder’s smoothing of TR: `ATRₜ = ATRₜ₋₁ + (TRₜ − ATRₜ₋₁) / period`; default period 14. Expresses “typical move per bar” for volatility-aware thresholds.
- **ROC (Rate of Change)**: `(Close_t / Close_{t-p}) − 1` over lookback `p`.
- **Candlestick parts**: body = `|Close−Open|`; upper wick = `High − max(Open,Close)`; lower wick = `min(Open,Close) − Low`.
- **Patterns**:  
  - Head & Shoulders (H&S): peak–valley–peak–valley–peak with head above shoulders, neckline break downward.  
  - Inverse H&S: mirrored, breakout upward.  
  - Double Top/Bottom (DT/DB): two similar peaks/troughs separated by a pullback/rally; confirmation when price crosses neckline.  
  - Ascending Triangle: flat-ish resistance, rising support, upside breakout near apex.
- **HLC3/ATR rationale**: ATR normalizes rules across assets; HLC3 reduces intra-bar noise vs Close-only.

## Computer vision terms
- **Edge**: location of strong intensity gradient in an image.
- **Canny Edge Detector**: smooth → gradient magnitude/angle → non-maximum suppression → hysteresis with two thresholds; outputs thin, coherent edges.
- **Morphological Closing**: dilation followed by erosion; fills small gaps and connects nearby edge fragments.
- **Contour**: connected boundary in a binary image; convex 4-point contours approximate rectangular frames.
- **Laplacian (2nd derivative) energy**: magnitude of second-order gradients; used here as a “texture activity” proxy to find content-rich regions when edges fail.
- **Homography**: 3×3 projective transform mapping four source points (quadrilateral) to four target points (rectangle), preserving straight lines but correcting perspective; applied to rectify chart frames.
- **HOG (Histogram of Oriented Gradients)**: counts gradient orientations within cells; robust to small illumination changes, encodes shape/texture.
- **Gradient histogram**: Sobel-derived magnitudes weighted into angle bins (0–180°) for global directionality.
- **Hough Transform**: votes in (ρ, θ) space for lines passing through edge pixels; peaks indicate dominant lines; useful for gridlines/trendlines.
- **ORB (Oriented FAST and Rotated BRIEF)**: FAST corner detector + orientation-normalized binary BRIEF descriptors; efficient, rotation/scale tolerant.
- **BoVW (Bag of Visual Words)**: cluster local descriptors (e.g., ORB) via KMeans to form a vocabulary; each image represented by a histogram of visual word counts (optionally L1/L2 normalized).
- **LAB equalization**: histogram equalization on LAB color space’s L (lightness) channel to normalize illumination while preserving color semantics.

## Modeling & metrics
- **RandomForest**: ensemble of decision trees with bootstrap samples and feature subsampling; here with `class_weight=balanced`.
- **ROC-AUC**: probability a random positive scores above a random negative (0.5 = random, 1.0 = perfect).
- **PR-AUC**: area under precision–recall curve; more informative under class imbalance.
- **F1 score**: harmonic mean of precision and recall at a chosen threshold.
- **GroupKFold**: cross-validation that keeps all samples of a group (symbol) in the same fold to prevent leakage.

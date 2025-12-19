## Pipeline (`configs/pipeline.yaml`)
| Field | Description | Default/Example |
|-------|-------------|-----------------|
| `meta.comment` | Free-text note for the pipeline. | "End-to-end labeling + feature extraction pipeline config" |
| `labeling.units.prefer_atr_over_percent` | Choose ATR thresholds over percent when both exist. | true |
| `labeling.labels_post.dedup_time_overlap_bars` | Time-based NMS for overlapping detections. | 10 |
| `labeling.swing_points.window` | Half-window for swing detection. | 4 |
| `labeling.swing_points.min_distance_bars` | Minimum bars between swings. | 4 |
| `labeling.swing_points.prominence_atr` | ATR multiplier for swing prominence. | 0.9 |
| `labeling.swing_points.atr_period` | ATR period (Wilder RMA). | 14 |
| `labeling.patterns.head_and_shoulders.geometry.duration.min_bars/max_bars` | Allowed span in bars. | 24 / 180 |
| `...head_above_shoulders_min_atr/pct` | Head prominence vs shoulders (ATR / %). | 0.45 / 0.0045 |
| `...shoulder_height_similarity_pct` | Max shoulder height diff (%). | 35 |
| `...shoulder_timing_similarity_pct` | Min timing symmetry (%). | 75 |
| `...neckline.max_slope_deg` | Neckline slope cap (deg). | 18 |
| `head_and_shoulders.breakout.side` | Break direction. | "down" |
| `...confirm.threshold_atr/percent` | Breakout depth in ATR/% units. | 0.08 / 0.05 |
| `...confirm.within_bars` | Breakout confirmation window. | 22 |
| `ascending_triangle.geometry.duration.min_bars/max_bars` | Allowed span. | 30 / 150 |
| `...max_peak_diff_pct` | Peak similarity cap. | 0.01 |
| `...max_resistance_slope_deg` | Resistance flatness cap. | 8 |
| `...min_support_slope_deg` | Support slope floor. | 10 |
| `...min_lows` | Minimum lows between peaks. | 2 |
| `ascending_triangle.breakout.side` | Break direction. | "up" |
| `...max_time_to_break_pct` | Break before % of apex. | 0.75 |
| `...confirm.threshold_atr/percent` | Breakout depth ATR/% | 0.1 / 0.05 |
| `double_top.geometry.peaks` | Number of peaks. | 2 |
| `...peak_height_similarity_pct` | Max peak diff %. | 8 |
| `...min_height_atr` | Min structure height (ATR). | 1.1 |
| `...valley_required` | Require valley. | true |
| `...valley_drop_min_atr/pct` | Valley depth ATR/% floors. | 1.2 / 0.015 |
| `...valley_min_bars_from_peaks` | Spacing to valley. | 3 |
| `...max_pairs_per_peak` | Pairing cap. | 5 |
| `...dynamic_thresholds.valley_drop_min_height_pct` | Floor vs pattern height (%). | 18 |
| `...dynamic_thresholds.breakout_height_pct` | Breakout floor vs height (%). | 9 |
| `...dynamic_thresholds.atr_floor_mult` | ATR floor multiplier. | 0.6 |
| `...dynamic_thresholds.vol_lookback_bars` | Volume lookback. | 36 |
| `...duration.min_bars/max_bars` | Span bounds. | 24 / 110 |
| `double_top.breakout.side` | Break direction. | "down" |
| `...require_confirmation` | Enforce breakout. | true |
| `...confirm.threshold_atr/percent` | Breakout depth ATR/% | 0.8 / 0.30 |
| `...confirm.within_bars` | Breakout window. | 14 |
| `...confirm.volume_mult` | Volume spike multiplier. | 1.2 |
| `double_top.label_allow_pre_breakout` | Emit positives pre-break? | false |
| `double_top.scoring.weight_valley_depth_atr/weight_span` | Score weights. | 1.0 / 0.4 |
| `double_bottom` | Mirrors double_top, but breakout `side: up`, `peak_rise_min_atr/pct` replace valley drops; similar confirmation and scoring. | See file |
| `features.ta.atr_period` | ATR period for TA block. | 14 |
| `features.ta.roc_periods` | ROC lookbacks. | [5,10,20] |
| `features.ta.ma_periods` | MA lookbacks. | [10,20,50] |
| `features.candles.enabled` | Toggle candle stats. | true |
| `features.swings.enabled` | Toggle swing summaries. | true |
| `features.swings.pivot_left/right/min_sep` | Pivot heuristics. | 3 / 3 / 4 |
| `features.pattern_proxies.enabled` | Toggle heuristic proxies. | true |
| `cv_features.hough.*` | Canny/Hough/HOG/grad/template params. | e.g., canny_low=40, canny_high=140, grad_bins=9 |
| `cv_features.bovw.enabled` | Enable ORB+BoVW. | true/false |
| `cv_features.bovw.vocab_path` | Joblib KMeans path. | reports/cv_vocab/bovw_k64_seed42.joblib |
| `cv_features.bovw.n_clusters` | Vocab size. | 64 |
| `dataset.manifest_path` | Default manifest path. | reports/runs/latest/render_manifest.json |
| `models.random_forest.*` | RF defaults. | n_estimators=600, min_samples_leaf=2, max_depth=None, random_state=42 |

## Patterns (`configs/patterns.yaml`)
Hierarchical config with `common` base, `overrides` per symbol/timeframe, and `patterns` definitions.
- `meta.schema_version/timeframe/comment`: metadata.
- `common.atr`: period/source/smoothing (rma).
- `common.units.prefer_atr_over_percent`: priority when both units exist.
- `common.pivots`: method (prominence), ATR scaling, min_prominence_atr (1.2), min_distance_bars (8), smoothing window, adaptive prominence fallback, caps (`max_pivots_per_window`), `ignore_inside_gaps`.
- `common.lines`: regression mode (ransac/ols), RMSD caps, near-horizontal/parallel angle tolerances.
- `common.windows`: min/max/step bars for scanning; `nms.iou_threshold/keep_top_k` for dedup; `labels_post.dedup_time_overlap_bars` for time NMS.
- `overrides`: per-symbol/timeframe adjustments (e.g., crypto raises pivot prominence, equities force percent-based thresholds and easier breakouts).
- `patterns.head_and_shoulders`: required_peaks, order, prominence (0.5 ATR / 0.7%), min_height_atr, shoulder similarity/timing, neckline slope (12°), duration 28–200, breakout confirm (0.10 ATR/% within 14), invalidations, trade plan.
- `patterns.inverse_head_and_shoulders`: inherits H&S with relaxed head depth, bullish breakout within 10 bars.
- `patterns.double_top/double_bottom`: geometry (similarity 8%, valley/peak depth), dynamic thresholds, duration 24–110, neckline type=level, breakout confirmation, scoring weights, invalidation, trade plan.
- `patterns.descending_triangle`: descending upper trendline (≥3 touches), flat lower boundary RMSD cap, duration 24–200 with apex constraint, bearish breakout (0.3 ATR), optional Hough aids (disabled in main pipeline).
- Disabled twins: ascending/symmetric triangles at bottom of file (`enabled: false`).

## Render (`configs/render.yaml`)
| Section | Key | Meaning (default) |
|---------|-----|-------------------|
| `canvas` | width/height/dpi/bg_color | Image size (960×640, 120 dpi, white). |
| `chart` | type | "candlestick" fixed. |
|  | margin_pct | Price padding (7%). |
|  | wick_thickness_px/body_thickness_px | Uniform random pick within ranges [1,2] / [3,5]. |
|  | bull_color/bear_color | Palette lists for jitter. |
|  | grid.enabled/alpha | Grid on/off jitter; alpha 0.2 when enabled. |
| `axes` | show_ticks/font_family/font_size | Randomized ticks/fonts (DejaVu/Arial/Liberation) and sizes [9,11,13]. |
| `overlays` | moving_averages/volume/watermark | All false by default to keep frames clean. |
| `style_jitter` | blur_sigma/noise_std | Gaussian blur and noise ranges (0–0.5, 0–0.01). |
|  | contrast_gamma | Gamma jitter (0.92–1.08). |
|  | jpeg_quality | For optional JPEG exports (85–100). |
|  | slight_perspective | false (rectilinear). |
| `export` | format/compress_level/naming.schema | PNG, compress_level 3, schema `{symbol}_{tf}_{startts}_{endts}_{seed}.png`. |

## Backtest (`configs/backtest.yaml`)
| Section | Key | Meaning |
|---------|-----|---------|
| `universes.crypto` | symbols/timeframe/session | BTC-USD, ETH-USD, 1h, 24/7 UTC. |
| `universes.equities_etf` | symbols/timeframe/session | AAPL,NVDA,GOOGL,TSLA,AMZN,QQQ,SPY; 1h; RTH-only America/New_York. |
| `costs` | commission_bps/slippage_bps/borrow_fee_bps_daily | 1.0 / 1.5 / 0.0 |
| `risk` | per_trade_risk_pct/max_concurrent_positions/capital | 0.5%, max 5 positions, initial 100k, reinvest profits. |
| `execution` | entry_fill/stop_type/target_type/gap_handling | next_open, stop_market, limit, gap worse_price. |
| `walk_forward` | train_start/train_end/validate_end/test_end/roll_years | 2023-11-10 → 2025-03-31/08-15/10-31, roll 1y. |
| `report` | metrics | ["CAGR","Sharpe","Sortino","MaxDD","Calmar","PF","HitRate","AvgR","Exposure","Turnover"] |
|  | plots | equity_curve, drawdown, per_class_contribution, monthly_heatmap |

## Other defaults
- `requirements.txt`: OpenCV, scikit-learn, pandas, mplfinance, Pillow, ccxt, yfinance, joblib, yaml, numpy.
- `dataset.manifest_path` points at `reports/runs/latest/render_manifest.json`; override with `--manifest` in CLI calls.

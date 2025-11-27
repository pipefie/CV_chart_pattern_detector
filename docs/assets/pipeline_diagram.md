```mermaid
flowchart LR
    A[OHLCV parquet] --> B[Regime scan]
    B --> C[Deterministic labelers]
    C --> D[Render PNG + sidecar]
    D --> E[Standardize (homography + LAB)]
    E --> F[CV features (Hough/HOG/grad/contours/ORB)]
    C --> G[Structural/TA features]
    F --> H[Feature CSVs]
    G --> H
    H --> I[RandomForest train/eval]
    I --> J[Metrics & manifests]
```

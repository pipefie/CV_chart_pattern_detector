### Baseline RandomForest (structural features dropped)
| Pattern | ROC-AUC | PR-AUC | F1 @ th | Threshold | Source |
|---------|-------:|-------:|--------:|-----------:|--------|
| Head & Shoulders | 0.916 | 0.670 | 0.615 | 0.60 | reports/baselines/hs |
| Double Top | 0.841 | 0.889 | 0.822 | 0.30 | reports/baselines/dt |
| Double Bottom | 0.820 | 0.843 | 0.807 | 0.30 | reports/baselines/db |
| Ascending Triangle | 0.831 | 0.738 | 0.634 | 0.35 | reports/baselines/tri |

### BoVW comparisons (representative configs)
| Pattern | ROC-AUC | PR-AUC | F1 @ th | Notes |
|---------|-------:|-------:|--------:|-------|
| Head & Shoulders | ~0.909 | ~0.681 | ~0.615 @0.60 | Small PR lift; optional |
| Double Top | ~0.845–0.849 | ~0.898 | ~0.81–0.83 @0.35 | Slight gain |
| Double Bottom | ~0.833–0.835 | ~0.854 | ~0.81 @0.35–0.40 | Slight gain |
| Ascending Triangle | ~0.81 | ~0.666 | ~0.62 @0.35 | Underperforms baseline |

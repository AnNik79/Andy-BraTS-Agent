# Test comparison

Research experiment; no clinical validation.

| Method | BraTS mean Dice (WT/TC/ET) | Configured mean Dice | WT | TC | ET |
|---|---:|---:|---:|---:|---:|
| average | 0.87220 | 0.85649 | 0.89154 | 0.87663 | 0.84843 |
| boundary | 0.86344 | 0.85026 | 0.89014 | 0.86263 | 0.83757 |
| cnn | 0.86844 | 0.85308 | 0.88975 | 0.87074 | 0.84484 |
| final | 0.87344 | 0.85888 | 0.89479 | 0.87679 | 0.84872 |
| highres | 0.81344 | 0.78720 | 0.80420 | 0.83077 | 0.80536 |
| transformer | 0.86602 | 0.84705 | 0.88666 | 0.86631 | 0.84510 |
| vote | 0.87312 | 0.85740 | 0.89400 | 0.87560 | 0.84976 |

| Method | WT HD95 (mm) | TC HD95 (mm) | ET HD95 (mm) |
|---|---:|---:|---:|
| average | 18.686 | 15.567 | 13.097 |
| boundary | 18.393 | 17.223 | 13.833 |
| cnn | 18.950 | 13.520 | 11.923 |
| final | 17.279 | 14.421 | 12.527 |
| highres | 47.914 | 34.394 | 26.407 |
| transformer | 17.239 | 13.148 | 10.177 |
| vote | 17.221 | 12.452 | 10.919 |

See metrics.csv for per-patient and region results, disagreement_errors.csv for error enrichment and AUROC.
Cohort correlation treats patients as observations. Voxel statistics are descriptive, not independent-sample significance tests.

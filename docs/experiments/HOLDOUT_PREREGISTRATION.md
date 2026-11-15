# Pre-registration: fresh holdout suite (Round 3)

Registered on 2026-11-15, **before** any of the datasets below was downloaded or any model was run on
them. The library code in `src/egmm/` was frozen in the same commit that introduced this file
(`git log --diff-filter=A -- docs/experiments/HOLDOUT_PREREGISTRATION.md` shows it); no file under
`src/egmm/` may change between that commit and the holdout run.

## Datasets (OpenML ids; numeric, no missing values, not in the primary or small suites)

| # | name | OpenML id | rows | features |
|---|------|-----------|------|----------|
| 1 | kin8nm | 189 | 8,192 | 8 |
| 2 | cpu_act | 197 | 8,192 | 21 |
| 3 | bank32nh | 833 | 8,192 | 32 |
| 4 | texture | 40499 | 5,500 | 40 |
| 5 | GesturePhaseSegmentationProcessed | 4538 | 9,873 | 32 |
| 6 | eye_movements | 1044 | 10,936 | 27 |
| 7 | Ailerons | 296 | 13,750 | 40 |
| 8 | eeg-eye-state | 1471 | 14,980 | 14 |
| 9 | elevators | 846 | 16,599 | 18 |
| 10 | house_16H | 574 | 22,784 | 16 |
| 11 | 2dplanes | 215 | 40,768 | 10 |
| 12 | isolet | 300 | 7,797 | 617 |
| 13 | volkert | 41166 | 58,310 | 180 |
| 14 | helena | 41169 | 65,196 | 27 |
| 15 | Run_or_walk_information | 40922 | 88,588 | 6 |
| 16 | dionis | 41167 | 416,188 | 60 |

Excluded before download: `vehicle_sensIT` (357), stored as sparse ARFF (unsupported by the shared loader).

## Fixed protocol

* Features: numeric columns of the OpenML default feature set (the target is never a feature); rows with
  non-finite values dropped; 70/15/15 train/validation/test split with seed `700 + index`; zero-variance
  training columns dropped; no PCA.
* K = 8 (N_train < 5k), 12 (< 10k), 16 (< 30k), 24 (< 100k), 32 (otherwise); covariance = diag iff D > 64.
* Methods, seeds (0, 1, 2), timing (median of 5 refits for PyTorch methods, single fit for scikit-learn),
  online-EM tuning on the validation split, and statistics (Wilcoxon on 3-seed means, Holm within the
  panel, Friedman ranks) are exactly those of the primary benchmark (`experiments/run_benchmark_suite.py`,
  `experiments/analyze_results.py`).
* ARI/NMI are computed only for categorical targets with 2-50 classes.
* Every dataset is reported; none may be dropped after results are seen.

Code: `experiments/holdout_suite.py`. Output: `docs/experiments/holdout_head_to_head.csv`.

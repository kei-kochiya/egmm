# e-GMM (`EGMM`) Experimental Findings

All numbers in this repository's paper and in this note are regenerated from the CSV logs in this
folder by [`experiments/analyze_results.py`](../../experiments/analyze_results.py), which writes
[`stats_summary.txt`](stats_summary.txt), [`small_dataset_stats.txt`](small_dataset_stats.txt) and the
LaTeX tables/macros in [`paper/generated/`](../../paper/generated/). Treat `stats_summary.txt` as the
authoritative source; this note only explains what was run and why.

## What changed in this version (and why)

| Change | Code | Evidence |
| :--- | :--- | :--- |
| Greedy k-means++ seeding on `min(N, max(2048, 128K))` points + 10 Lloyd steps (shared by every PyTorch baseline) | `kernels.init_kmeans_pp(n_local_trials=None)`, `EGMM._initialize_moments` | `design_studies/seeding_variants.csv` (validation NLL only) |
| Smooth logistic-in-log N epoch / polish budgets replace the `N < 15k / 100k` (full) and `30k / 100k` (diag) brackets | `scaling.auto_hyperparameters(schedule_law="smooth")` | `design_studies/schedule_law_ab.csv`, ablation `EGMM_bracket_law` |
| Single-pass device-side standardization (every host row crosses PCIe once; bit-identical when it fits a 40%-of-free-memory budget) | `EGMM._fit_standardizer_and_transform` | `tests/test_round3_improvements.py`, phase table (`syn_scale400k_d256_diag`) |
| Exact float64 scoring (`score`, `score_samples`, `predict*`, `bic`, `aic`) for EGMM and all PyTorch baselines | `EGMM._set_exact_scoring_params`, `_iter_weighted_log_prob` | review finding: float32 expanded diagonal quadratic form cancels when variances hit `reg_covar` |
| Fair timing: `gc.collect()` + ~60 ms GPU clock warm-up before every timed GPU fit | `run_benchmark_suite.gpu_clock_warmup` | seed-1 outliers after long CPU sklearn fits in an earlier run |
| sklearn baselines: `max_iter=50` on every dataset and seed (an earlier run capped `N > 150k` full at 20 iterations, seed 0 only); `StandardizedSklearnGMM` now uses sklearn's default `init_params="kmeans"` and the same 1e-12 std floor as EGMM | `run_benchmark_suite.py`, `baselines.StandardizedSklearnGMM` | review finding |
| `OnlineGMM_matched_polish` receives exactly EGMM's resolved polish budget and `s_init` | `run_benchmark_suite.run_ablation_suite` | review finding (an earlier run gave it 3 vs 5 / 6 vs 12 steps on 10 datasets) |
| New sensitivity baseline `TorchBatchGMM_fullinit` (seeding on all N points) and `EGMM_polish0` inside the head-to-head run | `baselines.TorchBatchGMMFullInit` | review finding |

## Final outcome (see `stats_summary.txt`)

* Primary suite (33 datasets, 23 of which informed development): e-GMM beats `TorchBatchGMM` 24/0/9
  (Holm p = 7.6e-4) at 2.35x median speed, is indistinguishable from standardized scikit-learn (21/0/12,
  p = 0.29) at 32.5x, and beats tuned online EM 26/1/6.
* Pre-registered holdout (16 unseen OpenML datasets, library code frozen before download; content
  fingerprint `ea2a6d95...391f` from `experiments/code_fingerprint.py`): the quality advantage over batch
  EM does **not** carry over (9/0/7 vs `TorchBatchGMM`, 8/0/8 vs scikit-learn, both p = 1.0 after Holm;
  Hodges-Lehmann 95% CIs about -1 to +2 nats, so parity is not established either), the speed advantage
  does (2.54x, 37x; ~15x of the latter is GPU vs CPU), and e-GMM still beats tuned online EM (13/0/3,
  p = 0.012). The closing polish does not help on the holdout and collapses components on discrete-valued
  data (ailerons); eeg_eye_state's NLL is decided by one outlier test row. Raw scikit-learn failed to initialize on `dionis` (all 3 seeds); the
  failure is recorded in the CSV (`error` column) and excluded only from that method's comparison.
* Systems decomposition (systems run, all methods interleaved in one session): CUDA graph alone 1.95x;
  algorithm alone (both eager) 1.26x; combined 2.54x over `TorchBatchGMM`. A control batch EM with e-GMM's
  float32 inner Cholesky (`TorchBatchGMM_f32chol`, defined in the harness, not the library) is as fast as
  `TorchBatchGMM` (median time ratio 1.00), so batch EM's float64 Cholesky is not the reason for the gap.
  The split depends on size: N <= 1e5 (26 datasets) graph 2.04x / algorithm 1.12x; N > 1e5 (7 datasets)
  graph 1.57x / algorithm 2.11x (total 4.39x). e-GMM needs ~4x fewer E-step passes but not fewer M-steps
  (median 42 covariance factorizations vs 37.7 batch-EM iterations). Most holdout fits are latency-bound
  (batch EM at 1-5 ms per iteration on 15 of 16 datasets).

## Negative results worth knowing

* **Closing refinement does not close the large-N gap to 50-iteration scikit-learn.** One or two exact
  full-batch polish steps, or 2-3 EM steps on a 64k/128k subsample, after the zero-polish EMA phase change
  the four large synthetic mixtures by at most ~0.07 nats, and subsampled refinement hurts `creditcard`
  by ~5.5 nats (`design_studies/closing_refinement_large_n.csv`). Exact full-batch steps remove stochastic
  tail noise, so an e-means-style closing average cannot do better: the gap is the optimization basin,
  which better seeding addresses.
* **float32 vs float64 training** with identical random streams changes test NLL by at most ~0.2 nats on
  the eight diagonal datasets (`design_studies/precision_f32_vs_f64.csv`); the large float32 errors found by
  the earlier float32 errors affected *scoring*, which is now exact.
* **The synthetic calibration sweep does not determine the deployed constants** (`scaling_law_fit_summary.csv`:
  log-linear R^2 = 0.11 diag / 0.18 full; median-optimal alpha_0 = 0.85 diag, 0.45 full vs ~0.8 deployed
  for full at high gamma_df). The rules are hand-designed heuristics.

## Development-set disclosure

The batch/epoch/polish rules were iterated during development on 23 of the 33 primary datasets. The other
10 (`jannis`, `numerai28.6`, `higgs_98k`, `philippine`, `gas_drift`, `har`, `skin_nonskin`, `creditcard`,
`syn_scale300k_k64_full`, `syn_scale400k_d256_diag`) are reported as a separate panel in `stats_summary.txt`
and Table 1C of the paper. The small-N rules were introduced alongside the 16-dataset small evaluation.
The seeding and budget choices of this version were selected on the validation splits of all 33 primary datasets (with the evaluation seeds), so the whole primary suite is development data; only the pre-registered holdout is confirmatory.

## Reproduce

```bash
export PYTHONPATH="src;."          # use "src:." on Linux/macOS
python experiments/fit_scaling_laws.py
python experiments/run_benchmark_suite.py --stages h2h,abl,phase,pareto,sys
python experiments/run_small_dataset_eval.py
python experiments/design_studies/run_design_studies.py all
python experiments/analyze_results.py
python experiments/plot_results.py
```

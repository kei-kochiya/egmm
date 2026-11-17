# Next-version experiments: findings (run 2026-11-15, 20:01-21:51, RTX 5070)

Source: `dev.csv`, `holdout.csv`, `diag.csv`, `large.csv` and `summary.txt` in this folder
(`experiments/next/run_next.py`, `analyze_next.py`). 2,171 fits; 3 failures, all of the float32-Cholesky
control `TorchBatchGMM_f32chol` on near-singular holdout data (recorded, excluded only for that method).
`dev`, `diag` and `large` are development evidence; `holdout` rows are post hoc.

> Sections 1-7 report the first follow-up sessions. Their speed numbers are **superseded** by the re-timing in
> section 8 (batch EM checking convergence every iteration), which the paper uses; the scikit-learn NLL
> comparison in the paper is seed-matched (-1.17 / -0.37), not the 3-seed mean used in section 7.

## 1. Stronger batch-EM baselines
* The CUDA-graph batch EM (`TorchBatchGMM_graph`) is the fastest batch-EM variant on 29/33 development
  and 15/16 holdout datasets (1.8x / 2.9x faster than the paper's `TorchBatchGMM`). Removing host syncs
  without a graph (`_fast`) barely helps. At large N with full covariance the graph variant is *slower*
  than plain batch EM (wasted iterations between convergence checks), so comparisons use the fastest
  variant per dataset.
* e-GMM vs the fastest batch-EM variant per dataset (median speedup):
  development 1.22x (N <= 1e5: 1.06x, 14/26 faster; N > 1e5: 2.52x, 7/7); holdout 1.03x (9/16, p = 0.50;
  N <= 1e5: 1.00x); large N 3.20x (8/8; 1.5x-2.9x on synthetic diagonal, 2.9x-4.0x on synthetic full,
  20x on HIGGS and 22x on SUSY).
* **Implication:** on small and medium datasets the paper's 2.5x speedup is an engineering effect that a
  graph-captured batch EM erases. The defensible speed claim is at N > 1e5, where e-GMM's ~4x fewer data
  passes give 2.5x-22x over the best batch EM, with equal or better NLL (large: 6/2/0).
* The sync-free and graph variants factorize covariances in float32, so on real data their NLL differs from
  `TorchBatchGMM`'s beyond the 1e-3 tie band on some datasets, in both directions (graph vs eager W/T/L:
  development 4/24/5, holdout 8/5/3; neither significant). Quality comparisons therefore use the fastest
  variant's own NLL (development 24/0/9, holdout 10/0/6, large N 6/2/0 for e-GMM).

## 2. CUDA graph vs algorithm (decomposition)
* Removing the per-step alpha sync from eager e-GMM changes the graph factor only from 1.79x to 1.75x
  (development), so the old decomposition was not materially inflated by it.
* Sync-free eager e-GMM vs batch EM: 1.41x (dev, vs `TorchBatchGMM`), 1.25x (vs `_fast`), 0.61x (vs `_graph`).

## 3. Single-factor schedule ablation (development)
* Cosine vs Robbins-Monro (t+2)^-0.7 inside the full e-GMM pipeline: 26/1/6, median -0.134 nats,
  Holm p = 0.0055. Cosine vs constant step (same mean): 23/0/10, median -0.022, Holm p = 0.0093.
  The cosine schedule has a measurable, mostly small, benefit; most of it is over Robbins-Monro decay.

## 4. Holdout, post hoc: what drives e-GMM's advantage over online EM?
* vs tuned online EM: 13/0/3 (as in the paper). vs online EM with e-GMM's exact polish budget and
  s_init: 11/0/5, median -0.43, p = 0.10 (n.s.). The advantage is largely the closing polish.
* vs e-GMM without revival: 10/5/1, median -0.35, Holm p = 0.036. Revival helps out of sample.

## 5. High-dimensional diagonal diagnostics (8 development + volkert, isolet)
* alpha_0 = 0.85 (above the deployed 0.68 cap) improves 7 of 10 datasets, by 1-9 nats on volkert, isolet,
  gas_drift, philippine, har and syn_highdim128 (isolet then beats batch EM by ~5.6 nats; volkert's gap
  shrinks from 5.4 to 3.2), but hurts spambase by 37 nats (degenerate, discrete-like data).
* Doubling the EMA epochs helps volkert/isolet by ~2 nats but hurts har and spambase.
* The diagonal polish's early stop never triggers (tol = 0 changes nothing); removing polish hurts everywhere.
* **Implication:** the alpha_0 cap is the main suspect; a fix needs a guard for degenerate features and must
  be confirmed on a fresh holdout.

## 6. Restarts at equal compute (development + large N)
* Best-of-R e-GMM (selected by validation NLL) within the time of the fastest batch EM:
  development vs `_graph` 26/1/6, median -0.36 nats (mean R = 1.57); vs `TorchBatchGMM` 30/1/2;
  large N 7/1/0 to 8/0/0, median -0.58 to -0.61 nats. Library `n_init=3` costs ~0.9x the sum of 3 single fits.
* Promising for "better than batch EM", but the single-start development advantage did not replicate on
  the holdout, so this needs a new pre-registered holdout before any claim.

## 7. Large N (synthetic 1M-10M rows, SUSY 3.5M, HIGGS 7.7M; full and diagonal)
* e-GMM is faster than every batch-EM variant on all 8 datasets and never worse in NLL (6 wins, 2 ties);
  HIGGS: -4.08 nats at 20x; SUSY: -0.48 nats at 22x (batch EM hits its 50-iteration cap on millions of rows).
* Standardized scikit-learn at 1M rows (seed 0): 173 s (full) and 43 s (diagonal) vs 0.30 s / 0.11 s for
  e-GMM; NLL better for e-GMM on full (-1.03), worse on diagonal (+0.60).

## 8. Re-timing with every-iteration convergence checks (speed2, 2026-11-16, 133 min; speed2b, 59 min)
Source: `speed2_{dev,holdout,large}.csv`, `speed2b_large.csv` (no failed fits). These replace the
check-interval *estimate* used in an earlier draft; the paper's speed numbers come from here.
* Fastest batch EM per dataset (eager, sync-light or graph, checking convergence every 5 or every iteration):
  e-GMM median 1.15x on development N <= 1e5 (17/26 faster), 1.00x on holdout N <= 1e5 (8/15),
  2.67x on development N > 1e5 (7/7), 1.8-22.6x (median 3.2x) at 1-10M rows (8/8).
  Checking every iteration matters little on small data (graph-5 / graph-1 time 1.02) and ~1.18x at large N.
* Equal quality at 1-10M rows: on all 5 full-covariance datasets no batch-EM run (2-20 iterations, default
  stopping, or seeded on all rows) reaches e-GMM's test NLL within 0.01 nats; on the 3 diagonal mixtures batch
  EM matches it at 1.3-2.2x e-GMM's time. Batch EM with the same 2 passes takes ~1.1x e-GMM's time and is worse
  on all 8 (median -0.6 nats; HIGGS -8.4). Full-data seeding does not close the gap (6/0/2 for e-GMM).
* e-GMM with 3 epochs is worse than 2 on 3 of 8 (HIGGS +1.3 nats): more epochs do not always help.
* e-GMM float32 vs float64 training on the 25 full-covariance development datasets: 19 ties, max |d| 0.40
  nats (covtype_581k); float64 takes 2.4x as long.
* The fixed-budget graph runs in speed2 include one warm-up iteration; speed2b times each budget in both the
  graph and the sync-light implementation and the analysis keeps the faster one.

## 9. Checks after the second set of reviews (speed2c, speed2d, speed2e; 2026-11-16)
* 500 iterations without early stopping (seed 0): batch EM still 0.29-1.09 nats worse than e-GMM on the
  full-covariance large datasets (4.8 on HIGGS); unchanged after 100 iterations on the synthetic mixtures,
  still slowly improving on SUSY and HIGGS.
* Revival: e-GMM vs e-GMM without revival at large N 5/3/0 (median -0.32); without revival e-GMM is level
  with batch EM (2/1/2). Batch EM with e-GMM's revival operator (after iterations 2, 4, 6, 8) gains a median
  0.29 nats but still loses 5/0/0 (median -0.55) at ~5-22x e-GMM's time. Tuned stepwise online EM at large N
  loses 5/0/0 (median -0.54) at 1.6x e-GMM's time. So online phase + revival together beat batch EM.
* HIGGS without its four 3-valued b-tag features: e-GMM only -0.11 nats vs batch EM (3/3 seeds), 20x faster.
* Diagonal mixtures with larger batch-EM chunks (256k/1M rows): e-GMM's lead falls to 1.3-1.5x.
* The large synthetic generators contain 0-3 true components below the 0.05/K revival threshold.
* On the high-dimensional diagonal datasets the deployed alpha_0 is the rule's floor (0.40-0.48), not the
  0.68 cap; the alpha_0 = 0.85 diagnostic also raised alpha_min (0.04 -> 0.085).

## 10. Checks after the third set of reviews (speed2f, speed2g, oracle; 2026-11-17)
* Batch EM with revival after each of its first 20 iterations (<= 0.3 dead components left): e-GMM still
  wins 5/0/0 (median -0.55; HIGGS -2.40; 14/15 seeds; 4/4 without HIGGS) at ~11x e-GMM's time.
* Batch EM on a 250k / 1M-row subsample: e-GMM better 5/0/0 / 4/0/0 (medians -0.51 / -0.39); the 250k fit is
  a bit faster than e-GMM, the 1M fit 2.4x slower. Stepwise online EM with revival: 5/0/0 (median -0.43), 1.6x.
* Schedule at 1-10M rows: cosine vs Robbins-Monro 5/2/1 (median -0.28), vs constant step 8/0/0 (median -0.09).
* HIGGS with the b-tags dequantized: e-GMM -0.16 nats vs batch EM (3/3 seeds), 19x faster.
* Oracle (true generating mixture): e-GMM 0.57-0.69 nats above it on the full-covariance synthetic sets,
  batch EM 1.17-1.76; on the diagonal sets both 0.80-1.92.


## 11. Checks after the fourth set of reviews (speed2h; 2026-11-17, 8 min)
* Logging bug: `TorchBatchGMM.fit` (frozen library) resets `revivals_count_` after the EM loop, so the
  `revivals` column of the batch-EM revival controls in speed2e/speed2f is 0. Fixed in `TorchBatchGMMRevival.fit`
  (unit test `test_revival_controls_log_their_split_count_cpu`); NLLs and dead counts were never affected.
* With the count fixed, batch EM with revival at four checkpoints revives 0-11.7 components per fit (HIGGS 11.7,
  synthetic 0-2), against 2.3-24.3 for e-GMM: batch EM leaves few components dead, so revival has little to do.
* Matched-intensity control (`TorchBatchGMM_forced`): at the same four checkpoints the lowest-mass components
  are marked dead and re-split by the library's revival operator, so batch EM makes at least as many split moves
  as e-GMM (3-24 per fit). e-GMM still wins 5/0/0 (median -0.87; HIGGS -1.87; 14/15 seeds; 4/0/0 without
  HIGGS) at ~8.6x e-GMM's time. Forced splits vs revival only for batch EM: better on HIGGS/SUSY, worse on the
  three synthetic mixtures (2/0/3). The number of split moves alone does not explain e-GMM's advantage.
* Speed reference sensitivity (no new run): restricted to batch-EM variants that stop at the float64 run's
  iteration on every seed, e-GMM's median speedup at N <= 1e5 is 1.30x (dev) and 1.83x (holdout) instead of
  1.15x and 1.00x, so the fastest-variant reference is conservative.
* Single e-GMM start vs best of 3 batch-EM starts (validation NLL) at large N: 4/0/1 on full covariance (one
  loss, 0.07 nats on syn3m_full), 4/0/4 overall; now reported in Section 5.3.

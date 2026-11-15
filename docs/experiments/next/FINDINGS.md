# Next-version experiments: findings (run 2026-11-15, 20:01-21:51, RTX 5070)

Source: `dev.csv`, `holdout.csv`, `diag.csv`, `large.csv` and `summary.txt` in this folder
(`experiments/next/run_next.py`, `analyze_next.py`). 2,171 fits; 3 failures, all of the float32-Cholesky
control `TorchBatchGMM_f32chol` on near-singular holdout data (recorded, excluded only for that method).
`dev`, `diag` and `large` are development evidence; `holdout` rows are post hoc.

## 1. Stronger batch-EM baselines (the main reviewer request)
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

# Next version of the paper: stronger baselines, large N, restarts at equal compute

Development work for a revision or resubmission. Nothing here changes the frozen library (`src/egmm`) or
the submitted paper; results land in `docs/experiments/next/`.

## Questions

1. **Baseline strength.** How much of e-GMM's speedup survives against a GPU batch EM without avoidable
   host synchronizations (`TorchBatchGMM_fast`), and against one with a CUDA-graph-captured iteration
   (`TorchBatchGMM_graph`)? (`experiments/next/baselines_fast.py`; both stop at the same iteration as
   `TorchBatchGMM`, checked by `tests/test_next_baselines.py`.)
2. **Large N.** Does the algorithmic advantage (fewer data passes) grow at 10^6-10^7 training rows, where
   batch EM is no longer launch-bound? Synthetic mixtures (1M, 3M, 10M rows) need no download. Real large
   datasets are read from `data/large/`: SUSY from UCI (`SUSY.csv.gz`, 5M rows x 18), and HIGGS from the
   Hugging Face mirror `jxie/higgs` (`data/large/higgs_hf/*.parquet`, SHA-256 checked): the 11M UCI rows in
   their original order with every feature z-scored, which leaves all methods' fits and NLL differences
   unchanged (every method standardizes internally) and shifts absolute NLLs by a constant.
3. **Restarts at equal compute.** Can best-of-R e-GMM restarts, within batch EM's time budget, reach a
   better test likelihood than batch EM? Evaluated offline from independent single-start fits, with two
   selection rules (training log-likelihood; validation NLL).

4. **External review (review.txt) requests**, all without library changes (`experiments/next/variants.py`):
   eager e-GMM without the per-step alpha synchronization (`EGMM_eager_hostalpha`); single-factor schedule
   ablations inside the full e-GMM pipeline (`EGMM_sched_rm`: alpha_t = (t + 2)^-0.7; `EGMM_sched_const`:
   the cosine schedule's mean); on the holdout (post hoc) `OnlineGMM_tuned`, `OnlineGMM_matched_polish`
   and `EGMM_no_revival`; and a `diag` stage on the 8 diagonal development datasets plus volkert and isolet
   (`alpha = 0.85`, twice the epochs with the same polish budget, polish without early stopping, no polish).

## Protocol

* `python experiments/next/run_next.py --stages dev,holdout,diag,large` (GPU), then
  `python experiments/next/analyze_next.py` -> `docs/experiments/next/summary.txt`.
* The revised paper's numbers and tables (`tab_speed_next`, `tab_large`, `tab_diag`, macros `\rX...`) come from
  `paper_results.py`, which `experiments/analyze_results.py` calls; `verify_tables.py` covers them too.
* Same harness as the paper (`evaluate_model`: GPU clock warm-up, median of 5 timed refits, 3 seeds,
  shuffled method order); every stage interleaves all its methods in one session, so speedups are never
  cross-session. Failed fits are recorded in the CSV `error` column.
* Smoke test without a GPU: `CUDA_VISIBLE_DEVICES=-1 python experiments/next/run_next.py --smoke --device cpu`
  (on Windows an empty `CUDA_VISIBLE_DEVICES` does not hide the GPU; use `-1`).

## Status of the evidence

* `dev` (33 development datasets) and `large` (synthetic) are **development** evidence.
* `holdout` is a **post-hoc timing** run on the pre-registered holdout (NLLs are deterministic and match
  the paper's runs); it contains no restarts, so the old holdout is not used to develop the restart idea.
* A claim that restarts beat batch EM needs a **new pre-registered holdout** (new datasets fixed before any
  fit, library frozen, hypothesis and margin stated in advance), as in the paper.

## Decision rules (fixed before the GPU runs)

* Report every batch-EM variant; the paper's headline baseline becomes the fastest variant on each
  dataset if it is faster than `TorchBatchGMM` on most datasets.
* Restarts: the primary rule is selection by validation NLL; "beats batch EM" requires W > L, Holm
  p < 0.05 against the fastest batch-EM variant on the development suite before a holdout is planned.
* Large N: report per-dataset times and NLLs; no significance test (n = 6 synthetic datasets).

## Re-timing after the mock reviews: `speed2` and `speed2b` (run 2026-11-16)

The first sessions' sync-light and graph variants read the convergence test only every 5 iterations (up to 4
extra iterations; the graph variant also runs one discarded warm-up iteration). The paper's speed results now
come from a re-timing that adds variants checking every iteration:

    python experiments/next/run_next.py --stages speed2      # speed2_{dev,holdout,large}.csv  (~133 min)
    python experiments/next/run_next.py --stages speed2b     # speed2b_large.csv               (~60 min)

`speed2` adds `_fast1`/`_graph1`, logs executed iterations and GPU clock/temperature/power per (dataset, seed),
an `EGMM_f64` run on full-covariance development datasets, and at large N full-data-seeded batch EM
(`TorchBatchGMM_fullinit`). `speed2b` times fixed batch-EM budgets (2-20 iterations) in both the graph and the
sync-light implementation next to e-GMM with 1-3 epochs. Analysis: `speed2_results.py` (called from
`paper_results.py`); figures: `plot_speed.py`. Results: `docs/experiments/next/FINDINGS.md`, section 8.

## Runtime estimate (RTX 5070)

`dev` ~80-100 min, `holdout` ~25-30 min (includes online-EM tuning), `diag` ~20-30 min, `large` ~45-60 min
with SUSY and HIGGS (10M-row sets take ~2 GB of RAM each and are generated one at a time); ~3-3.5 h total.

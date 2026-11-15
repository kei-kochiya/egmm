# Next version of the paper: stronger baselines, large N, restarts at equal compute

Development work for a revision or resubmission. Nothing here changes the frozen library (`src/egmm`) or
the submitted paper; results land in `docs/experiments/next/`.

## Questions

1. **Baseline strength.** How much of e-GMM's speedup survives against a GPU batch EM without avoidable
   host synchronizations (`TorchBatchGMM_fast`), and against one with a CUDA-graph-captured iteration
   (`TorchBatchGMM_graph`)? (`experiments/next/baselines_fast.py`; both stop at the same iteration as
   `TorchBatchGMM`, checked by `tests/test_next_baselines.py`.)
2. **Large N.** Does the algorithmic advantage (fewer data passes) grow at 10^6-10^7 training rows, where
   batch EM is no longer launch-bound? Synthetic mixtures (1M, 3M, 10M rows) need no download; real large
   datasets (UCI HIGGS, SUSY) are read from `data/large/` only if the author downloads them.
3. **Restarts at equal compute.** Can best-of-R e-GMM restarts, within batch EM's time budget, reach a
   better test likelihood than batch EM? Evaluated offline from independent single-start fits, with two
   selection rules (training log-likelihood; validation NLL).

## Protocol

* `python experiments/next/run_next.py --stages dev,holdout,large` (GPU), then
  `python experiments/next/analyze_next.py` -> `docs/experiments/next/summary.txt`.
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

## Runtime estimate (RTX 5070)

`dev` ~60-75 min, `holdout` ~15-20 min, `large` ~30-45 min (10M-row sets take ~2 GB of RAM each and are
generated one at a time).

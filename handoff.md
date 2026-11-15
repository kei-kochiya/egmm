# Handoff: `e-GMM` (Round 3 → TMLR submission)

**Repository:** `https://github.com/kei-kochiya/egmm` (branch `main`) · **Workspace:** `d:\nckh`
**Venue:** Transactions on Machine Learning Research (TMLR), double-blind.
**Author:** Ho Tuan Khoi (first submission). Keep this file out of the supplementary material (it is excluded
automatically by `experiments/make_supplementary.py`).
**Environment (real, verifiable):** Python 3.14.7, PyTorch 2.14.1+cu132, scikit-learn 1.9.1, NVIDIA GeForce
RTX 5070 (12 GB), Intel Core i7-14700F, Windows 11.

---

## 1. Where things stand

* `paper/main.pdf` is built with the **official TMLR style** (`paper/tmlr.sty`, `paper/tmlr.bst`) in anonymous
  mode; 12 pages of main text (Sections 1-7), 26 pages with statements, references and appendices A-D.
  Results are ordered holdout first. Zero LaTeX errors, zero `Overfull \hbox`. The PDF contains no name/email/affiliation/GitHub link (checked by
  `python experiments/make_supplementary.py`).
* **Every number, table and statistic** in the paper is generated from `docs/experiments/*.csv` by
  `experiments/analyze_results.py` into `paper/generated/` (`numbers.tex` macros `\r...` + table bodies).
  Never type a number into `main.tex` by hand: re-run the script instead.
* Supplementary ZIP: `python experiments/make_supplementary.py` → `dist/supplementary_anonymized.zip`
  (anonymity scan of PDF and archive, SHA-256 manifest, library fingerprint, and a self-test that re-runs
  the analysis from the unpacked archive and checks all generated paper files are byte-identical). The
  archive ships `paper/generated/` and `experiments/verify_tables.py`, so reviewers can run the same check;
  exact versions are in `requirements-lock.txt`.
* Submission steps: `docs/submission/SUBMISSION_CHECKLIST.md`. Reviewer preparation:
  `docs/submission/REVIEWER_QA_PREP.md`.
* Tests: `PYTHONPATH=src pytest -q` → 51 passed (incl. CUDA-graph-vs-eager and sync-free seeding checks).

## 2. What changed in Round 3 (summary; details in `docs/experiments/egmm_findings.md`)

Library (`src/egmm`, frozen at commit `7d6b9c9` before the holdout data was downloaded):
greedy k-means++ seeding on `max(2048, 128K)` points + 10 Lloyd steps (shared by all PyTorch baselines);
smooth logistic-in-log-N epoch/polish budgets (`schedule_law="bracket"` reproduces the old rules);
single-pass device-side standardization; exact float64 scoring; `TorchBatchGMMFullInit` sensitivity
baseline; sklearn wrappers with identical initialization (`init_params="kmeans"`) and std floor.

Harness / analysis: GPU clock warm-up + median of 5 timed refits + randomized method order; uncapped
sklearn (`max_iter=50`) on every dataset and seed; exactly matched `OnlineGMM_matched_polish`; N-scaled
online-EM grid for small N; systems stage (`EGMM_eager`) and data-pass cost; Holm correction within every
panel; failed baseline fits recorded and disclosed (never dropped); design-study and holdout scripts.

Paper: rewritten around generated macros; audit fixes (wrong statistics, unfair baselines, e-means and
theorem wording, references); development-set and calibration disclosure; pre-registered holdout section;
systems decomposition; TMLR template, anonymization, appendix, reproducibility statement.

## 3. Final results (authoritative: `docs/experiments/stats_summary.txt`; p-values Holm-adjusted)

| | e-GMM vs TorchBatchGMM | vs StdSklearnGMM (CPU) | vs tuned online EM |
| :--- | :--- | :--- | :--- |
| Development suite (33; called "primary" in the pre-registration) | 24/0/9, p = 7.6e-4, 2.35x faster | 21/0/12, p = 0.29, 32.5x faster | 26/1/6, p = 3.1e-4 |
| **Pre-registered holdout (16 unseen)** | **9/0/7, p = 1.0, 2.54x faster** | **8/0/8, p = 1.0, 37x faster** | **13/0/3, p = 0.012** |

* Honest positioning: on unseen data e-GMM is **not significantly different** from batch EM / scikit-learn
  (HL 95% CIs about -1 to +2 nats, so parity is not established either) at 2.5x / ~37x lower fit time
  (~15x of the latter is GPU vs CPU), with skewed differences (losses of 3.8-5.4 nats on 3 holdout datasets,
  incl. both diagonal ones). It beats tuned online EM, but without its closing polish not significantly
  (12/0/4, raw p = 0.074). It is not a more accurate optimizer than batch EM.
* Speed decomposition (systems run, re-run with a float32-Cholesky batch-EM control): CUDA graph 1.95x;
  algorithm (both eager) 1.26x; total 2.54x. The control shows batch EM's float64 Cholesky costs nothing
  (ratio 1.00). By size: N <= 1e5 graph 2.04x / algorithm 1.12x; N > 1e5 graph 1.57x / algorithm 2.11x.
  e-GMM needs ~4x fewer E-step passes but not fewer M-steps (median 42 vs 37.7).
* Revival: dead components 0.03 vs 2.38 without revival (ablation), 1.09 for TorchBatchGMM.
* Known weaknesses to defend: slower than online EM in single-run time; lowest-to-mid ARI; hand-designed
  constants (synthetic sweep R^2 = 0.11 / 0.18); mixed large-N quality vs sklearn on two synthetic sets;
  raw sklearn failed to initialize on `dionis` (disclosed).

## 3b. Mock review (round 1) and what changed

Three independent reviewer agents (methods/theory, empirical rigor, systems/reproducibility) all said
"leaning reject, close to the boundary". Every factual claim was verified against the code/CSVs before
acting. The paper was revised accordingly: new title (no "collapse-resistant"), online-EM framing
(Titterington; Sato & Ishii; Cappé & Moulines), non-Robbins-Monro disclosure, Theorem -> Lemma with
simulation (0/5,000 classical-split PSD violations), heuristic rules disclosed, holdout robustness table
(Hodges-Lehmann CIs, per-dimension/sign tests, exclusions, converged batch-EM reference
`experiments/converged_reference.py`), holdout polish/dead-component results, full list of
pre-registration deviations (dionis re-run after a scikit-learn crash, failed-cell rule, two pre-cached
datasets), speed decomposition with session-to-session variability and GPU-vs-CPU share, expanded
Limitations. The questions and honest answers are in `docs/submission/REVIEWER_QA_PREP.md`.
Library code fingerprint (unchanged since the freeze): `ea2a6d957ddc...8eab391f`.

## 3c. Mock review rounds 2-4 (12 reviewer agents in total, Opus and Sonnet)

Round 2 (action editor, statistics, theory): ~14 distinct major issues, all verified and fixed (skewed holdout
tail, "parity" wording, per-dimension robustness, polish and online-EM claims, selective design-study
reporting, Pareto and small-dataset misstatements, positioning against mini-batch/stochastic EM, 20 new
references, s_init is a step-size reparametrization). Round 3 (generalist, systems, numbers auditor): 2
distinct majors (timing variability understated; speed decomposition confounded by batch EM's float64
Cholesky - a control run showed it is not). Round 4 on the shortened version (structure, content
preservation, adversarial): no critical issue; fixed the abstract numbers missing from the body, the size
dependence of the speed split, latency/stopping-rule caveats, covtype/letter discrete features, the ailerons
description, and scoped the conclusion. Verdicts moved from "leaning reject" (round 1) to "leaning accept".
Open suggestion from the adversarial reviewer (not done; needs the author's decision): a stronger,
synchronization-free or graph-captured batch-EM baseline, which could shrink the 2.5x headline.

## 4. Reproduce everything

```bash
export PYTHONPATH="src;."          # "src:." on Linux/macOS
python experiments/fit_scaling_laws.py
python experiments/run_benchmark_suite.py --stages h2h,abl,phase,pareto,sys
python experiments/run_small_dataset_eval.py
python experiments/design_studies/run_design_studies.py all
python experiments/holdout_suite.py          # pre-registered; do not change src/egmm before running it
python experiments/converged_reference.py
python experiments/design_studies/split_psd_check.py
python experiments/analyze_results.py
python experiments/verify_tables.py         # checks paper/generated against the logs
python experiments/plot_results.py && cp docs/experiments/fig_*.png paper/figures/
cd paper && tectonic -X compile main.tex
```

## 5. Next steps

1. Submission (user): follow `docs/submission/SUBMISSION_CHECKLIST.md`.
2. Library hardening (after the research round): license (user's choice), packaging, input validation,
   docs, CI, without changing numerics.
3. Possible follow-up paper (only after this one is decided): "better than batch EM" via convergence-aware
   polish / equal-compute restarts / split-merge, evaluated on a *second* pre-registered holdout.

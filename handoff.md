# Researcher Handoff: `e-GMM` Review, Scaling Law Upgrade & Extended Medium/Large Evaluation

**Repository:** `https://github.com/kei-kochiya/egmm` (branch `main`, latest verified commit prior to this handoff: `9a1e3f0`)  
**Workspace:** `d:\nckh`  
**Status at handoff:** Unit tests pass (`25 passed in 3.98s`), 23-dataset primary benchmark + 16-dataset external small benchmark + 11 standalone single-panel figures + Overleaf manuscript (`paper/main.tex`) are complete and committed. However, several methodology, scaling-law, systems, and reviewer-rigor gaps remain open for strict auditing and next-phase research.

---

## 1. Current Architecture & File Map

| Component | Path | Purpose & Key Notes |
| :--- | :--- | :--- |
| Kernels | `src/egmm/kernels.py` | Shift-invariant centered mixed-precision E-step (`e_step_batch`, `float32` likelihoods / `float64` sufficient-stat reconstruction), pre-projected Cholesky GEMMs (`prepare_precisions`, `compute_log_gaussian_prob_from_precisions`), M-step projection (`moments_to_params_and_precisions`, branchless `allow_sync=False` CUDA-Graph path & `fast_cholesky_f32`). |
| Revival | `src/egmm/revival.py` | Vectorized donor eigen-split (`revive_dead_components`): top-k donor pairing (`M * (spread + reg_covar)`), 5-step batched `float32` power iteration on CUDA / exact `float64` `eigh` on CPU, split separation `split_delta=0.5`. |
| Auto Scaling Laws | `src/egmm/scaling.py` | Closed-form `auto_hyperparameters` + `compute_cosine_alpha`. Uses piecewise thresholds and `eff_df = B / (D * K)` (`diag` and `full` branches). **Primary improvement target.** |
| Estimator API | `src/egmm/estimator.py` | Drop-in `sklearn` `EGMM` (`fit`, `partial_fit`, `score`, `score_samples`, `predict`, `predict_proba`, `bic`, `aic`, `sample`), CUDA Graph step capture (`fit` lines 533–626), adaptive CPU/CUDA switch (`_resolve_device` lines 77–85). |
| Baselines | `src/egmm/baselines.py` | `TorchBatchGMM`, `CappeMoulinesOnlineGMM`, `NaiveMinibatchGMM`, `EMeansInitGMM` (subclasses of `EGMM` sharing `kernels.py`). |
| Benchmark Suites | `experiments/datasets.py`, `experiments/run_benchmark_suite.py`, `experiments/run_small_dataset_eval.py` | 23 primary datasets (3 seeds) + Pareto curves (6 datasets) + Wilcoxon/Friedman-Nemenyi stats; 16 unseen small datasets ($N_{\text{train}} \in [105, 3831]$). |
| Scaling & Plots | `experiments/fit_scaling_laws.py`, `experiments/plot_results.py`, `experiments/profile_kernels.py` | Synthetic sweep (`scaling_sweep.csv`, 600 runs) + standalone plot generator (11 PNGs in `docs/experiments/` and `paper/figures/`). |
| Reference | `docs/references/e_means/` | Mirrored Meta FAIR `e-means` (`arXiv:2611.09501`) code (`base.py`, `e_means_torch.py`, `torch_backend.py`, `e_means.py`, `README.md`, `NOTES.md`). |
| Manuscript | `paper/main.tex`, `paper/references.bib`, `paper/figures/` | Self-contained Overleaf two-column LaTeX draft (verified structurally; not compiled via local `pdflatex` because TeX is not installed locally). |

---

## 2. Verified Baseline Metrics at Handoff

Hardware/runtime observed in this workspace: Windows 11, Python 3.14.7, PyTorch 2.14.1+cu132, CUDA 13.2, NVIDIA GeForce RTX 5070, `scikit-learn` 1.9.1.

- **Unit tests:** `PYTHONPATH=src pytest -q` -> `25 passed`.
- **23-dataset primary suite (3-seed means, `docs/experiments/stats_summary.txt`):**
  - `EGMM` vs `TorchBatchGMM`: Test NLL **19/0/4** wins (`med_dNLL = -0.3691`, Wilcoxon $p = 2.15 \times 10^{-4}$); Speed **23/23** faster (`med = 2.33x`, `max = 6.72x` on `miniboone`, `5.89x` on `covtype_581k`, $p = 2.38 \times 10^{-7}$).
  - `EGMM` vs `OnlineGMM_tuned`: NLL **19/1/3** ($p = 7.32 \times 10^{-5}$); median speedup `0.77x`.
  - `EGMM` vs `NaiveMinibatchGMM` / `EMeansInitGMM`: NLL **22/0/1** ($p = 7.87 \times 10^{-6}$) / **23/0/0** ($p = 2.38 \times 10^{-7}$).
  - `EGMM` vs `SklearnGMM` (CPU `float64`): NLL **11/0/12** ($p = 0.964$, unstandardized `reg_covar` differs); median speedup **56.07x** (max **560.91x** on `covtype_581k`, where `SklearnGMM` was capped at `max_iter=15`, `seed=0` only).
  - Friedman $\chi^2_F = 41.957$ ($p = 6.01 \times 10^{-8}$), Nemenyi $\mathrm{CD}_{0.05} = 1.572$; mean NLL ranks: `EGMM=1.913`, `SklearnGMM=2.826`, `TorchBatchGMM=3.261`, `OnlineGMM_tuned=3.435`, `EMeansInitGMM=4.609`, `NaiveMinibatchGMM=4.957`.
  - Dead components (mean across 23 datasets): `EGMM = 0.10` (`3.46` mean revivals) vs `TorchBatchGMM = 1.87`, `OnlineGMM_tuned = 2.68`, `SklearnGMM = 1.16`.
  - Labeled-subset clustering (22 datasets with `y_test`): mean ARI `EGMM = 0.3879` vs `OnlineGMM_tuned = 0.4254`, `TorchBatchGMM = 0.4056`, `SklearnGMM = 0.4098`; mean NMI `EGMM = 0.5692` vs `SklearnGMM = 0.5794`.
- **16 external small datasets ($N_{\text{train}} \in [105, 3831]$, zero tuning, `docs/experiments/small_dataset_stats.txt`):**
  - `EGMM_auto` vs `TorchBatchGMM`: NLL **10/0/6** ($p = 0.0739$), Speed **1.88x** ($p = 6.10 \times 10^{-5}$).
  - `EGMM_auto` vs `SklearnGMM`: NLL **5/0/11** ($p = 0.4037$), Speed **1.68x** ($p = 6.29 \times 10^{-3}$).

---

## 3. Known Technical Issues, Confounds, and Missed Gaps for Strict Review

Any strict reviewer (or follow-up researcher) MUST inspect these concrete issues directly visible in the current codebase and experiments:

### A. Scaling Law & "Auto" Hyperparameter Weaknesses (`src/egmm/scaling.py` & `experiments/fit_scaling_laws.py`)
1. **Manual piecewise heuristics instead of regressed power laws:** Unlike `e-means` (`docs/references/e_means/base.py`: `_BATCH_LAWS`, `_ALPHA_LAW`, `_INIT_USAGE = 2**-16`), `src/egmm/scaling.py` still uses hand-tuned piecewise cutoffs (`N < 3000, 10000, 15000, 25000, 30000, 100000`) and a hand-picked linear ramp `t_df = clip((eff_df - 1.5)/1.5, 0, 1)`. Running `analyze_and_fit` on `docs/experiments/scaling_sweep.csv` only achieves OLS $R^2 = 0.331$ (`diag`) and $R^2 = 0.186$ (`full`), and those OLS coefficients are **not** used in `scaling.py`.
2. **Revival silently disabled in diagonal mode:** `src/egmm/scaling.py:80` sets `rev_points = 0 if revive_every == "auto" else -1` for `diag` covariances! Consequently, **zero revivals occur on any `diag` dataset** (`miniboone`, `spambase`, `syn_highdim128_50k`, `syn_scale500k_diag` all log `rev=0.00` and leave dead components `0.33–1.00`), contradicting any blanket claim of universal collapse prevention across both covariance types.
3. **Hardcoded `diag` $\alpha_0 = 0.42$ despite sweep evidence:** `fig_scaling_diag.png` and `analyze_and_fit` show optimal `diag` $\alpha_0$ repeatedly hitting `0.70` (the top of `alpha_grid`), yet `scaling.py:77` hardcodes `alpha_0 = 0.42` and `fit_scaling_laws.py` capped `alpha_grid` at `0.70` without ever sweeping `init_usage_scale`, `polish_steps`, `revive_points`, or larger $N > 128\text{k}$.

### B. Experimental Confounds & Coverage Gaps (`experiments/`)
4. **Full-batch polish confound (`estimator.py:505`):**
   `use_polish = type(self) is EGMM and hp.polish_steps > 0` runs 5–6 exact full-batch EM iterations (`full`, $N \le 100\text{k}$) or up to 12 (`diag`, $N \le 100\text{k}$) ONLY for `EGMM`, while subclass baselines `OnlineGMM_tuned` and `NaiveMinibatchGMM` get `use_polish = False` and `use_init_scale = False`. On $N \le 100\text{k}$, reviewers will rightly demand ablations isolating:
   - `EGMM (polish_steps=0)` vs `OnlineGMM_tuned`,
   - `OnlineGMM_tuned + matched polish` vs `EGMM`,
   - `EGMM` without `init_usage_scale` vs with `init_usage_scale`,
   - `EGMM` without `revive_dead` vs with `revive_dead`.
5. **Insufficient large & diverse real medium/large datasets:** Only 3 datasets in the primary suite have $N_{\text{train}} > 100\text{k}$ (`covtype_581k` real, `syn_scale200k`, `syn_scale500k_diag`), and `diag` covariance is only tested on 4/23 primary datasets (`spambase`, `miniboone`, `syn_highdim128_50k`, `syn_scale500k_diag`). Since `polish_steps == 0` ONLY triggers when $N > 100\text{k}$, large-$N$ evaluation ($100\text{k}$ to $2\text{M}+$ rows across multiple real domains) is currently under-sampled!
6. **`SklearnGMM` normalization & iteration mismatch:** `SklearnGMM` regularizes `reg_covar=1e-6` in raw unstandardized units while PyTorch models regularize in standardized unit-variance space (producing $+13.15$ to $-106.96$ NLL gaps unrelated to optimization). Moreover, `covtype_581k` capped `SklearnGMM` at `max_iter=15` (`seed=0` only). Add a `StandardizedSklearnGMM` (`StandardScaler` + `GaussianMixture` with Jacobian correction) alongside raw `SklearnGMM`.
7. **Warmup & seeding wall-clock accounting:** `run_benchmark_suite.py:98-104` warms up CUDA kernels/CUDA-Graph before dataset loops, and passes explicit `device="cuda"` even on small datasets (`digits_pca32`, `segment`), whereas default users call `EGMM()` (`device="auto"`). Measure cold-start vs warm-start overhead and report seeding (`kmeans++`), EMA graph replay, revival, and polish time breakdowns separately.
8. **ARI / Likelihood trade-off uninvestigated:** `EGMM` lowers held-out NLL on 19/23 datasets yet scores lower mean ARI (`0.3879` vs `0.4254` for `OnlineGMM_tuned`). Test whether splitting high-mass Gaussian modes over-fragments unimodal ground-truth classes, and evaluate whether BIC / held-out density vs purity-weighted donor criteria behaves differently.

### C. Code & Systems Edge Cases (`src/egmm/`)
9. **Unchunked `diag` memory path (`kernels.py:62-73`):** `compute_log_gaussian_prob_from_precisions` chunks full-covariance GEMMs via `max_work_elements=32_000_000`, whereas the `diag` branch materializes full `(B, K)` intermediate `(Xc * Xc) @ prec.T` and `Xc @ (mc * prec).T` without chunking — dangerous when scoring (`predict_proba`, `score_samples`) or polishing millions of rows on GPU.
10. **`partial_fit` inconsistencies (`estimator.py:726-810`):** Standardizer `_x_mean_f64` / `_x_std_f64` is frozen from the very first `partial_fit` batch only, `_partial_total_steps` ignores dataset scale (`max(20, max_epochs * 10)`), `init_usage_scale` is ignored, and CUDA Graph is unused.
11. **Multi-epoch padding duplication (`estimator.py:415-426`):** `_build_epoch_batches` pads short remainder samples up to `steps_per_epoch * batch_size` by wrapping `perm[:pad]`. When `N` is slightly below a multiple of `batch_size`, a few rows can be weighted slightly differently if `steps_per_epoch` is small (worth checking exact unbiasedness / random permutation vs remainder handling).
12. **CUDA Graph safety on non-contiguous GPU tensor inputs & CPU fallback timing:** Verify `fit(X)` on pre-resident CUDA tensors (avoiding unnecessary host-device roundtrips in `_sync_public_attributes` if optional), multi-seed variance ($5+$ seeds on noisy datasets), and test `torch.float64` mode speed/accuracy.
13. **LaTeX Overleaf compile verification:** `paper/main.tex` passed structural/math/ref/CSV audits, **not** an actual `pdflatex` + `bibtex` build (no TeX binary installed locally). Compile on Overleaf or in a TeX Live environment and inspect every page, caption, and `Algorithm 1` / table break visually!

---

## 4. Recommended Next Research Plan

1. **Full Audit First:** Re-run `pytest -q`, review `src/egmm/*.py`, `experiments/*.py`, and `paper/main.tex` line-by-line against Theorem 1 / Proposition 1 and look for any numerical bugs, silent failure modes, or paper overclaims.
2. **Add Component Ablations + Fair Sklearn Baseline:** Measure `EGMM` vs `EGMM (no polish)`, `EGMM (no init_usage_scale)`, `EGMM (no revival)`, `OnlineGMM_tuned (with matched polish)`, and `StandardizedSklearnGMM`.
3. **Expand Medium & Large Datasets (Target 35–45 Total Datasets):** Pull additional real tabular and embedding datasets via OpenML/sklearn covering $N_{\text{train}} \in [30\text{k}, 2\text{M}]$, wide $D \in [64, 512]$ (`diag` + PCA-reduced `full`), and high $K \in [32, 256]$ (e.g., `higgs`, `susy`, `hepmass`, `poker`, `connect-4`, `kddcup99`, `skin_nonskin`, `airlines`, `jannis`, `nomao`, `gas_drift`, `sensorless_drive`). Include a proper `diag`-revival benchmark!
4. **Build True Meta-FAIR-Style Auto Scaling Laws for `e-GMM`:**
   - Expand `experiments/fit_scaling_laws.py` to sweep `(B, alpha_0, init_usage_scale, T, revive_points, polish_steps)` across `N in [2k .. 1M]`, both `diag` (with revival enabled!) and `full`, plus multi-regime data structures (isotropic vs anisotropic, imbalanced vs balanced).
   - Fit smooth closed-form laws `(B, alpha_0, alpha_min, init_usage_scale)` maximizing held-out Pareto optimality against a compute budget `T` on calibration benchmarks disjoint from evaluation datasets, and eliminate brittle hardcoded cutoffs cleanly.
5. **Fix Identified Systems Bugs & Update Tests:** Chunk `diag` scoring in `kernels.py`, resolve `diag` revival policy in `scaling.py`, address any `partial_fit` / edge-case findings, and extend `tests/`.
6. **Re-run All Benchmarks, Regenerate Figures, Update Paper & Compile PDF:** Refresh CSVs and figures, add the ablation table, expanded large-$N$ table, and fitted scaling-law section to `paper/main.tex`, compile PDF via Overleaf / `pdflatex` to inspect layout visually, and commit cleanly.

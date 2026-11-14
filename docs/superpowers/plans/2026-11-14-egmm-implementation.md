# e-GMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, test, empirically calibrate, and rigorously benchmark `EGMM`—a zero-tuning, GPU/CPU drop-in replacement for `sklearn.mixture.GaussianMixture` using EMA sufficient statistics, cosine decay, eigen-split component revival, and empirical scaling laws.

**Architecture:** `egmm` separates fast batched PyTorch tensor kernels (`kernels.py`), moment-preserving dead-component eigen-splitting (`revival.py`), scaling-law parameter selection (`scaling.py`), the `sklearn`-compatible `EGMM` estimator (`estimator.py`), and fair baselines (`baselines.py`). Experiments run in `experiments/` strictly separating scaling-law calibration from the 20+ dataset Pareto and Wilcoxon/CD evaluation suite.

**Tech Stack:** Python 3.14, PyTorch 2.14 (`cu132`, RTX 5070), NumPy, SciPy, scikit-learn, pytest, matplotlib, pandas.

**Spec:** `docs/superpowers/specs/2026-11-14-egmm-design.md`

## Global Constraints

- Strict TDD: write a failing `pytest` test before any production code.
- Zero test leakage: standardization and scaling-law calibration never touch evaluation test splits.
- Preserve `sklearn.mixture.GaussianMixture` drop-in compatibility (`fit`, `predict`, `predict_proba`, `score_samples`, `score`, `bic`, `aic`, `sample`, `weights_`, `means_`, `covariances_`, `precisions_cholesky_`).
- Report only observed command outputs, exact dataset names, and real statistical test results.

---

### Task 1: Project Scaffolding & Batched E-Step / Moment Kernels (`kernels.py`)

**Files:**
- Create: `src/egmm/__init__.py`
- Create: `src/egmm/kernels.py`
- Create: `tests/test_kernels.py`

**Interfaces:**
- Produces: `compute_log_gaussian_prob`, `e_step_batch`, `moments_to_params`, `params_to_moments`, `init_kmeans_pp`

- [ ] **Step 1: Write failing tests in `tests/test_kernels.py`** verifying:
  - `compute_log_gaussian_prob` matches `scipy.stats.multivariate_normal.logpdf` to `<1e-5` for both `diag` and `full` covariances.
  - `e_step_batch` returns exact normalized responsibilities summing to 1 and valid batch sufficient statistics `(m_bar, s_bar, q_bar)`.
  - `moments_to_params` + `params_to_moments` round-trip preserves PSD covariances even under extreme near-collinear inputs via `reg_covar`.
- [ ] **Step 2: Run `pytest tests/test_kernels.py` and confirm failure.**
- [ ] **Step 3: Implement `src/egmm/kernels.py` and `src/egmm/__init__.py`** using Cholesky precision caching and fast batched quadratic forms (`torch.einsum` / triangular solves) for both `diag` and `full`.
- [ ] **Step 4: Run `pytest tests/test_kernels.py` and verify all pass.**
- [ ] **Step 5: Profile E-step speed on RTX 5070 (`experiments/profile_kernels.py`)** comparing quadratic precision expansion vs triangular Cholesky solve so we pick the fastest exact kernel before building the solver.
- [ ] **Step 6: Commit.**

### Task 2: Moment-Preserving Eigen-Split Revival (`revival.py`)

**Files:**
- Create: `src/egmm/revival.py`
- Create: `tests/test_revival.py`

**Interfaces:**
- Consumes: `moments_to_params`, `params_to_moments` from `egmm.kernels`
- Produces: `revive_dead_components`

- [ ] **Step 1: Write failing tests in `tests/test_revival.py`** verifying:
  - Dead components (`M_k < dead_threshold / K`) are revived by splitting high-score donors along their principal eigenvector (`full`) or max-variance coordinate (`diag`).
  - Total mixture mass $\sum_k M_k$, total mixture mean $\sum_k S_k$, and total mixture second moment $\sum_k Q_k$ are preserved to `<1e-5` after splitting!
  - Handles edge case where multiple components die simultaneously (`len(dead) > len(donors)` or `K=1`).
- [ ] **Step 2: Run `pytest tests/test_revival.py` and confirm failure.**
- [ ] **Step 3: Implement `src/egmm/revival.py`.**
- [ ] **Step 4: Run `pytest tests/test_revival.py` and verify pass.**
- [ ] **Step 5: Commit.**

### Task 3: Scaling Laws & Cosine Schedules (`scaling.py`) + Drop-in `EGMM` Estimator (`estimator.py`)

**Files:**
- Create: `src/egmm/scaling.py`
- Create: `src/egmm/estimator.py`
- Create: `tests/test_estimator_api.py`

**Interfaces:**
- Consumes: `kernels.py`, `revival.py`
- Produces: `compute_cosine_alpha`, `auto_hyperparameters`, `EGMM`

- [ ] **Step 1: Write failing tests in `tests/test_estimator_api.py`** verifying:
  - Full `sklearn.mixture.GaussianMixture` API compatibility (`fit`, `fit_predict`, `predict`, `predict_proba`, `score_samples`, `score`, `bic`, `aic`, `sample`, `partial_fit`).
  - Exact parameter unscaling when data has large offsets (`mean=1e5`, `scale=1e3` or `1e-3`): recovered `means_` and `covariances_` match ground truth and `score(X_test)` matches `sklearn`!
  - Works on both CPU and CUDA (`RTX 5070`) for both `diag` and `full`.
  - Recovers anisotropic / imbalanced / overlapping synthetic mixtures with dead-component revival active (`revivals_count_ >= 0` and no collapsed components).
- [ ] **Step 2: Run `pytest tests/test_estimator_api.py` and confirm failure.**
- [ ] **Step 3: Implement `src/egmm/scaling.py` (initial prior heuristics, later updated by Task 5 empirical fit) and `src/egmm/estimator.py`.**
- [ ] **Step 4: Run `pytest tests/test_estimator_api.py` and verify pass.**
- [ ] **Step 5: Commit.**

### Task 4: Fair Baselines Suite (`baselines.py`)

**Files:**
- Create: `src/egmm/baselines.py`
- Create: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `kernels.py`
- Produces: `TorchBatchGMM`, `CappeMoulinesOnlineGMM`, `NaiveMinibatchGMM`, `EMeansInitGMM`

- [ ] **Step 1: Write failing tests in `tests/test_baselines.py`** verifying:
  - `TorchBatchGMM` produces monotonically non-decreasing lower bound (`score(X)`) and matches `sklearn.mixture.GaussianMixture` log-likelihood on synthetic data.
  - `CappeMoulinesOnlineGMM`, `NaiveMinibatchGMM`, and `EMeansInitGMM` implement unified `.fit(X)` / `.score(X)` / `.predict_proba(X)` APIs.
- [ ] **Step 2: Run `pytest tests/test_baselines.py` and confirm failure.**
- [ ] **Step 3: Implement `src/egmm/baselines.py`.**
- [ ] **Step 4: Run `pytest tests/test_baselines.py` and verify pass.**
- [ ] **Step 5: Commit.**

### Task 5: Out-of-Distribution Scaling-Law Discovery Study (`experiments/fit_scaling_laws.py`)

**Files:**
- Create: `experiments/fit_scaling_laws.py`

- [ ] **Step 1: Run systematic sweep** over synthetic Gaussian mixtures spanning $N \in [5\text{k}, 500\text{k}]$, $K \in [8, 128]$, $D \in [8, 128]$, epochs $T \in [1, 10]$, batch sizes $B \in [256, 16384]$, initial decay $\alpha_0 \in [0.02, 0.8]$ on dedicated calibration seeds (disjoint from evaluation suite).
- [ ] **Step 2: Fit empirical power-law formulas** for $B^*(N, K, D)$ and $\alpha_0^*(N, K, D, T, B)$ via non-linear least squares (`scipy.optimize.curve_fit`), measure out-of-sample $R^2$ on held-out calibration configs, and plug the fitted constants into `src/egmm/scaling.py`.
- [ ] **Step 3: Re-run full unit tests (`pytest -v`) and commit.**

### Task 6: Rigorous 20+ Dataset Verification, Ablations, Pareto & Stats (`experiments/run_benchmark_suite.py`, `experiments/plot_results.py`)

**Files:**
- Create: `experiments/run_benchmark_suite.py`
- Create: `experiments/plot_results.py`
- Create: `docs/experiments/egmm_findings.md`

- [ ] **Step 1: Assemble 20+ diverse datasets** (8 hard synthetic regimes + 12+ real `sklearn` / `fetch_openml` datasets across Small/Medium/Large scales and numerical/high-D regimes), all using strict train/val/test splits.
- [ ] **Step 2: Run nested / validation-tuned baseline sweeps** for `CappeMoulinesOnlineGMM` and time-budget Pareto sweeps across all methods (`SklearnGMM`, `TorchBatchGMM`, `OnlineGMM`, `NaiveMinibatchGMM`, `EMeansInitGMM`, `EGMM`).
- [ ] **Step 3: Audit failure modes!** If `EGMM` underperforms on any regime (e.g., high-$D$ `full`, small-$N$ overhead, CPU fallback speed), diagnose root cause, refine algorithm/kernels, re-test with TDD, and re-run benchmarks.
- [ ] **Step 4: Compute paired Wilcoxon signed-rank tests and Friedman + Nemenyi Critical Difference ranks**, generate publication plots, and write `docs/experiments/egmm_findings.md`.
- [ ] **Step 5: Run final verification (`pytest -v` + benchmark integrity checks) and commit.**

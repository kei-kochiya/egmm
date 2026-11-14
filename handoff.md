# Researcher Handoff: `e-GMM` Complete Audit, Wedderburn Revival, Scaling Laws & 33-Dataset Verification

**Repository:** `https://github.com/kei-kochiya/egmm` (branch `main`)  
**Workspace:** `d:\nckh`  
**Status:** All 13 prior audit items resolved and verified. Unit tests pass (`34 passed`), 33-dataset primary benchmark + 5 fairness ablations + 8-dataset cold/warm runtime phase breakdown + 16-dataset external small benchmark + 11 standalone single-panel figures + compiled PDF (`paper/main.tex` -> `paper/main.pdf`, 11 pages, zero overfull boxes) are complete and verified.

---

## 1. Summary of Resolved Audit Items & Algorithmic Upgrades

1. **Wedderburn Rank-1 Eigen-Split Revival (`src/egmm/revival.py`, Theorem 1 in `paper/main.tex`):**
   - Upgraded CUDA 5-step power iteration from $\boldsymbol{\delta} = \Delta \sqrt{\mathbf{v}^\top \boldsymbol{\Sigma}_d \mathbf{v}}\,\mathbf{v}$ (which could violate PSD when $\mathbf{v}$ is an approximate eigenvector) to the **Wedderburn rank-1 reduction vector** $\boldsymbol{\delta} = \Delta \frac{\boldsymbol{\Sigma}_d \mathbf{v}}{\sqrt{\mathbf{v}^\top \boldsymbol{\Sigma}_d \mathbf{v}}}$, guaranteeing exact Loewner PSD conservation $\boldsymbol{\Sigma}_d - \boldsymbol{\delta}\boldsymbol{\delta}^\top \succeq (1-\Delta^2)\boldsymbol{\Sigma}_d \succeq 0$ for *any* approximate direction $\mathbf{v}$.
   - Fixed `diag` donor selection spread to use full covariance trace $\mathrm{tr}(\boldsymbol{\Sigma}_d) = \sum_j \Sigma_{d,jj}$ instead of $\max_j \Sigma_{d,jj}$, and enabled `diag` dead-component revival (`rev_points = 2` in early $[0.15, 0.45]$ window).
2. **Memory-Bounded Chunking for `diag` & Full-Batch Polish/Scoring (`src/egmm/kernels.py`, `src/egmm/estimator.py`):**
   - Added automatic workspace chunking (`max_work_elements = 32_000_000`) to the `diag` branch of `compute_log_gaussian_prob_from_precisions` (using chunk-invariant `shift = means.mean(dim=0, keepdim=True)`) and `e_step_batch`, plus chunked full-batch polish (`polish_chunk = 32768`) and chunked `score_samples`/`predict_proba`/`predict`.
   - Fixed `_build_epoch_batches` to draw an unbiased prefix of length `steps_per_epoch * batch_size` without duplicating remainder samples.
3. **Decoupled Degrees-of-Freedom Scaling Law & Explicit Ablation Controls (`src/egmm/scaling.py`, `src/egmm/estimator.py`, `src/egmm/baselines.py`):**
   - Exposed `polish_steps: int | str = "auto"` and `init_usage_scale: float | str = "auto"` on `EGMM`, `CappeMoulinesOnlineGMM`, and `NaiveMinibatchGMM`, removing the `type(self) is EGMM` gate.
   - Added `StandardizedSklearnGMM` (`StandardScaler` + `GaussianMixture` + exact log-Jacobian adjustment) to isolate coordinate standardization from optimization.
   - Added smooth degrees-of-freedom scaling for `diag` ($\gamma_{\mathrm{df,diag}} = B / (K\sqrt{D})$) and calibrated `experiments/fit_scaling_laws.py` with `polish_steps=0` across 10 disjoint synthetic mixtures ($N$ up to $350\text{k}$).

---

## 2. Verified 33-Dataset Benchmark & Fairness Ablation Summary

- **All 33 Primary Datasets ($n=33, k=7$, 3 seeds each, `docs/experiments/stats_summary.txt`):**
  - `EGMM` vs `TorchBatchGMM`: **26/0/7** in Test NLL ($\text{med}\,\Delta\text{NLL} = -0.4288$, $p = 6.36 \times 10^{-6}$), **30/33** faster ($\text{med} = 2.68\times$, up to $8.26\times$, $p = 3.45 \times 10^{-7}$).
  - `EGMM` vs `OnlineGMM_tuned`: **28/0/5** in Test NLL ($\text{med}\,\Delta\text{NLL} = -0.9848$, $p = 2.11 \times 10^{-7}$).
  - `EGMM` vs `NaiveMinibatchGMM` / `EMeansInitGMM`: **32/0/1** ($p = 7.68 \times 10^{-9}$) / **32/0/1** ($p = 1.63 \times 10^{-8}$).
  - `EGMM` vs `StandardizedSklearnGMM` (CPU): **17/0/16** ($\text{med}\,\Delta\text{NLL} = -0.0366$, $p = 0.313$), **$48.21\times$ median speedup** ($p = 2.33 \times 10^{-10}$).
  - `EGMM` vs `SklearnGMM` (Raw CPU): **14/0/19** ($\text{med}\,\Delta\text{NLL} = +0.0378$, $p = 0.860$), **$57.86\times$ median speedup**.
  - Friedman $\chi^2_F = 80.922$ ($p = 2.30 \times 10^{-15}$), Nemenyi $\mathrm{CD}_{0.05} = 1.568$; mean NLL ranks: `EGMM = 2.485` (#1), `StandardizedSklearnGMM = 2.576`, `SklearnGMM = 3.030`, `TorchBatchGMM = 4.030`, `OnlineGMM_tuned = 4.485`, `EMeansInitGMM = 5.394`, `NaiveMinibatchGMM = 6.000`.
- **Large-Scale Zero-Polish Subset ($N_{\text{train}} > 100\text{k}, n=7$, `polish_steps == 0`):**
  - `EGMM` vs `TorchBatchGMM`: **6/0/1** ($\text{med}\,\Delta\text{NLL} = -1.0330$), **$3.93\times$ median speedup**.
  - `EGMM` vs `OnlineGMM_tuned`: **6/0/1** ($\text{med}\,\Delta\text{NLL} = -1.5134$), **$1.06\times$ median speedup**.
  - `EGMM` vs `StandardizedSklearnGMM`: **$161.55\times$ median speedup**.
- **Fairness Ablations ($n=33$, `docs/experiments/benchmark_ablations.csv`):**
  - `EGMM` vs `EGMM_no_revival`: **26/4/3** ($\text{med}\,\Delta\text{NLL} = -0.5132$, $p = 7.58 \times 10^{-6}$, mean dead components `0.05` vs `4.49`).
  - `EGMM` vs `OnlineGMM_matched_polish`: **26/0/7** ($\text{med}\,\Delta\text{NLL} = -0.5962$, $p = 2.30 \times 10^{-6}$, $1.22\times$ speedup).
  - `EGMM_polish0` vs `TorchBatchGMM`: **21/0/12** ($\text{med}\,\Delta\text{NLL} = -0.2591$, $p = 0.0491$, $2.30\times$ speedup).
  - `EGMM_polish0` vs `OnlineGMM_tuned`: **25/0/8** ($\text{med}\,\Delta\text{NLL} = -0.5313$, $p = 3.73 \times 10^{-4}$).

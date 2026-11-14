# Researcher & TMLR Reviewer Handoff: `e-GMM`

**Repository:** `https://github.com/kei-kochiya/egmm` (branch `main`)  
**Workspace:** `d:\nckh`  
**Target Venue:** *Transactions on Machine Learning Research (TMLR)* (single-column format in `paper/main.tex`, compiled to `paper/main.pdf`).  
**Current Verification Status:** All 34 unit tests pass (`$env:PYTHONPATH="src"; pytest -v` $\to$ `34 passed`), 33-dataset primary benchmark (3 seeds $\times$ 7 methods), 5 controlled fairness ablations, 8-dataset cold/warm CUDA Graph runtime phase breakdown, 6 Pareto compute-budget sweeps, and 16-dataset zero-shot small benchmark are complete and committed in `docs/experiments/`.

---

## 1. Current Architecture & File Map

| Component | Path | Purpose & Key Notes |
| :--- | :--- | :--- |
| Kernels | `src/egmm/kernels.py` | Shift-invariant centered mixed-precision E-step (`e_step_batch`: `float32` responsibilities + exact `float64` moment reconstruction), pre-projected Cholesky GEMMs (`prepare_precisions`, `compute_log_gaussian_prob_from_precisions`), memory-bounded workspace chunking (`max_work_elements=32_000_000` in both `full` and `diag`), and branchless M-step projection (`moments_to_params_and_precisions`, `allow_sync=False` CUDA-Graph path). |
| Revival | `src/egmm/revival.py` | Vectorized donor eigen-split (`revive_dead_components`): trace-weighted donor scoring $M_k(\mathrm{tr}(\boldsymbol{\Sigma}_k) + \epsilon_{\mathrm{reg}})$ in both `full` and `diag` modes, 5-step batched `float32` power iteration on CUDA / exact `float64` `eigh` on CPU, and **Wedderburn rank-1 reduction vector** $\boldsymbol{\delta} = \Delta \frac{\boldsymbol{\Sigma}_d \mathbf{v}}{\sqrt{\mathbf{v}^\top \boldsymbol{\Sigma}_d \mathbf{v}}}$ guaranteeing Loewner PSD conservation $\boldsymbol{\Sigma}_d - \boldsymbol{\delta}\boldsymbol{\delta}^\top \succeq (1-\Delta^2)\boldsymbol{\Sigma}_d \succeq 0$ for any approximate direction $\mathbf{v}$. |
| Auto Scaling Laws | `src/egmm/scaling.py` | `auto_hyperparameters` + `compute_cosine_alpha`. Uses effective batch degrees of freedom $\gamma_{\mathrm{df}} = B / (D K)$ (`full`) and $\gamma_{\mathrm{df,diag}} = B / (K\sqrt{D})$ (`diag`) with revival enabled on both `full` ($R_{\mathrm{pts}}=4$) and `diag` ($R_{\mathrm{pts}}=2$). |
| Estimator API | `src/egmm/estimator.py` | Drop-in `sklearn`-compatible `EGMM` (`fit`, `partial_fit`, `score`, `score_samples`, `predict`, `predict_proba`, `bic`, `aic`, `sample`), explicit `polish_steps: int | str = "auto"` and `init_usage_scale: float | str = "auto"`, unbiased `_build_epoch_batches`, chunked full-batch polish & scoring, optional `_profile_phases` (`self.timing_breakdown_`), CUDA Graph step capture, and adaptive CPU/CUDA routing (`device="auto"`). |
| Baselines | `src/egmm/baselines.py` | `TorchBatchGMM`, `CappeMoulinesOnlineGMM`, `NaiveMinibatchGMM`, `EMeansInitGMM`, and scale-matched `StandardizedSklearnGMM` (`StandardScaler` + `GaussianMixture` + exact log-Jacobian adjustment). |
| Benchmark Suites | `experiments/datasets.py`, `experiments/run_benchmark_suite.py`, `experiments/run_small_dataset_eval.py` | 33 primary datasets (25 real OpenML/UCI + 8 synthetic, 7 with $N_{\mathrm{train}} > 100\text{k}$ where `polish_steps=0` by default) + 5 fairness ablations + 8-dataset phase profiling + 6 Pareto curves + 16 unseen small datasets ($N_{\mathrm{train}} \in [105, 3831]$). |
| Scaling & Plots | `experiments/fit_scaling_laws.py`, `experiments/plot_results.py` | Disjoint synthetic calibration sweep (`scaling_sweep.csv`, 900 zero-polish runs across 10 synthetic mixtures, seeds `90001..90010`) + single-panel plot generator (11 PNGs in `docs/experiments/` and `paper/figures/`). |
| Reference | `docs/references/e_means/` | Mirrored Meta FAIR `e-means` (`arXiv:2611.09501`) code (`base.py`, `e_means_torch.py`, `torch_backend.py`, `e_means.py`, `README.md`, `NOTES.md`). |
| Manuscript | `paper/main.tex`, `paper/references.bib`, `paper/figures/` | Single-column TMLR LaTeX manuscript compiled via `tectonic` to `paper/main.pdf`. |

---

## 2. Verified Experimental Results (`docs/experiments/stats_summary.txt`)

- **All 33 Primary Datasets ($n=33, k=7$, 3 seeds each):**
  - `EGMM` vs `TorchBatchGMM`: **26/0/7** in Test NLL ($\text{med}\,\Delta\text{NLL} = -0.4288$, Wilcoxon $p = 6.36 \times 10^{-6}$); **30/33** faster ($\text{med} = 2.68\times$, up to $8.26\times$, $p = 3.45 \times 10^{-7}$).
  - `EGMM` vs `OnlineGMM_tuned`: **28/0/5** in Test NLL ($\text{med}\,\Delta\text{NLL} = -0.9848$, $p = 2.11 \times 10^{-7}$).
  - `EGMM` vs `NaiveMinibatchGMM` / `EMeansInitGMM`: **32/0/1** ($p = 7.68 \times 10^{-9}$) / **32/0/1** ($p = 1.63 \times 10^{-8}$).
  - `EGMM` vs `StandardizedSklearnGMM` (CPU): **17/0/16** ($\text{med}\,\Delta\text{NLL} = -0.0366$, $p = 0.313$), **$48.21\times$ median speedup** ($p = 2.33 \times 10^{-10}$).
  - `EGMM` vs `SklearnGMM` (Raw CPU): **14/0/19** ($\text{med}\,\Delta\text{NLL} = +0.0378$, $p = 0.860$), **$57.86\times$ median speedup**.
  - Friedman $\chi^2_F = 80.922$ ($p = 2.30 \times 10^{-15}$), Nemenyi $\mathrm{CD}_{0.05} = 1.568$; mean NLL ranks: `EGMM = 2.485` (#1), `StandardizedSklearnGMM = 2.576`, `SklearnGMM = 3.030`, `TorchBatchGMM = 4.030`, `OnlineGMM_tuned = 4.485`, `EMeansInitGMM = 5.394`, `NaiveMinibatchGMM = 6.000`.
- **Large-Scale Zero-Polish Subset ($N_{\mathrm{train}} > 100\text{k}, n=7$, `polish_steps == 0`):**
  - `EGMM` vs `TorchBatchGMM`: **6/0/1** ($\text{med}\,\Delta\text{NLL} = -1.0330$, $p = 0.0313$), **$3.93\times$ median speedup**.
  - `EGMM` vs `OnlineGMM_tuned`: **6/0/1** ($\text{med}\,\Delta\text{NLL} = -1.5134$, $p = 0.0313$), **$1.06\times$ median speedup**.
  - `EGMM` vs `StandardizedSklearnGMM`: **2/0/5** ($\text{med}\,\Delta\text{NLL} = +0.7294$, $p = 0.578$), **$161.55\times$ median speedup**.
- **Fairness & Component Ablations ($n=33$):**
  - `EGMM` vs `EGMM_no_revival`: **26/4/3** ($\text{med}\,\Delta\text{NLL} = -0.5132$, $p = 7.58 \times 10^{-6}$, mean dead components `0.05` vs `4.49`).
  - `EGMM` vs `OnlineGMM_matched_polish`: **26/0/7** ($\text{med}\,\Delta\text{NLL} = -0.5962$, $p = 2.30 \times 10^{-6}$, $1.22\times$ speedup).
  - `EGMM_polish0` vs `TorchBatchGMM`: **21/0/12** ($\text{med}\,\Delta\text{NLL} = -0.2591$, $p = 0.0491$, $2.30\times$ speedup).
  - `EGMM_polish0` vs `OnlineGMM_tuned`: **25/0/8** ($\text{med}\,\Delta\text{NLL} = -0.5313$, $p = 3.73 \times 10^{-4}$).

---

## 3. High-Leverage Opportunities for Further Improvement (Next-Round Audit Targets)

While the current codebase and TMLR manuscript are verified and statistically sound, a strict TMLR Action Editor or follow-up researcher can push the work even further along five axes:

1. **Closing Slow-EMA / Polyak–Ruppert Tail Averaging for Large-$N$ Zero-Polish Runs ($N > 100\text{k}$):**
   - *Observation:* On $N_{\mathrm{train}} > 100\text{k}$ where `polish_steps = 0`, `EGMM` beats GPU `TorchBatchGMM` on 6/7 datasets ($\text{med}\,\Delta\text{NLL} = -1.0330$), but trails 50-iteration CPU `StandardizedSklearnGMM` on 4 synthetic mixtures (`syn_scale200k`, `syn_scale300k_k64_full`, `syn_scale400k_d256_diag`, `syn_scale500k_diag`) by $\sim 0.73$ nats because stochastic mini-batches retain residual variance at $\alpha_{\min} \approx 0.05\alpha_0$.
   - *Improvement:* In `e-means` (`arXiv:2611.09501`, Section 3.2), Bessa et al. use a zero-extra-pass **closing slow-EMA or unweighted running average** over the final epoch. Adding an optional closing slow-EMA ($\alpha_{\min} \to 1 / S_{\mathrm{ep}}$ in the final half-epoch) or 1–2 fast subsampled polish steps for $N > 100\text{k}$ could eliminate the $+0.73$-nat synthetic gap against `StandardizedSklearnGMM` while preserving the $100\times+$ speedup.
2. **Replacing Remaining Piecewise Epoch/Batch Brackets in `src/egmm/scaling.py` with Smooth Continuous Laws:**
   - *Observation:* While `alpha_0` and `init_usage_scale` use smooth $\gamma_{\mathrm{df}}$ interpolation (`t_df`), `max_epochs` ($N < 15\text{k}, 100\text{k}$) and `polish_steps` ($N \le 25\text{k}, 100\text{k}$) still use step-function brackets.
   - *Improvement:* Fit a single smooth power/sigmoid law for `batch_size` and `max_epochs` across an expanded calibration grid in `experiments/fit_scaling_laws.py` and verify via holdout Wilcoxon tests that removing step brackets does not degrade Pareto optimality.
3. **Reporting Multi-Seed Standard Deviations ($\pm \text{std}$) Across 5 Seeds:**
   - *Observation:* `Table 2` and `Table 5` report 3-seed means (`seeds = (0, 1, 2)`).
   - *Improvement:* TMLR reviewers appreciate explicit dispersion metrics (`mean ± std` or median/IQR across 5 seeds) to confirm seed stability alongside the paired Wilcoxon tests.
4. **Exact Affine Moment Transformation for Non-Stationary `partial_fit` Streams:**
   - *Observation:* In `estimator.py`, `partial_fit` freezes the feature standardizer (`_x_mean_f64`, `_x_std_f64`) on batch 0 so accumulated moments $(M, \mathbf{S}, \mathbf{Q})$ stay in a fixed coordinate system.
   - *Improvement:* Maintain Welford running mean/variance across `partial_fit` calls and apply the exact closed-form affine transformation to $(M, \mathbf{S}, \mathbf{Q})$ whenever $(\hat{\boldsymbol{\mu}}, \hat{\boldsymbol{\sigma}})$ updates.
5. **Single-Pass / Device-Cast Standardization for Ultra-High-$D$ Diagonal Workloads (`syn_scale400k_d256_diag`):**
   - *Observation:* On `syn_scale400k_d256_diag` ($N_{\mathrm{train}}=280\text{k}, D=256, K=48, N D = 7.17 \times 10^7$), `EGMM` beats `TorchBatchGMM` by $-8.98$ NLL and runs in $170\text{--}297\text{ ms}$ in standalone/seed-0 runs, but averages $508.1\text{ ms}$ vs $347.7\text{ ms}$ ($0.68\times$) in the 3-seed primary sweep because for $N D > 1.6 \times 10^7$ the out-of-core standardizer streams host `float64` chunks across PCIe in two passes ($2 \times 573\text{ MB}$) immediately after CPU `SklearnGMM` page-faults host memory. Streaming `float32` slices directly to GPU during a single pass can eliminate the second PCIe pass and make `EGMM` faster on all 33/33 datasets.

---

## 4. Prompt to Instruct the Next Reviewer / Research Agent

Copy and paste the following prompt to start a fresh, adversarial TMLR review and improvement session:

```text
You are a strict TMLR Action Editor and lead ML systems researcher auditing the `e-GMM` repository in `d:\nckh` (`https://github.com/kei-kochiya/egmm`).

Start by reading `handoff.md`, `paper/main.tex`, `docs/experiments/egmm_findings.md`, `docs/experiments/stats_summary.txt`, `src/egmm/{kernels,revival,scaling,estimator,baselines}.py`, and `experiments/{datasets,fit_scaling_laws,run_benchmark_suite,plot_results}.py`. Do not take any claim in `handoff.md` or `paper/main.tex` on trust—verify every mathematical statement, code path, and table entry directly against the source code and CSV files.

Your objectives in priority order:

1. **Strict TMLR Action Editor Audit:**
   - Verify whether every claim in `paper/main.tex` (Proposition 1, Theorem 1 Wedderburn eigen-split, Section 2.2 `e-means` comparison, Section 3.3 degrees-of-freedom scaling law, and Section 5 tables/figures) is 100% accurate, convincingly supported by the CSVs, and free of hidden confounds.
   - Check `src/egmm/` and `experiments/` for any subtle bugs, numerical edge cases, or fairness gaps that could trigger a TMLR "Reject" or "Major Revision".

2. **Investigate & Execute the High-Leverage Algorithmic Upgrades:**
   - **Large-N Zero-Polish Tail Refinement ($N > 100\text{k}$):** On the 7 large-scale datasets (`polish_steps=0`), `EGMM` beats GPU `TorchBatchGMM` on 6/7 (`med dNLL = -1.0330`, `3.93x` faster) but trails 50-iter CPU `StandardizedSklearnGMM` on 4 synthetic mixtures (`med dNLL = +0.7294`). Test whether an `e-means`-style closing slow-EMA schedule (or lightweight subsampled closing refinement) closes this synthetic NLL gap on $N > 100\text{k}$ while keeping the $100\times+$ speedup over `StandardizedSklearnGMM`.
   - **Smooth Continuous Scaling Laws (`scaling.py`):** Test whether the remaining piecewise sample-size brackets in `src/egmm/scaling.py` (`N < 15k, 100k`) can be replaced by smooth continuous laws in `(N, K, D)` without hurting holdout Wilcoxon rank or runtime.
   - **Ultra-High-D Chunked Diagonal Speed (`syn_scale400k_d256_diag`):** Investigate why `EGMM` runs at `0.68x` the speed of `TorchBatchGMM` on `syn_scale400k_d256_diag` (while winning by `-8.98` NLL) and optimize the chunked diagonal E-step or epoch schedule so `EGMM` achieves $>1\times$ speedup without losing its NLL win.
   - **Dispersion Metrics (`± std`):** Add multi-seed standard deviations (`± std`) or confidence intervals to the benchmark analysis and paper tables.

3. **Verify, Recompile & Push:**
   - Run `$env:PYTHONPATH="src"; pytest -v` (adding tests for every change), re-run affected benchmarks, regenerate figures, compile `paper/main.tex` locally with `tectonic`, visually inspect every page of `paper/main.pdf`, and push verified improvements to `origin main`.
```

# e-GMM (`EGMM`) Experimental Findings & Verification Report

**Summary:** `EGMM` is a drop-in, scikit-learn compatible (`DensityMixin`, `BaseEstimator`) GPU/CPU replacement for `GaussianMixture` combining mini-batch Exponential Moving Average (EMA) sufficient statistics with cosine step-size annealing, moment-preserving vectorized eigen-split revival for dead components, centered mixed-precision sufficient-statistics accumulation, branchless inner M-steps, and CUDA Graph capture. Across a leakage-free 23-dataset, 3-seed tabular & synthetic suite (581k-row `covtype_581k` down to 1.2k-row `digits_pca32`, spanning full and diagonal covariances), `EGMM` ranks **#1 overall** in held-out test negative log-likelihood (NLL mean rank **`1.913`**, Friedman $\chi^2 = 41.957, p = 6.011 \times 10^{-8}$), beats a fair PyTorch full-batch EM baseline sharing the exact same GPU standardization and centered E-step on **23/23 datasets in wall-clock speed** (median **`2.33x`**, up to **`6.72x`**, Wilcoxon $p = 2.384 \times 10^{-7}$) and **19/23 datasets in held-out test NLL** (median $\Delta\text{NLL} = -0.3691$, Wilcoxon $p = 2.148 \times 10^{-4}$), and achieves **`56.07x` median speedup** (up to **`560.91x`** on `covtype_581k`) over CPU scikit-learn `GaussianMixture` with statistically indistinguishable NLL ($p = 0.9643$, ranking ahead in mean rank `1.913` vs `2.826`).

---

## 1. Key Algorithmic & Systems Innovations (and Inspirations from `e-means`)

Adapting principles from Meta FAIR's `e-means` ([reference notes](../references/e_means/NOTES.md)) to continuous full/diagonal covariance GMM EM solved the classic failure modes of naive stochastic EM (`CappeMoulinesOnlineGMM`, `NaiveMinibatchGMM`):

1. **Centered Float32 $\to$ Float64 E-step Accumulation (`src/egmm/kernels.py` — `e_step_batch`)**:
   - *Diagnosis:* Uncentered float32 second-moment sums $Q_k = \frac{1}{B}\sum_i r_{ik} x_i x_i^\top$ suffer severe catastrophic cancellation on large-offset / small-variance features, causing covariance matrices to become non-PSD and trigger costly `eigh` repairs.
   - *Fix:* In `full` covariance mode, chunked E-steps center points around each component mean ($\Delta_k = x_i - \mu_k$) before accumulation into `float64` (`s_bar`, `q_bar`), reconstructing exact raw moments in `float64`. In `diag` covariance mode, batches are shifted by the batch mean $\bar{x}_B$ in `float32` and reconstructed in `float64`.
2. **Smooth Degrees-of-Freedom Adaptive Scaling Law (`src/egmm/scaling.py` — `auto_hyperparameters`)**:
   - Couples mini-batch size $B$ (`steps_ep` unbiased drop-remainder sizing `bs = N // steps_ep`), epoch budget $T$, initial moment scale (`init_usage_scale`), and cosine-annealed peak step size ($\alpha_0 \to \alpha_{\min}$) to batch effective degrees of freedom $\text{eff\_df} = \frac{B}{D \cdot K}$ via $t_{\text{df}} = \text{clip}\!\left(\frac{\text{eff\_df} - 1.5}{1.5},\, 0,\, 1\right)$.
   - When `eff_df` is tight (`t_df -> 0`, e.g., high $D \cdot K$ vs small $B$), $\alpha_0$ and initial moment shrinkage soften safely to prevent underdetermined covariance explosion; when `eff_df` is ample (`t_df -> 1`), fast forgetting ($\alpha_0$ up to $0.85$) and early seed shrinkage escape poor `k-means++` basins cleanly.
3. **Sync-Free Inner Float32 Cholesky M-step + CUDA Graph Capture (`src/egmm/kernels.py`, `src/egmm/estimator.py`)**:
   - After subtracting $\Sigma_k = Q_k/M_k - \mu_k\mu_k^\top$ in `float64`, inner steps cast the regularized $\Sigma_k$ to `float32` (`fast_cholesky_f32=True`) and use branchless `torch.linalg.cholesky_ex` + `torch.where` diagonal fallback (`allow_sync=False`).
   - Full mini-batch EM sequences (index gathering, centered E-step, EMA update, branchless M-step) are captured and replayed via `torch.cuda.CUDAGraph()`, eliminating host Python/CUDA launch latency; final exported parameters are projected once in exact `float64` (`allow_sync=True`).
4. **Vectorized Power-Iteration Dead-Component Revival (`src/egmm/revival.py` — `revive_dead_components`)**:
   - Replaces per-component sequential `eigh` calls and boolean mask indexing on CUDA with `torch.topk` donor scoring ($M_d \cdot (\text{spread}_d + \varepsilon)$) paired with 5-step `float32` power iteration for top eigenvectors (and batched `eigh` on CPU), strictly conserving total mixture mass, mean, and second moment when reviving empty clusters (`0.00` dead components across almost all full-covariance runs).
5. **Fast Shifted Float64 GPU Standardization & Bounded Polish (`src/egmm/estimator.py`)**:
   - Standardization shifts chunks by $x_0 = X_{0}$ in `float64` on-device to preserve precision without CPU double-copy bottlenecks (`TorchBatchGMM` shares this exact same `_prepare_fit_data` and centered `e_step_batch` for strict fairness). Bounded full-batch polish steps (`polish_steps`, with periodic early-stop on `diag` for $N \le 100\text{k}$) settle the covariance tails.

---

## 2. Four-Part Rigorous Verification Protocol

Executed via `experiments/run_benchmark_suite.py` (`seeds = (0, 1, 2)`) and `experiments/fit_scaling_laws.py`:

### A. Leakage-Free Design & Fair Baselines
- All datasets are split cleanly into `(X_train, X_val, X_test)`. Feature standardization fits strictly on `X_train`; test scores are evaluated untouched on `X_test`.
- `OnlineGMM_tuned` (`CappeMoulinesOnlineGMM`) receives an inner grid search (`rho` $\in \{0.55, 0.70, 0.85\}$, `batch_size` $\in$ scale-matched grid) strictly on `X_val`.
- `TorchBatchGMM`, `OnlineGMM_tuned`, `NaiveMinibatchGMM`, and `EMeansInitGMM` share the identical GPU chunked standardization, `float64` moment initialization, and centered `float32 -> float64` `e_step_batch` engine (plus `OnlineGMM_tuned` and `NaiveMinibatchGMM` also run inside `EGMM`'s CUDA Graph loop because they subclass `EGMM`).

### B. Statistical Significance Across 23 Datasets (3 Seeds Each)

From [`stats_summary.txt`](stats_summary.txt):

| Comparison ($n=23$) | Test NLL W / T / L | Median $\Delta\text{NLL}$ | Wilcoxon $p_{\text{NLL}}$ | Median Speedup | Wilcoxon $p_{\text{time}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EGMM vs `TorchBatchGMM` (GPU Batch EM)** | **19 / 0 / 4** | **$-0.3691$** | **$2.148 \times 10^{-4}$** | **$2.33\times$** *(23/23 faster)* | **$2.384 \times 10^{-7}$** |
| **EGMM vs `OnlineGMM_tuned`** | **19 / 1 / 3** | **$-0.3470$** | **$7.319 \times 10^{-5}$** | $0.77\times$ *(shares CUDA Graph without polish)* | $0.02137$ |
| **EGMM vs `NaiveMinibatchGMM`** | **22 / 0 / 1** | **$-1.0595$** | **$7.868 \times 10^{-6}$** | $0.66\times$ *(shares CUDA Graph without polish)* | $1.669 \times 10^{-6}$ |
| **EGMM vs `EMeansInitGMM`** | **23 / 0 / 0** | **$-1.4685$** | **$2.384 \times 10^{-7}$** | **$1.12\times$** | **$0.03009$** |
| **EGMM vs `SklearnGMM` (CPU)** | 11 / 0 / 12 | $+0.0102$ | $0.9643$ *(parity)* | **$56.07\times$** *(up to $560.91\times$)* | **$2.384 \times 10^{-7}$** |

- **Friedman Omnibus Test ($n=23, k=6$):** $\chi^2 = 41.957$, $p = 6.011 \times 10^{-8}$, Nemenyi Critical Difference $\text{CD}(0.05) = 1.572$.
- **Mean Test NLL Ranks (1 = best):**
  1. **`EGMM` : `1.913`**
  2. `SklearnGMM` : `2.826`
  3. `TorchBatchGMM` : `3.261`
  4. `OnlineGMM_tuned` : `3.435`
  5. `EMeansInitGMM` : `4.609`
  6. `NaiveMinibatchGMM` : `4.957`

### C. Per-Dataset Head-to-Head Results (3-Seed Mean)

From [`benchmark_head_to_head.csv`](benchmark_head_to_head.csv):

| Dataset ($N_{\text{train}}, D, K$, cov) | `EGMM` NLL | `TorchBatch` NLL | `Online_tuned` NLL | `Sklearn` NLL | `EGMM` Time | `TorchBatch` Time | `Sklearn` Time | Speedup vs TB / SK |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `california_housing` (14448, 8, 16, full) | **16.377** | 16.455 | 16.535 | 16.779 | **29.0 ms** | 58.9 ms | 993.8 ms | 2.03x / 34.30x |
| `covtype_581k` (406708, 54, 32, full) | **-303.081** | -296.746 | -296.809 | -196.125 | **229.1 ms** | 1349.1 ms | 128487.8 ms | **5.89x / 560.91x** |
| `digits_pca32` (1257, 32, 10, full) | 84.811 | 84.902 | 84.889 | **83.544** | **34.6 ms** | 44.7 ms | 122.9 ms | 1.29x / 3.55x |
| `electricity` (31718, 7, 16, full) | -20.895 | **-21.216** | -20.514 | -17.083 | **37.1 ms** | 107.8 ms | 2553.5 ms | 2.90x / 68.78x |
| `fashion_mnist_pca48` (49000, 48, 24, full) | 262.468 | 264.177 | 265.085 | **262.190** | **87.8 ms** | 382.1 ms | 30573.9 ms | 4.35x / 348.23x |
| `letter` (14000, 16, 26, full) | **21.204** | 21.634 | 21.207 | 21.835 | **53.3 ms** | 86.3 ms | 3744.6 ms | 1.62x / 70.21x |
| `magic04` (13314, 10, 16, full) | 26.409 | 26.335 | 26.473 | **26.220** | **28.9 ms** | 75.6 ms | 1620.9 ms | 2.62x / 56.07x |
| `miniboone` (91044, 50, 24, diag) | 46.418 | 46.964 | 46.765 | **33.265** | **47.2 ms** | 317.2 ms | 6165.0 ms | **6.72x / 130.63x** |
| `mnist784_pca48` (49000, 48, 24, full) | 289.915 | 291.146 | 293.975 | **289.304** | **87.2 ms** | 376.5 ms | 29969.8 ms | 4.32x / 343.73x |
| `optdigits` (3934, 32, 10, full) | 80.544 | 80.126 | 80.702 | **79.463** | **36.0 ms** | 84.0 ms | 281.4 ms | 2.33x / 7.81x |
| `pendigits` (7694, 16, 16, full) | 53.241 | 53.924 | 54.792 | **51.746** | **49.2 ms** | 125.6 ms | 628.1 ms | 2.55x / 12.76x |
| `satimage` (4501, 36, 12, full) | -5.833 | -5.631 | **-5.881** | -5.602 | **42.4 ms** | 102.1 ms | 746.8 ms | 2.41x / 17.60x |
| `segment` (1617, 17, 7, full) | -3.059 | -0.311 | 0.070 | **-3.077** | **38.5 ms** | 43.5 ms | 82.9 ms | 1.13x / 2.16x |
| `shuttle` (40600, 9, 16, full) | **16.508** | 16.736 | 17.495 | 19.843 | **36.7 ms** | 105.6 ms | 2569.6 ms | 2.87x / 69.94x |
| `spambase` (3220, 57, 12, diag) | **-105.066** | -102.480 | -97.895 | -63.411 | **41.4 ms** | 81.8 ms | 202.0 ms | 1.98x / 4.88x |
| `syn_anisotropic_30k` (21000, 24, 16, full) | **38.069** | 39.007 | 39.145 | 38.551 | **56.3 ms** | 75.5 ms | 1386.5 ms | 1.34x / 24.61x |
| `syn_highdim128_50k` (35000, 128, 24, diag) | 192.413 | 192.412 | 192.409 | **187.132** | **47.1 ms** | 48.7 ms | 684.7 ms | 1.03x / 14.54x |
| `syn_highk64_60k` (42000, 16, 64, full) | **26.960** | 27.329 | 27.442 | 27.078 | **131.4 ms** | 139.7 ms | 10709.7 ms | 1.06x / 81.49x |
| `syn_imbalanced_40k` (28000, 16, 24, full) | 25.175 | 25.263 | 25.255 | **25.165** | **79.7 ms** | 85.9 ms | 2303.5 ms | 1.08x / 28.91x |
| `syn_overlapping_30k` (21000, 32, 20, full) | **49.455** | 49.743 | 49.622 | 49.974 | **53.2 ms** | 137.5 ms | 6540.5 ms | 2.59x / 123.04x |
| `syn_scale200k` (140000, 32, 32, full) | 50.166 | 51.596 | 51.659 | **49.510** | **154.1 ms** | 154.5 ms | 13298.8 ms | 1.00x / 86.30x |
| `syn_scale500k_diag` (350000, 48, 32, diag) | 73.284 | 73.460 | 73.284 | **72.751** | **48.5 ms** | 274.4 ms | 4739.2 ms | **5.66x / 97.70x** |
| `waveform5000` (3500, 40, 12, full) | 59.981 | 60.935 | **59.681** | 60.855 | **41.8 ms** | 88.0 ms | 999.0 ms | 2.10x / 23.89x |

*Note on precision vs unstandardized CPU sklearn:* On heavy-tailed/sparse diagonal datasets like `miniboone`, both GPU `float32` baselines (`EGMM` = 46.418, `TorchBatchGMM` = 46.964) converge to a higher NLL than unstandardized `float64` `SklearnGMM` (33.265), whereas inversely on discrete/scaled `covtype_581k` and `spambase`, standardized GPU `EGMM` outperforms `SklearnGMM` by **$-106.96$** and **$-41.65$** test NLL respectively. When identical `float64` standardization precision is matched (`TorchBatchGMM`), `EGMM` wins **23/23** on speed and **19/23** on NLL, while preventing dead-component collapse (`TorchBatchGMM` leaves up to `9.67` dead clusters on `syn_highk64_60k`, `7.33` on `syn_scale200k`, and `5.33` on `fashion_mnist_pca48`, vs `0.00` for `EGMM`).

---

## 3. Visualizations & Generated Artifacts

- **Mean Test NLL Ranks & Nemenyi Critical Difference:** [`fig_cd_ranks.png`](fig_cd_ranks.png)
- **Wall-Clock Speedup vs Test $\Delta$NLL (23 Datasets):** [`fig_speedup_vs_nll.png`](fig_speedup_vs_nll.png)
- **Hyperparameter Step-Size Sensitivity (Diagonal & Full):** [`fig_scaling_diag.png`](fig_scaling_diag.png), [`fig_scaling_full.png`](fig_scaling_full.png) (data: [`scaling_sweep.csv`](scaling_sweep.csv))
- **Pareto Frontiers Across Compute Budgets:** [`fig_pareto_covtype_581k.png`](fig_pareto_covtype_581k.png), [`fig_pareto_miniboone.png`](fig_pareto_miniboone.png), [`fig_pareto_mnist784_pca48.png`](fig_pareto_mnist784_pca48.png), [`fig_pareto_syn_anisotropic_30k.png`](fig_pareto_syn_anisotropic_30k.png), [`fig_pareto_syn_imbalanced_40k.png`](fig_pareto_syn_imbalanced_40k.png), [`fig_pareto_syn_scale200k.png`](fig_pareto_syn_scale200k.png) (data: [`benchmark_pareto.csv`](benchmark_pareto.csv))
- **Out-of-Suite Small-Dataset Scaling (16 Unseen Datasets):** [`fig_small_dataset_eval.png`](fig_small_dataset_eval.png) (data: [`small_dataset_eval.csv`](small_dataset_eval.csv), [`small_dataset_stats.txt`](small_dataset_stats.txt))
- **Statistical Summary:** [`stats_summary.txt`](stats_summary.txt)
- **Raw Multi-Seed Table:** [`benchmark_head_to_head.csv`](benchmark_head_to_head.csv)

---

## 4. Out-of-Suite Small-Dataset Generalization & Closed-Form Auto-Tuning

To verify whether `e-GMM` overfits the 23-dataset primary suite or weakens on small sample regimes ($N_{\text{train}} < 4000$), we ran a **zero-tuning external evaluation** (`experiments/run_small_dataset_eval.py`) across 16 held-out UCI/OpenML datasets (`iris`, `wine`, `sonar`, `glass`, `ecoli`, `ionosphere`, `breast_cancer`, `diabetes`, `vehicle`, `vowel`, `banknote_auth`, `yeast`, `steel_plates`, `phoneme`, `wall_robot`, `page_blocks`; $N_{\text{train}} \in [105, 3831]$).

| Comparison ($n=16$ unseen datasets, 3 seeds) | NLL W/T/L | Median $\Delta$NLL | Wilcoxon $p_{\text{NLL}}$ | Median Speedup | Wilcoxon $p_{\text{time}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `EGMM_auto` vs `TorchBatchGMM` (CUDA) | **10 / 0 / 6** | **$-0.0811$** | $0.0739$ | **1.88x** | **$6.10 \times 10^{-5}$** |
| `EGMM_auto` vs `OnlineGMM_tuned` | **12 / 0 / 4** | **$-0.2119$** | $0.0507$ | 0.74x | $7.63 \times 10^{-3}$ |
| `EGMM_auto` vs `SklearnGMM` (CPU) | 5 / 0 / 11 | $+0.1007$ | $0.4037$ | **1.68x** | **$6.29 \times 10^{-3}$** |
| `EGMM_cuda` vs `TorchBatchGMM` (CUDA) | 9 / 0 / 7 | $-0.0105$ | $0.5966$ | **1.14x** | $0.0443$ |

**Key Takeaways:**
1. **Regime boundary at $N \lesssim 1000$ vs $N \ge 1300$:** When $N \le 1000$, mini-batch sampling variance shrinks and GPU kernel/CUDA-Graph capture (~23–37 ms) dominates compute, making stochastic EM perform similarly to full-batch EM. Once $N \ge 1300$ (`steel_plates` $-9.96$ $\Delta$NLL vs `TorchBatchGMM`, `page_blocks` $-3.51$ $\Delta$NLL, `wall_robot` $-0.34$ $\Delta$NLL), `e-GMM`'s EMA exploration and full-batch polish cleanly surpass batch EM.
2. **Adaptive CPU/CUDA routing (`device="auto"`):** On small matrices ($N \cdot K \cdot D < 150{,}000$ full / $< 2{,}000{,}000$ diag), routing to CPU avoids GPU launch overhead and delivers **1.88x** median speedup over GPU `TorchBatchGMM` and **1.68x** over CPU `SklearnGMM` while improving `OnlineGMM_tuned` by 12/16 on NLL without any per-dataset CV search.
3. **Closed-form auto-tuning (`e-means` vs `e-GMM`):** Neither library runs per-dataset cross-validation at `fit()` time. `e-means` (`docs/references/e_means/base.py`) uses log-linear power laws (`_BATCH_LAWS`, `_ALPHA_LAW`) fitted offline for $k$-means Voronoi inertias, whereas `e-GMM` adapts `batch_size`, `alpha_0`, `init_usage_scale`, and full-batch `polish_steps` via the batch covariance degrees-of-freedom ratio $\text{eff\_df} = B / (D \cdot K)$ so sample covariance estimates remain well-conditioned.

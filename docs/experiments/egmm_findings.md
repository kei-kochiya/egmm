# e-GMM (`EGMM`) Experimental Findings, Fairness Ablations & Verification Report

**Summary:** `EGMM` is a drop-in, scikit-learn compatible (`DensityMixin`, `BaseEstimator`) GPU/CPU replacement for `GaussianMixture` combining mini-batch Exponential Moving Average (EMA) sufficient statistics with cosine step-size annealing, moment-preserving Wedderburn rank-1 eigen-split revival for dead components, centered mixed-precision sufficient-statistics accumulation with memory-bounded batch chunking, branchless inner M-steps, and CUDA Graph capture. Across an expanded leakage-free **33-dataset, 3-seed** tabular & synthetic suite ($N_{\text{train}}$ up to $406{,}708$, $D$ up to $561$, $K$ up to $64$, spanning 25 full and 8 diagonal covariance tasks), `EGMM` ranks **#1 overall** in held-out test negative log-likelihood (NLL mean rank **`2.485`** across 7 methods, Friedman $\chi^2 = 80.922, p = 2.304 \times 10^{-15}$), beats a shared-kernel PyTorch full-batch EM baseline (`TorchBatchGMM`) on **26/33 datasets in held-out test NLL** (median $\Delta\text{NLL} = -0.4288$, Wilcoxon $p = 6.357 \times 10^{-6}$) and **30/33 datasets in wall-clock speed** (median **`2.68x`**, up to **`8.26x`**, Wilcoxon $p = 3.450 \times 10^{-7}$), beats validation-tuned Cappé–Moulines Online EM (`OnlineGMM_tuned`) on **28/33 datasets** (median $\Delta\text{NLL} = -0.9848$, $p = 2.111 \times 10^{-7}$), and achieves a **`48.21x` median speedup** over scale-matched CPU `StandardizedSklearnGMM` (**`57.86x`** over raw `SklearnGMM`) at statistically indistinguishable NLL ($p = 0.3127$).

Crucially, controlled fairness ablations prove that even with **zero full-batch polish (`polish_steps=0`)**, pure stochastic `EGMM_polish0` significantly outperforms `TorchBatchGMM` (**21/33 NLL**, median $\Delta\text{NLL} = -0.2591$, $p = 0.0491$, **`2.30x` speedup**) and `OnlineGMM_tuned` (**25/33 NLL**, $p = 3.729 \times 10^{-4}$), and on the **7 large-scale datasets ($N_{\text{train}} > 100\text{k}$, where `polish_steps == 0` by default)** achieves **6/7 NLL wins** (median $\Delta\text{NLL} = -1.0330$) with a **`3.93x` median speedup** over `TorchBatchGMM` and **`161.55x` median speedup** over `StandardizedSklearnGMM`.

---

## 1. Theoretical & Algorithmic Alignment with `e-means` (`arXiv:2611.09501`)

Direct inspection of Bessa et al. (2026), *"`e-means`: outperforming `k-means` (in one-tenth the epochs)"* (`arXiv:2611.09501`) and `faiss/contrib/e_means` reveals four foundational connections and second-order generalizations in `e-GMM`:

1. **From Convex-Hull Invariance (`e-means` Appendix E) to Schur-Complement PSD-Cone Invariance (`e-GMM` Theorem 1)**:
   - Bessa et al. (Appendix E, Figs. 16–17) prove that mini-batch SGD and Adam collapse at large batch sizes / learning rates because unconstrained gradient steps leave the convex hull of assigned points, whereas EMA is a convex combination of sufficient statistics and stays strictly inside the convex hull for all $\alpha_t \in (0, 1]$.
   - In `e-GMM`, parameters live in the positive-definite covariance cone rather than $\mathbb{R}^D$. Applying EMA in raw sufficient statistics $\Theta_k = (M_k, \mathbf{S}_k, \mathbf{Q}_k)$ where each batch satisfies $M_k \mathbf{Q}_k - \mathbf{S}_k \mathbf{S}_k^\top \succeq 0$ preserves Schur-complement PSD-cone invariance for all $\alpha_t \in (0, 1]$, provided floating-point cancellation in $\mathbf{Q}_k / M_k - \boldsymbol{\mu}_k \boldsymbol{\mu}_k^\top$ is eliminated via centered `float32 -> float64` accumulation (`src/egmm/kernels.py`).
2. **Factorization of `e-means` Step Law vs `e-GMM` Batch Degrees of Freedom (`src/egmm/scaling.py`)**:
   - `e-means` Eq. (6) ($\alpha_0 = 0.8244 \cdot K^{-0.4617} B^{0.9003} N^{-0.4498} T^{-0.1057}$) factors into $(B/N)^{0.45} (B/K)^{0.46} T^{-0.11}$: inverse steps per epoch ($B/N$) times points per cluster per batch ($B/K$).
   - While $k$-means centroids require only $O(1)$ points per cluster regardless of $D$ (allowing fixed `_INIT_USAGE = 2**-16`), estimating a $D$-dimensional covariance matrix requires replacing $B/K$ with the **effective batch degrees of freedom** $\gamma_{\text{df}} = B / (K D_{\text{eff}})$ ($D_{\text{eff}} = D$ for `full`, $D_{\text{eff}} = \sqrt{D}$ for `diag`), governing both $\alpha_0$ and initial Voronoi moment mass `init_usage_scale` $\in [0.25, 1.0]$.
3. **Why Multiplicative BF16 Jitter Fails for GMMs & Wedderburn Rank-1 Eigen-Splitting (`src/egmm/revival.py`)**:
   - `e-means` (Appendix H.1, Eq. 8) splits dead centroids via $\mathbf{c}_j \odot (1 \pm \varepsilon_{\text{BF16}})$, which fails near the origin ($\boldsymbol{\mu}_d \approx \mathbf{0}$) and leaves donor covariance $\boldsymbol{\Sigma}_d$ unsplit.
   - `e-GMM` replaces this with the **Wedderburn rank-1 reduction vector** $\boldsymbol{\delta} = \Delta \frac{\boldsymbol{\Sigma}_d \mathbf{v}}{\sqrt{\mathbf{v}^\top \boldsymbol{\Sigma}_d \mathbf{v}}}$, $\boldsymbol{\Sigma}'_d = \boldsymbol{\Sigma}_d - \boldsymbol{\delta}\boldsymbol{\delta}^\top + \epsilon_{\text{reg}}\mathbf{I}_D$, which conserves exact first and second raw mixture moments while guaranteeing Loewner PSD ordering $\boldsymbol{\Sigma}_d - \boldsymbol{\delta}\boldsymbol{\delta}^\top \succeq (1-\Delta^2)\boldsymbol{\Sigma}_d \succeq 0$ for *any* approximate GPU power-iteration vector $\mathbf{v}$.

---

## 2. Statistical Significance Across 33 Primary Datasets & Fairness Ablations

From [`stats_summary.txt`](stats_summary.txt), [`benchmark_head_to_head.csv`](benchmark_head_to_head.csv), and [`benchmark_ablations.csv`](benchmark_ablations.csv):

### A. All 33 Primary Datasets (3 Seeds Each)

| Comparison ($n=33$) | Test NLL W / T / L | Median $\Delta\text{NLL}$ | Wilcoxon $p_{\text{NLL}}$ | Median Speedup | Wilcoxon $p_{\text{time}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EGMM vs `TorchBatchGMM` (GPU Batch EM)** | **26 / 0 / 7** | **$-0.4288$** | **$6.357 \times 10^{-6}$** | **`2.68x`** *(30/33 faster)* | **$3.450 \times 10^{-7}$** |
| **EGMM vs `OnlineGMM_tuned`** | **28 / 0 / 5** | **$-0.9848$** | **$2.111 \times 10^{-7}$** | `0.66x` | $0.01931$ |
| **EGMM vs `NaiveMinibatchGMM`** | **32 / 0 / 1** | **$-1.5624$** | **$7.683 \times 10^{-9}$** | `0.60x` | $4.369 \times 10^{-4}$ |
| **EGMM vs `EMeansInitGMM`** | **32 / 0 / 1** | **$-1.4988$** | **$1.630 \times 10^{-8}$** | **`1.15x`** | **$2.581 \times 10^{-3}$** |
| **EGMM vs `StandardizedSklearnGMM` (CPU)** | **17 / 0 / 16** | **$-0.0366$** | $0.3127$ *(parity)* | **`48.21x`** *(up to `489.9x`)* | **$2.328 \times 10^{-10}$** |
| **EGMM vs `SklearnGMM` (Raw CPU)** | 14 / 0 / 19 | $+0.0378$ | $0.8599$ *(parity)* | **`57.86x`** *(up to `489.9x`)* | **$2.328 \times 10^{-10}$** |

- **Friedman Omnibus Test ($n=33, k=7$):** $\chi^2 = 80.922$, $p = 2.304 \times 10^{-15}$, Nemenyi Critical Difference $\text{CD}(0.05) = 1.568$.
- **Mean Test NLL Ranks (1 = best):**
  1. **`EGMM` : `2.485`**
  2. `StandardizedSklearnGMM` : `2.576`
  3. `SklearnGMM` : `3.030`
  4. `TorchBatchGMM` : `4.030`
  5. `OnlineGMM_tuned` : `4.485`
  6. `EMeansInitGMM` : `5.394`
  7. `NaiveMinibatchGMM` : `6.000`

### B. Large-Scale Zero-Polish Subset ($N_{\text{train}} > 100\text{k}, n=7$, `polish_steps == 0`)

Datasets: `covtype_581k` ($406.7\text{k}$), `creditcard` ($199.4\text{k}$), `skin_nonskin` ($171.5\text{k}$), `syn_scale200k` ($140\text{k}$), `syn_scale300k_k64_full` ($210\text{k}$), `syn_scale400k_d256_diag` ($280\text{k}$), `syn_scale500k_diag` ($350\text{k}$).

| Comparison ($n=7$, `polish_steps=0`) | Test NLL W / T / L | Median $\Delta\text{NLL}$ | Wilcoxon $p_{\text{NLL}}$ | Median Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **EGMM vs `TorchBatchGMM`** | **6 / 0 / 1** | **$-1.0330$** | **$0.03125$** | **`3.93x`** |
| **EGMM vs `OnlineGMM_tuned`** | **6 / 0 / 1** | **$-1.5134$** | **$0.03125$** | **`1.06x`** |
| **EGMM vs `NaiveMinibatchGMM`** | **7 / 0 / 0** | **$-1.5624$** | **$0.01563$** | `0.96x` |
| **EGMM vs `EMeansInitGMM`** | **6 / 0 / 1** | **$-1.1368$** | **$0.03125$** | **`3.02x`** |
| **EGMM vs `StandardizedSklearnGMM`** | 2 / 0 / 5 | $+0.7294$ | $0.5781$ | **`161.55x`** |

### C. Component & Fairness Ablations ($n=33$)

| Comparison ($n=33$) | Test NLL W / T / L | Median $\Delta\text{NLL}$ | Wilcoxon $p_{\text{NLL}}$ | Median Speedup | Dead Components (Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EGMM vs `EGMM_no_revival`** | **26 / 4 / 3** | **$-0.5132$** | **$7.579 \times 10^{-6}$** | `0.89x` | **`0.05` vs `4.49`** |
| **EGMM vs `OnlineGMM_matched_polish`** | **26 / 0 / 7** | **$-0.5962$** | **$2.302 \times 10^{-6}$** | **`1.22x`** | **`0.05` vs `4.29`** |
| **EGMM vs `EGMM_polish0`** | **21 / 7 / 5** | **$-0.0493$** | **$4.398 \times 10^{-4}$** | `0.68x` | `0.05` vs `0.04` |
| **EGMM vs `EGMM_no_init_scale`** | **17 / 3 / 13** | $-0.0044$ | $0.3672$ | `1.00x` | `0.05` vs `0.04` |
| **`EGMM_polish0` vs `TorchBatchGMM`** | **21 / 0 / 12** | **$-0.2591$** | **$0.04914$** | **`2.30x`** | **`0.04` vs `3.01`** |
| **`EGMM_polish0` vs `OnlineGMM_tuned`** | **25 / 0 / 8** | **$-0.5313$** | **$3.729 \times 10^{-4}$** | **`1.03x`** | **`0.04` vs `4.16`** |

---

## 3. Runtime Phase Breakdown & ARI vs NLL Split Trade-Off

- **Cold vs Warm Runtime Phase Breakdown ([`benchmark_phase_breakdown.csv`](benchmark_phase_breakdown.csv)):**
  - Across 8 profiled datasets, CUDA Graph warmup/capture takes `7.3–19.2 ms` in warm steady-state and replays 36–45 mini-batch EMA steps in `6.5–48.6 ms` (`204.7 ms` on 406.7k-row `covtype_581k`).
  - For $N_{\text{train}} > 100\text{k}$, `polish_ms = 0.0`, confirming that large-scale speedups (`3.72x–8.26x` over `TorchBatchGMM`, `89.6x–489.9x` over `StandardizedSklearnGMM`) come from pure CUDA-Graph EMA + Wedderburn revival.
- **ARI vs NLL Split Trade-Off (32 Labeled Datasets):**
  - `EGMM` achieves mean ARI `0.3251` and NMI `0.4873` (`0.05` dead components), whereas disabling revival (`EGMM_no_revival`) raises mean ARI to `0.3631` and NMI to `0.4945` (exceeding `TorchBatchGMM`'s `0.3467` and `OnlineGMM_tuned`'s `0.3602`) while degrading held-out test NLL by $+0.5132$ ($p = 7.58 \times 10^{-6}$, leaving `4.49` dead components). Leaving excess components dead avoids subdividing skewed classes into multiple Gaussian sub-lobes (preserving pairwise class Rand index ARI) at the cost of inferior continuous density estimation (NLL).

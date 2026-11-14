# e-GMM: Exponential Moving Average Gaussian Mixture Models — Design & Verification Spec

**Date:** 2026-11-14  
**Goal:** Design, implement, empirically validate, and benchmark **`e-GMM`**, a zero-tuning, GPU/CPU drop-in replacement for `sklearn.mixture.GaussianMixture` inspired by `e-means` ([arXiv:2611.09501](https://arxiv.org/abs/2611.09501)), achieving substantially lower wall-clock time **and** lower or equal held-out negative log-likelihood (NLL) across diverse data regimes under strict statistical testing.

---

## 1. Mathematical Formulation (`EGMM`)

Let $X \in \mathbb{R}^{N \times D}$ be the training dataset and $K$ be `n_components`.

### 1.1 Internal Standardization (Leakage-Free, Numerical Stability)
Computing uncentered second moments $Q_k = \mathbb{E}[x x^\top]$ and recovering $\Sigma_k = Q_k / M_k - \mu_k \mu_k^\top$ can suffer catastrophic floating-point cancellation when $\|\mu_k\| \gg \sigma_k$.  
To prevent cancellation without adding a second decay hyperparameter:
- Compute per-feature empirical mean $m_{\text{train}} \in \mathbb{R}^D$ and std $s_{\text{train}} \in \mathbb{R}^D$ **strictly on the training split** inside `fit` (or from the first batch in `partial_fit`).
- Transform $\tilde{x} = (x - m_{\text{train}}) / \max(s_{\text{train}}, 10^{-8})$ for internal EMA optimization in `float32` (or `float64` when requested).
- After optimization, unscale $(\tilde{\pi}_k, \tilde{\mu}_k, \tilde{\Sigma}_k)$ back to original coordinates so `weights_`, `means_`, `covariances_`, `score_samples`, `bic`, and `aic` are 100% drop-in compatible with `sklearn.mixture.GaussianMixture`.

### 1.2 Sufficient Statistics & EMA Update
Each component $k \in \{1, \dots, K\}$ maintains three EMA-smoothed sufficient statistics:
- Effective mass $M_k \in \mathbb{R}_{>0}$
- Linear sum $S_k \in \mathbb{R}^D$
- Uncentered second moment $Q_k$: shape $(K, D)$ for `covariance_type="diag"` or $(K, D, D)$ for `covariance_type="full"`.

For mini-batch $\mathcal{B}$ of size $B$ (normalized by batch size so statistics stay scale-invariant across $B$):
1. **E-step (batched log-responsibilities):**
   Compute $\log r_{ik} = \log \pi_k + \log \mathcal{N}(\tilde{x}_i \mid \tilde{\mu}_k, \tilde{\Sigma}_k)$ via cached Cholesky precision factors $L_k$ (updated once per mini-batch, $O(K D^3)$ for `full`, $O(K D)$ for `diag`), then $r_{ik} = \text{softmax}_k(\log r_{ik})$.
   - Batched quadratic form optimization for `full`: avoid $O(B K D^2)$ Python loops; compute Mahalanobis distances via batched matrix multiplications / `torch.linalg.solve_triangular` or quadratic expansion $x^\top P_k x - 2 x^\top (P_k \mu_k) + \mu_k^\top P_k \mu_k$!
   - **Important insight:** Using precision matrix $P_k = \Sigma_k^{-1}$ directly in the E-step lets us compute all $(B, K)$ log-densities using **batched GEMMs** (`X^2 @ P` or `(X @ L_k)`)! Specifically:
     $$\text{maha}_{ik} = \|\tilde{x}_i - \tilde{\mu}_k\|_{P_k}^2$$
     For `full`, precompute $P_k = \Sigma_k^{-1}$, $h_k = P_k \mu_k$, and $c_k = \mu_k^\top P_k \mu_k - \log|P_k|$ once per step ($O(K D^3)$). Then for the batch $\tilde{X} \in \mathbb{R}^{B \times D}$, if $B$ is large and $D$ moderate, $(\tilde{X} P_k) \odot \tilde{X}$ or batched `torch.einsum('bd,kde,be->bk', ...)` / chunked Cholesky solves keep GPU Tensor Cores saturated! We will benchmark exact E-step formulations in Task 2.
2. **Batch Sufficient Statistics (mean-normalized per batch):**
   $$\bar{m}_k = \frac{1}{B} \sum_{i \in \mathcal{B}} r_{ik}, \qquad \bar{s}_k = \frac{1}{B} \sum_{i \in \mathcal{B}} r_{ik} \tilde{x}_i$$
   $$\bar{q}_k = \begin{cases} \frac{1}{B} \sum_{i \in \mathcal{B}} r_{ik} \tilde{x}_i^2 & (\text{diag}) \\ \frac{1}{B} \sum_{i \in \mathcal{B}} r_{ik} \tilde{x}_i \tilde{x}_i^\top & (\text{full}) \end{cases}$$
3. **EMA Update with Cosine-Annealed Decay $\alpha_t$:**
   $$M_k \leftarrow (1 - \alpha_t) M_k + \alpha_t \bar{m}_k$$
   $$S_k \leftarrow (1 - \alpha_t) S_k + \alpha_t \bar{s}_k$$
   $$Q_k \leftarrow (1 - \alpha_t) Q_k + \alpha_t \bar{q}_k$$
   where at step $t \in \{0, \dots, T_{\text{steps}}-1\}$:
   $$\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_0 - \alpha_{\min})\left(1 + \cos\left(\frac{\pi t}{\max(1, T_{\text{steps}} - 1)}\right)\right), \qquad \alpha_{\min} = 0.1 \alpha_0.$$
   *(Plus optional short warmup / bias-correction during the very first few batches so initial seeded statistics do not carry stale initialization bias.)*
4. **M-step Projection:**
   $$\tilde{\pi}_k = \frac{M_k}{\sum_{j=1}^K M_j}, \qquad \tilde{\mu}_k = \frac{S_k}{\max(M_k, 10^{-12})}$$
   $$\tilde{\Sigma}_k = \text{PSD\_clamp}\!\left(\frac{Q_k}{\max(M_k, 10^{-12})} - \tilde{\mu}_k \tilde{\mu}_k^\top\right) + \epsilon_{\text{reg}} I$$
   Because $(M_k, S_k, Q_k)$ is a convex combination of valid batch empirical moments, $\frac{Q_k}{M_k} - \tilde{\mu}_k \tilde{\mu}_k^\top \succeq 0$ mathematically; tiny negative eigenvalues from floating-point rounding are clamped safely at $0$ before adding `reg_covar`.

### 1.3 Dead-Component & Collapsed-Covariance Revival (Eigen-Split)
Every `revive_every` steps (during the first ~80% of training so final cosine annealing settles cleanly):
- Identify **dead / starving components**: $M_k < \tau_{\text{dead}} \cdot \frac{1}{K}$ (e.g., $\tau_{\text{dead}} = 0.05$).
- Rank **donor components** by a splitting score: $M_d \cdot \lambda_{\max}(\tilde{\Sigma}_d)$ (prioritizing heavy, stretched components that likely cover multiple modes!).
- For each dead component $k$ paired with a donor $d$:
  - Compute the donor's principal eigenvector $v_1$ and eigenvalue $\lambda_1$ (for `diag`, simply the coordinate with largest variance).
  - Split donor $d$ into two children along $\pm \delta \sqrt{\lambda_1} v_1$ ($\delta = 0.5$):
    - $\tilde{\mu}_d' = \tilde{\mu}_d - \delta \sqrt{\lambda_1} v_1$, $\tilde{\mu}_k' = \tilde{\mu}_d + \delta \sqrt{\lambda_1} v_1$
    - Halve donor mass $M_d' = M_k' = \frac{1}{2} M_d$
    - Adjust covariance along $v_1$ by subtracting $\delta^2 \lambda_1 v_1 v_1^\top$ (preserving total mixture second moment $\frac{1}{2}(\Sigma' + \mu'_d {\mu'_d}^\top + \Sigma' + \mu'_k {\mu'_k}^\top) = \Sigma_d + \mu_d \mu_d^\top$!), and reconstruct exact consistent $(M, S, Q)$ buffers!
- This moment-preserving split guarantees smooth continuation without spiking NLL.

### 1.4 Initialization & Zero-Tuning Scaling Laws
- **Initialization:** Fast mini-batch $k$-means++ seeding on the first mini-batch (or subsample of $\min(N, \max(2048, 20 K))$ standardized points), initializing empirical moments via one soft/hard assignment pass so $M_k, S_k, Q_k$ start well-calibrated.
- **Scaling Laws (`batch_size="auto"`, `alpha="auto"`, `max_epochs="auto"`):**
  Derived empirically in a dedicated **out-of-distribution scaling-law sweep** (Task 5) over synthetic Gaussian mixtures with varying $(N, K, D, \text{epochs})$:
  $$B^* = \text{clip}\!\left(\text{round}(c_B \cdot K^{\beta_K} \cdot D^{\beta_D} \cdot N^{\beta_N}),\; B_{\min},\; \min(N, B_{\max})\right)$$
  $$\alpha_0^* = \text{clip}\!\left(c_\alpha \cdot \left(\frac{B^*}{N_{\text{eff}}}\right)^{\gamma_1} \cdot K^{\gamma_2} \cdot T^{-\gamma_3},\; \alpha_{\text{lo}},\; \alpha_{\text{hi}}\right)$$
  Crucially: the scaling law is fitted **only** on synthetic calibration distributions (never on the held-out evaluation benchmark datasets) to guarantee zero test leakage.

---

## 2. Drop-In API Specification

Class `EGMM` in `src/egmm/estimator.py` matching `sklearn.mixture.GaussianMixture`:
- **Parameters:** `n_components=1`, `covariance_type="full"` (`"full"` | `"diag"`), `tol=1e-3`, `reg_covar=1e-6`, `max_epochs="auto"` (or int), `batch_size="auto"` (or int), `alpha="auto"` (or float), `revive_dead=True`, `n_init=1`, `init_params="kmeans++"`, `device="auto"`, `dtype="float32"`, `random_state=None`, `verbose=0`.
- **Attributes:** `weights_`, `means_`, `covariances_`, `precisions_`, `precisions_cholesky_`, `converged_`, `n_iter_`, `lower_bound_`, `n_features_in_`, `revivals_count_`.
- **Methods:** `fit(X, y=None)`, `partial_fit(X, y=None)`, `fit_predict(X, y=None)`, `predict(X)`, `predict_proba(X)`, `score_samples(X)`, `score(X)`, `bic(X)`, `aic(X)`, `sample(n_samples=1)`.
- Return numpy arrays from public `sklearn`-compatible methods/attributes (accepting both `np.ndarray` and `torch.Tensor` inputs).

---

## 3. Baselines & Four-Part Verification Protocol

### Baselines (all evaluated under identical splits & hardware)
1. **`SklearnGMM`**: `sklearn.mixture.GaussianMixture` (CPU full-batch EM).
2. **`TorchBatchGMM`**: PyTorch GPU/CPU exact full-batch EM using the exact same E-step / Cholesky kernels (isolates pure *algorithmic* gain from CPU-vs-GPU hardware gain).
3. **`OnlineGMM` (Cappé & Moulines, 2009)**: PyTorch GPU/CPU Online EM with polynomial step size $\gamma_t = (t + t_0)^{-\rho}$ ($\rho \in (0.5, 1]$), tuned on inner validation split with equal tuning budget.
4. **`NaiveMinibatchGMM`**: Ablation — fixed-$\alpha$ EMA without cosine schedule or eigen-split revival.
5. **`EMeansInitGMM`**: Ablation — `e-means`-style $k$-means warm-start + short batch EM.

### Verification Protocol (Adapted from User's 4-Part Rigorous Protocol)
- **A. Leakage-Free Evaluation:**
  - Strict train / validation / test split (or 5-fold CV for smaller datasets). All scaling/standardization happens strictly inside each model's `fit(X_train)`—no test/val statistics ever touch training.
  - `EGMM` runs **zero-tuning** (`alpha="auto"`, `batch_size="auto"` from the pre-fit synthetic scaling law). `OnlineGMM` gets an inner validation tuning sweep over $(\rho, B)$ with matched budget on validation NLL before final evaluation on untouched `X_test`.
  - Match convergence/wall-clock/epoch budgets fairly.
- **B. Diverse Data Regimes (20+ Datasets):**
  - **Synthetic Hard Regimes (known ground-truth NLL & cluster labels for ARI/NMI):**
    - Overlapping anisotropic ellipses, high aspect-ratio covariance ($100:1$), severe cluster imbalance ($50:1$), high-D ($D \in \{64, 256\}$), many components ($K \in \{32, 64, 128\}$), large scale ($N \in \{200\text{k}, 1\text{M}\}$).
  - **Real Tabular & Embedding Datasets (`sklearn.datasets` + `fetch_openml`):**
    - Small (`<10k`): `digits`, `wine`, `breast_cancer`, OpenML small tabular sets (`segment`, `satimage`, `pendigits`, `waveform`, etc.).
    - Medium (`10k–200k`): `letter`, `adult` (numeric), `shuttle`, `fashion-mnist` (PCA-50 / raw), `mnist_784` (PCA-64), `california_housing`, `creditcard` / `magic04` / `electricity`.
    - Large (`200k–1M+`): `covtype` (`581,012 × 54`), `HIGGS` subset / `poker` / large `fetch_openml` or high-D embedding dataset.
- **C. Pareto Frontier (Speed vs. Quality):**
  - Measure **Held-Out Test NLL** (`-model.score(X_test)`) vs **Wall-Clock Training Time** across multiple budgets (`0.05s, 0.1s, 0.25s, 0.5s, 1s, 2s, 5s, 15s` or epoch budgets `0.5, 1, 2, 3, 5, 10, 25`).
- **D. Statistical Significance:**
  - Paired **Wilcoxon Signed-Rank Test** (`scipy.stats.wilcoxon`) across all $N \ge 20$ datasets comparing `EGMM` vs `TorchBatchGMM`, `SklearnGMM`, and `OnlineGMM` on both **Held-Out Test NLL** and **Time-to-Target-NLL**.
  - **Mean Rank & Critical Difference (CD) diagram** (Nemenyi post-hoc test after Friedman test) saved as publication-ready figures.

---

## 4. Project Directory Layout
```
d:\nckh\
├── pyproject.toml
├── docs/
│   ├── superpowers/specs/2026-11-14-egmm-design.md
│   ├── superpowers/plans/2026-11-14-egmm-implementation.md
│   └── experiments/egmm_findings.md
├── src/egmm/
│   ├── __init__.py
│   ├── kernels.py        # Fast batched E-step, sufficient stats, PSD projection
│   ├── revival.py        # Dead-component detection & moment-preserving eigen-split
│   ├── scaling.py        # Scaling-law formulas & schedules
│   ├── estimator.py      # Drop-in EGMM (sklearn-compatible)
│   └── baselines.py      # TorchBatchGMM, OnlineGMM, NaiveMinibatchGMM, EMeansInitGMM
├── tests/
│   ├── test_kernels.py
│   ├── test_revival.py
│   ├── test_estimator_api.py
│   └── test_baselines.py
└── experiments/
    ├── profile_kernels.py       # E-step microbenchmark (CPU & RTX 5070)
    ├── fit_scaling_laws.py      # Out-of-distribution synthetic sweep -> scaling laws
    ├── run_benchmark_suite.py   # 20+ datasets, Pareto frontiers, ablations, stats
    └── plot_results.py          # Figures (Pareto, scaling law, CD diagram, NLL curves)
```

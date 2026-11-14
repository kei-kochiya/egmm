# e-GMM

**Fast and Collapse-Resistant Expectation–Maximization for Gaussian Mixture Models via Centered EMA and Eigen-Split Revival.**

`EGMM` is a drop-in `scikit-learn`-compatible PyTorch/CUDA and CPU Gaussian Mixture estimator inspired by Meta FAIR's [`e-means`](docs/references/e_means/README.md) (`arXiv:2611.09501`), adapted for diagonal and full $D \times D$ covariance density modeling.

## Key Features

- **Centered Mixed-Precision E-Step ([`src/egmm/kernels.py`](src/egmm/kernels.py)):** Computes Mahalanobis distances and responsibilities in `float32` via fused pre-projected triangular Cholesky GEMMs, accumulating centered batch/component sufficient statistics and covariance updates in `float64` to prevent offset cancellation.
- **Degrees-of-Freedom Cosine EMA Scaling ([`src/egmm/scaling.py`](src/egmm/scaling.py)):** Automatically resolves batch size, epochs, peak EMA decay $\alpha_0$, initial moment scale $s_{\text{init}}$, and full-batch polish budget in closed form via batch degrees of freedom $B / (D \cdot K)$ with zero per-dataset cross-validation.
- **Moment-Preserving Eigen-Split Revival ([`src/egmm/revival.py`](src/egmm/revival.py)):** Revives starving components by splitting high-variance donors along their leading covariance eigenvector (5-step batched power iteration on CUDA, exact `eigh` on CPU).
- **CUDA Graph Step Capture & Adaptive Routing ([`src/egmm/estimator.py`](src/egmm/estimator.py)):** Branchless GPU M-step captured via `torch.cuda.CUDAGraph()`, with automatic CPU execution for small workloads (`device="auto"`).

## Installation & Quick Start

```bash
pip install -e ".[dev]"
```

```python
from egmm import EGMM

model = EGMM(n_components=16, covariance_type="full", random_state=0)
model.fit(X_train)
test_nll = -model.score(X_test)
```

## Verification & Benchmarks

- **Unit tests:** `pytest -q` (verifies kernels, eigen-split moment preservation, scikit-learn estimator API, baselines, and CUDA Graph / float32 precision stability).
- **23-dataset primary suite + Pareto frontiers:** `python experiments/run_benchmark_suite.py` (outputs saved in [`docs/experiments/`](docs/experiments/egmm_findings.md)).
- **16-dataset zero-tuning small-$N$ check:** `python experiments/run_small_dataset_eval.py`.
- **Scaling sensitivity sweep & plotting:** `python experiments/fit_scaling_laws.py` and `python experiments/plot_results.py`.

## Summary Results

Across 23 real and synthetic datasets (3 seeds each, NVIDIA RTX 5070 + CPU):
- **vs. shared-kernel GPU `TorchBatchGMM`:** **19/23** Test NLL wins (Wilcoxon $p = 2.15 \times 10^{-4}$), **23/23** speed wins (`2.33x` median, up to `6.72x`, $p = 2.38 \times 10^{-7}$), and `0.10` vs. `1.87` mean dead components.
- **vs. validation-tuned `OnlineGMM_tuned`:** **19/23** Test NLL wins ($p = 7.32 \times 10^{-5}$).
- **vs. CPU `sklearn.mixture.GaussianMixture`:** Indistinguishable Test NLL ($p = 0.964$) at **56.07x** median speedup (up to **560.91x** on `covtype_581k`).
- **Zero-shot 16 external small datasets ($N_{\text{train}} \in [105, 3831]$):** `device="auto"` delivers **1.88x** median speedup vs. `TorchBatchGMM` ($p = 6.10 \times 10^{-5}$) and **1.68x** vs. `SklearnGMM` ($p = 6.29 \times 10^{-3}$).

## Paper (Overleaf Ready)

The self-contained LaTeX manuscript, bibliography, and single-panel figures live in [`paper/`](paper/main.tex) (`main.tex`, `references.bib`, and `figures/*.png`). Zip `paper/` and upload directly to Overleaf (`pdfLaTeX`).

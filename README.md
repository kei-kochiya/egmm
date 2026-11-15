# e-GMM

**Fast and Collapse-Resistant Expectation–Maximization for Gaussian Mixture Models via Centered EMA and Wedderburn Eigen-Split Revival.**

`EGMM` is a drop-in `scikit-learn`-compatible PyTorch/CUDA and CPU Gaussian Mixture estimator inspired by Meta FAIR's [`e-means`](docs/references/e_means/README.md) (`arXiv:2611.09501`), adapted to diagonal and full $D \times D$ covariance density modeling.

## Key Features

- **Centered mixed-precision E-step ([`src/egmm/kernels.py`](src/egmm/kernels.py)):** log-densities and centered mini-batch statistics are `float32` GEMMs, promoted to `float64` for exact uncentered moment reconstruction and EMA blending of the global moments; all reported likelihoods (`score`, `score_samples`, `predict*`, `bic`, `aic`) are evaluated exactly in `float64`.
- **Seeding and smooth budgets ([`src/egmm/scaling.py`](src/egmm/scaling.py)):** greedy k-means++ (sync-free on CUDA) on `max(2048, 128K)` points; batch size, peak EMA rate $\alpha_0$, initial mass $s_{\text{init}}$ from batch degrees of freedom $B/(K D_{\text{eff}})$; epoch and closing-polish budgets are continuous logistic gates in $\log N$ (`schedule_law="bracket"` reproduces the legacy piecewise rules).
- **Moment-preserving Wedderburn eigen-split revival ([`src/egmm/revival.py`](src/egmm/revival.py)):** splits high-variance donors along a power-iteration (CUDA) or exact `eigh` (CPU) direction with a split vector that keeps the donor covariance PSD for any approximate direction.
- **Systems ([`src/egmm/estimator.py`](src/egmm/estimator.py)):** single-pass device-side standardization, branchless inner M-step captured in `torch.cuda.CUDAGraph()`, and automatic CPU execution for small workloads (`device="auto"`).

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

```bash
export PYTHONPATH="src;."          # "src:." on Linux/macOS
pytest -q
python experiments/fit_scaling_laws.py                                  # synthetic alpha_0 sweep
python experiments/run_benchmark_suite.py --stages h2h,abl,phase,pareto,sys # 33-dataset suite + systems run
python experiments/run_small_dataset_eval.py                            # 16 small datasets
python experiments/design_studies/run_design_studies.py all             # validation-only design studies
python experiments/design_studies/split_psd_check.py                    # split safety simulation (CPU)
python experiments/holdout_suite.py                                     # pre-registered holdout (frozen code)
python experiments/converged_reference.py                               # converged batch-EM reference (holdout)
python experiments/analyze_results.py                                   # ALL statistics + paper tables
python experiments/plot_results.py
```

`experiments/analyze_results.py` is the single source of truth: it writes
[`docs/experiments/stats_summary.txt`](docs/experiments/stats_summary.txt) and the LaTeX tables and number
macros in `paper/generated/` that the paper `\input`s, so the paper cannot drift from the CSV logs.
`python experiments/make_supplementary.py` builds the anonymized supplementary archive and self-tests it.
See [`docs/experiments/egmm_findings.md`](docs/experiments/egmm_findings.md) for what changed in this version,
negative results, and the development-set disclosure.

## Summary Results

Authoritative numbers: `docs/experiments/stats_summary.txt` (NVIDIA RTX 5070 + Intel i7-14700F, 3 seeds,
Holm-adjusted Wilcoxon p-values).

| | vs GPU batch EM (`TorchBatchGMM`) | vs CPU `scikit-learn` (standardized) | vs tuned online EM |
| :--- | :--- | :--- | :--- |
| 33 primary datasets | 24/0/9 wins/ties/losses, 2.35x faster | 21/0/12 (n.s.), 32.5x faster | 26/1/6 |
| 16 pre-registered unseen datasets | 9/0/7 (n.s.), 2.54x faster | 8/0/8 (n.s.), 37x faster | 13/0/3 (p = 0.012) |

In short: on unseen data e-GMM's held-out likelihood is not significantly different from full-batch EM
and scikit-learn (the holdout is too small to prove equivalence) at a fraction of the fit time, and it
clearly beats online EM. It is a fast default, not a more accurate optimizer. Exact versions, hardware and
runtimes: [`ENVIRONMENT.md`](ENVIRONMENT.md).

## Paper

The LaTeX manuscript lives in [`paper/`](paper/main.tex) (`main.tex`, `references.bib`, `figures/`, and
the auto-generated `generated/`). Compile with `tectonic -X compile paper/main.tex` or upload `paper/`
to Overleaf (`pdfLaTeX`).

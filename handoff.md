# Handoff: e-GMM (TMLR submission) — brief for AI agents

`e-GMM` = mini-batch online EM for Gaussian mixtures on GPU/CPU (PyTorch, scikit-learn-like API), plus a paper.
Repo `github.com/kei-kochiya/egmm`, branch `main`. This file is excluded from the supplementary archive.

## Status (2026-11-15)

* **Paper submitted to TMLR** (double-blind). Main text 12 pages (Sections 1-7); 26 pages with statements,
  references, appendices A-D. Results are ordered holdout first.
* **arXiv preprint** built in `dist/arxiv/` (25 pages, non-anonymous); waiting for arXiv endorsement (cs.LG).
* Four rounds of mock review (12 reviewer agents): last verdicts "leaning accept"; no open critical issue.
  Private reviewer Q&A and submission checklist: `docs/submission/`.

## Hard rules

1. **`src/egmm` is frozen** since commit `7d6b9c9` (holdout pre-registration). Do not change library code
   while the paper is under review. New baselines/variants go in `experiments/` as subclasses (example:
   `TorchBatchGMMF32Chol` in `run_benchmark_suite.py`). Check: `git diff 7d6b9c9 HEAD -- src/egmm` is empty;
   `python experiments/code_fingerprint.py` gives `ea2a6d95...391f`.
2. **Never type numbers into `paper/main.tex`.** Every number is a `\r...` macro in
   `paper/generated/numbers.tex` and every table body a `\gentable{...}`, all written by
   `experiments/analyze_results.py` from `docs/experiments/*.csv`. After any change run
   `analyze_results.py` then `verify_tables.py`.
3. **Anonymity.** `paper/main.pdf` and `dist/supplementary_anonymized.zip` must not identify the author;
   `experiments/make_supplementary.py` enforces this. `\codelink` and `\verifysentence` switch to the GitHub
   wording automatically under `\usepackage[preprint]{tmlr}`; never hard-code the repo URL.
4. **`docs/submission/` is private**: never commit it or ship it.
5. **Ask the author first** before publishing anything (arXiv, OpenReview, PyPI, tags/releases) or
   downloading data/packages.
6. **Holdout is pre-registered** (`docs/experiments/HOLDOUT_PREREGISTRATION.md`; never edit it, deviations go
   in `..._ADDENDUM.md`). Any new holdout analysis is post hoc and must be labelled so.

## Map

| Path | What |
| :--- | :--- |
| `src/egmm/` | library: `estimator.py` (EGMM), `kernels.py`, `revival.py`, `scaling.py` (rules), `baselines.py` |
| `experiments/run_benchmark_suite.py` | stages `h2h,abl,phase,pareto,sys` (development suite, 33 datasets) |
| `experiments/holdout_suite.py`, `converged_reference.py` | pre-registered holdout (16 OpenML datasets) |
| `experiments/analyze_results.py` | every statistic, table and macro; also writes `docs/experiments/stats_summary.txt` |
| `experiments/verify_tables.py` | re-runs the analysis and checks `paper/generated/` byte for byte |
| `experiments/make_supplementary.py` | anonymized ZIP + anonymity scan + self-test |
| `experiments/design_studies/`, `fit_scaling_laws.py`, `run_small_dataset_eval.py`, `plot_results.py` | side studies, figures |
| `docs/experiments/` | CSV logs (source of truth), `stats_summary.txt`, `egmm_findings.md`, pre-registration |
| `paper/` | `main.tex`, `references.bib`, TMLR style, `generated/`, `figures/`, `make_arxiv.py` |
| `dist/`, `data/` (git-ignored) | submission ZIP + arXiv bundle; OpenML cache (337 MB, needed for reruns) |
| `docs/references/e_means/` | mirrored third-party e-means code, reference only |

## Commands (Windows; use `src:.` on Linux)

```bash
export PYTHONPATH="src;."
pytest -q                                                    # 51 tests, incl. CUDA
python experiments/run_benchmark_suite.py --stages sys       # ~40 min; h2h ~75 min (CPU sklearn)
python experiments/analyze_results.py && python experiments/verify_tables.py
python experiments/plot_results.py && cp docs/experiments/fig_*.png paper/figures/
cd paper && tectonic -X compile main.tex                     # this machine: tectonic.exe under ~/.gemini/antigravity/.../scratch/tectonic/
python experiments/make_supplementary.py                     # -> dist/supplementary_anonymized.zip
TECTONIC=/path/to/tectonic python paper/make_arxiv.py        # -> dist/arxiv/ (preprint PDF + source ZIP)
```

Full reproduction order and runtimes: `ENVIRONMENT.md`; exact versions: `requirements-lock.txt`.

## Results (authoritative: `docs/experiments/stats_summary.txt`; p Holm-adjusted)

| | vs GPU batch EM (`TorchBatchGMM`) | vs CPU scikit-learn (standardized) | vs tuned online EM |
| :--- | :--- | :--- | :--- |
| Holdout (16, frozen code) | 9/0/7, p = 1.0, 2.5x faster; losses 3.8-5.4 nats on 3 datasets | 8/0/8, p = 1.0, 37x (~15x is GPU vs CPU) | 13/0/3, p = 0.012 (12/0/4 n.s. without polish) |
| Development (33) | 24/0/9, p = 7.6e-4, 2.35x | 21/0/12, p = 0.29, 32.5x | 26/1/6 |

Speed split (systems run): CUDA graph 1.95x, algorithm 1.26x, total 2.54x; N <= 1e5 graph 2.04x / algorithm
1.12x, N > 1e5 graph 1.57x / algorithm 2.11x. Batch EM's float64 Cholesky is not the cause (control ratio 1.00).
Positioning: as accurate as batch EM on unseen data, faster; not a more accurate optimizer.

## Open items

1. **Next version (in progress): `experiments/next/`** (plan, protocol and decision rules in its README).
   Stronger batch-EM baselines (`TorchBatchGMM_fast`, sync-light; `TorchBatchGMM_graph`, CUDA-graph
   captured; both checked against `TorchBatchGMM` by `tests/test_next_baselines.py`), a large-N suite
   (synthetic 1M/3M/10M rows; real HIGGS/SUSY only if the author downloads them into `data/large/`), and
   best-of-R restarts at equal compute. Run `run_next.py --stages dev,holdout,large` (GPU, ~2-2.5 h), then
   `analyze_next.py`. Results are development evidence; a restarts claim needs a new pre-registered holdout.
2. Other plausible requests: external GPU GMM library comparison (needs a package install, ask first),
   another GPU/OS, peak memory, sensitivity to K.
3. arXiv: after endorsement upload `dist/arxiv/egmm_arxiv_source.zip`; authors field "Tuan Khoi Ho";
   license CC BY; cs.LG + stat.ML cross-list.
4. After the review: library hardening (license is the author's choice; packaging, input validation, docs,
   CI; no numerics changes); follow-up paper on "better than batch EM" (equal-compute restarts,
   convergence-aware polish) with a second pre-registered holdout.

## Pitfalls

* The Bash tool here collapses `\\` in inline commands and heredocs (LaTeX `\texttt` became a tab once).
  Edit LaTeX with file-edit tools or with Python scripts written to a file.
* Benchmark stages rewrite their CSV after every dataset; do not analyze a CSV while its stage runs.
* GPU timings drift between sessions (per-dataset speedups up to 2x, absolute times up to ~3x); compare
  methods only within one interleaved session.
* Set `PYTHONIOENCODING=utf-8` when printing PDF text on the Windows console.

# Handoff: e-GMM (TMLR submission) — brief for AI agents

`e-GMM` = mini-batch online EM for Gaussian mixtures on GPU/CPU (PyTorch, scikit-learn-like API), plus a paper.
Repo `github.com/kei-kochiya/egmm`, branch `main`. This file is excluded from the supplementary archive.

## Status (2026-11-16)

* **Paper submitted to TMLR** (double-blind, Long submission) on 2026-11-15; the submitted version is kept
  privately in `docs/submission/submitted_tmlr_2026-11-15/` (source commit 80ee180).
* **Revised version in `paper/` (not uploaded to OpenReview; the author decides when).** Main text ends on
  p.14 (Long submission; keep it from growing). It adds the follow-up runs of `experiments/next/` (stronger
  batch EM, 1-10M-row datasets with equal-pass / equal-quality / revival checks, schedule and revival
  ablations), an evidence map (Table 1), and a method glossary.
* **Mock reviews:** two waves of 3 reviewers on the revision, all "leaning accept, no blocking issue" after
  fixes. Wave 3 is pending (the author will ask). Remaining non-blocking requests: tighter Section 5
  (fewer statistics in the main text), Linux/datacenter GPU check, tuned Robbins-Monro / averaged online EM.
* **arXiv dropped** by the author; `dist/` was deleted. `python experiments/make_supplementary.py`
  recreates the anonymized supplementary zip when a revision is uploaded.

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
| `data/` (git-ignored) | OpenML cache (337 MB, needed for reruns); `data/large/` HIGGS/SUSY (3.7 GB) |
| `docs/references/e_means/` | mirrored third-party e-means code, reference only |

## Commands (Windows; use `src:.` on Linux)

```bash
export PYTHONPATH="src;."
pytest -q                                                    # 51 tests, incl. CUDA
python experiments/run_benchmark_suite.py --stages sys       # ~40 min; h2h ~75 min (CPU sklearn)
python experiments/analyze_results.py && python experiments/verify_tables.py
python experiments/plot_results.py && cp docs/experiments/fig_*.png paper/figures/
cd paper && tectonic -X compile main.tex                     # this machine: C:/Users/Admin/tools/tectonic/tectonic.exe (0.15.0)
python experiments/make_supplementary.py                     # -> dist/supplementary_anonymized.zip
```

Full reproduction order and runtimes: `ENVIRONMENT.md`; exact versions: `requirements-lock.txt`.

## Results (authoritative: `docs/experiments/stats_summary.txt`; p Holm-adjusted)

| | vs GPU batch EM (`TorchBatchGMM`) | vs CPU scikit-learn (standardized) | vs tuned online EM |
| :--- | :--- | :--- | :--- |
| Holdout (16, frozen code) | 9/0/7, p = 1.0, 2.5x faster; losses 3.8-5.4 nats on 3 datasets | 8/0/8, p = 1.0, 37x (~15x is GPU vs CPU) | 13/0/3, p = 0.012 (12/0/4 n.s. without polish) |
| Development (33) | 24/0/9, p = 7.6e-4, 2.35x | 21/0/12, p = 0.29, 32.5x | 26/1/6 |

Speed split (systems run): CUDA graph 1.95x, algorithm 1.26x, total 2.54x. **Revision (follow-up runs,
`docs/experiments/next/`):** against a CUDA-graph-captured batch EM, e-GMM is about as fast at N <= 1e5
(1.06x dev, 1.00x holdout) and faster above (dev N > 1e5: 2.52x measured, >= 2.36x estimated; 1-10M rows:
median 3.2x measured, >= 2.9x estimated, 1.3-22x; HIGGS 20x, SUSY 22x, both to batch EM's default stopping rule).
Quality is always compared with float64 `TorchBatchGMM` (the faster variants use float32 Cholesky).
Positioning: as accurate as batch EM on unseen data; faster only at large N; not a more accurate optimizer.

## Open items

1. **Next: review wave 3** (when the author asks): 3 fresh reviewers (Opus/Sonnet) on `paper/main.pdf`, verify
   every claim against code and logs, fix. All follow-up GPU stages are done (`run_next.py` stages speed2,
   speed2b-e; results in `docs/experiments/next/FINDINGS.md` sections 8-9). Optional, non-blocking runs:
   schedule ablation at large N and on the holdout, a tuned Robbins-Monro / averaged online EM, more real large
   datasets / K / D, peak memory, a Linux or datacenter GPU.
2. Other plausible requests: external GPU GMM library comparison (needs a package install, ask first),
   another GPU/OS, peak memory, sensitivity to K.
3. Disk: `data/` is ~4 GB (git-ignored): HIGGS Parquet 2.0 GB, SUSY 0.9 GB plus a redundant original
   download `data/large/susy.zip` (0.9 GB), OpenML cache 0.3 GB. Large-N reruns need `data/large/`.
4. After the review: library hardening (license is the author's choice; packaging, input validation, docs,
   CI; no numerics changes); follow-up paper on "better than batch EM" (restarts, convergence-aware polish,
   an alpha_0 guard for diagonal mode) with a second pre-registered holdout that states a primary hypothesis
   and margin and uses a public timestamp (OSF/Zenodo).

## Pitfalls

* The Bash tool here collapses `\\` in inline commands and heredocs (LaTeX `\texttt` became a tab once).
  Edit LaTeX with file-edit tools or with Python scripts written to a file.
* Benchmark stages rewrite their CSV after every dataset; do not analyze a CSV while its stage runs.
* GPU timings drift between sessions (per-dataset speedups up to 2x, absolute times up to ~3x); compare
  methods only within one interleaved session.
* Set `PYTHONIOENCODING=utf-8` when printing PDF text on the Windows console.
* `CUDA_VISIBLE_DEVICES=""` does not hide the GPU on Windows; use `-1` for CPU-only runs.
* Revision numbers live in `experiments/next/paper_results.py` (called by `analyze_results.py`); the speed
  figure comes from `experiments/next/plot_speed.py` (copy the PNG to `paper/figures/`).

# Environment used for every reported result

| Component | Version / setting |
| :--- | :--- |
| OS | Windows 11 Pro |
| CPU | Intel Core i7-14700F (20 cores, 28 threads) |
| GPU | NVIDIA GeForce RTX 5070, 12 GB (driver 610.88) |
| Python | 3.14.7 |
| PyTorch | 2.14.1+cu132 (CUDA 13.2, cuDNN 9.24) |
| NumPy / SciPy / pandas | 2.5.3 / 1.18.1 / 3.0.5 |
| scikit-learn | 1.9.1 (default threading: OpenBLAS 24 threads, OpenMP 28) |
| PyTorch numerics | TF32 off (`torch.backends.cuda.matmul.allow_tf32 = False`, float32 matmul precision `"highest"`; PyTorch defaults, not changed by the code); CPU threads 20 (intra-op and inter-op defaults) |

Exact package versions: [`requirements-lock.txt`](requirements-lock.txt). Check that the released logs reproduce
every table and number of the paper: `python experiments/verify_tables.py`.

Timing protocol: every timed PyTorch fit is preceded by `gc.collect()` and ~60 ms of GPU GEMMs (clock
warm-up), synchronized, repeated 5 times (median reported; 3 times in the Pareto stage); scikit-learn fits are
timed once. The head-to-head, small-dataset and systems stages run untimed warm-up fits first, so process
start-up is excluded. The machine was not otherwise isolated; between independent sessions per-dataset
speedups varied by up to about 2x and absolute times by more (see the paper, Section 4).

## Approximate wall-clock cost of the full reproduction

| Command | Time on the machine above |
| :--- | :--- |
| `python experiments/fit_scaling_laws.py` | ~2 min |
| `python experiments/run_benchmark_suite.py --stages h2h` | ~75 min (mostly CPU scikit-learn) |
| `--stages abl` / `phase` / `pareto` / `sys` | ~15 / ~2 / ~15 / ~15 min |
| `python experiments/run_small_dataset_eval.py` | ~5 min |
| `python experiments/design_studies/run_design_studies.py all` | ~15 min |
| `python experiments/holdout_suite.py` | ~45 min (`dionis` dominates) |
| `python experiments/converged_reference.py` | ~5 min |
| `python experiments/design_studies/split_psd_check.py` | ~1 min (CPU) |
| `python experiments/analyze_results.py` | seconds (regenerates every statistic from the CSV logs) |

Datasets are downloaded from OpenML (ids listed in `experiments/datasets.py`,
`experiments/run_small_dataset_eval.py`, `experiments/holdout_suite.py`) on first use and cached under
`data/openml`. The analysis alone needs no GPU and no downloads.

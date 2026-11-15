"""Pre-registered fresh holdout suite (Round 3).

The dataset list, the K / covariance rule, the split seeds, and the methods below were fixed and committed
*before* any of these datasets was downloaded or any e-GMM run on them; the library code (``src/egmm``) was
frozen at the same commit. None of these datasets was used during development of any e-GMM rule.

Selection rule: numeric OpenML datasets without missing values, not in the primary or small suites, with
5k <= N <= 420k rows, covering regression benchmarks (features only), sensor, vision-feature, AutoML and
physiological data. ``vehicle_sensIT`` (id 357) was excluded before download because it is stored as sparse
ARFF, which the shared loader cannot read as a dataframe.

Fixed rules (never tuned):
  * K = 8 (N_train < 5k), 12 (< 10k), 16 (< 30k), 24 (< 100k), 32 (otherwise)
  * covariance = "diag" if D > 64 else "full"
  * 70/15/15 split with seed 700 + index; no PCA; zero-variance train columns dropped (shared loader)

Run from the repository root (``PYTHONPATH=src;.`` on Windows):
    python experiments/holdout_suite.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.datasets import DatasetSplit, _load_openml_numeric

HOLDOUT_SPECS: list[tuple[str, int]] = [
    ("kin8nm", 189),
    ("cpu_act", 197),
    ("bank32nh", 833),
    ("texture", 40499),
    ("gesture_phase", 4538),
    ("eye_movements", 1044),
    ("ailerons", 296),
    ("eeg_eye_state", 1471),
    ("elevators", 846),
    ("house_16H", 574),
    ("2dplanes", 215),
    ("isolet", 300),
    ("volkert", 41166),
    ("helena", 41169),
    ("run_or_walk", 40922),
    ("dionis", 41167),
]


def holdout_k(n_train: int) -> int:
    if n_train < 5_000:
        return 8
    if n_train < 10_000:
        return 12
    if n_train < 30_000:
        return 16
    if n_train < 100_000:
        return 24
    return 32


def holdout_cov(n_features: int) -> str:
    return "diag" if n_features > 64 else "full"


def load_holdout_suite(data_home: str = "data/openml") -> list[DatasetSplit]:
    Path(data_home).mkdir(parents=True, exist_ok=True)
    out: list[DatasetSplit] = []
    for idx, (name, did) in enumerate(HOLDOUT_SPECS):
        # Load once with placeholder K/cov, then apply the fixed rules to the resulting split.
        ds = _load_openml_numeric(
            name=name,
            data_id=did,
            data_home=data_home,
            n_components=8,
            regime="holdout",
            recommended_cov="full",
            seed=700 + idx,
        )
        n_tr, d = ds.X_train.shape
        ds.n_components = holdout_k(n_tr)
        ds.recommended_cov = holdout_cov(d)
        # Clustering metrics only make sense for categorical targets with few classes.
        if ds.y_test is not None and not (2 <= len(np.unique(ds.y_test)) <= 50):
            ds.y_test = None
        out.append(ds)
    return out


if __name__ == "__main__":
    from experiments.run_benchmark_suite import run_head_to_head

    suite = load_holdout_suite()
    print("Holdout suite:")
    for ds in suite:
        print(f"  {ds.name:16s} N_train={ds.X_train.shape[0]:7d} D={ds.X_train.shape[1]:4d} K={ds.n_components:2d} {ds.recommended_cov}")
    df = run_head_to_head(suite, seeds=(0, 1, 2), out_csv="docs/experiments/holdout_head_to_head.csv")
    df.to_csv("docs/experiments/holdout_head_to_head.csv", index=False)

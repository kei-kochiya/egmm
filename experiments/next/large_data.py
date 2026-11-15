"""Large-N datasets (10^6-10^7 training rows) for the next version, built in float32 without float64 copies.

Synthetic mixtures are generated here (no download). Real large datasets are read only from local files
placed in ``data/large/`` by the author (e.g. UCI HIGGS.csv.gz, SUSY.csv.gz); nothing is downloaded here.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from experiments.datasets import DatasetSplit

ROOT = Path(__file__).resolve().parents[2]

# name, N_train, D, K, covariance, anisotropy, seed
SYNTHETIC_LARGE = [
    ("syn1m_full", 1_000_000, 32, 32, "full", 12.0, 31001),
    ("syn1m_diag", 1_000_000, 64, 32, "diag", 10.0, 31002),
    ("syn3m_full", 3_000_000, 32, 32, "full", 12.0, 31003),
    ("syn3m_diag", 3_000_000, 64, 32, "diag", 10.0, 31004),
    ("syn10m_full", 10_000_000, 32, 32, "full", 12.0, 31005),
    ("syn10m_diag", 10_000_000, 48, 32, "diag", 10.0, 31006),
]


def _split_float32(X: np.ndarray, y: np.ndarray | None, seed: int):
    """70/15/15 split by a seeded permutation (no float64 copy, unlike the small-suite splitter)."""
    n = X.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(0.70 * n), int(0.85 * n)
    tr, va, te = perm[:a], perm[a:b], perm[b:]
    return X[tr], X[va], X[te], (y[te] if y is not None else None)


def make_large_synthetic(name, n_train, n_features, n_components, cov_type, anisotropy, seed,
                         dirichlet_alpha: float = 1.0, center_scale: float = 2.2) -> DatasetSplit:
    """Same generative family as the development suite's synthetic mixtures, written chunk-wise in float32."""
    rng = np.random.default_rng(seed)
    n_total = int(round(n_train / 0.70))
    weights = rng.dirichlet(np.full(n_components, dirichlet_alpha))
    counts = rng.multinomial(n_total, weights)
    means = rng.normal(scale=center_scale, size=(n_components, n_features))
    scales = np.geomspace(1.0 / math.sqrt(anisotropy), math.sqrt(anisotropy), n_features)
    X = np.empty((n_total, n_features), dtype=np.float32)
    y = np.empty(n_total, dtype=np.int32)
    pos = 0
    for k, cnt in enumerate(counts):
        rot = None if cov_type == "diag" else np.linalg.qr(rng.normal(size=(n_features, n_features)))[0]
        done = 0
        while done < cnt:
            m = min(cnt - done, 1_000_000)
            Z = rng.standard_normal(size=(m, n_features)).astype(np.float32) * scales.astype(np.float32)
            if rot is not None:
                Z = Z @ rot.T.astype(np.float32)
            X[pos:pos + m] = Z + means[k].astype(np.float32)
            y[pos:pos + m] = k
            pos += m
            done += m
    X_tr, X_va, X_te, y_te = _split_float32(X, y, seed)
    del X
    return DatasetSplit(name=name, regime="large", X_train=X_tr, X_val=X_va, X_test=X_te,
                        n_components=n_components, recommended_cov=cov_type, y_test=y_te)


# Real large datasets: (file name in data/large/, label column index to drop, K, covariance)
REAL_LARGE = {
    "higgs": ("HIGGS.csv.gz", 0, 32, "full"),  # UCI HIGGS: 11,000,000 rows x (1 label + 28 features)
    "susy": ("SUSY.csv.gz", 0, 32, "full"),    # UCI SUSY: 5,000,000 rows x (1 label + 18 features)
}


def load_real_large(key: str, seed: int = 41001, max_rows: int | None = None) -> DatasetSplit | None:
    """Read a local CSV(.gz) chunk-wise into float32; returns None when the file is not present."""
    fname, label_col, K, cov = REAL_LARGE[key]
    path = ROOT / "data" / "large" / fname
    if not path.exists():
        return None
    import pandas as pd

    parts, labels = [], []
    for chunk in pd.read_csv(path, header=None, chunksize=1_000_000, dtype=np.float32, nrows=max_rows):
        arr = chunk.to_numpy(dtype=np.float32, copy=False)
        labels.append(arr[:, label_col].astype(np.int32))
        parts.append(np.delete(arr, label_col, axis=1))
    X = np.concatenate(parts)
    y = np.concatenate(labels)
    keep = np.isfinite(X).all(axis=1)
    X, y = X[keep], y[keep]
    X_tr, X_va, X_te, y_te = _split_float32(X, y, seed)
    return DatasetSplit(name=key, regime="large_real", X_train=X_tr, X_val=X_va, X_test=X_te,
                        n_components=K, recommended_cov=cov, y_test=y_te)


def iter_large_suite(include_real: bool = True, sizes: tuple[str, ...] = ("1m", "3m", "10m")):
    """Yield datasets one at a time (each 10M-row set is ~2 GB in float32; never hold all of them)."""
    for name, n, d, k, cov, aniso, seed in SYNTHETIC_LARGE:
        if any(f"syn{s}_" in name for s in sizes):
            yield make_large_synthetic(name, n, d, k, cov, aniso, seed)
    if include_real:
        for key in REAL_LARGE:
            ds = load_real_large(key)
            if ds is not None:
                yield ds

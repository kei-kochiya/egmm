"""Benchmark dataset suite (22+ datasets across Small, Medium, Large, and Synthetic Hard regimes)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    fetch_covtype,
    fetch_openml,
    load_digits,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplit:
    name: str
    regime: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    n_components: int
    recommended_cov: str
    y_test: np.ndarray | None = None


def _split_and_maybe_pca(
    X: np.ndarray,
    y: np.ndarray | None,
    seed: int = 42,
    pca_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    X = np.asarray(X, dtype=np.float64)
    # Remove any non-finite rows safely
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y_clean = y[mask] if y is not None else None

    X_tr, X_tmp, idx_tr, idx_tmp = train_test_split(
        X, np.arange(len(X)), test_size=0.30, random_state=seed
    )
    X_va, X_te, idx_va, idx_te = train_test_split(
        X_tmp, idx_tmp, test_size=0.50, random_state=seed
    )
    y_te = y_clean[idx_te] if y_clean is not None else None

    if pca_dim is not None and X_tr.shape[1] > pca_dim:
        # Fit PCA strictly on train to prevent leakage
        pca = PCA(n_components=pca_dim, random_state=seed)
        X_tr = pca.fit_transform(X_tr)
        X_va = pca.transform(X_va)
        X_te = pca.transform(X_te)

    # Drop zero-variance columns based strictly on train
    stds = X_tr.std(axis=0)
    keep = stds > 1e-10
    if not keep.all():
        X_tr = X_tr[:, keep]
        X_va = X_va[:, keep]
        X_te = X_te[:, keep]

    return X_tr, X_va, X_te, y_te


def _make_hard_synthetic(
    name: str,
    regime: str,
    n_samples: int,
    n_features: int,
    n_components: int,
    seed: int,
    cov_type: str = "full",
    anisotropy: float = 20.0,
    dirichlet_alpha: float = 1.0,
    center_scale: float = 2.2,
    diag_only: bool = False,
) -> DatasetSplit:
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.full(n_components, dirichlet_alpha))
    counts = rng.multinomial(n_samples, weights)
    means = rng.normal(scale=center_scale, size=(n_components, n_features))

    X_parts = []
    y_parts = []
    for k, cnt in enumerate(counts):
        if cnt == 0:
            continue
        scales = np.geomspace(1.0 / math.sqrt(anisotropy), math.sqrt(anisotropy), n_features)
        Z = rng.normal(size=(cnt, n_features)) * scales
        if not diag_only:
            Q_raw = rng.normal(size=(n_features, n_features))
            Q_orth, _ = np.linalg.qr(Q_raw)
            pts = Z @ Q_orth.T + means[k]
        else:
            pts = Z + means[k]
        X_parts.append(pts)
        y_parts.append(np.full(cnt, k, dtype=np.int64))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    X_tr, X_va, X_te, y_te = _split_and_maybe_pca(X, y, seed=seed)
    return DatasetSplit(
        name=name,
        regime=regime,
        X_train=X_tr,
        X_val=X_va,
        X_test=X_te,
        n_components=n_components,
        recommended_cov=cov_type,
        y_test=y_te,
    )


def _load_openml_numeric(
    name: str,
    data_id: int,
    data_home: str,
    n_components: int,
    regime: str,
    recommended_cov: str,
    pca_dim: int | None = None,
    max_rows: int | None = None,
    seed: int = 42,
) -> DatasetSplit:
    bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=data_home, parser="auto")
    df = bunch.data.copy()
    # Keep numeric columns only (or ordinal-encode object/category cleanly)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        for col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                df[col] = df[col].astype("category").cat.codes.astype(np.float64)
        num_df = df
    X = num_df.to_numpy(dtype=np.float64)
    y = None
    if bunch.target is not None:
        try:
            y = bunch.target.astype("category").cat.codes.to_numpy()
        except Exception:
            y = None

    if max_rows is not None and len(X) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X = X[idx]
        if y is not None:
            y = y[idx]

    X_tr, X_va, X_te, y_te = _split_and_maybe_pca(X, y, seed=seed, pca_dim=pca_dim)
    return DatasetSplit(
        name=name,
        regime=regime,
        X_train=X_tr,
        X_val=X_va,
        X_test=X_te,
        n_components=n_components,
        recommended_cov=recommended_cov,
        y_test=y_te,
    )


def load_benchmark_suite(data_home: str = "data/openml") -> list[DatasetSplit]:
    """Load 22 diverse datasets (14 real OpenML/sklearn + 8 hard synthetic regimes)."""
    Path(data_home).mkdir(parents=True, exist_ok=True)
    datasets: list[DatasetSplit] = []

    # --- 1. Built-in sklearn datasets (fast, guaranteed offline-ready) ---
    dig = load_digits()
    X_tr, X_va, X_te, y_te = _split_and_maybe_pca(dig.data, dig.target, seed=101, pca_dim=32)
    datasets.append(DatasetSplit("digits_pca32", "real_small", X_tr, X_va, X_te, 10, "full", y_te))

    cal = fetch_california_housing(data_home=data_home)
    X_tr, X_va, X_te, _ = _split_and_maybe_pca(cal.data, None, seed=102)
    datasets.append(DatasetSplit("california_housing", "real_medium", X_tr, X_va, X_te, 16, "full"))

    cov = fetch_covtype(data_home=data_home)
    X_tr, X_va, X_te, y_te = _split_and_maybe_pca(cov.data, cov.target, seed=103)
    datasets.append(DatasetSplit("covtype_581k", "real_large", X_tr, X_va, X_te, 32, "full", y_te))

    # --- 2. OpenML real datasets across Small / Medium / High-D / Large ---
    openml_specs = [
        # (name, openml_id, K, regime, cov, pca_dim, max_rows)
        ("segment", 40984, 7, "real_small", "full", None, None),
        ("optdigits", 28, 10, "real_small", "full", 32, None),
        ("satimage", 182, 12, "real_small", "full", None, None),
        ("waveform5000", 60, 12, "real_small", "full", None, None),
        ("spambase", 44, 12, "real_small", "diag", None, None),
        ("pendigits", 32, 16, "real_medium", "full", None, None),
        ("letter", 6, 26, "real_medium", "full", None, None),
        ("magic04", 1120, 16, "real_medium", "full", None, None),
        ("electricity", 151, 16, "real_medium", "full", None, None),
        ("shuttle", 40685, 16, "real_medium", "full", None, None),
        ("mnist784_pca48", 554, 24, "real_highd", "full", 48, None),
        ("fashion_mnist_pca48", 40996, 24, "real_highd", "full", 48, None),
        ("miniboone", 41150, 24, "real_large", "diag", None, None),
    ]
    for idx, (name, did, K, reg, cov_t, pdim, mrows) in enumerate(openml_specs):
        try:
            ds = _load_openml_numeric(
                name=name,
                data_id=did,
                data_home=data_home,
                n_components=K,
                regime=reg,
                recommended_cov=cov_t,
                pca_dim=pdim,
                max_rows=mrows,
                seed=200 + idx,
            )
            datasets.append(ds)
        except Exception as exc:
            print(f"[WARN] Failed loading OpenML {name} (id={did}): {exc}")

    # --- 3. Hard synthetic regimes (disjoint seeds 20001..20008 from scaling calibration!) ---
    datasets.extend(
        [
            _make_hard_synthetic(
                "syn_anisotropic_30k", "syn_hard", 30_000, 24, 16, 20001, "full", anisotropy=50.0
            ),
            _make_hard_synthetic(
                "syn_imbalanced_40k",
                "syn_hard",
                40_000,
                16,
                24,
                20002,
                "full",
                anisotropy=15.0,
                dirichlet_alpha=0.3,
            ),
            _make_hard_synthetic(
                "syn_overlapping_30k",
                "syn_hard",
                30_000,
                32,
                20,
                20003,
                "full",
                anisotropy=12.0,
                center_scale=1.0,
            ),
            _make_hard_synthetic(
                "syn_highk64_60k", "syn_hard", 60_000, 16, 64, 20004, "full", anisotropy=10.0
            ),
            _make_hard_synthetic(
                "syn_highdim128_50k",
                "syn_highd",
                50_000,
                128,
                24,
                20005,
                "diag",
                anisotropy=8.0,
                diag_only=True,
            ),
            _make_hard_synthetic(
                "syn_scale200k", "syn_large", 200_000, 32, 32, 20006, "full", anisotropy=12.0
            ),
            _make_hard_synthetic(
                "syn_scale500k_diag",
                "syn_large",
                500_000,
                48,
                32,
                20007,
                "diag",
                anisotropy=10.0,
                diag_only=True,
            ),
        ]
    )
    return datasets

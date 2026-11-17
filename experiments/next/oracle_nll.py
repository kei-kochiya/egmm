"""Oracle test NLL of the large synthetic mixtures under their true generating parameters.

    python experiments/next/oracle_nll.py   # -> docs/experiments/next/oracle_large.csv   (CPU only, ~10 min)

Replays the random stream of ``large_data.make_large_synthetic`` to recover each component's mean, covariance
(``rot @ diag(scales**2) @ rot.T``; diagonal: ``diag(scales**2)``) and mixture weight, rebuilds the same test
split, and evaluates the mean negative log-likelihood per test row of the true mixture (raw units, nats), the
reference that no fitted model can beat in expectation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.next.large_data import SYNTHETIC_LARGE, make_large_synthetic  # noqa: E402


def true_parameters(n_train, n_features, n_components, cov_type, anisotropy, seed, dirichlet_alpha=1.0, center_scale=2.2):
    rng = np.random.default_rng(seed)
    n_total = int(round(n_train / 0.70))
    weights = rng.dirichlet(np.full(n_components, dirichlet_alpha))
    counts = rng.multinomial(n_total, weights)
    means = rng.normal(scale=center_scale, size=(n_components, n_features))
    scales = np.geomspace(1.0 / math.sqrt(anisotropy), math.sqrt(anisotropy), n_features)
    covs = []
    for cnt in counts:
        rot = None if cov_type == "diag" else np.linalg.qr(rng.normal(size=(n_features, n_features)))[0]
        done = 0
        while done < cnt:  # advance the stream exactly as the generator does
            m = min(cnt - done, 1_000_000)
            rng.standard_normal(size=(m, n_features))
            done += m
        base = np.diag(scales ** 2)
        covs.append(base if rot is None else rot @ base @ rot.T)
    return weights, means, np.stack(covs)


def mixture_nll(X, weights, means, covs, chunk=200_000):
    K, D = means.shape
    chol = np.linalg.cholesky(covs)
    logdet = 2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
    inv_chol = np.linalg.inv(chol)
    total = 0.0
    for st in range(0, X.shape[0], chunk):
        x = X[st:st + chunk].astype(np.float64)
        lp = np.empty((x.shape[0], K))
        for k in range(K):
            z = (x - means[k]) @ inv_chol[k].T
            lp[:, k] = np.log(weights[k]) - 0.5 * (D * math.log(2 * math.pi) + logdet[k] + (z * z).sum(axis=1))
        mx = lp.max(axis=1, keepdims=True)
        total += float((mx[:, 0] + np.log(np.exp(lp - mx).sum(axis=1))).sum())
    return -total / X.shape[0]


def main() -> None:
    rows = []
    for name, n, d, k, cov, aniso, seed in SYNTHETIC_LARGE:
        w, mu, sig = true_parameters(n, d, k, cov, aniso, seed)
        ds = make_large_synthetic(name, n, d, k, cov, aniso, seed)
        rows.append(dict(dataset=name, oracle_test_nll=mixture_nll(ds.X_test, w, mu, sig), n_test=ds.X_test.shape[0]))
        print(rows[-1], flush=True)
        del ds
    out = ROOT / "docs" / "experiments" / "next" / "oracle_large.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

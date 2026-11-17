"""The stronger batch-EM baselines (experiments/next/baselines_fast.py) must stop at the same iteration and
reach the same likelihood as TorchBatchGMM; they only remove host synchronizations and launch overhead."""

import numpy as np
import pytest
import torch

from egmm.baselines import TorchBatchGMM
from experiments.next.baselines_fast import TorchBatchGMMFast, TorchBatchGMMGraph


def _data(seed=0, n=6000, d=8, k=5):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(k, d))
    X = np.concatenate([rng.normal(size=(n // k, d)) * rng.uniform(0.5, 2.0, d) + c for c in centers])
    return X[rng.permutation(len(X))].astype(np.float32)


def _fit(cls, X, device, cov, **kw):
    return cls(n_components=5, covariance_type=cov, max_iter=50, tol=1e-3, device=device, random_state=0, **kw).fit(X)


@pytest.mark.parametrize("cov", ["full", "diag"])
@pytest.mark.parametrize("check_every", [1, 3, 5])
def test_fast_matches_reference_cpu(cov, check_every):
    X = _data()
    ref = _fit(TorchBatchGMM, X, "cpu", cov)
    fast = _fit(TorchBatchGMMFast, X, "cpu", cov, check_every=check_every)
    assert fast.n_iter_ == ref.n_iter_
    assert fast.converged_ == ref.converged_
    assert abs(fast.score(X) - ref.score(X)) < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph path needs a GPU")
@pytest.mark.parametrize("cov", ["full", "diag"])
def test_graph_matches_reference_cuda(cov):
    X = _data(seed=1)
    ref = _fit(TorchBatchGMM, X, "cuda", cov)
    for cls in (TorchBatchGMMFast, TorchBatchGMMGraph):
        m = _fit(cls, X, "cuda", cov)
        assert m.n_iter_ == ref.n_iter_
        assert abs(m.score(X) - ref.score(X)) < 1e-3


def test_revival_controls_log_their_split_count_cpu():
    """TorchBatchGMM.fit resets revivals_count_ after the EM loop; the revival controls must restore it, and the
    matched-intensity control must make exactly n_forced split moves (well-separated data: no natural deaths)."""
    from experiments.next.baselines_fast import TorchBatchGMMForcedSplit, TorchBatchGMMRevival

    X = _data(seed=2)
    rev = _fit(TorchBatchGMMRevival, X, "cpu", "full")
    assert rev.revivals_count_ == rev.n_revivals_
    forced = _fit(TorchBatchGMMForcedSplit, X, "cpu", "full", n_forced=6)
    assert forced.revivals_count_ == forced.n_revivals_ == 6
    assert np.isfinite(forced.score(X))

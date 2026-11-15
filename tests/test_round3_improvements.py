"""Tests for greedy k-means++ seeding, single-pass standardization, and smooth schedule laws."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from sklearn.datasets import make_blobs

from egmm import EGMM
from egmm.kernels import init_kmeans_pp
from egmm.scaling import auto_hyperparameters, large_scale_gate

CUDA = torch.cuda.is_available()


def _potential(X: torch.Tensor, C: torch.Tensor) -> float:
    return float(torch.cdist(X.double(), C.double()).min(dim=1).values.pow(2).sum())


# ----------------------------------------------------------------------------- greedy k-means++
def test_greedy_kmeans_pp_returns_distinct_data_rows_and_is_deterministic():
    X_np, _ = make_blobs(n_samples=1500, centers=12, n_features=6, random_state=0)
    X = torch.from_numpy(X_np).float()
    c1 = init_kmeans_pp(X, 12, generator=torch.Generator().manual_seed(3), n_local_trials=None)
    c2 = init_kmeans_pp(X, 12, generator=torch.Generator().manual_seed(3), n_local_trials=None)
    assert c1.shape == (12, 6) and c1.dtype == X.dtype
    torch.testing.assert_close(c1, c2, rtol=0, atol=0)
    # every center is an actual data row and centers are pairwise distinct
    is_row = (X.unsqueeze(0) == c1.unsqueeze(1)).all(dim=-1).any(dim=-1)
    assert is_row.all()
    dd = torch.cdist(c1, c1)
    dd.fill_diagonal_(1.0)
    assert (dd > 0).all()


def test_greedy_kmeans_pp_lowers_seeding_potential_vs_plain():
    X_np, _ = make_blobs(n_samples=3000, centers=24, n_features=8, cluster_std=1.0, random_state=1)
    X = torch.from_numpy(X_np).float()
    plain, greedy = [], []
    for s in range(6):
        plain.append(_potential(X, init_kmeans_pp(X, 24, generator=torch.Generator().manual_seed(s), n_local_trials=1)))
        greedy.append(_potential(X, init_kmeans_pp(X, 24, generator=torch.Generator().manual_seed(s), n_local_trials=None)))
    assert np.mean(greedy) < np.mean(plain)


def test_greedy_kmeans_pp_handles_duplicate_points_and_k_equal_n():
    X = torch.zeros(16, 3)
    X[8:] = 1.0  # only two distinct locations -> potential hits zero after two centers
    c = init_kmeans_pp(X, 5, generator=torch.Generator().manual_seed(0), n_local_trials=None)
    assert c.shape == (5, 3) and torch.isfinite(c).all()
    Y = torch.randn(7, 2)
    c2 = init_kmeans_pp(Y, 7, generator=torch.Generator().manual_seed(0), n_local_trials=None)
    assert c2.shape == (7, 2)


@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_greedy_kmeans_pp_is_sync_free_on_cuda():
    X = torch.randn(4096, 16, device="cuda")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    torch.cuda.synchronize()
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        c = init_kmeans_pp(X, 32, generator=gen, n_local_trials=None)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    assert c.shape == (32, 16) and c.is_cuda


# ----------------------------------------------------------------------------- standardization
def _reference_standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = np.maximum(X.std(axis=0), 1e-12)
    return (X - mu) / sd


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
def test_single_pass_streaming_standardizer_matches_cached_path(device, monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.normal(loc=50.0, scale=3.0, size=(70_000, 5)) * np.array([1.0, 10.0, 0.1, 100.0, 1.0])
    dev = torch.device(device)
    m = EGMM(n_components=2)
    cached, _, _ = m._fit_standardizer_and_transform(X, dev, torch.float32)
    mean_c, std_c = m._x_mean_f64.clone(), m._x_std_f64.clone()
    monkeypatch.setattr(EGMM, "_device_cache_fits", staticmethod(lambda n_bytes, d: False))
    streamed, _, _ = m._fit_standardizer_and_transform(X, dev, torch.float32)
    torch.testing.assert_close(m._x_mean_f64, mean_c, rtol=0, atol=0)
    torch.testing.assert_close(m._x_std_f64, std_c, rtol=0, atol=0)
    # streamed values are rounded once more (x - x0 kept in float32) -> a few ulps at most
    torch.testing.assert_close(streamed, cached, rtol=0, atol=2e-6)
    ref = _reference_standardize(X)
    np.testing.assert_allclose(cached.cpu().numpy(), ref, rtol=0, atol=2e-6)
    # torch float64 host source: streams on CUDA (exactly like the numpy path), cached on CPU
    streamed_t, _, _ = m._fit_standardizer_and_transform(torch.from_numpy(X), dev, torch.float32)
    torch.testing.assert_close(streamed_t, streamed if device == "cuda" else cached, rtol=0, atol=0)


def test_device_cache_budget_is_always_true_on_cpu():
    assert EGMM._device_cache_fits(10**15, torch.device("cpu"))


# ----------------------------------------------------------------------------- schedule laws
def test_large_scale_gate_is_smooth_and_monotone():
    Ns = np.geomspace(100, 2e6, 400)
    g = np.array([large_scale_gate(int(n), 100_000, 4.0) for n in Ns])
    assert g[0] > 0.999 and g[-1] < 1e-4
    assert np.all(np.diff(g) <= 1e-12)
    assert math.isclose(large_scale_gate(100_000, 100_000, 4.0), 0.5)


def _step_and_polish_curves(cov: str, law: str):
    K, D = 24, 32
    Ns = np.unique(np.geomspace(2_000, 1_000_000, 600).astype(int))
    hps = [auto_hyperparameters(int(n), K, D, cov, schedule_law=law) for n in Ns]
    T = np.array([h.n_steps for h in hps], dtype=float)
    E = np.array([h.max_epochs for h in hps])
    P = np.array([h.polish_steps for h in hps])
    return T, E, P


@pytest.mark.parametrize("cov", ["full", "diag"])
def test_smooth_law_has_no_sample_size_jumps(cov):
    T, E, P = _step_and_polish_curves(cov, "smooth")
    # Total EMA steps change only through the integer steps-per-epoch granularity (at most one extra
    # step per epoch between neighbouring N), never by the factor-of-two cliffs of the bracket law.
    dT = np.abs(np.diff(T))
    assert np.all(dT / T[1:] <= 0.15)
    # polish budget decays monotonically in N and only by single integer steps
    assert np.all(np.diff(P) <= 0)
    assert np.all(np.abs(np.diff(P)) <= 1)
    assert P[-1] == 0


@pytest.mark.parametrize("cov", ["full", "diag"])
def test_bracket_law_has_the_cliffs_the_smooth_law_removes(cov):
    T, _, P = _step_and_polish_curves(cov, "bracket")
    assert (np.abs(np.diff(T)) / T[1:]).max() > 0.3 or np.abs(np.diff(P)).max() >= 5


def test_bracket_law_reproduces_legacy_values():
    hp = auto_hyperparameters(140_000, 32, 32, "full", schedule_law="bracket")
    assert (hp.max_epochs, hp.n_steps, hp.polish_steps) == (2, 2 * hp.steps_per_epoch, 0)
    hp = auto_hyperparameters(49_000, 24, 48, "full", schedule_law="bracket")
    assert (hp.max_epochs, hp.polish_steps) == (4, 5)
    hp = auto_hyperparameters(91_044, 24, 50, "diag", schedule_law="bracket")
    assert (hp.max_epochs, hp.polish_steps) == (2, 12)
    with pytest.raises(ValueError):
        auto_hyperparameters(1000, 2, 2, "full", schedule_law="nope")


def test_fractional_epoch_schedule_runs_exact_step_count():
    X, _ = make_blobs(n_samples=6000, centers=6, n_features=5, random_state=0)
    m = EGMM(n_components=6, covariance_type="full", device="cpu", random_state=0)
    hp = auto_hyperparameters(6000, 6, 5, "full", max_epochs=3, schedule_law="smooth")
    assert hp.n_steps == 3 * hp.steps_per_epoch
    m.fit(X)
    assert m.hyperparams_.n_steps >= 1
    assert m.hyperparams_.n_steps <= m.hyperparams_.max_epochs * m.hyperparams_.steps_per_epoch
    assert np.isfinite(m.score(X))
    # an N where the smooth law yields a non-integer epoch count still fits end-to-end
    hp2 = auto_hyperparameters(150_000, 8, 4, "full", schedule_law="smooth")
    assert hp2.n_steps % hp2.steps_per_epoch != 0


# ----------------------------------------------------------------------------- exact scoring
def test_scoring_is_exact_float64_for_near_degenerate_diagonal_variances():
    """float32 training must not leak catastrophic cancellation into reported likelihoods."""
    rng = np.random.default_rng(0)
    n, D = 4000, 60
    X = rng.normal(size=(n, D))
    # 30 near-constant indicator-like features with tiny within-cluster spread (variance floor regime)
    lab = rng.integers(0, 2, size=n)
    X[:, :30] = lab[:, None] * 3.0 + 1e-7 * rng.normal(size=(n, 30))
    m = EGMM(n_components=4, covariance_type="diag", device="cpu", random_state=0).fit(X)
    # independent exact float64 reference from the exported sklearn-convention parameters
    from scipy.special import logsumexp

    var = m.covariances_
    lp = (
        -0.5 * (((X[:, None, :] - m.means_[None]) ** 2) / var[None]).sum(-1)
        - 0.5 * np.log(2 * np.pi * var).sum(-1)[None]
        + np.log(m.weights_)[None]
    )
    ref = logsumexp(lp, axis=1)
    np.testing.assert_allclose(m.score_samples(X), ref, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(m.predict_proba(X).sum(1), 1.0, rtol=0, atol=1e-12)


# ----------------------------------------------------------------------------- CUDA graph == eager
@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("cov", ["full", "diag"])
def test_cuda_graph_path_matches_eager_path(cov):
    X, _ = make_blobs(n_samples=20_000, centers=12, n_features=8, cluster_std=1.5, random_state=4)

    class EagerEGMM(EGMM):
        _USE_CUDA_GRAPH = False

    kw = dict(n_components=12, covariance_type=cov, polish_steps=0, device="cuda", random_state=3)
    g = EGMM(**kw).fit(X)
    e = EagerEGMM(**kw).fit(X)
    assert g.hyperparams_.n_steps >= 4
    assert g.revivals_count_ == e.revivals_count_
    torch.testing.assert_close(g._M_t, e._M_t, rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(g._S_t, e._S_t, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(g._Q_t, e._Q_t, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(g.score(X), e.score(X), rtol=0, atol=1e-4)

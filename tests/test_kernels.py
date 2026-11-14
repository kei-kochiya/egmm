import numpy as np
import pytest
import torch
from scipy.stats import multivariate_normal

from egmm.kernels import (
    compute_log_gaussian_prob,
    e_step_batch,
    init_kmeans_pp,
    moments_to_params,
    params_to_moments,
)


@pytest.mark.parametrize("cov_type", ["diag", "full"])
def test_compute_log_gaussian_prob_matches_scipy(cov_type):
    rng = np.random.default_rng(42)
    N, K, D = 25, 4, 6
    X_np = rng.normal(size=(N, D)).astype(np.float64)
    means_np = rng.normal(scale=2.0, size=(K, D)).astype(np.float64)

    if cov_type == "diag":
        covs_np = rng.uniform(0.3, 2.5, size=(K, D)).astype(np.float64)
    else:
        A = rng.normal(size=(K, D, D)).astype(np.float64)
        covs_np = np.einsum("kij,klj->kil", A, A) + 0.5 * np.eye(D)[None, :, :]

    expected = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        cov_k = np.diag(covs_np[k]) if cov_type == "diag" else covs_np[k]
        expected[:, k] = multivariate_normal.logpdf(X_np, mean=means_np[k], cov=cov_k)

    X = torch.from_numpy(X_np)
    means = torch.from_numpy(means_np)
    covs = torch.from_numpy(covs_np)

    got = compute_log_gaussian_prob(X, means, covs, covariance_type=cov_type).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("cov_type", ["diag", "full"])
def test_e_step_batch_sufficient_statistics_exact(cov_type):
    rng = np.random.default_rng(7)
    B, K, D = 32, 3, 5
    X = torch.from_numpy(rng.normal(size=(B, D)).astype(np.float64))
    weights = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    means = torch.from_numpy(rng.normal(size=(K, D)).astype(np.float64))
    if cov_type == "diag":
        covs = torch.ones((K, D), dtype=torch.float64) * 1.2
    else:
        covs = torch.eye(D, dtype=torch.float64).unsqueeze(0).repeat(K, 1, 1) * 1.2

    r, log_prob_norm, m_bar, s_bar, q_bar = e_step_batch(
        X, weights, means, covs, covariance_type=cov_type
    )

    assert r.shape == (B, K)
    assert torch.isfinite(log_prob_norm)
    torch.testing.assert_close(r.sum(dim=1), torch.ones(B, dtype=torch.float64))
    torch.testing.assert_close(m_bar.sum(), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(m_bar, r.mean(dim=0))
    torch.testing.assert_close(s_bar, (r.T @ X) / B)
    if cov_type == "diag":
        torch.testing.assert_close(q_bar, (r.T @ (X * X)) / B)
    else:
        expected_q = torch.einsum("bk,bd,be->kde", r, X, X) / B
        torch.testing.assert_close(q_bar, expected_q)


@pytest.mark.parametrize("cov_type", ["diag", "full"])
def test_moments_params_round_trip_and_psd_guard(cov_type):
    rng = np.random.default_rng(123)
    K, D = 4, 5
    weights = torch.tensor([0.1, 0.3, 0.4, 0.2], dtype=torch.float64)
    means = torch.from_numpy(rng.normal(size=(K, D)).astype(np.float64))
    if cov_type == "diag":
        covs = torch.from_numpy(rng.uniform(0.2, 2.0, size=(K, D)).astype(np.float64))
    else:
        A = rng.normal(size=(K, D, D)).astype(np.float64)
        covs = torch.from_numpy(
            np.einsum("kij,klj->kil", A, A) + 0.3 * np.eye(D)[None, :, :]
        )

    M, S, Q = params_to_moments(weights, means, covs, covariance_type=cov_type)
    w2, m2, c2 = moments_to_params(M, S, Q, covariance_type=cov_type, reg_covar=1e-6)

    torch.testing.assert_close(w2, weights)
    torch.testing.assert_close(m2, means)
    reg = 1e-6 if cov_type == "diag" else 1e-6 * torch.eye(D, dtype=torch.float64).unsqueeze(0)
    torch.testing.assert_close(c2, covs + reg, rtol=1e-5, atol=1e-5)

    # Collinear / negative numerical perturbation must still yield strict PSD
    Q_bad = S.unsqueeze(-1) * means.unsqueeze(-2) if cov_type == "full" else S * means
    if cov_type == "full":
        Q_bad = Q_bad - 1e-2 * torch.eye(D, dtype=torch.float64).unsqueeze(0)
    else:
        Q_bad = Q_bad - 1e-2
    _, _, c_safe = moments_to_params(M, S, Q_bad, covariance_type=cov_type, reg_covar=1e-5)
    if cov_type == "diag":
        assert (c_safe >= 1e-5 - 1e-12).all()
    else:
        eigvals = torch.linalg.eigvalsh(c_safe)
        assert (eigvals >= 1e-5 - 1e-9).all()


def test_init_kmeans_pp_shapes_and_distinct():
    torch.manual_seed(0)
    X = torch.randn(200, 8)
    centers = init_kmeans_pp(X, n_components=10, generator=torch.Generator().manual_seed(0))
    assert centers.shape == (10, 8)
    assert torch.isfinite(centers).all()
    dists = torch.cdist(centers, centers)
    dists.fill_diagonal_(1.0)
    assert (dists > 1e-6).all()


def test_moments_to_params_and_precisions_float32_high_dim_ill_conditioned():
    from egmm.kernels import moments_to_params_and_precisions

    torch.manual_seed(0)
    K, D = 8, 96
    weights = torch.full((K,), 1.0 / K, dtype=torch.float32)
    means = torch.randn(K, D, dtype=torch.float32) * 3.0
    # Construct rank-1 / ill-conditioned second moments with tiny negative rounding error in float32
    covs = torch.zeros(K, D, D, dtype=torch.float32)
    v = torch.randn(K, D, 1, dtype=torch.float32)
    covs = 20.0 * torch.bmm(v, v.transpose(-1, -2))
    covs[0] -= 1e-4 * torch.eye(D, dtype=torch.float32)
    M, S, Q = params_to_moments(weights, means, covs, covariance_type="full")
    w, m, c, pf, ldh = moments_to_params_and_precisions(
        M, S, Q, covariance_type="full", reg_covar=1e-6
    )
    assert torch.isfinite(w).all()
    assert torch.isfinite(m).all()
    assert torch.isfinite(c).all()
    assert torch.isfinite(pf).all()
    assert torch.isfinite(ldh).all()


@pytest.mark.parametrize("cov_type", ["diag", "full"])
def test_compute_log_gaussian_prob_and_estep_chunking_exact(cov_type):
    from egmm.kernels import compute_log_gaussian_prob_from_precisions, prepare_precisions

    rng = np.random.default_rng(99)
    B, K, D = 250, 6, 8
    X = torch.from_numpy(rng.normal(loc=50.0, scale=2.0, size=(B, D)).astype(np.float32))
    weights = torch.full((K,), 1.0 / K, dtype=torch.float32)
    means = torch.from_numpy(rng.normal(loc=50.0, scale=2.0, size=(K, D)).astype(np.float32))
    if cov_type == "diag":
        covs = torch.from_numpy(rng.uniform(0.4, 2.5, size=(K, D)).astype(np.float32))
    else:
        A = rng.normal(size=(K, D, D)).astype(np.float32)
        covs = torch.from_numpy(np.einsum("kij,klj->kil", A, A) + 0.5 * np.eye(D, dtype=np.float32)[None, :, :])

    pf, ldh = prepare_precisions(covs, covariance_type=cov_type)
    lp_full = compute_log_gaussian_prob_from_precisions(
        X, means, pf, ldh, covariance_type=cov_type, max_work_elements=32_000_000
    )
    lp_chunk = compute_log_gaussian_prob_from_precisions(
        X, means, pf, ldh, covariance_type=cov_type, max_work_elements=300
    )
    torch.testing.assert_close(lp_chunk, lp_full, rtol=1e-5, atol=1e-5)

    r1, ll1, m1, s1, q1 = e_step_batch(
        X, weights, means, prec_factor=pf, log_det_half=ldh,
        covariance_type=cov_type, stats_dtype=torch.float64, max_work_elements=32_000_000
    )
    r2, ll2, m2, s2, q2 = e_step_batch(
        X, weights, means, prec_factor=pf, log_det_half=ldh,
        covariance_type=cov_type, stats_dtype=torch.float64, max_work_elements=300
    )
    torch.testing.assert_close(r2, r1, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ll2, ll1, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(m2, m1, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(s2, s1, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(q2, q1, rtol=1e-6, atol=1e-6)



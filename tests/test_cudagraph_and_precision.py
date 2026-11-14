"""Targeted tests for centered E-step precision, branchless M-step, adaptive scaling, and CUDA Graph."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.datasets import make_blobs

from egmm import EGMM
from egmm.kernels import e_step_batch, moments_to_params_and_precisions
from egmm.scaling import auto_hyperparameters


def test_centered_e_step_batch_prevents_offset_cancellation():
    rng = np.random.default_rng(123)
    # Offset = 2000.0 with std = 0.05: naive float32 (r.T @ (X*X)) loses ~1e-2 relative variance accuracy.
    X_np = 2000.0 + 0.05 * rng.standard_normal((2048, 4))
    X_f32 = torch.from_numpy(X_np.astype(np.float32))
    weights = torch.tensor([1.0], dtype=torch.float32)
    means = torch.full((1, 4), 2000.0, dtype=torch.float32)
    covs_f32 = torch.eye(4, dtype=torch.float32).unsqueeze(0) * (0.05**2)

    _, _, m_bar, s_bar, q_bar = e_step_batch(
        X_f32, weights, means, covariances=covs_f32, covariance_type="full", stats_dtype=torch.float64
    )
    assert m_bar.dtype == torch.float64
    assert s_bar.dtype == torch.float64
    assert q_bar.dtype == torch.float64

    mu_est = s_bar[0] / m_bar[0]
    cov_est = q_bar[0] / m_bar[0] - torch.outer(mu_est, mu_est)
    cov_true = torch.from_numpy(np.cov(X_np, rowvar=False, bias=True))
    torch.testing.assert_close(cov_est, cov_true, atol=1e-5, rtol=1e-3)


def test_moments_to_params_sync_free_and_f32_cholesky_fallback():
    K, D = 3, 4
    M = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float64)
    means = torch.randn(K, D, dtype=torch.float64)
    S = M.unsqueeze(-1) * means
    eye = torch.eye(D, dtype=torch.float64).unsqueeze(0).expand(K, D, D)
    Q = M.view(K, 1, 1) * (0.5 * eye + torch.bmm(means.unsqueeze(-1), means.unsqueeze(-2)))
    # Inject a strictly non-PSD covariance slice on component 1
    Q[1] = M[1] * (torch.outer(means[1], means[1]) - 0.5 * torch.eye(D, dtype=torch.float64))

    w, mu, covs, pf, ldh = moments_to_params_and_precisions(
        M,
        S,
        Q,
        covariance_type="full",
        reg_covar=1e-6,
        allow_sync=False,
        fast_cholesky_f32=True,
    )
    assert pf.dtype == torch.float32
    assert ldh.dtype == torch.float32
    assert torch.isfinite(w).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(pf).all()
    assert torch.isfinite(ldh).all()
    assert torch.isfinite(covs).all()


def test_scaling_law_eff_df_smooth_adaptation():
    low_df = auto_hyperparameters(1437, 16, 32, covariance_type="full")
    high_df = auto_hyperparameters(8793, 16, 16, covariance_type="full")
    diag_hp = auto_hyperparameters(20000, 16, 20, covariance_type="diag")

    assert 0.10 <= high_df.init_usage_scale <= low_df.init_usage_scale <= 1.0
    assert low_df.alpha_0 < high_df.alpha_0
    assert diag_hp.revive_points > 0
    assert high_df.revive_points > 0
    assert diag_hp.polish_steps > 0


def test_scaling_law_decouples_alpha_and_exposes_polish_and_init_scale():
    hp_auto = auto_hyperparameters(20000, 16, 16, covariance_type="full")
    hp_manual_a = auto_hyperparameters(20000, 16, 16, covariance_type="full", alpha=0.55)
    # Passing manual alpha must NOT silently disable polish_steps or init_usage_scale
    assert hp_manual_a.polish_steps == hp_auto.polish_steps
    assert hp_manual_a.init_usage_scale == pytest.approx(hp_auto.init_usage_scale)

    # Explicit overrides for polish_steps and init_usage_scale
    hp_override = auto_hyperparameters(
        20000, 16, 16, covariance_type="full", polish_steps=0, init_usage_scale=1.0
    )
    assert hp_override.polish_steps == 0
    assert hp_override.init_usage_scale == 1.0

    # Batch size should never exceed cap (8192 for full, 16384 for diag) when auto
    hp_99k = auto_hyperparameters(99999, 16, 16, covariance_type="full")
    assert hp_99k.batch_size <= 8192



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CUDA Graph tests")
def test_egmm_cuda_graph_consecutive_fits_and_partial_fit():
    X, _ = make_blobs(n_samples=1500, n_features=8, centers=4, cluster_std=0.9, random_state=42)
    X_tr, X_te = X[:1200].astype(np.float32), X[1200:].astype(np.float32)

    m1 = EGMM(n_components=4, covariance_type="full", n_init=2, device="cuda", random_state=7).fit(X_tr)
    m2 = EGMM(n_components=4, covariance_type="full", n_init=2, device="cuda", random_state=7).fit(X_tr)
    m_diag = EGMM(n_components=4, covariance_type="diag", device="cuda", random_state=7).fit(X_tr)
    # Test explicit indivisible batch_size + revival trigger inside CUDA Graph training
    m_odd = EGMM(
        n_components=6,
        covariance_type="full",
        batch_size=250,
        max_epochs=4,
        revive_every=2,
        device="cuda",
        random_state=11,
    ).fit(X_tr)

    assert np.isfinite(m1.score(X_te))
    assert np.isfinite(m_diag.score(X_te))
    assert np.isfinite(m_odd.score(X_te))
    np.testing.assert_allclose(m1.score(X_te), m2.score(X_te), atol=1e-4)

    # Incremental partial_fit on GPU should continue seamlessly
    m1.partial_fit(X_te)
    assert np.isfinite(m1.score(X_te))

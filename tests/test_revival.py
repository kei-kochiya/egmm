import pytest
import torch

from egmm.kernels import moments_to_params, params_to_moments
from egmm.revival import revive_dead_components


@pytest.mark.parametrize("cov_type", ["diag", "full"])
def test_revive_dead_components_preserves_moments_when_dead_mass_zero(cov_type):
    K, D = 4, 5
    weights = torch.tensor([0.8, 0.2, 0.0, 0.0], dtype=torch.float64)
    means = torch.tensor(
        [
            [2.0, -1.0, 0.5, 0.0, 1.0],
            [-2.0, 1.0, -0.5, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    if cov_type == "diag":
        covs = torch.ones((K, D), dtype=torch.float64) * 0.5
        covs[0, 2] = 9.0  # Strong variance along dim 2
    else:
        covs = torch.eye(D, dtype=torch.float64).unsqueeze(0).repeat(K, 1, 1) * 0.5
        # Stretch component 0 along [1, 1, 0, 0, 0] / sqrt(2)
        v = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64) / (2.0**0.5)
        covs[0] = covs[0] + 8.0 * torch.outer(v, v)

    M, S, Q = params_to_moments(weights, means, covs, covariance_type=cov_type)
    # Zero out dead components completely
    M[2:] = 0.0
    S[2:] = 0.0
    Q[2:] = 0.0

    M_tot_before = M.sum().clone()
    S_tot_before = S.sum(dim=0).clone()
    Q_tot_before = Q.sum(dim=0).clone()

    M_new, S_new, Q_new, n_revived = revive_dead_components(
        M,
        S,
        Q,
        covariance_type=cov_type,
        reg_covar=1e-6,
        dead_ratio=0.05,
    )

    assert n_revived == 2
    assert (M_new > 0.05 / K).all()
    torch.testing.assert_close(M_new.sum(), M_tot_before)
    torch.testing.assert_close(S_new.sum(dim=0), S_tot_before)
    torch.testing.assert_close(Q_new.sum(dim=0), Q_tot_before)

    # All resulting covariances must be strictly positive definite
    _, new_means, new_covs = moments_to_params(
        M_new, S_new, Q_new, covariance_type=cov_type, reg_covar=1e-6
    )
    assert torch.isfinite(new_means).all()
    if cov_type == "diag":
        assert (new_covs > 1e-6).all()
    else:
        eigvals = torch.linalg.eigvalsh(new_covs)
        assert (eigvals > 1e-6).all()


@pytest.mark.parametrize("cov_type", ["diag", "full"])
def test_revive_dead_components_noop_when_all_healthy(cov_type):
    K, D = 3, 4
    weights = torch.tensor([0.3, 0.4, 0.3], dtype=torch.float64)
    means = torch.randn(K, D, dtype=torch.float64)
    if cov_type == "diag":
        covs = torch.ones((K, D), dtype=torch.float64)
    else:
        covs = torch.eye(D, dtype=torch.float64).unsqueeze(0).repeat(K, 1, 1)

    M, S, Q = params_to_moments(weights, means, covs, covariance_type=cov_type)
    M2, S2, Q2, n_revived = revive_dead_components(
        M, S, Q, covariance_type=cov_type, reg_covar=1e-6, dead_ratio=0.05
    )
    assert n_revived == 0
    torch.testing.assert_close(M2, M)
    torch.testing.assert_close(S2, S)
    torch.testing.assert_close(Q2, Q)

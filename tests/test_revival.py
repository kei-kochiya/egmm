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


def test_revive_dead_components_wedderburn_psd_on_ill_conditioned_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required to test GPU power-iteration Wedderburn deflation")
    # Construct a covariance where top two eigenvalues are close (100.0 and 80.0)
    # so 5-step power iteration starting from e_0 retains a strong component (0.8^5 = 0.328)
    # along the second eigenvector. Naive v*v^T subtraction violates Loewner bound Sigma' >= (1-delta^2)*Sigma,
    # whereas Wedderburn deflation u = Sigma v / sqrt(v^T Sigma v) guarantees Sigma' >= (1-delta^2)*Sigma.
    D = 4
    Q_orth = torch.tensor(
        [
            [0.70710678, 0.70710678, 0.0, 0.0],
            [0.70710678, -0.70710678, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device="cuda",
        dtype=torch.float64,
    )
    eigs = torch.tensor([100.0, 80.0, 10.0, 1.0], device="cuda", dtype=torch.float64)
    cov0 = Q_orth @ torch.diag(eigs) @ Q_orth.T
    cov0 = 0.5 * (cov0 + cov0.T)

    weights = torch.tensor([0.99, 0.01], device="cuda", dtype=torch.float64)
    means = torch.zeros((2, D), device="cuda", dtype=torch.float64)
    covs = torch.stack([cov0, torch.eye(D, device="cuda", dtype=torch.float64)], dim=0)
    M, S, Q = params_to_moments(weights, means, covs, covariance_type="full")
    M[1] = 0.0
    S[1] = 0.0
    Q[1] = 0.0

    M_new, S_new, Q_new, n_rev = revive_dead_components(
        M, S, Q, covariance_type="full", reg_covar=0.0, dead_ratio=0.05, split_delta=0.5
    )
    assert n_rev == 1
    mu_new = S_new[0] / M_new[0]
    cov_new = Q_new[0] / M_new[0] - torch.outer(mu_new, mu_new)
    # Loewner ordering: Sigma' - (1 - delta^2) Sigma must be PSD (>= 0)
    loewner_diff = cov_new - (1.0 - 0.5**2) * cov0
    min_loewner_eig = float(torch.linalg.eigvalsh(loewner_diff).min().item())
    assert min_loewner_eig >= -1e-4



def test_revive_dead_components_diag_uses_trace_spread():
    # Donor 0 has total variance 10.0 spread across 5 dims (2.0 each).
    # Donor 1 has total variance 3.5 with a single coordinate spike of 3.1.
    # Trace-based spread scoring should pick Donor 0 (mass 0.5 * 10 = 5.0) over Donor 1 (0.5 * 3.5 = 1.75).
    K, D = 3, 5
    weights = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    means = torch.zeros((K, D), dtype=torch.float64)
    covs = torch.tensor(
        [
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [3.1, 0.1, 0.1, 0.1, 0.1],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    M, S, Q = params_to_moments(weights, means, covs, covariance_type="diag")
    M[2] = 0.0
    S[2] = 0.0
    Q[2] = 0.0
    M_new, _, _, n_rev = revive_dead_components(M, S, Q, covariance_type="diag", dead_ratio=0.05)
    assert n_rev == 1
    # Donor 0 was split into (0, 2), so M_new[0] == 0.25 and M_new[1] remains 0.50
    torch.testing.assert_close(M_new[0], torch.tensor(0.25, dtype=torch.float64))
    torch.testing.assert_close(M_new[1], torch.tensor(0.50, dtype=torch.float64))


"""Dead-component detection and moment-preserving eigen-split revival for e-GMM."""

from __future__ import annotations

import math

import torch

from egmm.kernels import CovarianceType, params_to_moments


def revive_dead_components(
    M: torch.Tensor,
    S: torch.Tensor,
    Q: torch.Tensor,
    covariance_type: CovarianceType = "full",
    reg_covar: float = 1e-6,
    dead_ratio: float = 0.05,
    split_delta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Revive dead/starving components by splitting high-mass/high-variance donors.

    Preserves total mixture mass, mean, and second moment when dead components have zero mass,
    and preserves total mixture mass while keeping covariances strictly PSD in general.
    """
    K = M.shape[0]
    if K <= 1:
        return M, S, Q, 0

    total_mass = float(M.sum().item())
    if total_mass <= 0.0:
        return M, S, Q, 0

    dead_thresh = dead_ratio * (total_mass / K)
    dead_indices = torch.nonzero(M < dead_thresh, as_tuple=False).flatten().tolist()
    if not dead_indices:
        return M, S, Q, 0

    # Recover unregularized moments for active components.
    D = S.shape[1]
    M_work = M.clone()
    active_mask = M_work > 1e-12
    means_work = torch.zeros_like(S)
    means_work[active_mask] = S[active_mask] / M_work[active_mask].unsqueeze(-1)

    if covariance_type == "diag":
        covs_work = torch.zeros_like(Q)
        covs_work[active_mask] = (
            Q[active_mask] / M_work[active_mask].unsqueeze(-1)
            - means_work[active_mask] * means_work[active_mask]
        ).clamp_min_(0.0)
    else:
        covs_work = torch.zeros_like(Q)
        if active_mask.any():
            m_act = means_work[active_mask]
            cov_act = Q[active_mask] / M_work[active_mask].view(-1, 1, 1) - torch.bmm(
                m_act.unsqueeze(-1), m_act.unsqueeze(-2)
            )
            covs_work[active_mask] = 0.5 * (cov_act + cov_act.transpose(-1, -2))

    # Compute donor scores: M_j * lambda_max(Sigma_j)
    if covariance_type == "diag":
        max_eigs = covs_work.max(dim=-1).values
    else:
        max_eigs = torch.linalg.eigvalsh(covs_work)[:, -1].clamp_min_(0.0)

    n_revived = 0
    for k in dead_indices:
        scores = M_work * (max_eigs + reg_covar)
        scores = torch.where(M_work > 2.0 * dead_thresh, scores, torch.full_like(scores, -1.0))
        d = int(torch.argmax(scores).item())
        if float(scores[d].item()) <= 0.0:
            break

        if covariance_type == "diag":
            j_star = int(torch.argmax(covs_work[d]).item())
            lambda_1 = float(covs_work[d, j_star].clamp_min(0.0).item())
            v_1 = torch.zeros(D, device=S.device, dtype=S.dtype)
            v_1[j_star] = 1.0
        else:
            eigvals, eigvecs = torch.linalg.eigh(covs_work[d])
            lambda_1 = float(eigvals[-1].clamp_min(0.0).item())
            v_1 = eigvecs[:, -1]

        shift = (split_delta * math.sqrt(max(lambda_1, 0.0))) * v_1
        m_half = 0.5 * (M_work[d] + M_work[k])
        mu_d = means_work[d].clone()

        means_work[d] = mu_d - shift
        means_work[k] = mu_d + shift
        M_work[d] = m_half
        M_work[k] = m_half

        if covariance_type == "diag":
            cov_new = covs_work[d].clone()
            cov_new[j_star] = max(0.0, float(cov_new[j_star].item()) - (split_delta**2) * lambda_1)
            covs_work[d] = cov_new
            covs_work[k] = cov_new.clone()
            max_eigs[d] = float(cov_new.max().item())
            max_eigs[k] = max_eigs[d]
        else:
            cov_new = covs_work[d] - ((split_delta**2) * lambda_1) * torch.outer(v_1, v_1)
            cov_new = 0.5 * (cov_new + cov_new.transpose(-1, -2))
            covs_work[d] = cov_new
            covs_work[k] = cov_new.clone()
            new_max = float(torch.linalg.eigvalsh(cov_new)[-1].clamp_min(0.0).item())
            max_eigs[d] = new_max
            max_eigs[k] = new_max

        n_revived += 1

    if n_revived == 0:
        return M, S, Q, 0

    M_new, S_new, Q_new = params_to_moments(
        M_work, means_work, covs_work, covariance_type=covariance_type
    )
    return M_new, S_new, Q_new, n_revived

"""Dead-component detection and moment-preserving eigen-split revival for e-GMM."""

from __future__ import annotations

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
    """Revive dead/starving components via vectorized high-variance donor eigen-splitting.

    Preserves total mixture mass, mean, and second moment when dead components have zero mass,
    avoiding boolean active-mask indexing and per-component cuSOLVER launches on CUDA by using
    top-k donor pairing and float32 power iteration (exact batched ``eigh`` on CPU).
    """
    K = M.shape[0]
    if K <= 1:
        return M, S, Q, 0

    total_mass_t = M.sum()
    total_mass = float(total_mass_t.item())
    if total_mass <= 0.0:
        return M, S, Q, 0

    dead_thresh = dead_ratio * (total_mass / K)
    dead_mask = M < dead_thresh
    if not dead_mask.any():
        return M, S, Q, 0

    eligible_mask = M > 2.0 * dead_thresh
    if not eligible_mask.any():
        return M, S, Q, 0

    dead_idx = torch.nonzero(dead_mask, as_tuple=False).flatten()
    D = S.shape[1]
    M_work = M.clone()
    M_safe = M_work.clamp_min(1e-12)
    means_work = S / M_safe.unsqueeze(-1)

    if covariance_type == "diag":
        covs_work = (Q / M_safe.unsqueeze(-1) - means_work * means_work).clamp_min_(0.0)
        spreads = covs_work.max(dim=-1).values
    else:
        covs_work = Q / M_safe.view(K, 1, 1) - torch.bmm(
            means_work.unsqueeze(-1), means_work.unsqueeze(-2)
        )
        covs_work = 0.5 * (covs_work + covs_work.transpose(-1, -2))
        spreads = torch.diagonal(covs_work, dim1=-2, dim2=-1).sum(dim=-1).clamp_min_(0.0)

    scores = torch.where(eligible_mask, M_work * (spreads + reg_covar), torch.full_like(M_work, -1.0))
    n_elig = int(eligible_mask.sum().item())
    n_pair = min(int(dead_idx.numel()), n_elig)
    if n_pair == 0:
        return M, S, Q, 0

    top_scores, donor_idx = torch.topk(scores, k=n_pair)
    valid = top_scores > 0.0
    if not valid.any():
        return M, S, Q, 0
    donor_idx = donor_idx[valid]
    dead_idx = dead_idx[: int(donor_idx.numel())]
    n_rev = int(dead_idx.numel())
    rev_arange = torch.arange(n_rev, device=S.device)

    if covariance_type == "diag":
        d_covs = covs_work[donor_idx]
        j_star = torch.argmax(d_covs, dim=-1)
        lam1 = d_covs[rev_arange, j_star].clamp_min(0.0)
        v1 = torch.zeros((n_rev, D), device=S.device, dtype=S.dtype)
        v1[rev_arange, j_star] = 1.0
    elif S.device.type == "cuda":
        d_covs = covs_work[donor_idx].to(torch.float32)
        diags = torch.diagonal(d_covs, dim1=-2, dim2=-1)
        j_star = torch.argmax(diags, dim=-1)
        v = torch.zeros((n_rev, D), device=S.device, dtype=torch.float32)
        v[rev_arange, j_star] = 1.0
        for _ in range(5):
            v_next = torch.bmm(d_covs, v.unsqueeze(-1)).squeeze(-1)
            v = v_next / v_next.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        Av = torch.bmm(d_covs, v.unsqueeze(-1)).squeeze(-1)
        lam1 = (v * Av).sum(dim=-1).clamp_min(0.0).to(S.dtype)
        v1 = v.to(S.dtype)
    else:
        eigvals, eigvecs = torch.linalg.eigh(covs_work[donor_idx].to(torch.float64))
        lam1 = eigvals[:, -1].clamp_min(0.0).to(S.dtype)
        v1 = eigvecs[:, :, -1].to(S.dtype)

    shifts = (split_delta * torch.sqrt(lam1)).unsqueeze(-1) * v1
    m_half = 0.5 * (M_work[donor_idx] + M_work[dead_idx])
    mu_d = means_work[donor_idx]
    means_work[donor_idx] = mu_d - shifts
    means_work[dead_idx] = mu_d + shifts
    M_work[donor_idx] = m_half
    M_work[dead_idx] = m_half

    if covariance_type == "diag":
        c_shrunk = (covs_work[donor_idx] - shifts * shifts).clamp_min(0.0)
        covs_work[donor_idx] = c_shrunk
        covs_work[dead_idx] = c_shrunk
    else:
        c_shrunk = covs_work[donor_idx] - torch.bmm(shifts.unsqueeze(-1), shifts.unsqueeze(-2))
        c_shrunk = 0.5 * (c_shrunk + c_shrunk.transpose(-1, -2))
        diag_idx = torch.arange(D, device=S.device)
        c_shrunk[:, diag_idx, diag_idx] = c_shrunk[:, diag_idx, diag_idx].clamp_min(0.0)
        covs_work[donor_idx] = c_shrunk
        covs_work[dead_idx] = c_shrunk

    M_new, S_new, Q_new = params_to_moments(
        M_work / M_work.sum().clamp_min(1e-12), means_work, covs_work, covariance_type=covariance_type
    )
    return M_new * total_mass_t, S_new * total_mass_t, Q_new * total_mass_t, n_rev

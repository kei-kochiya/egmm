"""Batched PyTorch kernels for e-GMM E-step, sufficient statistics, and PSD projection."""

from __future__ import annotations

import math
from typing import Literal

import torch

CovarianceType = Literal["diag", "full"]


def _validate_cov_type(covariance_type: str) -> None:
    if covariance_type not in ("diag", "full"):
        raise ValueError(
            f"Unsupported covariance_type={covariance_type!r}; expected 'diag' or 'full'."
        )


def prepare_precisions(
    covariances: torch.Tensor,
    covariance_type: CovarianceType = "full",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute precision Cholesky factors and half log-determinants.

    For ``diag``: returns ``(inv_std, log_det_half)`` where ``inv_std`` has shape ``(K, D)``
    and ``log_det_half = sum_d log(inv_std_{kd})``.
    For ``full``: ``covariances = L_cov @ L_cov^T``; returns ``(L_inv, log_det_half)`` where
    ``L_inv = L_cov^{-1}`` (lower triangular, shape ``(K, D, D)``) so
    ``(x - mu)^T Sigma^{-1} (x - mu) = ||(x - mu) @ L_inv^T||_2^2``, and
    ``log_det_half = -sum_d log(diag(L_cov)_{kd})``.
    """
    _validate_cov_type(covariance_type)
    if covariance_type == "diag":
        inv_std = torch.rsqrt(covariances.clamp_min(1e-30))
        log_det_half = torch.log(inv_std).sum(dim=1)
        return inv_std, log_det_half

    K, D, _ = covariances.shape
    L_cov = torch.linalg.cholesky(covariances)
    diag = torch.diagonal(L_cov, dim1=-2, dim2=-1)
    log_det_half = -torch.log(diag).sum(dim=-1)
    eye = torch.eye(D, device=covariances.device, dtype=covariances.dtype).expand(K, D, D)
    L_inv = torch.linalg.solve_triangular(L_cov, eye, upper=False)
    return L_inv, log_det_half


def compute_log_gaussian_prob_from_precisions(
    X: torch.Tensor,
    means: torch.Tensor,
    prec_factor: torch.Tensor,
    log_det_half: torch.Tensor,
    covariance_type: CovarianceType = "full",
    max_work_elements: int = 32_000_000,
) -> torch.Tensor:
    """Compute log N(X | means_k, covs_k) of shape (B, K) using precomputed precision factors."""
    _validate_cov_type(covariance_type)
    B, D = X.shape
    K = means.shape[0]
    const = -0.5 * D * math.log(2.0 * math.pi)

    if covariance_type == "diag":
        # Shift by batch mean for shift-invariance in quadratic expansion.
        shift = X.mean(dim=0, keepdim=True)
        Xc = X - shift
        mc = means - shift
        prec = prec_factor * prec_factor  # (K, D)
        quad_x = (Xc * Xc) @ prec.T  # (B, K)
        cross = Xc @ (mc * prec).T  # (B, K)
        quad_m = (mc * mc * prec).sum(dim=1).unsqueeze(0)  # (1, K)
        maha = (quad_x - 2.0 * cross + quad_m).clamp_min_(0.0)
        return const + log_det_half.unsqueeze(0) - 0.5 * maha

    # Full covariance: L_inv has shape (K, D, D), so (X - mu_k) @ L_inv_k^T
    # A_mat maps (B, D) -> (B, K * D) in a single GEMM.
    A_mat = prec_factor.permute(2, 0, 1).reshape(D, K * D)
    mu_proj = torch.bmm(means.unsqueeze(1), prec_factor.transpose(-1, -2)).squeeze(1)  # (K, D)

    elems_per_row = max(1, K * D)
    chunk_b = max(1, min(B, max_work_elements // elems_per_row))
    if chunk_b >= B:
        Z = (X @ A_mat).view(B, K, D)
        diff = Z - mu_proj.unsqueeze(0)
        maha = (diff * diff).sum(dim=-1)
    else:
        maha = torch.empty((B, K), device=X.device, dtype=X.dtype)
        for start in range(0, B, chunk_b):
            end = min(B, start + chunk_b)
            Z = (X[start:end] @ A_mat).view(end - start, K, D)
            diff = Z - mu_proj.unsqueeze(0)
            maha[start:end] = (diff * diff).sum(dim=-1)

    return const + log_det_half.unsqueeze(0) - 0.5 * maha


def compute_log_gaussian_prob(
    X: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    covariance_type: CovarianceType = "full",
) -> torch.Tensor:
    """Compute log N(X | means_k, covariances_k) of shape (B, K)."""
    prec_factor, log_det_half = prepare_precisions(covariances, covariance_type=covariance_type)
    return compute_log_gaussian_prob_from_precisions(
        X, means, prec_factor, log_det_half, covariance_type=covariance_type
    )


def e_step_batch(
    X: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor | None = None,
    covariance_type: CovarianceType = "full",
    prec_factor: torch.Tensor | None = None,
    log_det_half: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run batched E-step and return (r, mean_log_prob, m_bar, s_bar, q_bar)."""
    _validate_cov_type(covariance_type)
    B, D = X.shape
    if prec_factor is None or log_det_half is None:
        if covariances is None:
            raise ValueError("Either covariances or (prec_factor, log_det_half) must be provided.")
        prec_factor, log_det_half = prepare_precisions(
            covariances, covariance_type=covariance_type
        )

    log_prob = compute_log_gaussian_prob_from_precisions(
        X, means, prec_factor, log_det_half, covariance_type=covariance_type
    )
    log_weights = torch.log(weights.clamp_min(1e-30))
    weighted_log_prob = log_prob + log_weights.unsqueeze(0)
    log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)
    r = torch.exp(weighted_log_prob - log_prob_norm.unsqueeze(1))

    inv_b = 1.0 / B
    m_bar = r.sum(dim=0) * inv_b
    s_bar = (r.T @ X) * inv_b
    if covariance_type == "diag":
        q_bar = (r.T @ (X * X)) * inv_b
    else:
        # Weighted outer products: sum_b r_{bk} x_b x_b^T / B via batched GEMM.
        rX = r.T.unsqueeze(-1) * X.unsqueeze(0)  # (K, B, D)
        q_bar = torch.bmm(
            rX.transpose(-1, -2), X.unsqueeze(0).expand(means.shape[0], B, D)
        ) * inv_b

    return r, log_prob_norm.mean(), m_bar, s_bar, q_bar


def moments_to_params(
    M: torch.Tensor,
    S: torch.Tensor,
    Q: torch.Tensor,
    covariance_type: CovarianceType = "full",
    reg_covar: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project EMA sufficient statistics (M, S, Q) to valid GMM parameters (weights, means, covs)."""
    _validate_cov_type(covariance_type)
    M_safe = M.clamp_min(1e-12)
    weights = M_safe / M_safe.sum()
    means = S / M_safe.unsqueeze(-1)

    if covariance_type == "diag":
        covs = (Q / M_safe.unsqueeze(-1)) - (means * means)
        covs = covs.clamp_min(0.0) + reg_covar
        return weights, means, covs

    K, D = means.shape
    second = Q / M_safe.view(K, 1, 1)
    covs = second - torch.bmm(means.unsqueeze(-1), means.unsqueeze(-2))
    covs = 0.5 * (covs + covs.transpose(-1, -2))
    eye = torch.eye(D, device=covs.device, dtype=covs.dtype).unsqueeze(0)
    covs_reg = covs + reg_covar * eye

    # Fast path: Cholesky succeeds on almost all convex EMA updates; fall back to eigh only if needed.
    _, info = torch.linalg.cholesky_ex(covs_reg)
    bad_mask = info != 0
    if bad_mask.any():
        bad_covs = covs[bad_mask]
        eigvals, eigvecs = torch.linalg.eigh(bad_covs)
        eigvals = eigvals.clamp_min(0.0) + reg_covar
        repaired = torch.bmm(eigvecs * eigvals.unsqueeze(-2), eigvecs.transpose(-1, -2))
        covs_reg[bad_mask] = 0.5 * (repaired + repaired.transpose(-1, -2))

    return weights, means, covs_reg


def moments_to_params_and_precisions(
    M: torch.Tensor,
    S: torch.Tensor,
    Q: torch.Tensor,
    covariance_type: CovarianceType = "full",
    reg_covar: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-pass projection from (M, S, Q) to (weights, means, covs, prec_factor, log_det_half).

    Avoids running a duplicate Cholesky decomposition for ``full`` covariance.
    """
    _validate_cov_type(covariance_type)
    M_safe = M.clamp_min(1e-12)
    weights = M_safe / M_safe.sum()
    means = S / M_safe.unsqueeze(-1)

    if covariance_type == "diag":
        covs = ((Q / M_safe.unsqueeze(-1)) - (means * means)).clamp_min(0.0) + reg_covar
        inv_std = torch.rsqrt(covs)
        log_det_half = torch.log(inv_std).sum(dim=1)
        return weights, means, covs, inv_std, log_det_half

    K, D = means.shape
    second = Q / M_safe.view(K, 1, 1)
    covs = second - torch.bmm(means.unsqueeze(-1), means.unsqueeze(-2))
    covs = 0.5 * (covs + covs.transpose(-1, -2))
    eye_d = torch.eye(D, device=covs.device, dtype=covs.dtype)
    covs_reg = covs + reg_covar * eye_d.unsqueeze(0)

    L_cov, info = torch.linalg.cholesky_ex(covs_reg)
    bad_mask = info != 0
    if bad_mask.any():
        bad_covs = covs[bad_mask]
        eigvals, eigvecs = torch.linalg.eigh(bad_covs)
        eigvals = eigvals.clamp_min(0.0) + reg_covar
        repaired = torch.bmm(eigvecs * eigvals.unsqueeze(-2), eigvecs.transpose(-1, -2))
        covs_reg[bad_mask] = 0.5 * (repaired + repaired.transpose(-1, -2))
        L_cov = torch.linalg.cholesky(covs_reg)

    diag = torch.diagonal(L_cov, dim1=-2, dim2=-1)
    log_det_half = -torch.log(diag).sum(dim=-1)
    L_inv = torch.linalg.solve_triangular(L_cov, eye_d.expand(K, D, D), upper=False)
    return weights, means, covs_reg, L_inv, log_det_half


def params_to_moments(
    weights: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    covariance_type: CovarianceType = "full",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert normalized GMM parameters (weights, means, covs) to sufficient statistics (M, S, Q)."""
    _validate_cov_type(covariance_type)
    M = weights.clone()
    S = M.unsqueeze(-1) * means
    if covariance_type == "diag":
        Q = M.unsqueeze(-1) * (covariances + means * means)
    else:
        second = covariances + torch.bmm(means.unsqueeze(-1), means.unsqueeze(-2))
        Q = M.view(-1, 1, 1) * second
    return M, S, Q


def init_kmeans_pp(
    X: torch.Tensor,
    n_components: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Fast k-means++ seeding on tensor X (N, D) returning initial centers (K, D)."""
    N, D = X.shape
    if n_components > N:
        # Sample with replacement if dataset smaller than K
        idx = torch.randint(0, N, (n_components,), generator=generator, device=X.device)
        return X[idx].clone()

    centers = torch.empty((n_components, D), device=X.device, dtype=X.dtype)
    first_idx = int(torch.randint(0, N, (1,), generator=generator, device=X.device).item())
    centers[0] = X[first_idx]

    x_sq = (X * X).sum(dim=1)
    c0_sq = (centers[0] * centers[0]).sum()
    min_dist_sq = (x_sq + c0_sq - 2.0 * (X @ centers[0])).clamp_min_(0.0)

    for k in range(1, n_components):
        total = min_dist_sq.sum()
        if not torch.isfinite(total) or total <= 0.0:
            next_idx = int(torch.randint(0, N, (1,), generator=generator, device=X.device).item())
        else:
            probs = min_dist_sq / total
            next_idx = int(torch.multinomial(probs, 1, generator=generator).item())
        centers[k] = X[next_idx]
        ck_sq = (centers[k] * centers[k]).sum()
        dist_sq = (x_sq + ck_sq - 2.0 * (X @ centers[k])).clamp_min_(0.0)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

    return centers

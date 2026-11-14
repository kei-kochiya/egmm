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
    stats_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run batched E-step with centered sufficient-statistics accumulation.

    Centering around the batch mean (``diag``) or component means (``full``) prevents
    floating-point offset cancellation when ``X`` is in float32 and statistics are
    accumulated in float64 via ``stats_dtype``. Returns ``(r, mean_log_prob, m_bar, s_bar, q_bar)``.
    """
    _validate_cov_type(covariance_type)
    B, D = X.shape
    K = means.shape[0]
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

    out_dtype = X.dtype if stats_dtype is None else stats_dtype
    inv_b = 1.0 / B
    m_bar = (r.sum(dim=0) * inv_b).to(out_dtype)

    if covariance_type == "diag":
        xb_mean = X.mean(dim=0)
        Xc = X - xb_mean.unsqueeze(0)
        delta_s = ((r.T @ Xc) * inv_b).to(out_dtype)
        var_c = ((r.T @ (Xc * Xc)) * inv_b).to(out_dtype)
        xb_out = xb_mean.to(out_dtype)
        s_bar = m_bar.unsqueeze(-1) * xb_out.unsqueeze(0) + delta_s
        q_bar = (
            var_c
            + m_bar.unsqueeze(-1) * (xb_out * xb_out).unsqueeze(0)
            + xb_out.unsqueeze(0) * delta_s
            + delta_s * xb_out.unsqueeze(0)
        )
        return r, log_prob_norm.mean(), m_bar, s_bar, q_bar

    means_e = means.to(X.dtype) if means.dtype != X.dtype else means
    means_out = means.to(out_dtype) if means.dtype != out_dtype else means
    max_elems = 24_000_000
    chunk_b = max(1, min(B, max_elems // max(1, K * D)))
    delta_accum = torch.zeros((K, D), device=X.device, dtype=out_dtype)
    cov_accum = torch.zeros((K, D, D), device=X.device, dtype=out_dtype)
    r_T = r.T

    for st in range(0, B, chunk_b):
        en = min(B, st + chunk_b)
        Xc = X[st:en]
        rc = r_T[:, st:en]
        diff = Xc.unsqueeze(0) - means_e.unsqueeze(1)
        r_diff = rc.unsqueeze(-1) * diff
        delta_accum = delta_accum + r_diff.sum(dim=1).to(out_dtype)
        cov_accum = cov_accum + torch.bmm(r_diff.transpose(-1, -2), diff).to(out_dtype)

    delta_out = delta_accum * inv_b
    cov_out = cov_accum * inv_b
    s_bar = m_bar.unsqueeze(-1) * means_out + delta_out
    q_bar = (
        cov_out
        + torch.bmm(means_out.unsqueeze(-1), s_bar.unsqueeze(-2))
        + torch.bmm(delta_out.unsqueeze(-1), means_out.unsqueeze(-2))
    )
    return r, log_prob_norm.mean(), m_bar, s_bar, q_bar


def _repair_non_psd_covariances(
    covs: torch.Tensor,
    covs_reg: torch.Tensor,
    bad_mask: torch.Tensor,
    reg_covar: float,
) -> torch.Tensor:
    """Repair non-PD covariance matrices using float64 eigh + condition-safe floor."""
    bad_covs = covs[bad_mask].to(torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(bad_covs)
    if covs.dtype == torch.float32:
        max_eig = eigvals[:, -1:].clamp_min(1.0)
        floor = torch.maximum(
            torch.full_like(max_eig, reg_covar), 2e-6 * max_eig
        )
    else:
        floor = torch.full_like(eigvals[:, -1:], reg_covar)
    eigvals = torch.maximum(eigvals, floor)
    repaired = torch.bmm(eigvecs * eigvals.unsqueeze(-2), eigvecs.transpose(-1, -2))
    repaired = 0.5 * (repaired + repaired.transpose(-1, -2))
    covs_reg = covs_reg.clone()
    covs_reg[bad_mask] = repaired.to(dtype=covs.dtype)
    return covs_reg


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
        covs_reg = _repair_non_psd_covariances(covs, covs_reg, bad_mask, reg_covar)

    return weights, means, covs_reg


def moments_to_params_and_precisions(
    M: torch.Tensor,
    S: torch.Tensor,
    Q: torch.Tensor,
    covariance_type: CovarianceType = "full",
    reg_covar: float = 1e-6,
    allow_sync: bool = True,
    fast_cholesky_f32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-pass projection from (M, S, Q) to (weights, means, covs, prec_factor, log_det_half).

    Supports branchless ``allow_sync=False`` diagonal fallback (CUDA-Graph capturable) and
    optional float32 inner Cholesky decomposition via ``fast_cholesky_f32=True``.
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

    chol_dtype = torch.float32 if fast_cholesky_f32 else covs.dtype
    chol_in = covs_reg.to(chol_dtype) if chol_dtype != covs_reg.dtype else covs_reg
    eye_chol = torch.eye(D, device=covs.device, dtype=chol_dtype)

    L_cov, info = torch.linalg.cholesky_ex(chol_in)
    if not allow_sync:
        bad = (info != 0).view(K, 1, 1)
        diag_safe = torch.diagonal(chol_in, dim1=-2, dim2=-1).clamp_min(reg_covar)
        L_cov = torch.where(bad, torch.diag_embed(torch.sqrt(diag_safe)), L_cov)
    elif (info != 0).any():
        covs_reg = _repair_non_psd_covariances(covs, covs_reg, info != 0, reg_covar)
        chol_in = covs_reg.to(chol_dtype) if chol_dtype != covs_reg.dtype else covs_reg
        L_cov, info = torch.linalg.cholesky_ex(chol_in)
        jitter = max(reg_covar, 1e-5)
        while (info != 0).any() and jitter <= 1e-1:
            still_bad = info != 0
            covs_reg[still_bad] = covs_reg[still_bad] + jitter * eye_d.unsqueeze(0)
            chol_in = covs_reg.to(chol_dtype) if chol_dtype != covs_reg.dtype else covs_reg
            L_cov, info = torch.linalg.cholesky_ex(chol_in)
            jitter *= 10.0

    diag = torch.diagonal(L_cov, dim1=-2, dim2=-1)
    log_det_half = -torch.log(diag).sum(dim=-1)
    L_inv = torch.linalg.solve_triangular(L_cov, eye_chol.expand(K, D, D), upper=False)
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
    orig_device = X.device
    orig_dtype = X.dtype
    N, D = X.shape
    if n_components > N:
        idx = torch.randint(0, N, (n_components,), generator=generator, device=orig_device)
        return X[idx].clone()

    # For seed buffers on CUDA, running the sequential K-step loop on CPU avoids K*2 sync stalls.
    work_X = X.detach().cpu() if (orig_device.type == "cuda" and N <= 8192) else X
    work_gen = None
    if generator is not None and work_X.device != orig_device:
        work_gen = torch.Generator(device="cpu")
        work_gen.manual_seed(int(generator.initial_seed()))
    else:
        work_gen = generator

    centers = torch.empty((n_components, D), device=work_X.device, dtype=orig_dtype)
    first_idx = int(torch.randint(0, N, (1,), generator=work_gen, device=work_X.device).item())
    centers[0] = work_X[first_idx]

    x_sq = (work_X * work_X).sum(dim=1)
    c0_sq = (centers[0] * centers[0]).sum()
    min_dist_sq = (x_sq + c0_sq - 2.0 * (work_X @ centers[0])).clamp_min_(0.0)

    for k in range(1, n_components):
        total = min_dist_sq.sum()
        if not torch.isfinite(total) or total <= 0.0:
            next_idx = int(
                torch.randint(0, N, (1,), generator=work_gen, device=work_X.device).item()
            )
        else:
            probs = min_dist_sq / total
            next_idx = int(torch.multinomial(probs, 1, generator=work_gen).item())
        centers[k] = work_X[next_idx]
        ck_sq = (centers[k] * centers[k]).sum()
        dist_sq = (x_sq + ck_sq - 2.0 * (work_X @ centers[k])).clamp_min_(0.0)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

    return centers.to(device=orig_device, dtype=orig_dtype)

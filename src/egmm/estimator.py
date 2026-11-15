"""Drop-in scikit-learn compatible e-GMM estimator."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils.validation import check_is_fitted

from egmm.kernels import (
    CovarianceType,
    compute_log_gaussian_prob_from_precisions,
    e_step_batch,
    init_kmeans_pp,
    moments_to_params_and_precisions,
    params_to_moments,
    prepare_precisions,
)
from egmm.revival import revive_dead_components
from egmm.scaling import auto_hyperparameters, compute_cosine_alpha


class EGMM(DensityMixin, BaseEstimator):
    """Exponential Moving Average Gaussian Mixture Model (e-GMM).

    Drop-in replacement for ``sklearn.mixture.GaussianMixture`` supporting CPU and CUDA
    with mini-batch EMA sufficient statistics, cosine decay, dead-component eigen-splitting,
    and automatic scaling-law hyperparameters.

    Fitted-attribute semantics (where they differ from ``GaussianMixture``):

    * ``n_iter_`` = number of EMA epochs actually started (``ceil(T / S_ep)``) + full-batch polish steps run.
    * ``converged_`` is only set by the diagonal-covariance polish loop (|dLL| < ``tol`` checked every 3
      polish steps); stochastic EMA training itself runs a fixed step budget and reports ``False``.
    * ``lower_bound_`` is the mean training log-likelihood (original units) before the last polish M-step
      when polish ran with ``n_init=1``; otherwise it is evaluated on a stride subsample of <= 65,536 rows.
    * ``score``/``score_samples``/``predict``/``predict_proba``/``bic``/``aic`` are evaluated in float64.
    """

    # Seeding: greedy k-means++ on n_seed = min(N, max(2048, 128 K)) points + 10 Lloyd steps
    # (selected on validation NLL only; shared by every PyTorch baseline that inherits EGMM).
    _SEED_MIN_POINTS = 2048
    _SEED_POINTS_PER_COMPONENT = 128
    _SEED_LLOYD_ITERS = 10
    # Epoch / closing-polish budget law ("smooth" logistic gates in log N, or legacy "bracket").
    _SCHEDULE_LAW = "smooth"
    # Capture the inner EMA step in a CUDA graph on CUDA (tests flip this to compare with the eager path).
    _USE_CUDA_GRAPH = True

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: CovarianceType = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_epochs: int | str = "auto",
        batch_size: int | str = "auto",
        alpha: float | str = "auto",
        revive_dead: bool = True,
        revive_every: int | str = "auto",
        dead_ratio: float = 0.05,
        polish_steps: int | str = "auto",
        init_usage_scale: float | str = "auto",
        n_init: int = 1,
        init_params: str = "kmeans++",
        device: str = "auto",
        dtype: str = "float32",
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.revive_dead = revive_dead
        self.revive_every = revive_every
        self.dead_ratio = dead_ratio
        self.polish_steps = polish_steps
        self.init_usage_scale = init_usage_scale
        self.n_init = n_init
        self.init_params = init_params
        self.device = device
        self.dtype = dtype
        self.random_state = random_state
        self.verbose = verbose

    def _resolve_dtype(self) -> torch.dtype:
        if self.dtype == "float64":
            return torch.float64
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"Unsupported dtype={self.dtype!r}; use 'float32' or 'float64'.")

    def _resolve_device(self, n_samples: int, n_features: int) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                if self.covariance_type == "full" and n_samples * self.n_components * n_features >= 150_000:
                    return torch.device("cuda")
                if n_samples * self.n_components * n_features >= 2_000_000 or n_samples >= 16384:
                    return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.device)

    def _extract_array_shape(self, X: np.ndarray | torch.Tensor) -> tuple[int, int]:
        if isinstance(X, torch.Tensor):
            if X.ndim != 2:
                raise ValueError(f"Expected 2D tensor, got shape {tuple(X.shape)}.")
            N, D = int(X.shape[0]), int(X.shape[1])
        else:
            arr = np.asarray(X)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {arr.shape}.")
            N, D = int(arr.shape[0]), int(arr.shape[1])
        if N < 1:
            raise ValueError("Expected at least 1 sample.")
        return N, D

    def _standardize_to_device(
        self,
        X: np.ndarray | torch.Tensor,
        device: torch.device,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        N, D = self._extract_array_shape(X)
        mean_64 = self._x_mean_f64.to(device=device, dtype=torch.float64).unsqueeze(0)
        inv_std_64 = (1.0 / self._x_std_f64.to(device=device, dtype=torch.float64)).unsqueeze(0)
        chunk = 65536
        X_std = torch.empty((N, D), device=device, dtype=target_dtype)

        if isinstance(X, torch.Tensor):
            X_src = X.to(device=device) if (device.type == "cuda" and X.dtype == torch.float32 and target_dtype == torch.float32) else X
            for st in range(0, N, chunk):
                en = min(N, st + chunk)
                Xc = X_src[st:en].to(device=device, dtype=torch.float64)
                X_std[st:en] = ((Xc - mean_64) * inv_std_64).to(dtype=target_dtype)
            return X_std

        arr = np.asarray(X)
        if arr.dtype == np.float32 and target_dtype == torch.float32:
            X_dev = torch.from_numpy(np.ascontiguousarray(arr)).to(device=device)
            for st in range(0, N, chunk):
                en = min(N, st + chunk)
                Xc = X_dev[st:en].to(torch.float64)
                X_std[st:en] = ((Xc - mean_64) * inv_std_64).to(dtype=torch.float32)
            return X_std

        if N * D <= 16_000_000:
            X_dev = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float64)).to(device=device)
            for st in range(0, N, chunk):
                en = min(N, st + chunk)
                X_std[st:en] = ((X_dev[st:en] - mean_64) * inv_std_64).to(dtype=target_dtype)
            return X_std

        for st in range(0, N, chunk):
            en = min(N, st + chunk)
            Xc = torch.from_numpy(np.ascontiguousarray(arr[st:en], dtype=np.float64)).to(device=device)
            X_std[st:en] = ((Xc - mean_64) * inv_std_64).to(dtype=target_dtype)
        return X_std

    @staticmethod
    def _device_cache_fits(n_bytes: int, device: torch.device) -> bool:
        """Whether a one-shot device copy of ``n_bytes`` source data fits in a safe memory budget.

        CPU tensors built via ``torch.from_numpy`` are zero-copy views, so CPU always fits. On CUDA we
        allow the cached single-transfer path when the raw source occupies at most 40% of free memory.
        """
        if device.type != "cuda":
            return True
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except Exception:  # pragma: no cover - defensive (e.g. exotic allocator backends)
            return n_bytes <= 128 * 2**20
        return n_bytes <= 0.40 * free_bytes

    def _fit_standardizer_and_transform(
        self,
        X: np.ndarray | torch.Tensor,
        device: torch.device,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int, int]:
        """Compute exact shifted float64 feature mean/std on device and return standardized tensor.

        Every host row crosses PCIe exactly once. When the raw source fits in a device-memory budget it is
        transferred in one copy and both the statistics pass and the standardization pass run on device;
        otherwise rows are streamed once in chunks, their ``x - x_0`` shifts are kept on device in the target
        dtype, and the final centering/scaling is applied in place after the statistics are known.
        """
        N, D = self._extract_array_shape(X)
        chunk = 65536
        sum1 = torch.zeros(D, device=device, dtype=torch.float64)
        sum2 = torch.zeros(D, device=device, dtype=torch.float64)
        X_cached: torch.Tensor | None = None
        X_shift: torch.Tensor | None = None

        if isinstance(X, torch.Tensor):
            keep_f32 = device.type == "cuda" and X.dtype == torch.float32 and target_dtype == torch.float32
            if X.device == device or keep_f32 or self._device_cache_fits(N * D * X.element_size(), device):
                X_cached = X.to(device=device)
            else:
                X_rows = lambda st, en: X[st:en].to(device=device, dtype=torch.float64)  # noqa: E731
                x0 = X[0:1].to(device=device, dtype=torch.float64)
        else:
            arr = np.asarray(X)
            if arr.dtype == np.float32 and target_dtype == torch.float32:
                X_cached = torch.from_numpy(np.ascontiguousarray(arr)).to(device=device)
            elif self._device_cache_fits(N * D * 8, device):
                X_cached = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float64)).to(device=device)
            else:
                X_rows = lambda st, en: torch.from_numpy(  # noqa: E731
                    np.ascontiguousarray(arr[st:en], dtype=np.float64)
                ).to(device=device)
                x0 = X_rows(0, 1)

        if X_cached is not None:
            x0 = X_cached[0:1].to(torch.float64)
            for st in range(0, N, chunk):
                en = min(N, st + chunk)
                dx = X_cached[st:en].to(torch.float64) - x0
                sum1 += dx.sum(dim=0)
                sum2 += (dx * dx).sum(dim=0)
        else:
            # Single streaming pass: accumulate shifted moments and keep x - x0 on device.
            X_shift = torch.empty((N, D), device=device, dtype=target_dtype)
            for st in range(0, N, chunk):
                en = min(N, st + chunk)
                dx = X_rows(st, en) - x0
                sum1 += dx.sum(dim=0)
                sum2 += (dx * dx).sum(dim=0)
                X_shift[st:en] = dx.to(dtype=target_dtype)

        mean_delta = sum1 / N
        mean_64 = x0.squeeze(0) + mean_delta
        var_64 = (sum2 / N - mean_delta * mean_delta).clamp_min(0.0)
        std_64 = torch.sqrt(var_64).clamp_min(1e-12)

        self._x_mean_f64 = mean_64
        self._x_std_f64 = std_64
        self._x_mean_np = mean_64.detach().cpu().numpy()
        self._x_std_np = std_64.detach().cpu().numpy()
        self._log_jac_t = (-torch.log(std_64).sum()).to(dtype=target_dtype)
        inv_std_b = (1.0 / std_64).unsqueeze(0)

        if X_cached is not None:
            mean_b = mean_64.unsqueeze(0)
            X_std = torch.empty((N, D), device=device, dtype=target_dtype)
            for st in range(0, N, chunk):
                en = min(N, st + chunk)
                Xc = X_cached[st:en].to(torch.float64) if X_cached.dtype != torch.float64 else X_cached[st:en]
                X_std[st:en] = ((Xc - mean_b) * inv_std_b).to(dtype=target_dtype)
            return X_std, N, D

        assert X_shift is not None
        delta_b = mean_delta.unsqueeze(0)
        for st in range(0, N, chunk):
            en = min(N, st + chunk)
            X_shift[st:en] = ((X_shift[st:en].to(torch.float64) - delta_b) * inv_std_b).to(dtype=target_dtype)
        return X_shift, N, D

    def _make_generator(self, device: torch.device, seed_offset: int = 0) -> torch.Generator | None:
        if self.random_state is None:
            return None
        gen = torch.Generator(device=device)
        gen.manual_seed(int(self.random_state) + seed_offset)
        return gen

    def _initialize_moments(
        self,
        X_std: torch.Tensor,
        generator: torch.Generator | None,
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, D = X_std.shape
        K = self.n_components
        device, dtype = X_std.device, X_std.dtype
        m_dtype = out_dtype if out_dtype is not None else dtype

        n_seed = min(N, max(self._SEED_MIN_POINTS, self._SEED_POINTS_PER_COMPONENT * K))
        if n_seed < N:
            perm = torch.randperm(N, generator=generator, device=device)[:n_seed]
            X_seed = X_std[perm]
        else:
            X_seed = X_std

        if self.init_params == "kmeans++":
            # Greedy k-means++ (2 + floor(ln K) D^2-sampled candidates per center, sync-free on CUDA).
            means = init_kmeans_pp(X_seed, K, generator=generator, n_local_trials=None)
            # Short Lloyd refinement on X_seed stabilizes initial Voronoi moments.
            for _ in range(self._SEED_LLOYD_ITERS):
                assign = torch.argmin(torch.cdist(X_seed, means), dim=1)
                one_hot = torch.nn.functional.one_hot(assign, num_classes=K).to(dtype=dtype)
                counts = one_hot.sum(dim=0)
                new_means = (one_hot.T @ X_seed) / counts.clamp_min(1.0).unsqueeze(-1)
                means = torch.where((counts > 0).unsqueeze(-1), new_means, means)
        elif self.init_params == "random":
            idx = torch.randperm(X_seed.shape[0], generator=generator, device=device)[:K]
            if idx.shape[0] < K:
                idx = torch.randint(0, X_seed.shape[0], (K,), generator=generator, device=device)
            means = X_seed[idx].clone()
        else:
            raise ValueError(f"Unsupported init_params={self.init_params!r}.")

        # Vectorized moment initialization in m_dtype (float64 prevents cancellation).
        X_seed_m = X_seed.to(dtype=m_dtype)
        means_m = means.to(dtype=m_dtype)
        dists = torch.cdist(X_seed, means)
        assign = torch.argmin(dists, dim=1)
        one_hot_m = torch.nn.functional.one_hot(assign, num_classes=K).to(dtype=m_dtype)
        counts = one_hot_m.sum(dim=0)  # (K,)
        inv_n = 1.0 / max(1, n_seed)

        M_init = counts * inv_n
        S_init = (one_hot_m.T @ X_seed_m) * inv_n
        if self.covariance_type == "diag":
            Q_init = (one_hot_m.T @ (X_seed_m * X_seed_m)) * inv_n
        else:
            # Per-component Voronoi scatter, chunked over components to bound the (k, n_seed, D) workspace.
            Q_init = torch.empty((K, D, D), device=device, dtype=m_dtype)
            k_chunk = max(1, min(K, 16_000_000 // max(1, n_seed * D)))
            X_seed_b = X_seed_m.unsqueeze(0)
            for k0 in range(0, K, k_chunk):
                k1 = min(K, k0 + k_chunk)
                rX = one_hot_m[:, k0:k1].T.unsqueeze(-1) * X_seed_b  # (k, n_seed, D)
                Q_init[k0:k1] = torch.bmm(rX.transpose(-1, -2), X_seed_b.expand(k1 - k0, n_seed, D)) * inv_n

        # Patch any starving/singleton cluster with global variance
        bad = counts < 2
        if bad.any():
            global_var = X_seed_m.var(dim=0, unbiased=False).clamp_min(1e-2)
            fallback_m = max(inv_n, 1e-3 / K)
            M_init = torch.where(bad, torch.full_like(M_init, fallback_m), M_init)
            S_init = torch.where(bad.unsqueeze(-1), M_init.unsqueeze(-1) * means_m, S_init)
            if self.covariance_type == "diag":
                fallback_q = M_init.unsqueeze(-1) * (global_var.unsqueeze(0) + means_m * means_m)
                Q_init = torch.where(bad.unsqueeze(-1), fallback_q, Q_init)
            else:
                diag_gv = torch.diag(global_var).unsqueeze(0)
                fallback_q = M_init.view(K, 1, 1) * (
                    diag_gv + torch.bmm(means_m.unsqueeze(-1), means_m.unsqueeze(-2))
                )
                Q_init = torch.where(bad.view(K, 1, 1), fallback_q, Q_init)

        # Normalize total mass to 1.0
        mass_sum = M_init.sum().clamp_min(1e-12)
        return M_init / mass_sum, S_init / mass_sum, Q_init / mass_sum

    def _moments_to_estep_params(
        self,
        M: torch.Tensor,
        S: torch.Tensor,
        Q: torch.Tensor,
        estep_dtype: torch.dtype,
        *,
        allow_sync: bool = True,
        fast_cholesky_f32: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project float64 sufficient statistics to E-step dtype parameters while keeping float64 covariances."""
        weights, means, covs, prec_factor, log_det_half = moments_to_params_and_precisions(
            M,
            S,
            Q,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            allow_sync=allow_sync,
            fast_cholesky_f32=fast_cholesky_f32,
        )
        return (
            weights.to(dtype=estep_dtype),
            means.to(dtype=estep_dtype),
            covs,
            prec_factor.to(dtype=estep_dtype),
            log_det_half.to(dtype=estep_dtype),
        )

    def _sync_public_attributes(
        self,
        weights: torch.Tensor,
        means_std: torch.Tensor,
        covs_std: torch.Tensor,
        prec_factor_std: torch.Tensor,
        log_det_half_std: torch.Tensor,
    ) -> None:
        self._weights_t = weights
        self._means_std_t = means_std
        self._covs_std_t = covs_std
        self._prec_factor_std_t = prec_factor_std
        self._log_det_half_std_t = log_det_half_std

        std = self._x_std_f64.to(dtype=torch.float64, device=means_std.device)
        mean = self._x_mean_f64.to(dtype=torch.float64, device=means_std.device)

        w_np = weights.detach().cpu().numpy().astype(np.float64)
        m_orig = (
            (means_std.to(torch.float64) * std.unsqueeze(0) + mean.unsqueeze(0))
            .detach()
            .cpu()
            .numpy()
        )

        if self.covariance_type == "diag":
            c_orig_t = covs_std.to(torch.float64) * (std * std).unsqueeze(0)
            p_orig_t = 1.0 / c_orig_t.clamp_min(1e-30)
            pc_orig_t = torch.sqrt(p_orig_t)
        else:
            scale_mat = std.unsqueeze(-1) * std.unsqueeze(-2)
            c_orig_t = covs_std.to(torch.float64) * scale_mat.unsqueeze(0)
            c_orig_t = 0.5 * (c_orig_t + c_orig_t.transpose(-1, -2))
            # Precision in original space
            c_orig_f64 = c_orig_t.to(torch.float64)
            D = c_orig_f64.shape[-1]
            eye_d = torch.eye(D, device=c_orig_f64.device, dtype=torch.float64)
            L_cov_orig, info = torch.linalg.cholesky_ex(c_orig_f64)
            if (info != 0).any():
                eigvals, eigvecs = torch.linalg.eigh(c_orig_f64)
                eigvals = eigvals.clamp_min(1e-12)
                c_orig_f64 = torch.bmm(eigvecs * eigvals.unsqueeze(-2), eigvecs.transpose(-1, -2))
                c_orig_f64 = 0.5 * (c_orig_f64 + c_orig_f64.transpose(-1, -2))
                c_orig_t = c_orig_f64
                L_cov_orig = torch.linalg.cholesky(c_orig_f64)
            L_inv_orig = torch.linalg.solve_triangular(
                L_cov_orig, eye_d.expand_as(L_cov_orig), upper=False
            )
            pc_orig_t = L_inv_orig.transpose(-1, -2)
            p_orig_t = pc_orig_t @ pc_orig_t.transpose(-1, -2)

        self.weights_ = w_np
        self.means_ = m_orig
        self.covariances_ = c_orig_t.detach().cpu().numpy()
        self.precisions_ = p_orig_t.detach().cpu().numpy()
        self.precisions_cholesky_ = pc_orig_t.detach().cpu().numpy()
        self._set_exact_scoring_params(weights, means_std, covs_std)

    def _set_exact_scoring_params(
        self, weights: torch.Tensor, means_std: torch.Tensor, covs_std: torch.Tensor
    ) -> None:
        """Cache float64 precision factors so every scoring path evaluates log-densities exactly.

        Training uses float32 E-step factors for speed; reported likelihoods (``score``, ``score_samples``,
        ``predict``, ``predict_proba``, ``bic``, ``aic``) are evaluated in float64 from the final float64
        M-step covariances. This matters for diagonal mixtures whose variances sit near ``reg_covar``, where
        the expanded float32 quadratic form cancels catastrophically.
        """
        w64 = weights.to(torch.float64)
        w64 = w64 / w64.sum().clamp_min(1e-300)
        mu64 = means_std.to(torch.float64)
        c64 = covs_std.to(torch.float64)
        if self.covariance_type == "diag":
            pf64 = torch.rsqrt(c64.clamp_min(1e-300))
            ldh64 = torch.log(pf64).sum(dim=1)
        else:
            c64 = 0.5 * (c64 + c64.transpose(-1, -2))
            L64, info = torch.linalg.cholesky_ex(c64)
            if (info != 0).any():
                eigvals, eigvecs = torch.linalg.eigh(c64)
                c64 = torch.bmm(eigvecs * eigvals.clamp_min(1e-12).unsqueeze(-2), eigvecs.transpose(-1, -2))
                L64 = torch.linalg.cholesky(0.5 * (c64 + c64.transpose(-1, -2)))
            K, D = mu64.shape
            eye = torch.eye(D, device=c64.device, dtype=torch.float64).expand(K, D, D)
            pf64 = torch.linalg.solve_triangular(L64, eye, upper=False)
            ldh64 = -torch.log(torch.diagonal(L64, dim1=-2, dim2=-1)).sum(dim=-1)
        log_jac64 = -torch.log(self._x_std_f64.to(device=mu64.device, dtype=torch.float64)).sum()
        self._score_params_t = (torch.log(w64.clamp_min(1e-300)), mu64, pf64, ldh64, log_jac64)

    def _step_alpha(self, step: int, total_steps: int, hp: Any) -> float:
        return compute_cosine_alpha(step, total_steps, hp.alpha_0, hp.alpha_min)

    def _prepare_fit_data(
        self, X: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.device, torch.dtype, int, int]:
        dtype = self._resolve_dtype()
        N, D = self._extract_array_shape(X)
        device = self._resolve_device(N, D)

        self.n_features_in_ = D
        self._device_t = device
        self._dtype_t = dtype

        X_std, _, _ = self._fit_standardizer_and_transform(X, device=device, target_dtype=dtype)
        return X_std, device, dtype, N, D

    def _build_epoch_batches(
        self,
        n_samples: int,
        batch_size: int,
        steps_per_epoch: int,
        max_epochs: int,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> torch.Tensor:
        used_len = steps_per_epoch * batch_size
        epoch_batches = []
        for _ in range(max_epochs):
            perm = torch.randperm(n_samples, generator=generator, device=device)
            if used_len <= n_samples:
                epoch_batches.append(perm[:used_len].view(steps_per_epoch, batch_size))
            else:
                pad = used_len - n_samples
                extra_parts = [perm]
                rem = pad
                while rem > 0:
                    p2 = torch.randperm(n_samples, generator=generator, device=device)
                    take = min(rem, n_samples)
                    extra_parts.append(p2[:take])
                    rem -= take
                epoch_batches.append(torch.cat(extra_parts, dim=0).view(steps_per_epoch, batch_size))
        return torch.cat(epoch_batches, dim=0)

    def _build_schedule_and_revivals(
        self, total_steps: int, hp: Any, device: torch.device
    ) -> tuple[torch.Tensor, set[int]]:
        if type(self)._step_alpha is EGMM._step_alpha:
            t_idx = torch.arange(total_steps, device=device, dtype=torch.float64)
            frac = (
                (t_idx / max(1, total_steps - 1)).clamp_max(1.0)
                if total_steps > 1
                else torch.zeros_like(t_idx)
            )
            alphas_t = hp.alpha_min + 0.5 * (hp.alpha_0 - hp.alpha_min) * (
                1.0 + torch.cos(math.pi * frac)
            )
        else:
            alphas_t = torch.tensor(
                [self._step_alpha(t, total_steps, hp) for t in range(total_steps)],
                device=device,
                dtype=torch.float64,
            )

        if not self.revive_dead:
            rev_set: set[int] = set()
        elif hp.revive_points > 0:
            end_frac = 0.45 if self.covariance_type == "diag" else 0.70
            rev_set = (
                {int(round(f * (total_steps - 1))) for f in np.linspace(0.15, end_frac, hp.revive_points)}
                if total_steps >= 6
                else set()
            )
        elif hp.revive_points == -1:
            cutoff = int(0.8 * total_steps)
            rev_set = {t for t in range(cutoff) if (t + 1) % hp.revive_every == 0}
        else:
            rev_set = set()
        return alphas_t, rev_set

    def _eval_subset_lower_bound(
        self,
        X_std: torch.Tensor,
        weights: torch.Tensor,
        means: torch.Tensor,
        prec_factor: torch.Tensor,
        log_det_half: torch.Tensor,
    ) -> float:
        N = X_std.shape[0]
        stride = max(1, N // 65536)
        X_eval = X_std[::stride] if stride > 1 else X_std
        log_weights = torch.log(weights.clamp_min(1e-30)).unsqueeze(0)
        ll_sum = torch.zeros((), device=X_std.device, dtype=torch.float64)
        for st in range(0, X_eval.shape[0], 16384):
            en = min(X_eval.shape[0], st + 16384)
            lp = compute_log_gaussian_prob_from_precisions(
                X_eval[st:en],
                means,
                prec_factor,
                log_det_half,
                covariance_type=self.covariance_type,
            )
            ll_sum = ll_sum + torch.logsumexp(lp + log_weights, dim=1).sum().to(torch.float64)
        return (float(ll_sum.item()) / max(1, X_eval.shape[0])) + float(self._log_jac_t.item())

    def fit(self, X: np.ndarray | torch.Tensor, y: Any = None) -> EGMM:
        import time as _time

        prof = bool(getattr(self, "_profile_phases", False))
        t_start = _time.perf_counter() if prof else 0.0
        X_std, device, dtype, N, D = self._prepare_fit_data(X)
        if prof and device.type == "cuda":
            torch.cuda.synchronize(device)
        t_std = (_time.perf_counter() - t_start) if prof else 0.0

        hp = auto_hyperparameters(
            n_samples=N,
            n_components=self.n_components,
            n_features=D,
            covariance_type=self.covariance_type,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            alpha=self.alpha,
            revive_every=self.revive_every,
            init_usage_scale=self.init_usage_scale,
            polish_steps=self.polish_steps,
            schedule_law=self._SCHEDULE_LAW,
        )
        self.hyperparams_ = hp
        steps_ep = hp.steps_per_epoch
        total_steps = hp.n_steps
        fast_chol = dtype == torch.float32
        alphas_t, rev_set = self._build_schedule_and_revivals(total_steps, hp, device)
        use_polish = hp.polish_steps > 0
        use_init_scale = hp.init_usage_scale != 1.0

        best_lower_bound = -math.inf
        best_state: tuple[torch.Tensor, ...] | None = None
        best_revivals = 0
        best_iters = 0
        best_converged = False
        t_seed_acc = 0.0
        t_cap_acc = 0.0
        t_ema_acc = 0.0
        t_pol_acc = 0.0

        for init_idx in range(max(1, self.n_init)):
            t0_s = _time.perf_counter() if prof else 0.0
            gen = self._make_generator(device, seed_offset=init_idx * 9973)
            M, S, Q = self._initialize_moments(X_std, generator=gen, out_dtype=torch.float64)
            if use_init_scale:
                M = M * hp.init_usage_scale
                S = S * hp.init_usage_scale
                Q = Q * hp.init_usage_scale

            weights, means, _, prec_factor, log_det_half = self._moments_to_estep_params(
                M, S, Q, estep_dtype=dtype, allow_sync=False, fast_cholesky_f32=fast_chol
            )
            batches = self._build_epoch_batches(
                N, hp.batch_size, steps_ep, hp.max_epochs, generator=gen, device=device
            )[:total_steps]
            if prof and device.type == "cuda":
                torch.cuda.synchronize(device)
            if prof:
                t_seed_acc += _time.perf_counter() - t0_s

            revivals_count = 0
            converged = False
            polish_done = 0

            if device.type == "cuda" and total_steps >= 4 and self._USE_CUDA_GRAPH:
                t0_c = _time.perf_counter() if prof else 0.0
                pos_buf = torch.zeros((), device=device, dtype=torch.long)
                M_buf = M.clone()
                S_buf = S.clone()
                Q_buf = Q.clone()
                w_buf = weights.clone()
                mu_buf = means.clone()
                pf_buf = prec_factor.clone()
                ldh_buf = log_det_half.clone()
                idx_buf = torch.empty((hp.batch_size,), device=device, dtype=torch.long)
                a_buf = torch.zeros((), device=device, dtype=torch.float64)

                def step_body() -> None:
                    torch.index_select(batches, 0, pos_buf, out=idx_buf.unsqueeze(0))
                    torch.index_select(alphas_t, 0, pos_buf, out=a_buf.unsqueeze(0))
                    xb = X_std.index_select(0, idx_buf)
                    _, _, dM, dS, dQ = e_step_batch(
                        xb,
                        w_buf,
                        mu_buf,
                        covariance_type=self.covariance_type,
                        prec_factor=pf_buf,
                        log_det_half=ldh_buf,
                        stats_dtype=torch.float64,
                    )
                    om = 1.0 - a_buf
                    M_buf.copy_(om * M_buf + a_buf * dM)
                    S_buf.copy_(om * S_buf + a_buf * dS)
                    Q_buf.copy_(om * Q_buf + a_buf * dQ)
                    w2, mu2, _, pf2, ldh2 = self._moments_to_estep_params(
                        M_buf,
                        S_buf,
                        Q_buf,
                        estep_dtype=dtype,
                        allow_sync=False,
                        fast_cholesky_f32=fast_chol,
                    )
                    w_buf.copy_(w2)
                    mu_buf.copy_(mu2)
                    pf_buf.copy_(pf2)
                    ldh_buf.copy_(ldh2)
                    pos_buf.add_(1)

                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device=device))
                with torch.cuda.stream(stream):
                    step_body()
                torch.cuda.current_stream(device=device).wait_stream(stream)

                pos_buf.zero_()
                M_buf.copy_(M)
                S_buf.copy_(S)
                Q_buf.copy_(Q)
                w_buf.copy_(weights)
                mu_buf.copy_(means)
                pf_buf.copy_(prec_factor)
                ldh_buf.copy_(log_det_half)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    step_body()
                if prof:
                    torch.cuda.synchronize(device)
                    t_cap_acc += _time.perf_counter() - t0_c

                t0_e = _time.perf_counter() if prof else 0.0
                for gstep in range(total_steps):
                    graph.replay()
                    if gstep in rev_set:
                        M_r, S_r, Q_r, n_rev = revive_dead_components(
                            M_buf,
                            S_buf,
                            Q_buf,
                            covariance_type=self.covariance_type,
                            reg_covar=self.reg_covar,
                            dead_ratio=self.dead_ratio,
                        )
                        revivals_count += n_rev
                        if n_rev > 0:
                            M_buf.copy_(M_r)
                            S_buf.copy_(S_r)
                            Q_buf.copy_(Q_r)
                            w2, mu2, _, pf2, ldh2 = self._moments_to_estep_params(
                                M_buf,
                                S_buf,
                                Q_buf,
                                estep_dtype=dtype,
                                allow_sync=False,
                                fast_cholesky_f32=fast_chol,
                            )
                            w_buf.copy_(w2)
                            mu_buf.copy_(mu2)
                            pf_buf.copy_(pf2)
                            ldh_buf.copy_(ldh2)

                del graph
                if prof:
                    torch.cuda.synchronize(device)
                    t_ema_acc += _time.perf_counter() - t0_e
                M, S, Q = M_buf, S_buf, Q_buf
                weights, means, prec_factor, log_det_half = w_buf, mu_buf, pf_buf, ldh_buf
            else:
                t0_e = _time.perf_counter() if prof else 0.0
                for gstep in range(total_steps):
                    xb = X_std.index_select(0, batches[gstep])
                    _, _, dM, dS, dQ = e_step_batch(
                        xb,
                        weights,
                        means,
                        covariance_type=self.covariance_type,
                        prec_factor=prec_factor,
                        log_det_half=log_det_half,
                        stats_dtype=torch.float64,
                    )
                    a = float(alphas_t[gstep].item())
                    om = 1.0 - a
                    M = om * M + a * dM
                    S = om * S + a * dS
                    Q = om * Q + a * dQ
                    if gstep in rev_set:
                        M, S, Q, n_rev = revive_dead_components(
                            M,
                            S,
                            Q,
                            covariance_type=self.covariance_type,
                            reg_covar=self.reg_covar,
                            dead_ratio=self.dead_ratio,
                        )
                        revivals_count += n_rev
                    weights, means, _, prec_factor, log_det_half = self._moments_to_estep_params(
                        M,
                        S,
                        Q,
                        estep_dtype=dtype,
                        allow_sync=False,
                        fast_cholesky_f32=fast_chol,
                    )
                if prof:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t_ema_acc += _time.perf_counter() - t0_e

            last_ll: float | None = None
            if use_polish:
                t0_p = _time.perf_counter() if prof else 0.0
                prev_ll: float | None = None
                polish_chunk = 32768
                for pit in range(hp.polish_steps):
                    if N <= polish_chunk:
                        _, ll, dM, dS, dQ = e_step_batch(
                            X_std,
                            weights,
                            means,
                            covariance_type=self.covariance_type,
                            prec_factor=prec_factor,
                            log_det_half=log_det_half,
                            stats_dtype=torch.float64,
                        )
                    else:
                        dM = torch.zeros_like(M)
                        dS = torch.zeros_like(S)
                        dQ = torch.zeros_like(Q)
                        ll_sum = torch.zeros((), device=device, dtype=torch.float64)
                        for st in range(0, N, polish_chunk):
                            en = min(N, st + polish_chunk)
                            bsz = en - st
                            _, b_ll, m_b, s_b, q_b = e_step_batch(
                                X_std[st:en],
                                weights,
                                means,
                                covariance_type=self.covariance_type,
                                prec_factor=prec_factor,
                                log_det_half=log_det_half,
                                stats_dtype=torch.float64,
                            )
                            frac = bsz / N
                            dM = dM + m_b * frac
                            dS = dS + s_b * frac
                            dQ = dQ + q_b * frac
                            ll_sum = ll_sum + b_ll.to(torch.float64) * bsz
                        ll = ll_sum / N
                    M, S, Q = dM, dS, dQ
                    weights, means, _, prec_factor, log_det_half = self._moments_to_estep_params(
                        M,
                        S,
                        Q,
                        estep_dtype=dtype,
                        allow_sync=False,
                        fast_cholesky_f32=fast_chol,
                    )
                    polish_done = pit + 1
                    if self.covariance_type == "diag" and (pit + 1) % 3 == 0:
                        llv = float(ll.item())
                        last_ll = llv
                        if prev_ll is not None and abs(llv - prev_ll) < self.tol:
                            converged = True
                            break
                        prev_ll = llv
                if last_ll is None:
                    last_ll = float(ll.item())
                if prof:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t_pol_acc += _time.perf_counter() - t0_p

            weights, means, covs, prec_factor, log_det_half = self._moments_to_estep_params(
                M, S, Q, estep_dtype=dtype, allow_sync=True, fast_cholesky_f32=False
            )
            if last_ll is not None and self.n_init <= 1:
                init_lb = last_ll + float(self._log_jac_t.item())
            else:
                init_lb = self._eval_subset_lower_bound(
                    X_std, weights, means, prec_factor, log_det_half
                )

            if init_lb > best_lower_bound or best_state is None:
                best_lower_bound = init_lb
                best_state = (M, S, Q, weights, means, covs, prec_factor, log_det_half)
                best_revivals = revivals_count
                best_iters = hp.max_epochs + polish_done  # epochs started (ceil) + polish steps
                best_converged = converged

        assert best_state is not None
        t0_ex = _time.perf_counter() if prof else 0.0
        M, S, Q, weights, means, covs, prec_factor, log_det_half = best_state
        self._M_t, self._S_t, self._Q_t = M, S, Q
        self._partial_step = total_steps
        self._partial_total_steps = total_steps
        self._hp = hp
        self.converged_ = best_converged
        self.n_iter_ = best_iters
        self.lower_bound_ = float(best_lower_bound)
        self.revivals_count_ = best_revivals
        self._sync_public_attributes(weights, means, covs, prec_factor, log_det_half)
        if prof:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_exp = _time.perf_counter() - t0_ex
            self.timing_breakdown_ = {
                "standardize_ms": t_std * 1e3,
                "seeding_ms": t_seed_acc * 1e3,
                "graph_capture_ms": t_cap_acc * 1e3,
                "ema_replay_ms": t_ema_acc * 1e3,
                "polish_ms": t_pol_acc * 1e3,
                "export_ms": t_exp * 1e3,
            }
        return self

    def partial_fit(self, X: np.ndarray | torch.Tensor, y: Any = None) -> EGMM:
        dtype = self._resolve_dtype()
        N_chunk, D = self._extract_array_shape(X)
        if not hasattr(self, "_x_mean_f64"):
            device = self._resolve_device(N_chunk, D)
            self.n_features_in_ = D
            self._device_t = device
            self._dtype_t = dtype
            X_std, _, _ = self._fit_standardizer_and_transform(X, device=device, target_dtype=dtype)
            gen = self._make_generator(device)
            self._M_t, self._S_t, self._Q_t = self._initialize_moments(
                X_std, generator=gen, out_dtype=torch.float64
            )
            self._hp = auto_hyperparameters(
                n_samples=max(N_chunk * 10, 4096),
                n_components=self.n_components,
                n_features=D,
                covariance_type=self.covariance_type,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                alpha=self.alpha,
                revive_every=self.revive_every,
                init_usage_scale=self.init_usage_scale,
                polish_steps=self.polish_steps,
                schedule_law=self._SCHEDULE_LAW,
            )
            if self._hp.init_usage_scale != 1.0:
                self._M_t = self._M_t * self._hp.init_usage_scale
                self._S_t = self._S_t * self._hp.init_usage_scale
                self._Q_t = self._Q_t * self._hp.init_usage_scale
            self._partial_step = 0
            self._partial_total_steps = max(20, self._hp.max_epochs * 10)
            self.revivals_count_ = 0
            self.n_iter_ = 0
            self.converged_ = False
        else:
            X_std = self._standardize_to_device(X, device=self._device_t, target_dtype=self._dtype_t)

        weights, means, covs, prec_factor, log_det_half = self._moments_to_estep_params(
            self._M_t,
            self._S_t,
            self._Q_t,
            estep_dtype=self._dtype_t,
        )

        bs = N_chunk if self.batch_size == "auto" else min(N_chunk, int(self.batch_size))
        last_lb = 0.0
        rev_cutoff = int(0.8 * self._partial_total_steps)
        for start in range(0, N_chunk, bs):
            Xb = X_std[start : min(N_chunk, start + bs)]
            _, batch_lb, m_bar, s_bar, q_bar = e_step_batch(
                Xb,
                weights,
                means,
                covariance_type=self.covariance_type,
                prec_factor=prec_factor,
                log_det_half=log_det_half,
                stats_dtype=torch.float64,
            )
            last_lb = float(batch_lb.item()) + float(self._log_jac_t.item())
            alpha_t = compute_cosine_alpha(
                self._partial_step,
                self._partial_total_steps,
                self._hp.alpha_0,
                self._hp.alpha_min,
            )
            one_minus = 1.0 - alpha_t
            self._M_t = one_minus * self._M_t + alpha_t * m_bar.to(torch.float64)
            self._S_t = one_minus * self._S_t + alpha_t * s_bar.to(torch.float64)
            self._Q_t = one_minus * self._Q_t + alpha_t * q_bar.to(torch.float64)
            if (
                self.revive_dead
                and self._partial_step < rev_cutoff
                and (self._partial_step + 1) % self._hp.revive_every == 0
            ):
                self._M_t, self._S_t, self._Q_t, n_rev = revive_dead_components(
                    self._M_t,
                    self._S_t,
                    self._Q_t,
                    covariance_type=self.covariance_type,
                    reg_covar=self.reg_covar,
                    dead_ratio=self.dead_ratio,
                )
                self.revivals_count_ += n_rev
            weights, means, covs, prec_factor, log_det_half = self._moments_to_estep_params(
                self._M_t,
                self._S_t,
                self._Q_t,
                estep_dtype=self._dtype_t,
            )
            self._partial_step += 1

        self.n_iter_ += 1
        self.lower_bound_ = last_lb
        self._sync_public_attributes(weights, means, covs, prec_factor, log_det_half)
        return self

    def _iter_weighted_log_prob(
        self, X: np.ndarray | torch.Tensor, max_work_elements: int = 16_000_000
    ):
        """Yield ``(start, end, log pi_k + log N(x | k) + J)`` chunks in exact float64 (original units)."""
        check_is_fitted(self, ["weights_", "means_", "covariances_"])
        X_std = self._standardize_to_device(X, device=self._device_t, target_dtype=torch.float64)
        log_w, mu, pf, ldh, log_jac = self._score_params_t
        N, D = X_std.shape
        K = self.n_components
        per_row = K * D if self.covariance_type == "full" else (K + D)
        chunk_b = max(1, min(N, max_work_elements // max(1, per_row)))
        bias = log_w.unsqueeze(0) + log_jac
        for st in range(0, N, chunk_b):
            en = min(N, st + chunk_b)
            lp = compute_log_gaussian_prob_from_precisions(
                X_std[st:en],
                mu,
                pf,
                ldh,
                covariance_type=self.covariance_type,
                max_work_elements=max_work_elements,
            )
            yield st, en, lp + bias

    def _estimate_weighted_log_prob(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        parts = [wlp for _, _, wlp in self._iter_weighted_log_prob(X)]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def score_samples(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        N, _ = self._extract_array_shape(X)
        out = np.empty(N, dtype=np.float64)
        for st, en, wlp in self._iter_weighted_log_prob(X):
            out[st:en] = torch.logsumexp(wlp, dim=1).detach().cpu().numpy()
        return out

    def score(self, X: np.ndarray | torch.Tensor, y: Any = None) -> float:
        return float(self.score_samples(X).mean())

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        N, _ = self._extract_array_shape(X)
        out = np.empty((N, self.n_components), dtype=np.float64)
        for st, en, wlp in self._iter_weighted_log_prob(X):
            out[st:en] = torch.softmax(wlp, dim=1).detach().cpu().numpy()
        return out

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        N, _ = self._extract_array_shape(X)
        out = np.empty(N, dtype=np.int64)
        for st, en, wlp in self._iter_weighted_log_prob(X):
            out[st:en] = torch.argmax(wlp, dim=1).detach().cpu().numpy()
        return out

    def fit_predict(self, X: np.ndarray | torch.Tensor, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)

    def _n_parameters(self) -> int:
        K = self.n_components
        D = self.n_features_in_
        if self.covariance_type == "full":
            cov_params = K * D * (D + 1) // 2
        else:
            cov_params = K * D
        mean_params = K * D
        return int(cov_params + mean_params + K - 1)

    def bic(self, X: np.ndarray | torch.Tensor) -> float:
        n_samples = X.shape[0]
        return float(-2.0 * self.score(X) * n_samples + self._n_parameters() * np.log(n_samples))

    def aic(self, X: np.ndarray | torch.Tensor) -> float:
        n_samples = X.shape[0]
        return float(-2.0 * self.score(X) * n_samples + 2.0 * self._n_parameters())

    def sample(self, n_samples: int = 1) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["weights_", "means_", "covariances_"])
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
        rng = np.random.default_rng(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights_)
        X_list = []
        y_list = []
        for k, count in enumerate(n_samples_comp):
            if count == 0:
                continue
            if self.covariance_type == "diag":
                samples = rng.normal(
                    loc=self.means_[k],
                    scale=np.sqrt(self.covariances_[k]),
                    size=(count, self.n_features_in_),
                )
            else:
                samples = rng.multivariate_normal(
                    mean=self.means_[k], cov=self.covariances_[k], size=count
                )
            X_list.append(samples)
            y_list.append(np.full(count, k, dtype=np.int64))
        X_out = np.vstack(X_list)
        y_out = np.concatenate(y_list)
        return X_out, y_out

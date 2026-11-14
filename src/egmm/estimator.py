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
    """

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

    def _to_float64_tensor(
        self, X: np.ndarray | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(device=device, dtype=torch.float64)
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}.")
        return torch.as_tensor(arr, dtype=torch.float64, device=device)

    def _standardize(
        self, X_f64: torch.Tensor, target_dtype: torch.dtype
    ) -> torch.Tensor:
        return ((X_f64 - self._x_mean_f64.unsqueeze(0)) / self._x_std_f64.unsqueeze(0)).to(
            dtype=target_dtype
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, D = X_std.shape
        K = self.n_components
        device, dtype = X_std.device, X_std.dtype

        n_seed = min(N, max(2048, 32 * K))
        if n_seed < N:
            perm = torch.randperm(N, generator=generator, device=device)[:n_seed]
            X_seed = X_std[perm]
        else:
            X_seed = X_std

        if self.init_params == "kmeans++":
            means = init_kmeans_pp(X_seed, K, generator=generator)
        elif self.init_params == "random":
            idx = torch.randperm(X_seed.shape[0], generator=generator, device=device)[:K]
            if idx.shape[0] < K:
                idx = torch.randint(0, X_seed.shape[0], (K,), generator=generator, device=device)
            means = X_seed[idx].clone()
        else:
            raise ValueError(f"Unsupported init_params={self.init_params!r}.")

        # Assign seed points to nearest centers to initialize empirical moments cleanly.
        dists = torch.cdist(X_seed, means)
        assign = torch.argmin(dists, dim=1)
        one_hot = torch.nn.functional.one_hot(assign, num_classes=K).to(dtype=dtype)
        counts = one_hot.sum(dim=0)

        global_var = X_seed.var(dim=0, unbiased=False).clamp_min(1e-2)
        weights = torch.empty(K, device=device, dtype=dtype)
        if self.covariance_type == "diag":
            covs = torch.empty((K, D), device=device, dtype=dtype)
        else:
            covs = torch.empty((K, D, D), device=device, dtype=dtype)

        inv_n = 1.0 / max(1, n_seed)
        eye_d = torch.eye(D, device=device, dtype=dtype)
        for k in range(K):
            mask = assign == k
            cnt = int(counts[k].item())
            if cnt >= 2:
                pts = X_seed[mask]
                means[k] = pts.mean(dim=0)
                diff = pts - means[k].unsqueeze(0)
                weights[k] = cnt * inv_n
                if self.covariance_type == "diag":
                    covs[k] = (diff * diff).mean(dim=0) + self.reg_covar
                else:
                    covs[k] = (diff.T @ diff) / cnt + self.reg_covar * eye_d
            else:
                weights[k] = max(inv_n, 1e-3 / K)
                if self.covariance_type == "diag":
                    covs[k] = global_var + self.reg_covar
                else:
                    covs[k] = torch.diag(global_var) + self.reg_covar * eye_d

        weights = weights / weights.sum()
        return params_to_moments(weights, means, covs, covariance_type=self.covariance_type)

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
            L_cov_orig = torch.linalg.cholesky(c_orig_t)
            D = c_orig_t.shape[-1]
            eye_d = torch.eye(D, device=c_orig_t.device, dtype=torch.float64).expand_as(L_cov_orig)
            L_inv_orig = torch.linalg.solve_triangular(L_cov_orig, eye_d, upper=False)
            pc_orig_t = L_inv_orig.transpose(-1, -2)
            p_orig_t = pc_orig_t @ pc_orig_t.transpose(-1, -2)

        self.weights_ = w_np
        self.means_ = m_orig
        self.covariances_ = c_orig_t.detach().cpu().numpy()
        self.precisions_ = p_orig_t.detach().cpu().numpy()
        self.precisions_cholesky_ = pc_orig_t.detach().cpu().numpy()

    def _step_alpha(self, step: int, total_steps: int, hp: Any) -> float:
        return compute_cosine_alpha(step, total_steps, hp.alpha_0, hp.alpha_min)

    def _prepare_fit_data(
        self, X: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.device, torch.dtype, int, int]:
        dtype = self._resolve_dtype()
        N, D = X.shape if isinstance(X, torch.Tensor) else np.asarray(X).shape
        device = self._resolve_device(N, D)
        X_f64 = self._to_float64_tensor(X, device=device)

        self.n_features_in_ = D
        self._device_t = device
        self._dtype_t = dtype

        # Fit standardization strictly on training data in float64 to avoid offset cancellation.
        self._x_mean_f64 = X_f64.mean(dim=0)
        self._x_std_f64 = X_f64.std(dim=0, unbiased=False).clamp_min(1e-12)
        self._log_jac_t = (-torch.log(self._x_std_f64).sum()).to(dtype=dtype)
        X_std = self._standardize(X_f64, target_dtype=dtype)
        return X_std, device, dtype, N, D

    def fit(self, X: np.ndarray | torch.Tensor, y: Any = None) -> EGMM:
        X_std, device, dtype, N, D = self._prepare_fit_data(X)

        hp = auto_hyperparameters(
            n_samples=N,
            n_components=self.n_components,
            n_features=D,
            covariance_type=self.covariance_type,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            alpha=self.alpha,
            revive_every=self.revive_every,
        )
        steps_per_epoch = max(1, math.ceil(N / hp.batch_size))
        total_steps = hp.max_epochs * steps_per_epoch

        best_lower_bound = -math.inf
        best_state: tuple[torch.Tensor, ...] | None = None
        best_revivals = 0
        best_epochs = 0
        best_converged = False

        for init_idx in range(max(1, self.n_init)):
            gen = self._make_generator(device, seed_offset=init_idx * 9973)
            M, S, Q = self._initialize_moments(X_std, generator=gen)
            weights, means, covs, prec_factor, log_det_half = moments_to_params_and_precisions(
                M, S, Q, covariance_type=self.covariance_type, reg_covar=self.reg_covar
            )

            global_step = 0
            revivals_count = 0
            prev_epoch_lb = -math.inf
            converged = False
            epochs_done = 0

            for epoch in range(hp.max_epochs):
                perm = torch.randperm(N, generator=gen, device=device)
                epoch_ll_sum = 0.0
                epoch_seen = 0

                for start in range(0, N, hp.batch_size):
                    end = min(N, start + hp.batch_size)
                    Xb = X_std[perm[start:end]]

                    _, batch_lb, m_bar, s_bar, q_bar = e_step_batch(
                        Xb,
                        weights,
                        means,
                        covariance_type=self.covariance_type,
                        prec_factor=prec_factor,
                        log_det_half=log_det_half,
                    )
                    bsz = end - start
                    epoch_ll_sum += float(batch_lb.item()) * bsz
                    epoch_seen += bsz

                    alpha_t = self._step_alpha(global_step, total_steps, hp)
                    one_minus = 1.0 - alpha_t
                    M = one_minus * M + alpha_t * m_bar
                    S = one_minus * S + alpha_t * s_bar
                    Q = one_minus * Q + alpha_t * q_bar

                    # Revive dead components only in the first 80% of steps so final cosine phase settles.
                    if (
                        self.revive_dead
                        and global_step < int(0.8 * total_steps)
                        and (global_step + 1) % hp.revive_every == 0
                    ):
                        M, S, Q, n_rev = revive_dead_components(
                            M,
                            S,
                            Q,
                            covariance_type=self.covariance_type,
                            reg_covar=self.reg_covar,
                            dead_ratio=self.dead_ratio,
                        )
                        revivals_count += n_rev

                    weights, means, covs, prec_factor, log_det_half = (
                        moments_to_params_and_precisions(
                            M, S, Q, covariance_type=self.covariance_type, reg_covar=self.reg_covar
                        )
                    )
                    global_step += 1

                epochs_done = epoch + 1
                epoch_lb = (epoch_ll_sum / max(1, epoch_seen)) + float(self._log_jac_t.item())
                if epoch >= 1 and abs(epoch_lb - prev_epoch_lb) < self.tol:
                    # Only stop early after at least half of scheduled cosine decay has passed.
                    if global_step >= total_steps // 2:
                        converged = True
                        prev_epoch_lb = epoch_lb
                        break
                prev_epoch_lb = epoch_lb

            if prev_epoch_lb > best_lower_bound or best_state is None:
                best_lower_bound = prev_epoch_lb
                best_state = (M, S, Q, weights, means, covs, prec_factor, log_det_half)
                best_revivals = revivals_count
                best_epochs = epochs_done
                best_converged = converged

        assert best_state is not None
        M, S, Q, weights, means, covs, prec_factor, log_det_half = best_state
        self._M_t, self._S_t, self._Q_t = M, S, Q
        self._partial_step = total_steps
        self._partial_total_steps = total_steps
        self._hp = hp
        self.converged_ = best_converged
        self.n_iter_ = best_epochs
        self.lower_bound_ = float(best_lower_bound)
        self.revivals_count_ = best_revivals
        self._sync_public_attributes(weights, means, covs, prec_factor, log_det_half)
        return self

    def partial_fit(self, X: np.ndarray | torch.Tensor, y: Any = None) -> EGMM:
        dtype = self._resolve_dtype()
        N_chunk, D = X.shape if isinstance(X, torch.Tensor) else np.asarray(X).shape
        if not hasattr(self, "_x_mean_f64"):
            device = self._resolve_device(N_chunk, D)
            X_f64 = self._to_float64_tensor(X, device=device)
            self.n_features_in_ = D
            self._device_t = device
            self._dtype_t = dtype
            self._x_mean_f64 = X_f64.mean(dim=0)
            self._x_std_f64 = X_f64.std(dim=0, unbiased=False).clamp_min(1e-12)
            self._log_jac_t = (-torch.log(self._x_std_f64).sum()).to(dtype=dtype)
            X_std = self._standardize(X_f64, target_dtype=dtype)
            gen = self._make_generator(device)
            self._M_t, self._S_t, self._Q_t = self._initialize_moments(X_std, generator=gen)
            self._hp = auto_hyperparameters(
                n_samples=max(N_chunk * 10, 4096),
                n_components=self.n_components,
                n_features=D,
                covariance_type=self.covariance_type,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                alpha=self.alpha,
                revive_every=self.revive_every,
            )
            self._partial_step = 0
            self._partial_total_steps = max(20, self._hp.max_epochs * 10)
            self.revivals_count_ = 0
            self.n_iter_ = 0
            self.converged_ = False
        else:
            X_f64 = self._to_float64_tensor(X, device=self._device_t)
            X_std = self._standardize(X_f64, target_dtype=self._dtype_t)

        weights, means, covs, prec_factor, log_det_half = moments_to_params_and_precisions(
            self._M_t,
            self._S_t,
            self._Q_t,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
        )

        bs = N_chunk if self.batch_size == "auto" else min(N_chunk, int(self.batch_size))
        last_lb = 0.0
        for start in range(0, N_chunk, bs):
            Xb = X_std[start : min(N_chunk, start + bs)]
            _, batch_lb, m_bar, s_bar, q_bar = e_step_batch(
                Xb,
                weights,
                means,
                covariance_type=self.covariance_type,
                prec_factor=prec_factor,
                log_det_half=log_det_half,
            )
            last_lb = float(batch_lb.item()) + float(self._log_jac_t.item())
            alpha_t = compute_cosine_alpha(
                self._partial_step,
                self._partial_total_steps,
                self._hp.alpha_0,
                self._hp.alpha_min,
            )
            one_minus = 1.0 - alpha_t
            self._M_t = one_minus * self._M_t + alpha_t * m_bar
            self._S_t = one_minus * self._S_t + alpha_t * s_bar
            self._Q_t = one_minus * self._Q_t + alpha_t * q_bar
            if self.revive_dead and (self._partial_step + 1) % self._hp.revive_every == 0:
                self._M_t, self._S_t, self._Q_t, n_rev = revive_dead_components(
                    self._M_t,
                    self._S_t,
                    self._Q_t,
                    covariance_type=self.covariance_type,
                    reg_covar=self.reg_covar,
                    dead_ratio=self.dead_ratio,
                )
                self.revivals_count_ += n_rev
            weights, means, covs, prec_factor, log_det_half = moments_to_params_and_precisions(
                self._M_t,
                self._S_t,
                self._Q_t,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
            )
            self._partial_step += 1

        self.n_iter_ += 1
        self.lower_bound_ = last_lb
        self._sync_public_attributes(weights, means, covs, prec_factor, log_det_half)
        return self

    def _estimate_weighted_log_prob(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        check_is_fitted(self, ["weights_", "means_", "covariances_"])
        X_f64 = self._to_float64_tensor(X, device=self._device_t)
        X_std = self._standardize(X_f64, target_dtype=self._dtype_t)
        log_prob = compute_log_gaussian_prob_from_precisions(
            X_std,
            self._means_std_t,
            self._prec_factor_std_t,
            self._log_det_half_std_t,
            covariance_type=self.covariance_type,
        )
        log_weights = torch.log(self._weights_t.clamp_min(1e-30))
        return log_prob + log_weights.unsqueeze(0) + self._log_jac_t

    def score_samples(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        wlp = self._estimate_weighted_log_prob(X)
        return torch.logsumexp(wlp, dim=1).detach().cpu().numpy().astype(np.float64)

    def score(self, X: np.ndarray | torch.Tensor, y: Any = None) -> float:
        return float(self.score_samples(X).mean())

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        wlp = self._estimate_weighted_log_prob(X)
        log_norm = torch.logsumexp(wlp, dim=1, keepdim=True)
        return torch.exp(wlp - log_norm).detach().cpu().numpy().astype(np.float64)

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        wlp = self._estimate_weighted_log_prob(X)
        return torch.argmax(wlp, dim=1).detach().cpu().numpy()

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

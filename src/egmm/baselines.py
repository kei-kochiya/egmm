"""Fair PyTorch baselines and ablations sharing exact kernels with EGMM."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.mixture import GaussianMixture

from egmm.estimator import EGMM
from egmm.kernels import (
    CovarianceType,
    e_step_batch,
    init_kmeans_pp,
    moments_to_params_and_precisions,
    params_to_moments,
)
from egmm.scaling import compute_cosine_alpha


class TorchBatchGMM(EGMM):
    """Exact full-batch EM running in the same PyTorch GPU/CPU engine as EGMM."""

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: CovarianceType = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        chunk_size: int = 16384,
        n_init: int = 1,
        init_params: str = "kmeans++",
        device: str = "auto",
        dtype: str = "float32",
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_epochs=max_iter,
            batch_size=chunk_size,
            alpha=1.0,
            revive_dead=False,
            n_init=n_init,
            init_params=init_params,
            device=device,
            dtype=dtype,
            random_state=random_state,
            verbose=verbose,
        )
        self.max_iter = max_iter
        self.chunk_size = chunk_size

    def _run_batch_em(
        self,
        X_std: torch.Tensor,
        M: torch.Tensor,
        S: torch.Tensor,
        Q: torch.Tensor,
        max_iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int, bool]:
        N = X_std.shape[0]
        weights, means, covs, prec_factor, log_det_half = self._moments_to_estep_params(
            M, S, Q, estep_dtype=X_std.dtype
        )
        prev_lb = -math.inf
        converged = False
        iters_done = 0

        for it in range(max_iters):
            M_acc = torch.zeros_like(M)
            S_acc = torch.zeros_like(S)
            Q_acc = torch.zeros_like(Q)
            ll_sum = torch.zeros((), device=X_std.device, dtype=torch.float64)

            for start in range(0, N, self.chunk_size):
                end = min(N, start + self.chunk_size)
                Xb = X_std[start:end]
                bsz = end - start
                _, batch_lb, m_bar, s_bar, q_bar = e_step_batch(
                    Xb,
                    weights,
                    means,
                    covariance_type=self.covariance_type,
                    prec_factor=prec_factor,
                    log_det_half=log_det_half,
                    stats_dtype=M.dtype,
                )
                frac = bsz / N
                M_acc += m_bar.to(M.dtype) * frac
                S_acc += s_bar.to(S.dtype) * frac
                Q_acc += q_bar.to(Q.dtype) * frac
                ll_sum = ll_sum + batch_lb.to(torch.float64) * bsz

            M, S, Q = M_acc, S_acc, Q_acc
            weights, means, covs, prec_factor, log_det_half = self._moments_to_estep_params(
                M, S, Q, estep_dtype=X_std.dtype
            )
            lb = (float(ll_sum.item()) / N) + float(self._log_jac_t.item())
            iters_done = it + 1
            if it >= 1 and abs(lb - prev_lb) < self.tol:
                converged = True
                prev_lb = lb
                break
            prev_lb = lb

        return M, S, Q, weights, means, covs, prec_factor, log_det_half, prev_lb, iters_done, converged

    def fit(self, X: np.ndarray | torch.Tensor, y: Any = None) -> TorchBatchGMM:
        X_std, device, dtype, N, D = self._prepare_fit_data(X)
        best_lb = -math.inf
        best_res = None

        for init_idx in range(max(1, self.n_init)):
            gen = self._make_generator(device, seed_offset=init_idx * 9973)
            M0, S0, Q0 = self._initialize_moments(X_std, generator=gen, out_dtype=torch.float64)
            res = self._run_batch_em(X_std, M0, S0, Q0, max_iters=self.max_iter)
            if res[8] > best_lb or best_res is None:
                best_lb = res[8]
                best_res = res

        assert best_res is not None
        M, S, Q, weights, means, covs, prec_factor, log_det_half, lb, iters_done, converged = best_res
        self._M_t, self._S_t, self._Q_t = M, S, Q
        self.converged_ = converged
        self.n_iter_ = iters_done
        self.lower_bound_ = float(lb)
        self.revivals_count_ = 0
        self._sync_public_attributes(weights, means, covs, prec_factor, log_det_half)
        return self


class TorchBatchGMMFullInit(TorchBatchGMM):
    """Sensitivity baseline: TorchBatchGMM seeded on the *full* training set.

    Greedy k-means++ and the 10 Lloyd steps run on all N points (as scikit-learn's KMeans-based
    initialization does) instead of EGMM's ``max(2048, 128 K)``-point seed subsample.
    """

    _SEED_POINTS_PER_COMPONENT = 1 << 40


class CappeMoulinesOnlineGMM(EGMM):
    """Cappé & Moulines (2009) Online EM with polynomial step size (t + t0)^(-rho)."""

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: CovarianceType = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_epochs: int | str = 5,
        batch_size: int | str = 1024,
        rho: float = 0.75,
        t0: float = 2.0,
        n_init: int = 1,
        init_params: str = "kmeans++",
        device: str = "auto",
        dtype: str = "float32",
        random_state: int | None = None,
        verbose: int = 0,
        polish_steps: int | str = 0,
        init_usage_scale: float | str = 1.0,
    ) -> None:
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_epochs=max_epochs,
            batch_size=batch_size,
            alpha=1.0,
            revive_dead=False,
            n_init=n_init,
            init_params=init_params,
            device=device,
            dtype=dtype,
            random_state=random_state,
            verbose=verbose,
            polish_steps=polish_steps,
            init_usage_scale=init_usage_scale,
        )
        self.rho = rho
        self.t0 = t0

    def _step_alpha(self, step: int, total_steps: int, hp: Any) -> float:
        return float(min(1.0, max(1e-4, (step + self.t0) ** (-self.rho))))


class NaiveMinibatchGMM(EGMM):
    """Ablation: constant-alpha EMA without cosine annealing or dead-component revival."""

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: CovarianceType = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_epochs: int | str = "auto",
        batch_size: int | str = "auto",
        alpha: float | str = 0.15,
        n_init: int = 1,
        init_params: str = "kmeans++",
        device: str = "auto",
        dtype: str = "float32",
        random_state: int | None = None,
        verbose: int = 0,
        polish_steps: int | str = 0,
        init_usage_scale: float | str = 1.0,
    ) -> None:
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_epochs=max_epochs,
            batch_size=batch_size,
            alpha=alpha,
            revive_dead=False,
            n_init=n_init,
            init_params=init_params,
            device=device,
            dtype=dtype,
            random_state=random_state,
            verbose=verbose,
            polish_steps=polish_steps,
            init_usage_scale=init_usage_scale,
        )

    def _step_alpha(self, step: int, total_steps: int, hp: Any) -> float:
        return float(hp.alpha_0)


class EMeansInitGMM(TorchBatchGMM):
    """Ablation: e-means style mini-batch EMA k-means warm-start + batch EM refinement."""

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: CovarianceType = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        kmeans_epochs: int = 2,
        em_iters: int = 10,
        batch_size: int = 1024,
        alpha_0: float = 0.3,
        device: str = "auto",
        dtype: str = "float32",
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=em_iters,
            chunk_size=16384,
            device=device,
            dtype=dtype,
            random_state=random_state,
        )
        self.kmeans_epochs = kmeans_epochs
        self.em_iters = em_iters
        self.batch_size = batch_size
        self.alpha_0 = alpha_0

    def fit(self, X: np.ndarray | torch.Tensor, y: Any = None) -> EMeansInitGMM:
        X_std, device, dtype, N, D = self._prepare_fit_data(X)
        K = self.n_components
        gen = self._make_generator(device)

        # Same greedy k-means++ seeding routine and seed-set size as EGMM / TorchBatchGMM.
        n_seed = min(N, max(self._SEED_MIN_POINTS, self._SEED_POINTS_PER_COMPONENT * K))
        seed_idx = torch.randperm(N, generator=gen, device=device)[:n_seed]
        centers = init_kmeans_pp(X_std[seed_idx], K, generator=gen, n_local_trials=None)

        bs = min(N, self.batch_size)
        steps_per_epoch = max(1, math.ceil(N / bs))
        total_steps = max(1, self.kmeans_epochs * steps_per_epoch)

        N_ema = torch.full((K,), 1.0 / K, device=device, dtype=dtype)
        S_ema = N_ema.unsqueeze(-1) * centers

        step = 0
        for _ in range(self.kmeans_epochs):
            perm = torch.randperm(N, generator=gen, device=device)
            for start in range(0, N, bs):
                Xb = X_std[perm[start : min(N, start + bs)]]
                assign = torch.argmin(torch.cdist(Xb, centers), dim=1)
                one_hot = torch.nn.functional.one_hot(assign, num_classes=K).to(dtype=dtype)
                n_bar = one_hot.mean(dim=0)
                s_bar = (one_hot.T @ Xb) / Xb.shape[0]
                a = compute_cosine_alpha(step, total_steps, self.alpha_0, 0.1 * self.alpha_0)
                N_ema = (1.0 - a) * N_ema + a * n_bar
                S_ema = (1.0 - a) * S_ema + a * s_bar
                centers = S_ema / N_ema.clamp_min(1e-12).unsqueeze(-1)
                step += 1

        # Build empirical covariances from Voronoi partition on subsample/full data
        n_cov = min(N, 32768)
        X_cov = X_std[:n_cov]
        assign = torch.argmin(torch.cdist(X_cov, centers), dim=1)
        one_hot = torch.nn.functional.one_hot(assign, num_classes=K).to(dtype=dtype)
        counts = one_hot.sum(dim=0)
        global_var = X_cov.var(dim=0, unbiased=False).clamp_min(1e-2)
        weights = torch.empty(K, device=device, dtype=dtype)
        eye_d = torch.eye(D, device=device, dtype=dtype)
        if self.covariance_type == "diag":
            covs = torch.empty((K, D), device=device, dtype=dtype)
        else:
            covs = torch.empty((K, D, D), device=device, dtype=dtype)

        for k in range(K):
            cnt = int(counts[k].item())
            if cnt >= 2:
                pts = X_cov[assign == k]
                diff = pts - centers[k].unsqueeze(0)
                weights[k] = cnt / n_cov
                if self.covariance_type == "diag":
                    covs[k] = (diff * diff).mean(dim=0) + self.reg_covar
                else:
                    covs[k] = (diff.T @ diff) / cnt + self.reg_covar * eye_d
            else:
                weights[k] = max(1.0 / n_cov, 1e-3 / K)
                covs[k] = (global_var + self.reg_covar) if self.covariance_type == "diag" else (torch.diag(global_var) + self.reg_covar * eye_d)

        weights = weights / weights.sum()
        M0, S0, Q0 = params_to_moments(weights, centers, covs, covariance_type=self.covariance_type)
        M, S, Q, w, m, c, pf, ldh, lb, iters_done, converged = self._run_batch_em(
            X_std, M0, S0, Q0, max_iters=self.em_iters
        )
        self._M_t, self._S_t, self._Q_t = M, S, Q
        self.converged_ = converged
        self.n_iter_ = iters_done
        self.lower_bound_ = float(lb)
        self.revivals_count_ = 0
        self._sync_public_attributes(w, m, c, pf, ldh)
        return self


class StandardizedSklearnGMM(BaseEstimator, DensityMixin):
    """Scale-invariant wrapper around sklearn.mixture.GaussianMixture.

    Applies feature-wise standardization with std floor 1e-12 (identical to EGMM
    and TorchBatchGMM) so that ``reg_covar=1e-6`` acts in standardized coordinates
    rather than raw coordinates, and adjusts all log-likelihoods by the exact
    change-of-variables log-Jacobian J = -sum(log(std)). The default ``init_params="kmeans"``
    is scikit-learn's own default, so this wrapper differs from raw ``GaussianMixture`` only
    by the standardization.
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: CovarianceType = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self._mean) / self._std

    def fit(self, X: np.ndarray, y: Any = None) -> StandardizedSklearnGMM:
        X_arr = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = int(X_arr.shape[1])
        self._mean = X_arr.mean(axis=0, keepdims=True)
        self._std = np.maximum(X_arr.std(axis=0, keepdims=True), 1e-12)
        self._log_jac = float(-np.log(self._std).sum())

        X_std = (X_arr - self._mean) / self._std
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            random_state=self.random_state,
        ).fit(X_std)

        self.weights_ = self._gmm.weights_.copy()
        self.means_ = self._gmm.means_ * self._std + self._mean
        if self.covariance_type == "diag":
            self.covariances_ = self._gmm.covariances_ * (self._std ** 2)
        else:
            scale_mat = self._std.T @ self._std
            self.covariances_ = self._gmm.covariances_ * scale_mat[None, :, :]
        self.converged_ = bool(self._gmm.converged_)
        self.n_iter_ = int(self._gmm.n_iter_)
        self.lower_bound_ = float(self._gmm.lower_bound_) + self._log_jac
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self._gmm.score_samples(self._standardize(X)) + self._log_jac

    def score(self, X: np.ndarray, y: Any = None) -> float:
        return float(np.mean(self.score_samples(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._gmm.predict_proba(self._standardize(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._gmm.predict(self._standardize(X))

    def _n_parameters(self) -> int:
        return int(self._gmm._n_parameters())

    def bic(self, X: np.ndarray) -> float:
        X_arr = np.asarray(X, dtype=np.float64)
        return float(-2.0 * self.score(X_arr) * X_arr.shape[0] + self._n_parameters() * math.log(X_arr.shape[0]))

    def aic(self, X: np.ndarray) -> float:
        X_arr = np.asarray(X, dtype=np.float64)
        return float(-2.0 * self.score(X_arr) * X_arr.shape[0] + 2.0 * self._n_parameters())


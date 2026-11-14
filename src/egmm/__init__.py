"""e-GMM: Exponential Moving Average Gaussian Mixture Models."""

__all__ = [
    "EGMM",
    "compute_log_gaussian_prob",
    "e_step_batch",
    "init_kmeans_pp",
    "moments_to_params",
    "params_to_moments",
]

from egmm.estimator import EGMM
from egmm.kernels import (
    compute_log_gaussian_prob,
    e_step_batch,
    init_kmeans_pp,
    moments_to_params,
    params_to_moments,
)

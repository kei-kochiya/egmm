"""Microbenchmark E-step and moment-update kernels on CPU and CUDA."""

import time
import torch

from egmm.kernels import e_step_batch, moments_to_params, params_to_moments


def bench_step(device: str, cov_type: str, B: int, K: int, D: int, iters: int = 50) -> float:
    torch.manual_seed(0)
    X = torch.randn(B, D, device=device, dtype=torch.float32)
    weights = torch.full((K,), 1.0 / K, device=device, dtype=torch.float32)
    means = torch.randn(K, D, device=device, dtype=torch.float32)
    if cov_type == "diag":
        covs = torch.ones(K, D, device=device, dtype=torch.float32)
    else:
        covs = torch.eye(D, device=device, dtype=torch.float32).unsqueeze(0).repeat(K, 1, 1)
    M, S, Q = params_to_moments(weights, means, covs, covariance_type=cov_type)

    # Warmup
    for _ in range(5):
        w, m, c = moments_to_params(M, S, Q, covariance_type=cov_type)
        _, _, m_bar, s_bar, q_bar = e_step_batch(X, w, m, c, covariance_type=cov_type)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        w, m, c = moments_to_params(M, S, Q, covariance_type=cov_type)
        _, _, m_bar, s_bar, q_bar = e_step_batch(X, w, m, c, covariance_type=cov_type)
        M = 0.9 * M + 0.1 * m_bar
        S = 0.9 * S + 0.1 * s_bar
        Q = 0.9 * Q + 0.1 * q_bar
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


if __name__ == "__main__":
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    configs = [
        ("diag", 2048, 32, 64),
        ("diag", 4096, 64, 128),
        ("full", 1024, 16, 32),
        ("full", 2048, 32, 64),
        ("full", 2048, 64, 128),
    ]
    for dev in devices:
        for cov_type, B, K, D in configs:
            ms = bench_step(dev, cov_type, B, K, D, iters=30)
            print(f"{dev:4s} | {cov_type:4s} | B={B:4d} K={K:2d} D={D:3d} -> {ms:.3f} ms/step")

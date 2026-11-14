"""Empirical scaling-law discovery sweep for e-GMM on calibration-only synthetic mixtures."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit

from egmm.baselines import TorchBatchGMM
from egmm.estimator import EGMM


def make_calibration_gmm_data(
    n_samples: int,
    n_components: int,
    n_features: int,
    seed: int,
    anisotropy: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate challenging anisotropic, imbalanced, overlapping Gaussian mixture."""
    rng = np.random.default_rng(seed)
    # Imbalanced Dirichlet weights
    weights = rng.dirichlet(np.full(n_components, 1.5))
    counts = rng.multinomial(n_samples, weights)

    # Overlapping centers
    means = rng.normal(scale=2.5, size=(n_components, n_features))
    X_parts = []
    for k, cnt in enumerate(counts):
        if cnt == 0:
            continue
        # Random orthogonal rotation + geometric spectrum of eigenvalues
        Q_raw = rng.normal(size=(n_features, n_features))
        Q_orth, _ = np.linalg.qr(Q_raw)
        scales = np.geomspace(1.0 / math.sqrt(anisotropy), math.sqrt(anisotropy), n_features)
        Z = rng.normal(size=(cnt, n_features)) * scales
        X_parts.append(Z @ Q_orth.T + means[k])

    X = np.vstack(X_parts)
    perm = rng.permutation(X.shape[0])
    X = X[perm]
    n_train = int(0.8 * X.shape[0])
    return X[:n_train], X[n_train:]


def run_sweep() -> pd.DataFrame:
    configs = [
        # (N_total, K, D, seed)
        (6_000, 8, 8, 10001),
        (18_000, 16, 16, 10002),
        (36_000, 24, 32, 10003),
        (75_000, 32, 48, 10004),
        (120_000, 16, 64, 10005),
        (160_000, 48, 24, 10006),
        (45_000, 12, 96, 10007),
    ]
    batch_grid = [512, 1024, 2048, 4096, 8192]
    alpha_grid = [0.05, 0.12, 0.25, 0.45, 0.70]
    epoch_grid = [2, 4, 6]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Warmup CUDA/cuBLAS/cuSOLVER handles before timing
        _w = torch.randn(512, 16, device="cuda")
        _ = EGMM(4, "full", batch_size=256, max_epochs=1, device="cuda").fit(_w)
        torch.cuda.synchronize()
    rows = []

    for cov_type in ["diag", "full"]:
        for N_tot, K, D, seed in configs:
            X_tr, X_va = make_calibration_gmm_data(N_tot, K, D, seed=seed)
            N_tr = X_tr.shape[0]
            X_tr_t = torch.from_numpy(X_tr).to(device=device, dtype=torch.float64)
            X_va_t = torch.from_numpy(X_va).to(device=device, dtype=torch.float64)

            # Reference batch EM (50 iters)
            t0 = time.perf_counter()
            ref = TorchBatchGMM(
                n_components=K,
                covariance_type=cov_type,
                max_iter=50,
                tol=1e-4,
                device=device,
                random_state=seed,
            ).fit(X_tr_t)
            if device == "cuda":
                torch.cuda.synchronize()
            ref_time = time.perf_counter() - t0
            ref_nll = -ref.score(X_va_t)

            for T in epoch_grid:
                for B in batch_grid:
                    if B > N_tr // 2:
                        continue
                    if N_tr > 20_000 and B < 1024:
                        continue
                    steps = T * math.ceil(N_tr / B)
                    if steps < 4 or steps > 120:
                        continue
                    for a0 in alpha_grid:
                        t1 = time.perf_counter()
                        m = EGMM(
                            n_components=K,
                            covariance_type=cov_type,
                            max_epochs=T,
                            batch_size=B,
                            alpha=a0,
                            tol=1e-6,  # disable early exit during sweep
                            device=device,
                            random_state=seed,
                        ).fit(X_tr_t)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        dt = time.perf_counter() - t1
                        nll = -m.score(X_va_t)
                        rows.append(
                            {
                                "cov_type": cov_type,
                                "N": N_tr,
                                "K": K,
                                "D": D,
                                "T": T,
                                "B": B,
                                "steps": steps,
                                "alpha_0": a0,
                                "val_nll": nll,
                                "ref_nll": ref_nll,
                                "nll_diff": nll - ref_nll,
                                "fit_time": dt,
                                "ref_time": ref_time,
                                "speedup": ref_time / max(dt, 1e-6),
                                "revivals": m.revivals_count_,
                            }
                        )
            print(
                f"Done {cov_type:4s} N={N_tr:6d} K={K:2d} D={D:2d} | "
                f"ref_nll={ref_nll:.3f} ({ref_time:.2f}s)",
                flush=True,
            )

    return pd.DataFrame(rows)


def analyze_and_fit(df: pd.DataFrame) -> None:
    for cov_type in ["diag", "full"]:
        sub = df[df["cov_type"] == cov_type].copy()
        print(f"\n=== Scaling Analysis ({cov_type}) ===")
        # For each (N, K, D, T), find best NLL and fastest config within 0.05 NLL of best
        winners = []
        for (N, K, D, T), grp in sub.groupby(["N", "K", "D", "T"]):
            best_row = grp.loc[grp["val_nll"].idxmin()]
            # Near-optimal pareto winner (within 0.03 NLL of best or better than ref)
            tol = 0.03
            near = grp[grp["val_nll"] <= best_row["val_nll"] + tol]
            fast_best = near.loc[near["fit_time"].idxmin()]
            winners.append(
                {
                    "N": N,
                    "K": K,
                    "D": D,
                    "T": T,
                    "B_best": best_row["B"],
                    "a0_best": best_row["alpha_0"],
                    "B_pareto": fast_best["B"],
                    "a0_pareto": fast_best["alpha_0"],
                    "nll_diff_best": best_row["nll_diff"],
                    "nll_diff_pareto": fast_best["nll_diff"],
                    "speedup_pareto": fast_best["speedup"],
                    "steps_pareto": fast_best["steps"],
                }
            )
        wdf = pd.DataFrame(winners)
        print(wdf.to_string(index=False))

        # Fit alpha_0* ~ c * steps^p1 * K^p2 * D^p3 * (B/N)^p4 in log space
        # Actually fit log(alpha_0) = a + b*log(steps) + c*log(B/(K*D))
        y_log = np.log(wdf["a0_best"].values)
        X_feat = np.column_stack(
            [
                np.ones(len(wdf)),
                np.log(wdf["steps_pareto"].values),
                np.log(wdf["B_best"].values / (wdf["K"].values * wdf["D"].values)),
                np.log(wdf["K"].values),
            ]
        )
        beta, _, _, _ = np.linalg.lstsq(X_feat, y_log, rcond=None)
        pred_log = X_feat @ beta
        ss_res = np.sum((y_log - pred_log) ** 2)
        ss_tot = np.sum((y_log - y_log.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        print(
            f"Fitted log(alpha_0) = {beta[0]:.3f} + {beta[1]:.3f}*log(steps) "
            f"+ {beta[2]:.3f}*log(B/(K*D)) + {beta[3]:.3f}*log(K) | R^2={r2:.3f}"
        )


if __name__ == "__main__":
    out_dir = Path("docs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = run_sweep()
    df.to_csv(out_dir / "scaling_sweep.csv", index=False)
    analyze_and_fit(df)

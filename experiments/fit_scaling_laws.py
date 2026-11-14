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
    dirichlet_alpha: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate challenging anisotropic, imbalanced, overlapping Gaussian mixture."""
    rng = np.random.default_rng(seed)
    # Imbalanced Dirichlet weights
    weights = rng.dirichlet(np.full(n_components, dirichlet_alpha))
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
        # (N_total, K, D, seed, anisotropy, dirichlet_alpha)
        (5_000, 8, 8, 90001, 6.0, 1.5),
        (15_000, 16, 16, 90002, 8.0, 1.2),
        (36_000, 24, 32, 90003, 10.0, 1.0),
        (75_000, 32, 48, 90004, 8.0, 1.2),
        (120_000, 16, 64, 90005, 12.0, 1.5),
        (160_000, 48, 24, 90006, 8.0, 0.9),
        (45_000, 12, 96, 90007, 10.0, 1.2),
        (250_000, 32, 32, 90008, 9.0, 1.0),
        (350_000, 64, 24, 90009, 8.0, 1.0),
        (90_000, 24, 128, 90010, 10.0, 1.2),
    ]
    batch_grid = [512, 1024, 2048, 4096, 8192, 16384]
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
        for N_tot, K, D, seed, aniso, dir_a in configs:
            if cov_type == "full" and D > 96:
                continue
            X_tr, X_va = make_calibration_gmm_data(
                N_tot, K, D, seed=seed, anisotropy=aniso, dirichlet_alpha=dir_a
            )
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
                    if cov_type == "full" and B > 8192:
                        continue
                    if N_tr > 20_000 and B < 1024:
                        continue
                    if N_tr > 120_000 and B < 2048:
                        continue
                    steps = T * math.ceil(N_tr / B)
                    if steps < 4 or steps > 120:
                        continue
                    eff_df = (
                        B / (K * math.sqrt(D))
                        if cov_type == "diag"
                        else B / (K * D)
                    )
                    for a0 in alpha_grid:
                        t1 = time.perf_counter()
                        m = EGMM(
                            n_components=K,
                            covariance_type=cov_type,
                            max_epochs=T,
                            batch_size=B,
                            alpha=a0,
                            polish_steps=0,  # calibrate pure stochastic EMA + revival without full-batch polish
                            init_usage_scale="auto",
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
                                "eff_df": eff_df,
                                "init_scale": m.hyperparams_.init_usage_scale,
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
                f"Done {cov_type:4s} N={N_tr:6d} K={K:2d} D={D:3d} | "
                f"ref_nll={ref_nll:.3f} ({ref_time:.2f}s)",
                flush=True,
            )

    return pd.DataFrame(rows)


def analyze_and_fit(df: pd.DataFrame) -> pd.DataFrame:
    fit_rows = []
    for cov_type in ["diag", "full"]:
        sub = df[df["cov_type"] == cov_type].copy()
        print(f"\n=== Scaling Analysis ({cov_type}) ===")
        print("Median Validation dNLL by (T, alpha_0):")
        piv = sub.pivot_table(index="T", columns="alpha_0", values="nll_diff", aggfunc="median")
        print(piv.to_string())

        # For each (N, K, D, T), find best NLL and fastest config within 0.03 NLL of best
        winners = []
        for (N, K, D, T), grp in sub.groupby(["N", "K", "D", "T"]):
            best_row = grp.loc[grp["val_nll"].idxmin()]
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
                    "eff_df_best": best_row["eff_df"],
                    "a0_best": best_row["alpha_0"],
                    "B_pareto": fast_best["B"],
                    "eff_df_pareto": fast_best["eff_df"],
                    "a0_pareto": fast_best["alpha_0"],
                    "nll_diff_best": best_row["nll_diff"],
                    "nll_diff_pareto": fast_best["nll_diff"],
                    "speedup_pareto": fast_best["speedup"],
                    "steps_pareto": fast_best["steps"],
                }
            )
        wdf = pd.DataFrame(winners)
        print(wdf.to_string(index=False))

        y_log = np.log(wdf["a0_best"].values)
        X_feat = np.column_stack(
            [
                np.ones(len(wdf)),
                np.log(wdf["steps_pareto"].values),
                np.log(wdf["eff_df_best"].values),
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
            f"+ {beta[2]:.3f}*log(eff_df) + {beta[3]:.3f}*log(K) | R^2={r2:.3f}"
        )

        # Softmin-weighted optimal alpha_0 across eff_df quartiles
        soft_rows = []
        for (N, K, D, T, B), grp in sub.groupby(["N", "K", "D", "T", "B"]):
            vals = grp["val_nll"].values
            a0s = grp["alpha_0"].values
            w = np.exp(-(vals - vals.min()) / 0.015)
            w /= w.sum()
            soft_rows.append(
                {
                    "eff_df": float(grp["eff_df"].iloc[0]),
                    "a0_soft": float(np.sum(w * a0s)),
                    "min_dnll": float(grp["nll_diff"].min()),
                }
            )
        sdf = pd.DataFrame(soft_rows)
        sdf["df_bin"] = pd.qcut(sdf["eff_df"], q=4)
        q_med = sdf.groupby("df_bin", observed=False)[["eff_df", "a0_soft", "min_dnll"]].median()
        print("Softmin-weighted optimal alpha_0 by eff_df quartile:")
        print(q_med.to_string())

        fit_rows.append(
            {
                "cov_type": cov_type,
                "intercept": float(beta[0]),
                "slope_log_steps": float(beta[1]),
                "slope_log_eff_df": float(beta[2]),
                "slope_log_K": float(beta[3]),
                "r2": float(r2),
                "a0_q1_low_df": float(q_med["a0_soft"].iloc[0]),
                "a0_q4_high_df": float(q_med["a0_soft"].iloc[-1]),
                "median_best_dnll": float(wdf["nll_diff_best"].median()),
                "median_pareto_dnll": float(wdf["nll_diff_pareto"].median()),
                "median_pareto_speedup": float(wdf["speedup_pareto"].median()),
            }
        )
    return pd.DataFrame(fit_rows)


if __name__ == "__main__":
    out_dir = Path("docs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = run_sweep()
    df.to_csv(out_dir / "scaling_sweep.csv", index=False)
    fit_df = analyze_and_fit(df)
    fit_df.to_csv(out_dir / "scaling_law_fit_summary.csv", index=False)

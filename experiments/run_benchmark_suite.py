"""Rigorous 4-part verification suite for e-GMM (leakage-free, 20+ datasets, Pareto, Wilcoxon/CD)."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

from egmm.baselines import (
    CappeMoulinesOnlineGMM,
    EMeansInitGMM,
    NaiveMinibatchGMM,
    TorchBatchGMM,
)
from egmm.estimator import EGMM
from egmm.revival import revive_dead_components
from experiments.datasets import DatasetSplit, load_benchmark_suite


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_model(
    model,
    ds: DatasetSplit,
    device: str,
) -> dict[str, float]:
    _sync(device)
    t0 = time.perf_counter()
    model.fit(ds.X_train)
    _sync(device)
    fit_time = time.perf_counter() - t0

    test_nll = float(-model.score(ds.X_test))
    val_nll = float(-model.score(ds.X_val))
    bic = float(model.bic(ds.X_test))
    ari = float("nan")
    nmi = float("nan")
    if ds.y_test is not None and len(np.unique(ds.y_test)) > 1:
        pred = model.predict(ds.X_test)
        ari = float(adjusted_rand_score(ds.y_test, pred))
        nmi = float(normalized_mutual_info_score(ds.y_test, pred))

    weights = getattr(model, "weights_", None)
    dead_cnt = int(np.sum(weights < (0.05 / ds.n_components))) if weights is not None else 0
    return {
        "fit_time": fit_time,
        "val_nll": val_nll,
        "test_nll": test_nll,
        "bic": bic,
        "ari": ari,
        "nmi": nmi,
        "dead_components": dead_cnt,
        "revivals": int(getattr(model, "revivals_count_", 0)),
        "n_iter": int(getattr(model, "n_iter_", 0)),
    }


def tune_online_em(ds: DatasetSplit, device: str, seed: int) -> tuple[float, int]:
    """Fair inner validation sweep for Cappé-Moulines Online EM on X_val only."""
    N = ds.X_train.shape[0]
    best_val = math.inf
    best_cfg = (0.75, min(N, 1024))
    epochs = 6 if N < 30_000 else 3
    bs_grid = [2048, 4096, 8192] if N >= 80_000 else ([1024, 2048, 4096] if N >= 20_000 else [512, 1024, 2048])
    for rho in [0.55, 0.70, 0.85]:
        for bs in bs_grid:
            if bs > N // 2 and N > 512:
                continue
            m = CappeMoulinesOnlineGMM(
                n_components=ds.n_components,
                covariance_type=ds.recommended_cov,
                max_epochs=epochs,
                batch_size=min(N, bs),
                rho=rho,
                device=device,
                random_state=seed,
            ).fit(ds.X_train)
            vnll = -m.score(ds.X_val)
            if np.isfinite(vnll) and vnll < best_val:
                best_val = vnll
                best_cfg = (rho, min(N, bs))
    return best_cfg


def run_head_to_head(datasets: list[DatasetSplit], seeds: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        _w = torch.randn(1024, 16, device="cuda")
        _ = EGMM(4, "full", batch_size=256, max_epochs=2, device="cuda").fit(_w)
        _M = torch.tensor([0.5, 0.5, 0.0, 0.0], device="cuda", dtype=torch.float64)
        _S = torch.randn(4, 16, device="cuda", dtype=torch.float64) * _M.unsqueeze(-1)
        _Q = torch.eye(16, device="cuda", dtype=torch.float64).unsqueeze(0).repeat(4, 1, 1) * _M.view(4, 1, 1)
        revive_dead_components(_M, _S, _Q, covariance_type="full")
        torch.cuda.synchronize()
    rows = []

    for ds in datasets:
        N, D = ds.X_train.shape
        K = ds.n_components
        cov = ds.recommended_cov

        # Tune OnlineGMM once on validation split for this dataset (seed=0)
        best_rho, best_bs = tune_online_em(ds, device=device, seed=0)

        for seed in seeds:
            models = {
                "EGMM": EGMM(
                    n_components=K,
                    covariance_type=cov,
                    device=device,
                    random_state=seed,
                ),
                "TorchBatchGMM": TorchBatchGMM(
                    n_components=K,
                    covariance_type=cov,
                    max_iter=50,
                    tol=1e-3,
                    device=device,
                    random_state=seed,
                ),
                "OnlineGMM_tuned": CappeMoulinesOnlineGMM(
                    n_components=K,
                    covariance_type=cov,
                    max_epochs=6 if N < 30_000 else 3,
                    batch_size=best_bs,
                    rho=best_rho,
                    device=device,
                    random_state=seed,
                ),
                "NaiveMinibatchGMM": NaiveMinibatchGMM(
                    n_components=K,
                    covariance_type=cov,
                    device=device,
                    random_state=seed,
                ),
                "EMeansInitGMM": EMeansInitGMM(
                    n_components=K,
                    covariance_type=cov,
                    kmeans_epochs=2,
                    em_iters=10,
                    batch_size=min(N, 8192),
                    device=device,
                    random_state=seed,
                ),
            }

            # Run sklearn CPU baseline (cap iters on >200k full datasets so CPU run completes cleanly)
            run_sk = not (N > 300_000 and cov == "full" and seed > 0)
            if run_sk:
                sk_iters = 15 if (N > 200_000 and cov == "full") else 50
                models["SklearnGMM"] = GaussianMixture(
                    n_components=K,
                    covariance_type=cov,
                    max_iter=sk_iters,
                    tol=1e-3,
                    reg_covar=1e-6,
                    random_state=seed,
                )

            for mname, model in models.items():
                m_dev = "cpu" if mname == "SklearnGMM" else device
                res = evaluate_model(model, ds, device=m_dev)
                res.update(
                    {
                        "dataset": ds.name,
                        "regime": ds.regime,
                        "N_train": N,
                        "D": D,
                        "K": K,
                        "cov_type": cov,
                        "seed": seed,
                        "method": mname,
                    }
                )
                rows.append(res)

        sub_ds = [r for r in rows if r["dataset"] == ds.name]
        df_ds = pd.DataFrame(sub_ds).groupby("method")[["test_nll", "fit_time"]].mean()
        eg = df_ds.loc["EGMM"]
        tb = df_ds.loc["TorchBatchGMM"]
        print(
            f"[{ds.name:20s} | N={N:6d} D={D:3d} K={K:2d} {cov:4s}] "
            f"EGMM: {eg['test_nll']:.4f} ({eg['fit_time']:.3f}s) vs "
            f"TorchBatch: {tb['test_nll']:.4f} ({tb['fit_time']:.3f}s)",
            flush=True,
        )
        pd.DataFrame(rows).to_csv("docs/experiments/benchmark_head_to_head.csv", index=False)

    return pd.DataFrame(rows)


def run_pareto_sweeps(datasets: list[DatasetSplit]) -> pd.DataFrame:
    """Measure Test NLL vs Wall-Clock Training Time across budgets on key datasets."""
    target_names = {
        "covtype_581k",
        "miniboone",
        "mnist784_pca48",
        "syn_anisotropic_30k",
        "syn_imbalanced_40k",
        "syn_scale200k",
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for ds in datasets:
        if ds.name not in target_names:
            continue
        N, D = ds.X_train.shape
        K = ds.n_components
        cov = ds.recommended_cov
        best_rho, best_bs = tune_online_em(ds, device=device, seed=0)

        for ep in [1, 2, 3, 5, 8, 12]:
            for mname, model in [
                (
                    "EGMM",
                    EGMM(
                        n_components=K,
                        covariance_type=cov,
                        max_epochs=ep,
                        tol=1e-7,
                        device=device,
                        random_state=0,
                    ),
                ),
                (
                    "OnlineGMM_tuned",
                    CappeMoulinesOnlineGMM(
                        n_components=K,
                        covariance_type=cov,
                        max_epochs=ep,
                        batch_size=best_bs,
                        rho=best_rho,
                        tol=1e-7,
                        device=device,
                        random_state=0,
                    ),
                ),
            ]:
                res = evaluate_model(model, ds, device=device)
                res.update({"dataset": ds.name, "budget": ep, "method": mname})
                rows.append(res)

        for it in [2, 5, 10, 20, 40, 80]:
            tb = TorchBatchGMM(
                n_components=K,
                covariance_type=cov,
                max_iter=it,
                tol=1e-7,
                device=device,
                random_state=0,
            )
            res = evaluate_model(tb, ds, device=device)
            res.update({"dataset": ds.name, "budget": it, "method": "TorchBatchGMM"})
            rows.append(res)

            if it <= 10 or (it <= 40 and N < 50_000) or N < 25_000:
                sk = GaussianMixture(
                    n_components=K,
                    covariance_type=cov,
                    max_iter=it,
                    tol=1e-7,
                    random_state=0,
                )
                res_sk = evaluate_model(sk, ds, device="cpu")
                res_sk.update({"dataset": ds.name, "budget": it, "method": "SklearnGMM"})
                rows.append(res_sk)

        pd.DataFrame(rows).to_csv("docs/experiments/benchmark_pareto.csv", index=False)
        print(f"Pareto done: {ds.name}", flush=True)

    return pd.DataFrame(rows)


def compute_statistical_tests(df: pd.DataFrame) -> str:
    """Wilcoxon signed-rank tests + Friedman/Nemenyi mean ranks across datasets."""
    agg = df.groupby(["dataset", "method"])[["test_nll", "fit_time"]].mean().reset_index()
    piv_nll = agg.pivot(index="dataset", columns="method", values="test_nll")
    piv_time = agg.pivot(index="dataset", columns="method", values="fit_time")

    lines = ["=== Statistical Significance Summary ==="]
    common_methods = [
        m
        for m in ["EGMM", "TorchBatchGMM", "OnlineGMM_tuned", "NaiveMinibatchGMM", "EMeansInitGMM", "SklearnGMM"]
        if m in piv_nll.columns
    ]

    # Pairwise Wilcoxon vs EGMM on Test NLL & Fit Time
    for base in common_methods:
        if base == "EGMM":
            continue
        pair = piv_nll[["EGMM", base]].dropna()
        pair_t = piv_time.loc[pair.index, ["EGMM", base]].dropna()
        diff_nll = pair["EGMM"] - pair[base]  # negative means EGMM has lower (better) NLL
        wins = int((diff_nll < -1e-3).sum())
        ties = int((diff_nll.abs() <= 1e-3).sum())
        losses = int((diff_nll > 1e-3).sum())

        stat_nll, p_nll = stats.wilcoxon(pair["EGMM"], pair[base], zero_method="zsplit")
        stat_t, p_t = stats.wilcoxon(pair_t["EGMM"], pair_t[base], zero_method="zsplit")
        med_speedup = float((pair_t[base] / pair_t["EGMM"]).median())
        lines.append(
            f"EGMM vs {base:18s} (n={len(pair):2d}): "
            f"NLL W/T/L={wins}/{ties}/{losses}, med_dNLL={diff_nll.median():+.4f}, "
            f"Wilcoxon p_nll={p_nll:.4g} | med_speedup={med_speedup:.2f}x, p_time={p_t:.4g}"
        )

    # Friedman & Mean Ranks on complete datasets
    comp = piv_nll[common_methods].dropna()
    ranks_nll = comp.rank(axis=1, ascending=True)
    mean_ranks = ranks_nll.mean(axis=0).sort_values()
    if len(comp) >= 3 and len(common_methods) >= 3:
        f_stat, f_p = stats.friedmanchisquare(*[comp[c].values for c in common_methods])
        k_m = len(common_methods)
        n_d = len(comp)
        # Nemenyi q_alpha (alpha=0.05) approximation via studentized range / sqrt(2)
        q_table = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}
        q_alpha = q_table.get(k_m, 2.85)
        cd = q_alpha * math.sqrt(k_m * (k_m + 1) / (6.0 * n_d))
        lines.append(f"\nFriedman test (n={n_d}, k={k_m}): chi2={f_stat:.3f}, p={f_p:.4g}, Nemenyi CD(0.05)={cd:.3f}")
    lines.append("Mean NLL Ranks (1=best):")
    for m, r in mean_ranks.items():
        lines.append(f"  {m:18s}: {r:.3f}")

    return "\n".join(lines)


if __name__ == "__main__":
    out_dir = Path("docs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_benchmark_suite()
    print(f"Loaded {len(datasets)} benchmark datasets.")

    df_h2h = run_head_to_head(datasets, seeds=(0, 1, 2))
    df_h2h.to_csv(out_dir / "benchmark_head_to_head.csv", index=False)

    summary_txt = compute_statistical_tests(df_h2h)
    print("\n" + summary_txt)
    (out_dir / "stats_summary.txt").write_text(summary_txt, encoding="utf-8")

    df_pareto = run_pareto_sweeps(datasets)
    df_pareto.to_csv(out_dir / "benchmark_pareto.csv", index=False)

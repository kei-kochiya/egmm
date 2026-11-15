"""e-GMM benchmark runner (head-to-head, ablations, phase profiling, Pareto). Statistics: analyze_results.py."""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

from egmm.baselines import (
    CappeMoulinesOnlineGMM,
    EMeansInitGMM,
    NaiveMinibatchGMM,
    StandardizedSklearnGMM,
    TorchBatchGMM,
    TorchBatchGMMFullInit,
)
from egmm.estimator import EGMM
from egmm.revival import revive_dead_components
from egmm.scaling import auto_hyperparameters
from experiments.datasets import DatasetSplit, load_benchmark_suite


class BracketLawEGMM(EGMM):
    """Ablation: legacy piecewise sample-size brackets for the epoch and closing-polish budgets."""

    _SCHEDULE_LAW = "bracket"


class EagerEGMM(EGMM):
    """Systems ablation: identical algorithm, eager PyTorch execution (no CUDA graph).

    Like TorchBatchGMM, the eager loop performs one host synchronization per step, so
    EGMM_eager vs TorchBatchGMM compares the two algorithms at the same engineering level and
    EGMM vs EGMM_eager isolates the CUDA-graph effect.
    """

    _USE_CUDA_GRAPH = False


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


_WARM_BUF: torch.Tensor | None = None


def gpu_clock_warmup(ms: float = 60.0) -> None:
    """Run ~60 ms of dense GEMMs so the GPU leaves its idle clock state before a timed fit.

    Applied identically before every GPU method's timed fit: long CPU-only phases (e.g. scikit-learn
    fits) otherwise let the GPU down-clock, inflating whichever GPU method happens to run next.
    """
    global _WARM_BUF
    if not torch.cuda.is_available():
        return
    if _WARM_BUF is None:
        _WARM_BUF = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < ms:
        _ = _WARM_BUF @ _WARM_BUF
        torch.cuda.synchronize()


GPU_TIMING_REPEATS = 5  # timed refits per (dataset, method, seed) for PyTorch methods; median is reported


def evaluate_model(
    model,
    ds: DatasetSplit,
    device: str,
    repeats: int = 1,
) -> dict[str, float]:
    """Fit ``repeats`` times with identical settings (results are deterministic given the seed) and report
    the median fit time; quality metrics come from the final fit."""
    times = []
    for _ in range(max(1, repeats)):
        gc.collect()
        if device == "cuda":
            gpu_clock_warmup()
        _sync(device)
        t0 = time.perf_counter()
        model.fit(ds.X_train)
        _sync(device)
        times.append(time.perf_counter() - t0)
    fit_time = float(np.median(times))

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
        "fit_time_min": float(np.min(times)),
        "fit_time_max": float(np.max(times)),
        "timing_repeats": len(times),
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
    if N >= 80_000:
        bs_grid = [2048, 4096, 8192]
    elif N >= 20_000:
        bs_grid = [1024, 2048, 4096]
    elif N >= 4096:
        bs_grid = [512, 1024, 2048]
    else:
        # Small N: scale the grid to N so every dataset gets 3 distinct batch sizes (9 configurations).
        bs_grid = sorted({max(16, N // 8), max(16, N // 4), max(16, N // 2)})
    for rho in [0.55, 0.70, 0.85]:
        for bs in bs_grid:
            if bs > max(16, N // 2):
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


def run_head_to_head(
    datasets: list[DatasetSplit],
    seeds: tuple[int, ...] = (0, 1, 2),
    out_csv: str = "docs/experiments/benchmark_head_to_head.csv",
) -> pd.DataFrame:
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
                "EGMM_polish0": EGMM(
                    n_components=K,
                    covariance_type=cov,
                    polish_steps=0,
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
                "TorchBatchGMM_fullinit": TorchBatchGMMFullInit(
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

            # CPU scikit-learn references: identical max_iter=50 / tol=1e-3 budget on every dataset and seed.
            sk_iters = 50
            models["StandardizedSklearnGMM"] = StandardizedSklearnGMM(
                n_components=K,
                covariance_type=cov,
                max_iter=sk_iters,
                tol=1e-3,
                reg_covar=1e-6,
                random_state=seed,
            )
            models["SklearnGMM"] = GaussianMixture(
                n_components=K,
                covariance_type=cov,
                max_iter=sk_iters,
                tol=1e-3,
                reg_covar=1e-6,
                random_state=seed,
            )

            # Randomized (but reproducible) method order per (dataset, seed): no method always follows
            # the long CPU scikit-learn fits.
            order = list(models.keys())
            np.random.default_rng(10_007 * seed + sum(map(ord, ds.name))).shuffle(order)
            for mname in order:
                model = models[mname]
                is_cpu = mname in ("SklearnGMM", "StandardizedSklearnGMM")
                m_dev = "cpu" if is_cpu else device
                res = evaluate_model(model, ds, device=m_dev, repeats=1 if is_cpu else GPU_TIMING_REPEATS)
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
        pd.DataFrame(rows).to_csv(out_csv, index=False)

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
                res = evaluate_model(model, ds, device=device, repeats=3)
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
            res = evaluate_model(tb, ds, device=device, repeats=3)
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


def run_ablation_suite(datasets: list[DatasetSplit], seeds: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    """Component & fairness ablations across all benchmark datasets."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for ds in datasets:
        N, D = ds.X_train.shape
        K = ds.n_components
        cov = ds.recommended_cov
        best_rho, best_bs = tune_online_em(ds, device=device, seed=0)
        egmm_hp = auto_hyperparameters(N, K, D, cov)

        for seed in seeds:
            abl_models = {
                "EGMM": EGMM(
                    n_components=K,
                    covariance_type=cov,
                    device=device,
                    random_state=seed,
                ),
                "EGMM_polish0": EGMM(
                    n_components=K,
                    covariance_type=cov,
                    polish_steps=0,
                    device=device,
                    random_state=seed,
                ),
                "EGMM_no_init_scale": EGMM(
                    n_components=K,
                    covariance_type=cov,
                    init_usage_scale=1.0,
                    device=device,
                    random_state=seed,
                ),
                "EGMM_no_revival": EGMM(
                    n_components=K,
                    covariance_type=cov,
                    revive_dead=False,
                    device=device,
                    random_state=seed,
                ),
                "EGMM_bracket_law": BracketLawEGMM(
                    n_components=K,
                    covariance_type=cov,
                    device=device,
                    random_state=seed,
                ),
                # Exactly EGMM's resolved closing-polish budget and initial moment scale.
                "OnlineGMM_matched_polish": CappeMoulinesOnlineGMM(
                    n_components=K,
                    covariance_type=cov,
                    max_epochs=6 if N < 30_000 else 3,
                    batch_size=best_bs,
                    rho=best_rho,
                    polish_steps=egmm_hp.polish_steps,
                    init_usage_scale=egmm_hp.init_usage_scale,
                    device=device,
                    random_state=seed,
                ),
            }
            order = list(abl_models.keys())
            np.random.default_rng(20_011 * seed + sum(map(ord, ds.name))).shuffle(order)
            for mname in order:
                model = abl_models[mname]
                res = evaluate_model(model, ds, device=device, repeats=GPU_TIMING_REPEATS)
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
        pd.DataFrame(rows).to_csv("docs/experiments/benchmark_ablations.csv", index=False)
    return pd.DataFrame(rows)


def run_systems_ablation(datasets: list[DatasetSplit], seeds: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    """Separate algorithmic from implementation effects in one harness pass (GPU methods only)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ds in datasets:
        N, D = ds.X_train.shape
        K = ds.n_components
        cov = ds.recommended_cov
        for seed in seeds:
            models = {
                "EGMM": EGMM(n_components=K, covariance_type=cov, device=device, random_state=seed),
                "EGMM_eager": EagerEGMM(n_components=K, covariance_type=cov, device=device, random_state=seed),
                "TorchBatchGMM": TorchBatchGMM(
                    n_components=K, covariance_type=cov, max_iter=50, tol=1e-3, device=device, random_state=seed
                ),
            }
            order = list(models.keys())
            np.random.default_rng(30_011 * seed + sum(map(ord, ds.name))).shuffle(order)
            for mname in order:
                res = evaluate_model(models[mname], ds, device=device, repeats=GPU_TIMING_REPEATS)
                res.update(
                    {"dataset": ds.name, "regime": ds.regime, "N_train": N, "D": D, "K": K, "cov_type": cov,
                     "seed": seed, "method": mname}
                )
                rows.append(res)
        pd.DataFrame(rows).to_csv("docs/experiments/benchmark_systems.csv", index=False)
    return pd.DataFrame(rows)


def run_phase_breakdown(datasets: list[DatasetSplit]) -> pd.DataFrame:
    """Profile cold-start vs warm-start and phase-by-phase runtime breakdown."""
    target_names = {
        "pendigits",
        "mnist784_pca48",
        "jannis",
        "miniboone",
        "creditcard",
        "covtype_581k",
        "syn_scale200k",
        "syn_scale500k_diag",
        "syn_scale400k_d256_diag",
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ds in datasets:
        if ds.name not in target_names:
            continue
        N, D = ds.X_train.shape
        K = ds.n_components
        cov = ds.recommended_cov
        m = EGMM(n_components=K, covariance_type=cov, device=device, random_state=0)
        m._profile_phases = True

        # Cold: first fit on this dataset's shapes (cuBLAS/cuSOLVER plan selection, caching-allocator
        # growth). Every fit captures a fresh CUDA graph, so "warm" differs only by those process caches.
        gpu_clock_warmup()
        m.fit(ds.X_train)
        tb_cold = dict(m.timing_breakdown_)
        tb_cold.update(
            {
                "dataset": ds.name,
                "N_train": N,
                "D": D,
                "K": K,
                "cov_type": cov,
                "start_mode": "cold",
                "polish_steps": m.hyperparams_.polish_steps,
            }
        )
        rows.append(tb_cold)

        # Warm: immediate refit of the same instance on the same data.
        gpu_clock_warmup()
        m.fit(ds.X_train)
        tb_warm = dict(m.timing_breakdown_)
        tb_warm.update(
            {
                "dataset": ds.name,
                "N_train": N,
                "D": D,
                "K": K,
                "cov_type": cov,
                "start_mode": "warm",
                "polish_steps": m.hyperparams_.polish_steps,
            }
        )
        rows.append(tb_warm)
    df_p = pd.DataFrame(rows)
    df_p.to_csv("docs/experiments/benchmark_phase_breakdown.csv", index=False)
    return df_p


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="e-GMM benchmark suite")
    ap.add_argument(
        "--stages",
        default="h2h,abl,phase,pareto,sys",
        help="comma-separated subset of h2h,abl,phase,pareto,sys (statistics: experiments/analyze_results.py)",
    )
    stages = set(ap.parse_args().stages.split(","))
    out_dir = Path("docs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_benchmark_suite()
    print(f"Loaded {len(datasets)} benchmark datasets.")

    if "h2h" in stages:
        df_h2h = run_head_to_head(datasets, seeds=(0, 1, 2))
        df_h2h.to_csv(out_dir / "benchmark_head_to_head.csv", index=False)
    if "abl" in stages:
        df_abl = run_ablation_suite(datasets, seeds=(0, 1, 2))
        df_abl.to_csv(out_dir / "benchmark_ablations.csv", index=False)
    if "phase" in stages:
        df_phase = run_phase_breakdown(datasets)
        df_phase.to_csv(out_dir / "benchmark_phase_breakdown.csv", index=False)
    if "sys" in stages:
        df_sys = run_systems_ablation(datasets, seeds=(0, 1, 2))
        df_sys.to_csv(out_dir / "benchmark_systems.csv", index=False)
    if "pareto" in stages:
        df_pareto = run_pareto_sweeps(datasets)
        df_pareto.to_csv(out_dir / "benchmark_pareto.csv", index=False)

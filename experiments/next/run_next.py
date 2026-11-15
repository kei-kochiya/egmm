"""Experiments for the next version of the paper (stronger baselines, large N, restarts at equal compute).

    python experiments/next/run_next.py --stages dev,holdout,diag,large   # GPU; ~3-3.5 h in total
    python experiments/next/run_next.py --smoke --device cpu              # CPU-only smoke test (minutes)

Stages (every stage interleaves all its methods in one session, in a shuffled order per (dataset, seed)):

* ``dev``: the 33 development datasets. e-GMM as four independent single-start fits per seed
  (``EGMM_r0`` = the standard fit; r = 1..3 use ``random_state = seed + 1000 r``) so that best-of-R
  restarts can be evaluated offline with either selection rule (training log-likelihood or validation
  NLL), plus the library's own ``n_init=3`` (``EGMM_n3``) to measure the real cost of restarts, and the
  batch-EM family ``TorchBatchGMM``, ``TorchBatchGMM_f32chol``, ``TorchBatchGMM_fast``, ``TorchBatchGMM_graph``.
* ``holdout``: post-hoc timing of e-GMM (graph and eager) against the four batch-EM variants on the
  16 pre-registered holdout datasets. No restarts here: the old holdout is not used to develop the
  restart idea (a new pre-registered holdout is needed for that claim).
* ``large``: synthetic mixtures with 1M, 3M and 10M training rows (and real large datasets if their files
  are in ``data/large/``): e-GMM (r0..r2 and n3), the fast and graph batch EM, ``TorchBatchGMM`` and, at
  N <= 1M only, standardized scikit-learn (seed 0, one timing).

All results go to ``docs/experiments/next/*.csv`` (rewritten after each dataset); analysis:
``experiments/next/analyze_next.py``. The library (``src/egmm``) is not modified.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from egmm.baselines import CappeMoulinesOnlineGMM, StandardizedSklearnGMM, TorchBatchGMM
from egmm.estimator import EGMM
from egmm.scaling import auto_hyperparameters
from experiments.datasets import DatasetSplit, load_benchmark_suite
from experiments.next.baselines_fast import TorchBatchGMMFast, TorchBatchGMMGraph
from experiments.next.variants import ConstantStepEGMM, EagerEGMMHostAlpha, RobbinsMonroEGMM
from experiments.run_benchmark_suite import EagerEGMM, TorchBatchGMMF32Chol, evaluate_model, tune_online_em

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "experiments" / "next"
SEEDS = (0, 1, 2)
N_SINGLE_RESTARTS = 4  # EGMM_r0..r3 on the development suite
REPEATS = 5


def batch_family(K, cov, device, seed):
    common = dict(n_components=K, covariance_type=cov, max_iter=50, tol=1e-3, device=device, random_state=seed)
    return {
        "TorchBatchGMM": TorchBatchGMM(**common),
        "TorchBatchGMM_f32chol": TorchBatchGMMF32Chol(**common),
        "TorchBatchGMM_fast": TorchBatchGMMFast(**common),
        "TorchBatchGMM_graph": TorchBatchGMMGraph(**common),
    }


def egmm_family(K, cov, device, seed, n_single, with_n3=True, with_eager=False):
    models = {f"EGMM_r{r}": EGMM(n_components=K, covariance_type=cov, device=device, random_state=seed + 1000 * r)
              for r in range(n_single)}
    if with_n3:
        models["EGMM_n3"] = EGMM(n_components=K, covariance_type=cov, device=device, random_state=seed, n_init=3)
    if with_eager:
        models["EGMM_eager"] = EagerEGMM(n_components=K, covariance_type=cov, device=device, random_state=seed)
        models["EGMM_eager_hostalpha"] = EagerEGMMHostAlpha(n_components=K, covariance_type=cov, device=device, random_state=seed)
    return models


def schedule_family(K, cov, device, seed):
    """Review request: single-factor schedule ablations inside the full e-GMM pipeline (development suite)."""
    return {
        "EGMM_sched_rm": RobbinsMonroEGMM(n_components=K, covariance_type=cov, device=device, random_state=seed),
        "EGMM_sched_const": ConstantStepEGMM(n_components=K, covariance_type=cov, device=device, random_state=seed),
    }


def online_family(ds, device, seed, tuned_cache):
    """Review request (holdout, post hoc): tuned online EM, online EM with e-GMM's exact polish budget and
    s_init (as in the paper's ablation), and e-GMM without revival."""
    N, D = ds.X_train.shape
    K, cov = ds.n_components, ds.recommended_cov
    if ds.name not in tuned_cache:
        tuned_cache[ds.name] = tune_online_em(ds, device=device, seed=0)
    rho, bs = tuned_cache[ds.name]
    hp = auto_hyperparameters(N, K, D, cov)
    ep = 6 if N < 30_000 else 3
    return {
        "OnlineGMM_tuned": CappeMoulinesOnlineGMM(n_components=K, covariance_type=cov, max_epochs=ep, batch_size=bs,
                                                  rho=rho, device=device, random_state=seed),
        "OnlineGMM_matched_polish": CappeMoulinesOnlineGMM(n_components=K, covariance_type=cov, max_epochs=ep, batch_size=bs,
                                                           rho=rho, polish_steps=hp.polish_steps,
                                                           init_usage_scale=hp.init_usage_scale, device=device,
                                                           random_state=seed),
        "EGMM_no_revival": EGMM(n_components=K, covariance_type=cov, revive_dead=False, device=device, random_state=seed),
    }


def diag_family(ds, device, seed):
    """Review request: which rule drives the high-dimensional diagonal losses? One factor at a time."""
    N, D = ds.X_train.shape
    K = ds.n_components
    hp = auto_hyperparameters(N, K, D, "diag")
    base = dict(n_components=K, covariance_type="diag", device=device, random_state=seed)
    return {
        "EGMM_r0": EGMM(**base),
        "EGMM_polish0": EGMM(**base, polish_steps=0),
        "EGMM_a085": EGMM(**base, alpha=0.85),                                   # sweep optimum; floor 0.085
        "EGMM_ep2x": EGMM(**base, max_epochs=2 * hp.max_epochs, polish_steps=hp.polish_steps),  # same polish budget
        "EGMM_notol": EGMM(**base, tol=0.0),                                     # polish runs all P steps
        "TorchBatchGMM": TorchBatchGMM(n_components=K, covariance_type="diag", max_iter=50, tol=1e-3, device=device, random_state=seed),
        "TorchBatchGMM_graph": TorchBatchGMMGraph(n_components=K, covariance_type="diag", max_iter=50, tol=1e-3, device=device, random_state=seed),
    }


def run_one(model, ds: DatasetSplit, device: str, repeats: int) -> dict:
    try:
        res = evaluate_model(model, ds, device=device, repeats=repeats)
        res["lower_bound"] = float(getattr(model, "lower_bound_", float("nan")))
        res["error"] = ""
    except Exception as exc:  # recorded, never dropped silently
        res = {"fit_time": float("nan"), "test_nll": float("nan"), "val_nll": float("nan"), "error": repr(exc)[:300]}
    return res


def warm_up(device: str) -> None:
    if device != "cuda":
        return
    w = torch.randn(4096, 16, device="cuda")
    for cov in ("full", "diag"):
        for m in list(egmm_family(4, cov, device, 0, 1, with_eager=True).values()) + list(batch_family(4, cov, device, 0).values()):
            m.fit(w)
    torch.cuda.synchronize()


def run_stage(name: str, datasets, make_models, device: str, repeats: int, seeds=SEEDS, extra=None) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        N, D = ds.X_train.shape
        for seed in seeds:
            models = make_models(ds, seed)
            if extra is not None:
                models.update(extra(ds, seed))
            order = list(models)
            np.random.default_rng(40_009 * seed + sum(map(ord, ds.name))).shuffle(order)
            for m in order:
                rep = 1 if m.startswith("StdSklearn") else repeats
                res = run_one(models[m], ds, "cpu" if m.startswith("StdSklearn") else device, rep)
                res.update(dict(dataset=ds.name, regime=ds.regime, N_train=N, D=D, K=ds.n_components,
                                cov_type=ds.recommended_cov, seed=seed, method=m, stage=name))
                rows.append(res)
                del models[m]
                gc.collect()
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
        print(f"[{name}] done {ds.name} ({N} x {D})", flush=True)
        del ds
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def stage_dev(device, repeats, smoke=False):
    dss = load_benchmark_suite() if not smoke else smoke_datasets()
    mk = lambda ds, seed: {**egmm_family(ds.n_components, ds.recommended_cov, device, seed, N_SINGLE_RESTARTS, with_eager=True),
                           **schedule_family(ds.n_components, ds.recommended_cov, device, seed),
                           **batch_family(ds.n_components, ds.recommended_cov, device, seed)}
    return run_stage("dev" if not smoke else "smoke_dev", dss, mk, device, repeats, seeds=SEEDS if not smoke else (0,))


def stage_holdout(device, repeats, smoke=False):
    if smoke:
        dss = smoke_datasets()
    else:
        from experiments.holdout_suite import load_holdout_suite
        dss = load_holdout_suite()
    cache: dict = {}
    mk = lambda ds, seed: {**egmm_family(ds.n_components, ds.recommended_cov, device, seed, 1, with_n3=False, with_eager=True),
                           **batch_family(ds.n_components, ds.recommended_cov, device, seed),
                           **online_family(ds, device, seed, cache)}
    return run_stage("holdout" if not smoke else "smoke_holdout", dss, mk, device, repeats, seeds=SEEDS if not smoke else (0,))


def stage_diag(device, repeats, smoke=False):
    """Diagonal-covariance diagnostics: the 8 diagonal development datasets plus the two diagonal holdout
    datasets (volkert, isolet; post hoc), so a fix is not tuned on the holdout alone."""
    if smoke:
        dss = [d for d in smoke_datasets() if d.recommended_cov == "diag"]
    else:
        from experiments.holdout_suite import load_holdout_suite
        dss = [d for d in load_benchmark_suite() if d.recommended_cov == "diag"]
        dss += [d for d in load_holdout_suite() if d.name in ("volkert", "isolet")]
    return run_stage("diag" if not smoke else "smoke_diag", dss, lambda ds, seed: diag_family(ds, device, seed),
                     device, repeats, seeds=SEEDS if not smoke else (0,))


def stage_large(device, repeats, smoke=False, sizes=("1m", "3m", "10m")):
    from experiments.next.large_data import iter_large_suite, make_large_synthetic
    if smoke:
        dss = [make_large_synthetic("smoke_large_full", 20_000, 8, 4, "full", 12.0, 1),
               make_large_synthetic("smoke_large_diag", 20_000, 16, 4, "diag", 10.0, 2)]
    else:
        dss = iter_large_suite(include_real=True, sizes=sizes)

    def mk(ds, seed):
        fam = batch_family(ds.n_components, ds.recommended_cov, device, seed)
        models = {**egmm_family(ds.n_components, ds.recommended_cov, device, seed, 3), **fam}
        if ds.X_train.shape[0] <= 1_000_000 and seed == 0:
            models["StdSklearnGMM"] = StandardizedSklearnGMM(n_components=ds.n_components, covariance_type=ds.recommended_cov,
                                                             max_iter=50, tol=1e-3, random_state=seed)
        return models

    return run_stage("large" if not smoke else "smoke_large", dss, mk, device, repeats, seeds=SEEDS if not smoke else (0,))


def smoke_datasets():
    from experiments.datasets import _make_hard_synthetic
    return [_make_hard_synthetic("smoke_full", "smoke", 6_000, 8, 6, 7, "full"),
            _make_hard_synthetic("smoke_diag", "smoke", 6_000, 16, 6, 8, "diag", diag_only=True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="dev,holdout,diag,large")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--large-sizes", default="1m,3m,10m")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic data, 1 seed, 1 repeat")
    args = ap.parse_args()
    repeats = 1 if args.smoke else args.repeats
    t0 = time.time()
    warm_up(args.device)
    stages = args.stages.split(",")
    if "dev" in stages:
        stage_dev(args.device, repeats, args.smoke)
    if "holdout" in stages:
        stage_holdout(args.device, repeats, args.smoke)
    if "diag" in stages:
        stage_diag(args.device, repeats, args.smoke)
    if "large" in stages:
        stage_large(args.device, 3 if not args.smoke else 1, args.smoke, tuple(args.large_sizes.split(",")))
    print(f"all stages done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

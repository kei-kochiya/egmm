"""Out-of-suite zero-shot small-dataset evaluation (16 unseen sklearn/OpenML datasets, N_total <= 5500)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.mixture import GaussianMixture

from egmm.baselines import CappeMoulinesOnlineGMM, TorchBatchGMM
from egmm.estimator import EGMM
from egmm.revival import revive_dead_components
from experiments.datasets import DatasetSplit, _load_openml_numeric, _split_and_maybe_pca
from experiments.run_benchmark_suite import evaluate_model, tune_online_em


def load_small_external_suite(data_home: str = "data/openml") -> list[DatasetSplit]:
    """Load 16 held-out small tabular datasets not present in the 23-dataset development suite."""
    Path(data_home).mkdir(parents=True, exist_ok=True)
    datasets: list[DatasetSplit] = []

    for name, loader, seed, K, cov in [
        ("iris", load_iris, 501, 3, "full"),
        ("wine", load_wine, 502, 3, "full"),
        ("breast_cancer", load_breast_cancer, 503, 4, "full"),
    ]:
        b = loader()
        X_tr, X_va, X_te, y_te = _split_and_maybe_pca(b.data, b.target, seed=seed)
        datasets.append(DatasetSplit(name, "ext_small", X_tr, X_va, X_te, K, cov, y_te))

    openml_small = [
        ("sonar", 40, 4, "diag"),
        ("glass", 41, 6, "diag"),
        ("ecoli", 39, 8, "diag"),
        ("ionosphere", 59, 6, "diag"),
        ("diabetes", 37, 6, "full"),
        ("vehicle", 54, 8, "full"),
        ("vowel", 307, 11, "full"),
        ("banknote_auth", 1462, 8, "full"),
        ("yeast", 181, 10, "diag"),
        ("steel_plates", 1504, 7, "full"),
        ("phoneme", 1489, 8, "full"),
        ("wall_robot", 1497, 8, "full"),
        ("page_blocks", 30, 8, "full"),
    ]
    for idx, (name, did, K, cov) in enumerate(openml_small):
        try:
            ds = _load_openml_numeric(
                name=name,
                data_id=did,
                data_home=data_home,
                n_components=K,
                regime="ext_small",
                recommended_cov=cov,
                seed=600 + idx,
            )
            datasets.append(ds)
        except Exception as exc:
            print(f"[WARN] Failed loading OpenML {name} (id={did}): {exc}")

    return datasets


def main() -> None:
    datasets = load_small_external_suite()
    print(f"Loaded {len(datasets)} external unseen small datasets.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        _w = torch.randn(1024, 16, device="cuda")
        _ = EGMM(4, "full", batch_size=256, max_epochs=2, device="cuda").fit(_w)
        _M = torch.tensor([0.5, 0.5, 0.0, 0.0], device="cuda", dtype=torch.float64)
        _S = torch.randn(4, 16, device="cuda", dtype=torch.float64) * _M.unsqueeze(-1)
        _Q = torch.eye(16, device="cuda", dtype=torch.float64).unsqueeze(0).repeat(4, 1, 1) * _M.view(4, 1, 1)
        revive_dead_components(_M, _S, _Q, covariance_type="full")
        torch.cuda.synchronize()

    seeds = (0, 1, 2)
    rows = []
    for ds in datasets:
        N_tr, D = ds.X_train.shape
        K, cov = ds.n_components, ds.recommended_cov
        best_rho, best_bs = tune_online_em(ds, device=device, seed=0)

        for seed in seeds:
            models = {
                "EGMM_cuda": EGMM(K, cov, device=device, random_state=seed),
                "EGMM_auto": EGMM(K, cov, device="auto", random_state=seed),
                "TorchBatchGMM": TorchBatchGMM(K, cov, max_iter=50, tol=1e-3, device=device, random_state=seed),
                "OnlineGMM_tuned": CappeMoulinesOnlineGMM(
                    K, cov, max_epochs=6, batch_size=min(N_tr, best_bs), rho=best_rho, device=device, random_state=seed
                ),
                "SklearnGMM": GaussianMixture(K, covariance_type=cov, max_iter=100, tol=1e-3, random_state=seed),
            }
            for mname, model in models.items():
                eval_dev = "cpu" if mname == "SklearnGMM" else (model._resolve_device(N_tr, D).type if mname == "EGMM_auto" else device)
                res = evaluate_model(model, ds, device=eval_dev, repeats=1 if mname == "SklearnGMM" else 5)
                res.update(
                    {
                        "dataset": ds.name,
                        "n_train": N_tr,
                        "n_features": D,
                        "n_components": K,
                        "cov_type": cov,
                        "seed": seed,
                        "method": mname,
                    }
                )
                rows.append(res)

        sub = pd.DataFrame([r for r in rows if r["dataset"] == ds.name])
        m_agg = sub.groupby("method")[["test_nll", "fit_time"]].mean()
        print(
            f"[{ds.name:15s} | N={N_tr:4d} D={D:2d} K={K:2d} {cov:4s}] "
            f"EGMM_cuda={m_agg.loc['EGMM_cuda','test_nll']:7.3f} ({m_agg.loc['EGMM_cuda','fit_time']*1e3:5.1f}ms) | "
            f"EGMM_auto={m_agg.loc['EGMM_auto','test_nll']:7.3f} ({m_agg.loc['EGMM_auto','fit_time']*1e3:5.1f}ms) | "
            f"TB={m_agg.loc['TorchBatchGMM','test_nll']:7.3f} ({m_agg.loc['TorchBatchGMM','fit_time']*1e3:5.1f}ms) | "
            f"SK={m_agg.loc['SklearnGMM','test_nll']:7.3f} ({m_agg.loc['SklearnGMM','fit_time']*1e3:5.1f}ms)",
            flush=True,
        )

    out_dir = Path("docs/experiments")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "small_dataset_eval.csv", index=False)

    agg = df.groupby(["dataset", "method"])[["test_nll", "fit_time"]].mean().reset_index()
    pn = agg.pivot(index="dataset", columns="method", values="test_nll")
    pt = agg.pivot(index="dataset", columns="method", values="fit_time")

    lines = [f"=== External Small-Dataset Generalization Summary (n={len(datasets)}) ==="]
    for eg_col in ["EGMM_cuda", "EGMM_auto"]:
        for base in ["TorchBatchGMM", "OnlineGMM_tuned", "SklearnGMM"]:
            pair = pn[[eg_col, base]].dropna()
            pair_t = pt.loc[pair.index, [eg_col, base]]
            diff = pair[eg_col] - pair[base]
            wins = int((diff < -1e-3).sum())
            ties = int((diff.abs() <= 1e-3).sum())
            losses = int((diff > 1e-3).sum())
            _, p_n = stats.wilcoxon(pair[eg_col], pair[base], zero_method="zsplit")
            _, p_t = stats.wilcoxon(pair_t[eg_col], pair_t[base], zero_method="zsplit")
            spd = float((pair_t[base] / pair_t[eg_col]).median())
            lines.append(
                f"{eg_col:10s} vs {base:16s}: NLL W/T/L={wins}/{ties}/{losses}, "
                f"med_dNLL={diff.median():+.4f}, p_nll={p_n:.4g} | med_spd={spd:.2f}x, p_time={p_t:.4g}"
            )
    txt = "\n".join(lines)
    print("\n" + txt)
    (out_dir / "small_dataset_stats.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Post-hoc sensitivity reference: batch EM run to convergence with full-data seeding.

Requested in review: the headline batch-EM baselines stop at max_iter=50 (TorchBatchGMM hits the cap on
about half of the holdout runs). This script fits TorchBatchGMM with full-data greedy k-means++ seeding
and max_iter=500 (tol=1e-3) on the pre-registered holdout suite, 3 seeds, as a *quality* reference
(timed once, no repeats). It is a post-hoc analysis and is labeled as such in the paper; it does not
change any pre-registered comparison.

    python experiments/converged_reference.py   # -> docs/experiments/holdout_converged_em.csv
"""

from __future__ import annotations

import pandas as pd
import torch

from egmm.baselines import TorchBatchGMMFullInit
from experiments.holdout_suite import load_holdout_suite
from experiments.run_benchmark_suite import evaluate_model


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ds in load_holdout_suite():
        N, D = ds.X_train.shape
        for seed in (0, 1, 2):
            m = TorchBatchGMMFullInit(
                n_components=ds.n_components,
                covariance_type=ds.recommended_cov,
                max_iter=500,
                tol=1e-3,
                device=device,
                random_state=seed,
            )
            res = evaluate_model(m, ds, device=device, repeats=1)
            res.update({"dataset": ds.name, "N_train": N, "D": D, "K": ds.n_components,
                        "cov_type": ds.recommended_cov, "seed": seed, "method": "BatchEM_converged"})
            rows.append(res)
            print(f"{ds.name:16s} seed={seed} nll={res['test_nll']:.4f} iters={res['n_iter']} t={res['fit_time']:.2f}s", flush=True)
        pd.DataFrame(rows).to_csv("docs/experiments/holdout_converged_em.csv", index=False)


if __name__ == "__main__":
    main()

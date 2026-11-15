"""Design studies behind the Round-3 changes (outputs in docs/experiments/design_studies/).

Selection decisions used the VALIDATION split only; test NLL is recorded for reference. Run from the
repository root with ``PYTHONPATH=src;.`` (Windows) or ``PYTHONPATH=src:.``:

    python experiments/design_studies/run_design_studies.py closing     # closing refinement (rejected)
    python experiments/design_studies/run_design_studies.py seeding     # greedy k-means++ seeding (adopted)
    python experiments/design_studies/run_design_studies.py law         # smooth vs bracket schedule law (adopted)
    python experiments/design_studies/run_design_studies.py precision   # float32 vs float64 training

``closing`` and ``seeding`` reproduce the studies as they were run: legacy seeding (k-means++ on
max(2048, 32K) points, 5 Lloyd steps) and the legacy bracket schedule law for the ``base`` arm.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from egmm.estimator import EGMM  # noqa: E402
from egmm.kernels import e_step_batch, init_kmeans_pp  # noqa: E402
from experiments.datasets import load_benchmark_suite  # noqa: E402
from experiments.run_benchmark_suite import gpu_clock_warmup  # noqa: E402

OUT = ROOT / "docs" / "experiments" / "design_studies"
LARGE = [
    "covtype_581k",
    "creditcard",
    "skin_nonskin",
    "syn_scale200k",
    "syn_scale300k_k64_full",
    "syn_scale400k_d256_diag",
    "syn_scale500k_diag",
]


class StudyEGMM(EGMM):
    """EGMM with explicit seeding knobs and an optional closing refinement on a random subsample."""

    def __init__(self, *args, seed_mult=32, lloyd=5, greedy=False, refine_steps=0, refine_sub=65536,
                 law="bracket", **kw):
        super().__init__(*args, **kw)
        self.seed_mult = seed_mult
        self.lloyd = lloyd
        self.greedy = greedy
        self.refine_steps = refine_steps
        self.refine_sub = refine_sub
        self.law = law

    @property
    def _SCHEDULE_LAW(self):  # type: ignore[override]
        return self.law

    @property
    def _SEED_POINTS_PER_COMPONENT(self):  # type: ignore[override]
        return self.seed_mult

    @property
    def _SEED_LLOYD_ITERS(self):  # type: ignore[override]
        return self.lloyd

    def _initialize_moments(self, X_std, generator, out_dtype=None):
        if self.greedy:
            return super()._initialize_moments(X_std, generator, out_dtype)
        # legacy classical k-means++ draw (n_local_trials=1); everything else is shared with EGMM
        import egmm.estimator as est_mod

        orig = est_mod.init_kmeans_pp
        est_mod.init_kmeans_pp = lambda X, K, generator=None, n_local_trials=None: orig(X, K, generator, 1)
        try:
            return super()._initialize_moments(X_std, generator, out_dtype)
        finally:
            est_mod.init_kmeans_pp = orig

    def _prepare_fit_data(self, X):
        out = super()._prepare_fit_data(X)
        self._X_std_cache = out[0]
        return out

    def fit(self, X, y=None):
        super().fit(X, y)
        X_std, self._X_std_cache = self._X_std_cache, None
        if self.refine_steps <= 0:
            return self
        N = X_std.shape[0]
        gen = self._make_generator(X_std.device, seed_offset=777)
        if self.refine_sub and self.refine_sub < N:
            X_r = X_std.index_select(0, torch.randperm(N, generator=gen, device=X_std.device)[: self.refine_sub])
        else:
            X_r = X_std
        M, S, Q = self._M_t, self._S_t, self._Q_t
        w, mu, _, pf, ldh = self._moments_to_estep_params(M, S, Q, estep_dtype=X_std.dtype, allow_sync=False,
                                                          fast_cholesky_f32=True)
        n = X_r.shape[0]
        for _ in range(self.refine_steps):
            dM, dS, dQ = torch.zeros_like(M), torch.zeros_like(S), torch.zeros_like(Q)
            for st in range(0, n, 32768):
                en = min(n, st + 32768)
                _, _, m_b, s_b, q_b = e_step_batch(X_r[st:en], w, mu, covariance_type=self.covariance_type,
                                                   prec_factor=pf, log_det_half=ldh, stats_dtype=torch.float64)
                f = (en - st) / n
                dM, dS, dQ = dM + m_b * f, dS + s_b * f, dQ + q_b * f
            M, S, Q = dM, dS, dQ
            w, mu, _, pf, ldh = self._moments_to_estep_params(M, S, Q, estep_dtype=X_std.dtype, allow_sync=False,
                                                              fast_cholesky_f32=True)
        w, mu, covs, pf, ldh = self._moments_to_estep_params(M, S, Q, estep_dtype=X_std.dtype, allow_sync=True)
        self._M_t, self._S_t, self._Q_t = M, S, Q
        self._sync_public_attributes(w, mu, covs, pf, ldh)
        return self


def timed_fit(model, X, cuda):
    gc.collect()
    if cuda:
        gpu_clock_warmup()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.fit(X)
    if cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def run(variants, names=None, seeds=(0, 1, 2), device="cuda", out_name="study.csv"):
    dss = load_benchmark_suite()
    if names is not None:
        dss = [d for d in dss if d.name in set(names)]
    cuda = torch.cuda.is_available()
    rows = []
    for ds in dss:
        for seed in seeds:
            for vname, kw in variants.items():
                kw = dict(kw)
                cls = kw.pop("cls", StudyEGMM)
                m = cls(n_components=ds.n_components, covariance_type=ds.recommended_cov, device=device,
                        random_state=seed, **kw)
                t = timed_fit(m, ds.X_train, cuda)
                rows.append(dict(dataset=ds.name, N=ds.X_train.shape[0], seed=seed, variant=vname,
                                 val_nll=float(-m.score(ds.X_val)), test_nll=float(-m.score(ds.X_test)),
                                 fit_ms=t * 1e3))
        pd.DataFrame(rows).to_csv(OUT / out_name, index=False)
        print(f"done {ds.name}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("closing", "all"):
        run({
            "base": dict(),
            "seed64x": dict(seed_mult=64),
            "seed256x_l10": dict(seed_mult=256, lloyd=10),
            "ep3": dict(max_epochs=3),
            "pol1": dict(polish_steps=1),
            "pol2": dict(polish_steps=2),
            "sub64k_r2": dict(refine_steps=2, refine_sub=65536),
            "sub128k_r3": dict(refine_steps=3, refine_sub=131072),
        }, names=LARGE, out_name="closing_refinement_large_n.csv")
    if which in ("seeding", "all"):
        run({
            "base": dict(),
            "greedy": dict(greedy=True),
            "greedy64x": dict(greedy=True, seed_mult=64),
            "greedy128x_l5": dict(greedy=True, seed_mult=128, lloyd=5),
            "greedy128x_l10": dict(greedy=True, seed_mult=128, lloyd=10),
            "greedy128x_l20": dict(greedy=True, seed_mult=128, lloyd=20),
            "greedy192x_l10": dict(greedy=True, seed_mult=192, lloyd=10),
            "greedy256x_l10": dict(greedy=True, seed_mult=256, lloyd=10),
            "plain256x_l10": dict(seed_mult=256, lloyd=10),
        }, device="auto", out_name="seeding_variants.csv")
    if which in ("law", "all"):
        run({
            "bracket": dict(greedy=True, seed_mult=128, lloyd=10, law="bracket"),
            "smooth": dict(greedy=True, seed_mult=128, lloyd=10, law="smooth"),
        }, device="auto", out_name="schedule_law_ab.csv")
    if which in ("precision", "all"):
        diag = ["spambase", "philippine", "har", "gas_drift", "miniboone", "syn_highdim128_50k",
                "syn_scale500k_diag", "syn_scale400k_d256_diag"]
        run({
            "float32": dict(cls=EGMM, dtype="float32"),
            "float64": dict(cls=EGMM, dtype="float64"),
        }, names=diag, seeds=(0, 1, 2, 3, 4), out_name="precision_f32_vs_f64.csv")

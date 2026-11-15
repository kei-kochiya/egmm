"""Analysis of the next-version experiments (run_next.py) -> docs/experiments/next/summary.txt.

    python experiments/next/analyze_next.py            # dev.csv, holdout.csv, large.csv
    python experiments/next/analyze_next.py --prefix smoke_

Reports, per stage: e-GMM vs each batch-EM variant (W/T/L with the 1e-3 tie band, median dNLL, Holm-adjusted
Wilcoxon p on 3-seed dataset means, median speedup); best-of-R restarts (selected by training
log-likelihood or by validation NLL; cost = sum of the R single-fit times, slightly conservative because
each single fit repeats standardization); and the equal-compute comparison (largest R whose cumulative
time fits within the baseline's time, R >= 1). These are development-suite analyses: any claim that
restarts beat batch EM needs a new pre-registered holdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiments.analyze_results import apply_holm, fmt_p_txt, pair_stats, wtl

ROOT = Path(__file__).resolve().parents[2]
NEXT = ROOT / "docs" / "experiments" / "next"
BATCH = ["TorchBatchGMM", "TorchBatchGMM_f32chol", "TorchBatchGMM_fast", "TorchBatchGMM_graph"]


def clean(df: pd.DataFrame, lines: list[str], tag: str) -> pd.DataFrame:
    if "error" in df.columns:
        bad = df[df["error"].fillna("") != ""]
        for r in bad.itertuples():
            lines.append(f"[{tag}] FAILED: {r.method} on {r.dataset} seed {r.seed}: {r.error}")
        df = df[df["error"].fillna("") == ""]
    return df


def pairs(df, first, bases, lines, title):
    rows = [(first, b, pair_stats(df, first, b)) for b in bases if b in set(df["method"]) and first in set(df["method"])]
    if not rows:
        return
    apply_holm(rows)
    lines.append(f"\n=== {title} (Holm within this block; raw in brackets) ===")
    for a, b, s in rows:
        lines.append(f"{a} vs {b:24s} n={s['n']:2d} W/T/L={s['W']}/{s['T']}/{s['L']} med_dNLL={s['med_dnll']:+.4f} "
                     f"p={fmt_p_txt(s['p_nll'])} [{fmt_p_txt(s['p_nll_raw'])}] | med_spd={s['med_spd']:.2f}x "
                     f"(faster on {s['n_faster']}/{s['n']}) p_time={fmt_p_txt(s['p_time'])}")


def restart_table(df: pd.DataFrame, n_max: int) -> pd.DataFrame:
    """Best-of-R over EGMM_r0..r{R-1} per (dataset, seed): test NLL under two selection rules, summed time."""
    single = df[df["method"].str.match(r"^EGMM_r\d+$")].copy()
    single["r"] = single["method"].str[6:].astype(int)
    out = []
    for (ds, seed), g in single.groupby(["dataset", "seed"]):
        g = g.sort_values("r")
        for R in range(1, n_max + 1):
            h = g[g["r"] < R]
            if len(h) < R:
                continue
            by_lb = h.loc[h["lower_bound"].idxmax()]
            by_val = h.loc[h["val_nll"].idxmin()]
            out.append(dict(dataset=ds, seed=seed, R=R, time=h["fit_time"].sum(),
                            test_lb=by_lb["test_nll"], test_val=by_val["test_nll"]))
    return pd.DataFrame(out)


def compare_restarts(rt: pd.DataFrame, df: pd.DataFrame, base: str, rule: str, lines: list[str]) -> None:
    b = df[df["method"] == base].groupby("dataset")[["test_nll", "fit_time"]].mean()
    for R in sorted(rt["R"].unique()):
        a = rt[rt["R"] == R].groupby("dataset")[[rule, "time"]].mean()
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        d = a.loc[common, rule] - b.loc[common, "test_nll"]
        w, t, l = wtl(d)
        p = stats.wilcoxon(a.loc[common, rule], b.loc[common, "test_nll"], zero_method="zsplit").pvalue if len(common) > 5 else float("nan")
        spd = (b.loc[common, "fit_time"] / a.loc[common, "time"]).median()
        lines.append(f"  best-of-{R} ({rule[5:]}) vs {base:22s} W/T/L={w}/{t}/{l} med_dNLL={d.median():+.4f} raw p={fmt_p_txt(p)} "
                     f"| median time ratio base/restarts={spd:.2f}x")


def equal_compute(rt: pd.DataFrame, df: pd.DataFrame, base: str, rule: str, lines: list[str]) -> None:
    """Per (dataset, seed): largest R whose summed time <= the baseline's time (at least 1)."""
    bt = df[df["method"] == base].set_index(["dataset", "seed"])[["fit_time", "test_nll"]]
    picks = []
    for (ds, seed), g in rt.groupby(["dataset", "seed"]):
        if (ds, seed) not in bt.index:
            continue
        tb, nb = bt.loc[(ds, seed)]
        ok = g[g["time"] <= tb]
        R = int(ok["R"].max()) if len(ok) else 1
        row = g[g["R"] == R].iloc[0]
        picks.append(dict(dataset=ds, seed=seed, R=R, nll=row[rule], base_nll=nb, time=row["time"], base_time=tb))
    if not picks:
        return
    pk = pd.DataFrame(picks).groupby("dataset").mean(numeric_only=True)
    d = pk["nll"] - pk["base_nll"]
    w, t, l = wtl(d)
    p = stats.wilcoxon(pk["nll"], pk["base_nll"], zero_method="zsplit").pvalue if len(pk) > 5 else float("nan")
    lines.append(f"  equal compute vs {base:22s} ({rule[5:]} selection): mean R={pk['R'].mean():.2f} (R=1 on "
                 f"{int((pk['R'] < 1.5).sum())}/{len(pk)} datasets) W/T/L={w}/{t}/{l} med_dNLL={d.median():+.4f} raw p={fmt_p_txt(p)}")


def large_table(df: pd.DataFrame, lines: list[str]) -> None:
    lines.append("\n=== Large-N: per-dataset 3-seed means (time in s; NLL) ===")
    g = df.groupby(["dataset", "method"])[["fit_time", "test_nll"]].mean().unstack()
    meta = df.groupby("dataset")[["N_train", "D", "K", "cov_type"]].first()
    for ds in meta.sort_values("N_train").index:
        n, dd, k, cov = meta.loc[ds]
        lines.append(f"{ds} (N={n}, D={dd}, K={k}, {cov})")
        for m in g.columns.levels[1]:
            if not np.isfinite(g.loc[ds, ("fit_time", m)]):
                continue
            t, nll = g.loc[ds, ("fit_time", m)], g.loc[ds, ("test_nll", m)]
            lines.append(f"    {m:24s} time={t:9.3f}s  NLL={nll:+.4f}")
        for base in ["TorchBatchGMM", "TorchBatchGMM_graph", "TorchBatchGMM_fast"]:
            if ("fit_time", base) in g.columns and np.isfinite(g.loc[ds, ("fit_time", base)]):
                lines.append(f"    speedup EGMM_r0 vs {base}: {g.loc[ds, ('fit_time', base)] / g.loc[ds, ('fit_time', 'EGMM_r0')]:.2f}x, "
                             f"dNLL={g.loc[ds, ('test_nll', 'EGMM_r0')] - g.loc[ds, ('test_nll', base)]:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()
    lines = ["Next-version experiments (development analyses; see experiments/next/README.md)"]
    for stage in ["dev", "holdout", "large"]:
        f = NEXT / f"{args.prefix}{stage}.csv"
        if not f.exists():
            continue
        df = clean(pd.read_csv(f), lines, stage)
        lines.append(f"\n##### stage {stage}: {df['dataset'].nunique()} datasets, {len(df)} fits")
        pairs(df, "EGMM_r0", BATCH, lines, f"{stage}: e-GMM (single start) vs batch EM variants")
        pairs(df, "TorchBatchGMM_graph", ["TorchBatchGMM", "TorchBatchGMM_fast"], lines, f"{stage}: batch EM variants")
        if "EGMM_eager" in set(df["method"]):
            pairs(df, "EGMM_r0", ["EGMM_eager"], lines, f"{stage}: CUDA graph effect")
        n_max = int(df["method"].str.extract(r"^EGMM_r(\d+)$")[0].dropna().astype(int).max() + 1) if df["method"].str.match(r"^EGMM_r\d+$").any() else 0
        if n_max >= 2:
            rt = restart_table(df, n_max)
            lines.append(f"\n=== {stage}: best-of-R restarts (R = 1..{n_max}) ===")
            for base in ["TorchBatchGMM", "TorchBatchGMM_graph"]:
                for rule in ["test_lb", "test_val"]:
                    compare_restarts(rt, df, base, rule, lines)
            lines.append(f"\n=== {stage}: equal compute ===")
            for base in ["TorchBatchGMM", "TorchBatchGMM_fast", "TorchBatchGMM_graph"]:
                for rule in ["test_lb", "test_val"]:
                    equal_compute(rt, df, base, rule, lines)
            if "EGMM_n3" in set(df["method"]) and n_max >= 3:
                n3 = df[df["method"] == "EGMM_n3"].groupby("dataset")["fit_time"].mean()
                s3 = rt[rt["R"] == 3].groupby("dataset")["time"].mean()
                c = n3.index.intersection(s3.index)
                lines.append(f"  library n_init=3 time / sum of 3 single fits: median {(n3[c] / s3[c]).median():.2f}")
        if stage == "large":
            large_table(df, lines)
    NEXT.mkdir(parents=True, exist_ok=True)
    out = NEXT / f"{args.prefix}summary.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

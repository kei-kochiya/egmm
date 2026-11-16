"""Macros and tables for the revised paper from the follow-up runs (docs/experiments/next/*.csv).

Called by experiments/analyze_results.py (the single source of every number in the paper). All comparisons
use 3-seed dataset means, the 1e-3 tie band and two-sided Wilcoxon tests exactly as in the main analysis.

* Time reference: "fastest batch EM" is, per dataset, the batch-EM variant with the lowest mean fit time among
  TorchBatchGMM, TorchBatchGMM_f32chol, TorchBatchGMM_fast and TorchBatchGMM_graph (failed fits excluded).
* Quality reference: always the float64 eager TorchBatchGMM. The faster variants factorize in float32, which
  changes their NLL (and stopping iteration) on some real datasets, so they are used for time only.
* Speed comparisons, the large-N tables and the time-to-quality results come from the speed2 re-timing
  (``speed2_results.py``), which adds batch-EM variants that check convergence every iteration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BATCH = ["TorchBatchGMM", "TorchBatchGMM_f32chol", "TorchBatchGMM_fast", "TorchBatchGMM_graph"]
SHORT = {"TorchBatchGMM": "eager", "TorchBatchGMM_f32chol": "f32chol", "TorchBatchGMM_fast": "fast",
         "TorchBatchGMM_graph": "graph"}
LARGE_N = 100_000
ARTIFACTS = ("eeg_eye_state", "ailerons")


def _load(exp: Path, stage: str) -> pd.DataFrame | None:
    f = exp / "next" / f"{stage}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d[d["error"].fillna("") == ""] if "error" in d.columns else d


def _add_best(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.Series]:
    src = df
    t = src[src["method"].isin(BATCH)].groupby(["dataset", "method"])["fit_time"].mean().unstack()
    best = t.idxmin(axis=1)
    rows = [src[(src["dataset"] == ds) & (src["method"] == m)].assign(method=name) for ds, m in best.items()]
    return pd.concat([df] + rows, ignore_index=True), best


def _tex(name: str) -> str:
    return name.replace("_", r"\_")


def _signed(v: float) -> str:
    """Signed 2-decimal number for tables; |v| < 0.005 prints as 0.00 (no spurious sign)."""
    if abs(v) < 0.005:
        return "$0.00$"
    return f"${v:+.2f}$" if abs(v) < 1e3 else f"${v:+.1e}$"


def _median_ci(x: pd.Series) -> tuple[float, float, float]:
    """Distribution-free (order-statistic) 95% interval for the median, with its exact coverage."""
    v = np.sort(np.asarray(x, dtype=float))
    n = len(v)
    k = max(j for j in range(1, n // 2 + 1) if stats.binom.cdf(j - 1, n, 0.5) <= 0.025)
    cov = 1.0 - 2.0 * stats.binom.cdf(k - 1, n, 0.5)
    return float(v[k - 1]), float(v[n - k]), 100.0 * cov


def _time_ratio(df: pd.DataFrame, ref: str, a: str = "EGMM_r0") -> pd.Series:
    g = df.groupby(["dataset", "method"])["fit_time"].mean().unstack()
    return g[ref] / g[a]


def add_next_results(exp: Path, gen: Path, macros: dict, lines: list[str]) -> None:
    from experiments.analyze_results import apply_holm, fmt_p, fmt_p_txt, pair_stats, wtl

    dev, ho, diag, large = (_load(exp, s) for s in ("dev", "holdout", "diag", "large"))
    if dev is None or ho is None or large is None:
        lines.append("\n(follow-up runs not found; revised-paper macros skipped)")
        return
    raw = [pd.read_csv(exp / "next" / f"{x}.csv") for x in ("dev", "holdout", "diag", "large")]
    macros["XNFits"] = f"{sum(len(r) for r in raw):,}".replace(",", "{,}")
    macros["XNFail"] = str(sum(int((r["error"].fillna("") != "").sum()) for r in raw))
    lines.append("\n=== Revised paper: follow-up runs (docs/experiments/next; one session per suite) ===")
    p = lambda v: fmt_p(v).strip("$")  # noqa: E731

    # registered holdout run: the one dataset on which e-GMM is slower than TorchBatchGMM
    hh = pd.read_csv(exp / "holdout_head_to_head.csv").groupby(["dataset", "method"])["fit_time"].mean().unstack()
    hr = (hh["TorchBatchGMM"] / hh["EGMM"]).sort_values()
    macros["HoSlowestTBName"], macros["HoSlowestTBSpd"] = _tex(hr.index[0]), f"{hr.iloc[0]:.2f}"
    macros["HoSlowerTBN"] = str(int((hr < 1).sum()))

    # fastest batch-EM variant per dataset in the first follow-up sessions (time budget for the restarts below;
    # the paper's speed comparisons use the speed2 re-timing, speed2_results.py)
    best_old = {}
    for tag, df in [("Dev", dev), ("Lg", large)]:
        best_old[tag] = _add_best(df, "BatchBest")[1]
    n_dev = dev.groupby("dataset")["N_train"].first()

    # ---------------- what the faster variants change: NLL (float32 factorization) and stopping iteration
    worst, best_gain, it_diff = [], [], {}
    for tag, df in [("Dev", dev), ("Ho", ho)]:
        q = pair_stats(df, "TorchBatchGMM_graph", "TorchBatchGMM")
        macros[f"X{tag}GraphQWTL"] = f"{q['W']}/{q['T']}/{q['L']}"
        macros[f"X{tag}GraphQP"] = p(q["p_nll"])
        g = df.groupby(["dataset", "method"])["test_nll"].mean().unstack()
        dd = (g["TorchBatchGMM_graph"] - g["TorchBatchGMM"]).drop(list(ARTIFACTS), errors="ignore")
        worst.append(dd.max())
        best_gain.append(-dd.min())
        it = df.pivot_table(index=["dataset", "seed"], columns="method", values="n_iter")
        x = (it["TorchBatchGMM_graph"] - it["TorchBatchGMM"]).dropna()
        it_diff[tag] = (int((x != 0).sum()), len(x), float(x.abs().max()))
        macros[f"XItDiff{tag}"] = f"{it_diff[tag][0]}/{it_diff[tag][1]}"
        lines.append(f"{tag}: graph vs TorchBatchGMM NLL W/T/L {q['W']}/{q['T']}/{q['L']} (raw p {fmt_p_txt(q['p_nll'])}), largest change "
                     f"without artifacts +{dd.max():.2f}/-{-dd.min():.2f}; stopping iteration differs on {it_diff[tag][0]}/{it_diff[tag][1]} runs "
                     f"(max {it_diff[tag][2]:.0f})")
    macros["XGraphQWorst"], macros["XGraphQGain"] = f"{max(worst):.1f}", f"{max(best_gain):.1f}"
    macros["XItDiffMax"] = f"{max(v[2] for v in it_diff.values()):.0f}"

    # graph factor and algorithm-alone factors
    for tag, df in [("Dev", dev), ("Ho", ho)]:
        for name, a, b in [("Graph", "EGMM_r0", "EGMM_eager"), ("GraphHost", "EGMM_r0", "EGMM_eager_hostalpha"),
                           ("AlgEager", "EGMM_eager_hostalpha", "TorchBatchGMM"), ("AlgFast", "EGMM_eager_hostalpha", "TorchBatchGMM_fast"),
                           ("AlgGraph", "EGMM_eager_hostalpha", "TorchBatchGMM_graph")]:
            macros[f"X{tag}{name}"] = f"{pair_stats(df, a, b)['med_spd']:.2f}"
        g = df.groupby(["dataset", "method"])["fit_time"].mean().unstack()
        macros[f"X{tag}GraphVsEager"] = f"{(g['TorchBatchGMM'] / g['TorchBatchGMM_graph']).median():.2f}"
    lines.append(f"graph factor (EGMM vs eager / vs eager without alpha sync): dev {macros['XDevGraph']}x / {macros['XDevGraphHost']}x; "
                 f"holdout {macros['XHoGraph']}x / {macros['XHoGraphHost']}x")

    # ---------------- single-factor schedule ablation (development), one Holm family of two, plus a split by N
    sched = [("EGMM_r0", b, pair_stats(dev, "EGMM_r0", b)) for b in ["EGMM_sched_rm", "EGMM_sched_const"]]
    apply_holm(sched)
    for (_, b, s), tag in zip(sched, ["Rm", "Const"]):
        macros[f"XSched{tag}WTL"] = f"{s['W']}/{s['T']}/{s['L']}"
        macros[f"XSched{tag}dNLL"] = f"{s['med_dnll']:+.3f}"
        macros[f"XSched{tag}P"] = p(s["p_nll"])
        lines.append(f"schedule ablation: cosine vs {b}: W/T/L {s['W']}/{s['T']}/{s['L']}, median dNLL {s['med_dnll']:+.4f}, Holm p {fmt_p_txt(s['p_nll'])}")
        for sub, sel in [("Small", n_dev[n_dev <= LARGE_N].index), ("Large", n_dev[n_dev > LARGE_N].index)]:
            ss = pair_stats(dev, "EGMM_r0", b, datasets=sel)
            macros[f"XSched{tag}{sub}WTL"] = f"{ss['W']}/{ss['T']}/{ss['L']}"
            macros[f"XSched{tag}{sub}dNLL"] = f"{ss['med_dnll']:+.3f}"
            macros[f"XSched{tag}{sub}P"] = p(ss["p_nll"])
            lines.append(f"  {sub} (n={ss['n']}): W/T/L {ss['W']}/{ss['T']}/{ss['L']}, median {ss['med_dnll']:+.4f}, raw p {fmt_p_txt(ss['p_nll'])}")

    # ---------------- holdout, post hoc: online EM with matched polish, revival (one Holm family of three)
    hpost = [("EGMM_r0", b, pair_stats(ho, "EGMM_r0", b)) for b in ["OnlineGMM_tuned", "OnlineGMM_matched_polish", "EGMM_no_revival"]]
    apply_holm(hpost)
    for (_, b, s), tag in zip(hpost, ["Online", "Matched", "NoRev"]):
        macros[f"XHo{tag}WTL"] = f"{s['W']}/{s['T']}/{s['L']}"
        macros[f"XHo{tag}dNLL"] = f"{s['med_dnll']:+.2f}"
        macros[f"XHo{tag}P"] = p(s["p_nll"])
        lines.append(f"holdout post hoc: e-GMM vs {b}: W/T/L {s['W']}/{s['T']}/{s['L']}, median dNLL {s['med_dnll']:+.3f}, Holm p {fmt_p_txt(s['p_nll'])}")
    cov_ho = ho.groupby("dataset")["cov_type"].first()
    full_ok = [x for x in cov_ho.index if cov_ho[x] == "full" and x not in ARTIFACTS]
    sm = pair_stats(ho, "EGMM_r0", "OnlineGMM_matched_polish", datasets=full_ok)
    macros["XHoMatchedFullWTL"], macros["XHoMatchedFulldNLL"] = f"{sm['W']}/{sm['T']}/{sm['L']}", f"{sm['med_dnll']:+.2f}"
    macros["XHoMatchedFullP"], macros["XnHoFullOk"] = p(sm["p_nll"]), str(sm["n"])
    gh = ho.groupby(["dataset", "method"])["test_nll"].mean().unstack()
    for ds in ("isolet", "volkert"):
        macros[f"XHoMatched{ds}"] = f"{gh.loc[ds, 'EGMM_r0'] - gh.loc[ds, 'OnlineGMM_matched_polish']:+.1f}"
    lines.append(f"  vs matched polish on full-covariance datasets without artifacts (n={sm['n']}): {sm['W']}/{sm['T']}/{sm['L']}, "
                 f"median {sm['med_dnll']:+.3f}, raw p {fmt_p_txt(sm['p_nll'])}")

    # ---------------- restarts with a fixed number R of starts (quality vs float64 TorchBatchGMM, time vs fastest)
    def restarts(df, best, tag, rs):
        single = df[df["method"].str.match(r"^EGMM_r\d+$")].copy()
        single["r"] = single["method"].str[6:].astype(int)
        tb = df[df["method"] == "TorchBatchGMM"].groupby(["dataset", "seed"])["test_nll"].mean()
        bt = df[df["method"].isin(BATCH)].groupby(["dataset", "seed", "method"])["fit_time"].mean()
        for R in rs:
            for rule, col, pick in [("val", "val_nll", "idxmin"), ("lb", "lower_bound", "idxmax")]:
                picks = []
                for (ds, seed), g in single.groupby(["dataset", "seed"]):
                    h = g.sort_values("r").iloc[:R]
                    if len(h) < R or (ds, seed, best[ds]) not in bt.index:
                        continue
                    i = getattr(h[col], pick)()
                    picks.append(dict(dataset=ds, nll=h.loc[i, "test_nll"], base=tb.loc[(ds, seed)],
                                      ratio=bt.loc[(ds, seed, best[ds])] / h["fit_time"].sum()))
                pk = pd.DataFrame(picks).groupby("dataset").mean(numeric_only=True)
                d = pk["nll"] - pk["base"]
                w, t, l_ = wtl(d)
                pv = stats.wilcoxon(pk["nll"], pk["base"], zero_method="zsplit").pvalue if len(pk) > 5 else float("nan")
                key = f"XRe{tag}{'Two' if R == 2 else 'Three'}{'' if rule == 'val' else 'Lb'}"
                macros[key + "WTL"] = f"{w}/{t}/{l_}"
                macros[key + "dNLL"] = f"{d.median():+.2f}"
                macros[key + "Time"] = f"{pk['ratio'].median():.2f}"
                macros[key + "TimeInv"] = f"{1.0 / pk['ratio'].median():.1f}"
                macros[key + "P"] = p(pv)
                lines.append(f"best-of-{R} ({rule}) {tag} vs TorchBatchGMM NLL: W/T/L {w}/{t}/{l_}, median {d.median():+.3f}, raw p "
                             f"{fmt_p_txt(pv)}; time ratio fastest batch / restarts {pk['ratio'].median():.2f}")

    restarts(dev, best_old["Dev"], "Dev", (2, 3))
    restarts(large, best_old["Lg"], "Lg", (2, 3))
    n3 = dev[dev["method"] == "EGMM_n3"].groupby("dataset")["fit_time"].mean()
    s3 = dev[dev["method"].isin(["EGMM_r0", "EGMM_r1", "EGMM_r2"])].groupby(["dataset", "method"])["fit_time"].mean().unstack().sum(axis=1)
    macros["XNinitRatio"] = f"{(n3 / s3.loc[n3.index]).median():.2f}"

    # ---------------- scikit-learn at 1M rows (first large-N session; compared with e-GMM from the same session)
    meta = large.groupby("dataset")[["N_train", "D", "K", "cov_type"]].first()
    sk = large[large["method"] == "StdSklearnGMM"]
    for ds in sk["dataset"].unique():
        cov = meta.loc[ds, "cov_type"].capitalize()
        ts = sk[sk["dataset"] == ds]["fit_time"].mean()
        eg = large[(large["dataset"] == ds) & (large["method"] == "EGMM_r0")]
        macros[f"XSk{cov}Time"] = f"{ts:.0f}"
        macros[f"XSk{cov}Ratio"] = f"{ts / eg[eg['seed'] == 0]['fit_time'].mean():.0f}"  # same seed as the single sklearn fit
        macros[f"XSk{cov}dNLL"] = f"{eg[eg['seed'] == 0]['test_nll'].mean() - sk[sk['dataset'] == ds]['test_nll'].mean():+.2f}"  # seed 0 (both)
        macros[f"XSk{cov}dNLLMean"] = f"{eg['test_nll'].mean() - sk[sk['dataset'] == ds]['test_nll'].mean():+.2f}"  # e-GMM 3-seed mean

    # ---------------- diagonal-covariance diagnostics (appendix table)
    if diag is not None:
        gd = diag.groupby(["dataset", "method"])["test_nll"].mean().unstack()
        md = diag.groupby("dataset")[["N_train", "D", "regime"]].first()
        cols = ["EGMM_polish0", "EGMM_a085", "EGMM_ep2x", "TorchBatchGMM"]  # notol is reported in the text (no effect)
        d_rows = []
        for ds in md.sort_values(["regime", "D"]).index:
            n, dd, reg = md.loc[ds]
            src = "holdout" if reg == "holdout" else "dev."
            cells = [_signed(gd.loc[ds, c] - gd.loc[ds, "EGMM_r0"]) for c in cols]
            d_rows.append(f"\\texttt{{{_tex(ds)}}} & {src} & {n:,} & {dd} & ".replace(",", "{,}") + " & ".join(cells) + " \\\\")
        (gen / "tab_diag.tex").write_text("\n".join(d_rows) + "\n", encoding="utf-8")
        da = gd["EGMM_a085"] - gd["EGMM_r0"]
        big = da[da < -1.0]
        macros["XDiagAImproved"], macros["XnDiag"] = str(len(big)), str(len(da))
        macros["XDiagAImpMin"], macros["XDiagAImpMax"] = f"{-big.max():.1f}", f"{-big.min():.1f}"
        macros["XDiagAWorst"], macros["XDiagAWorstName"] = f"{da.max():+.1f}", _tex(da.idxmax())
        sd = diag[diag["method"] == "EGMM_r0"].groupby("dataset")["test_nll"].std(ddof=1)
        macros["XDiagWorstSeedSD"] = f"{sd[da.idxmax()]:.0f}"
        for ds in ("volkert", "isolet"):
            if ds in gd.index:
                macros[f"XDiagAvsTB{ds}"] = f"{gd.loc[ds, 'EGMM_a085'] - gd.loc[ds, 'TorchBatchGMM']:+.1f}"
        # the rule's actual settings on these datasets (the diagnostics change more than one constant)
        from egmm.scaling import auto_hyperparameters

        meta_d = diag.groupby("dataset")[["N_train", "D", "K"]].first()
        hps = {ds: auto_hyperparameters(int(r.N_train), int(r.K), int(r.D), "diag") for ds, r in meta_d.iterrows()}
        hi_d = [ds for ds in hps if meta_d.loc[ds, "D"] >= 100]
        a0 = pd.Series({ds: hps[ds].alpha_0 for ds in hi_d})
        am = pd.Series({ds: hps[ds].alpha_min for ds in hi_d})
        macros["XDiagAlphaLo"], macros["XDiagAlphaHi"] = f"{a0.min():.2f}", f"{a0.max():.2f}"
        macros["XDiagAminLo"], macros["XDiagAminHi"] = f"{am.min():.3f}", f"{am.max():.3f}"
        r = meta_d.iloc[0]
        macros["XDiagAminAlt"] = f"{auto_hyperparameters(int(r.N_train), int(r.K), int(r.D), 'diag', alpha=0.85).alpha_min:.3f}"
        ratio = pd.Series({ds: auto_hyperparameters(int(meta_d.loc[ds, 'N_train']), int(meta_d.loc[ds, 'K']), int(meta_d.loc[ds, 'D']), "diag",
                                                    max_epochs=2 * hps[ds].max_epochs).total_steps / hps[ds].total_steps for ds in hps})
        macros["XDiagStepRatioLo"], macros["XDiagStepRatioHi"] = f"{ratio.min():.1f}", f"{ratio.max():.1f}"
        sd_all = diag[diag["method"] == "EGMM_r0"].groupby("dataset")["test_nll"].std(ddof=1)
        macros["XDiagIsoletSD"] = f"{sd_all.get('isolet', float('nan')):.1f}"
        lines.append(f"diag rule settings: alpha0 {a0.round(3).to_dict()}, alpha_min {am.round(3).to_dict()}; "
                     f"twice the epochs = {ratio.round(2).to_dict()} x the EMA steps")
        macros["XDiagNotolMax"] = f"{(gd['EGMM_notol'] - gd['EGMM_r0']).abs().max():.3f}"
        lines.append(f"diag: alpha0=0.85 vs deployed: improves >1 nat on {len(big)}/{len(da)} ({', '.join(big.index)}); worst {da.idxmax()} "
                     f"{da.max():+.2f} (its e-GMM seed SD {sd[da.idxmax()]:.1f}); polish without early stop max |d| "
                     f"{(gd['EGMM_notol'] - gd['EGMM_r0']).abs().max():.4f}")

    from experiments.next.speed2_results import add_speed2_results

    add_speed2_results(exp, gen, macros, lines)

"""Macros and tables for the revised paper from the follow-up runs (docs/experiments/next/*.csv).

Called by experiments/analyze_results.py (the single source of every number in the paper). All comparisons
use 3-seed dataset means, the 1e-3 tie band and two-sided Wilcoxon tests exactly as in the main analysis.

* Time reference: "fastest batch EM" is, per dataset, the batch-EM variant with the lowest mean fit time among
  TorchBatchGMM, TorchBatchGMM_f32chol, TorchBatchGMM_fast and TorchBatchGMM_graph (failed fits excluded).
* Quality reference: always the float64 eager TorchBatchGMM. The faster variants factorize in float32, which
  changes their NLL (and stopping iteration) on some real datasets, so they are used for time only.
* The sync-light and graph variants read the convergence test every 5 iterations and therefore execute up to
  4 iterations past the stopping iteration (the graph variant also runs one discarded warm-up iteration before
  capture). ``_adjust_check_interval`` estimates their time with a check every iteration by scaling the logged
  time by needed / executed iterations; it scales the fixed costs too, so it favors batch EM.
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
MAX_ITER, CHECK = 50, 5
ARTIFACTS = ("eeg_eye_state", "ailerons")


def _load(exp: Path, stage: str) -> pd.DataFrame | None:
    f = exp / "next" / f"{stage}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d[d["error"].fillna("") == ""] if "error" in d.columns else d


def _adjust_check_interval(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    n = d["n_iter"].astype(float)
    e = np.minimum(MAX_ITER, CHECK * np.ceil(n / CHECK))
    f = d["method"] == "TorchBatchGMM_fast"
    g = d["method"] == "TorchBatchGMM_graph"
    d.loc[f, "fit_time"] = d.loc[f, "fit_time"] * n[f] / e[f]
    d.loc[g, "fit_time"] = d.loc[g, "fit_time"] * (n[g] + 1.0) / (e[g] + 1.0)
    return d


def _add_best(df: pd.DataFrame, name: str, adjust: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    src = _adjust_check_interval(df) if adjust else df
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
    lines.append("\n=== Revised paper: follow-up runs (docs/experiments/next; one session per suite) ===")
    p = lambda v: fmt_p(v).strip("$")  # noqa: E731

    # registered holdout run: the one dataset on which e-GMM is slower than TorchBatchGMM
    hh = pd.read_csv(exp / "holdout_head_to_head.csv").groupby(["dataset", "method"])["fit_time"].mean().unstack()
    hr = (hh["TorchBatchGMM"] / hh["EGMM"]).sort_values()
    macros["HoSlowestTBName"], macros["HoSlowestTBSpd"] = _tex(hr.index[0]), f"{hr.iloc[0]:.2f}"
    macros["HoSlowerTBN"] = str(int((hr < 1).sum()))

    # ---------------- speed against the fastest batch-EM variant (logged and check-interval-adjusted)
    suites = {}
    for tag, df, label in [("Dev", dev, "development"), ("Ho", ho, "holdout (post hoc)"), ("Lg", large, "large N")]:
        dfb, best = _add_best(df, "BatchBest")
        dfb, best_adj = _add_best(dfb, "BatchBestAdj", adjust=True)
        n_ds = df.groupby("dataset")["N_train"].first()
        spd = _time_ratio(dfb, "BatchBest")
        spd_adj = _time_ratio(dfb, "BatchBestAdj")
        suites[tag] = (dfb, best, spd, spd_adj, n_ds)
        macros[f"Xn{tag}"] = str(len(spd))
        macros[f"X{tag}BestSpd"] = f"{spd.median():.2f}"
        macros[f"X{tag}BestAdj"] = f"{spd_adj.median():.2f}"
        macros[f"X{tag}GraphFastest"] = str(int((best == "TorchBatchGMM_graph").sum()))
        for sub, msk in [("Small", n_ds[spd.index] <= LARGE_N), ("Large", n_ds[spd.index] > LARGE_N)]:
            if msk.sum():
                macros[f"Xn{tag}{sub}"] = str(int(msk.sum()))
                macros[f"X{tag}{sub}Spd"] = f"{spd[msk].median():.2f}"
                macros[f"X{tag}{sub}Faster"] = str(int((spd[msk] > 1).sum()))
                macros[f"X{tag}{sub}Adj"] = f"{spd_adj[msk].median():.2f}"
                macros[f"X{tag}{sub}AdjMin"] = f"{spd_adj[msk].min():.2f}"
                macros[f"X{tag}{sub}AdjMax"] = f"{spd_adj[msk].max():.0f}"
                if msk.sum() >= 6:
                    lo, hi, cov = _median_ci(spd[msk])
                    macros[f"X{tag}{sub}CI"] = f"{lo:.2f}--{hi:.2f}"  # text mode
                    macros[f"X{tag}{sub}Cov"] = f"{cov:.0f}"
        lines.append(f"{label}: e-GMM vs fastest batch EM: median {spd.median():.2f}x (faster on {int((spd > 1).sum())}/{len(spd)}); "
                     f"with every-iteration checks (estimate) {spd_adj.median():.2f}x; fastest variant {best.value_counts().to_dict()}")
    macros["XDevBestSlower"] = str(int((suites["Dev"][2] <= 1).sum()))
    lg_spd, lg_adj = suites["Lg"][2], suites["Lg"][3]
    macros["XLgMinSpd"], macros["XLgMaxSpd"] = f"{lg_spd.min():.1f}", f"{lg_spd.max():.0f}"
    macros["XLgAdjMin"], macros["XLgAdjMax"] = f"{lg_adj.min():.1f}", f"{lg_adj.max():.0f}"
    lines.append(f"large N per-dataset speedups vs fastest: {lg_spd.round(2).to_dict()}; adjusted {lg_adj.round(2).to_dict()}")

    # Table 2: four strata; speed vs each variant, vs the fastest (logged and adjusted); quality vs float64 TorchBatchGMM
    rows = []
    devb, hob, lab = suites["Dev"][0], suites["Ho"][0], suites["Lg"][0]
    n_dev, n_ho = suites["Dev"][4], suites["Ho"][4]
    groups = [(r"Development, $N_{\mathrm{train}} \le 10^5$", devb[devb["dataset"].isin(n_dev[n_dev <= LARGE_N].index)]),
              (r"Development, $N_{\mathrm{train}} > 10^5$", devb[devb["dataset"].isin(n_dev[n_dev > LARGE_N].index)]),
              (r"Holdout, $N_{\mathrm{train}} \le 10^5$ (post hoc)", hob[hob["dataset"].isin(n_ho[n_ho <= LARGE_N].index)]),
              (r"Large $N$ ($1$--$10$\,M rows)", lab)]
    for name, d in groups:
        cells = []
        for b in ["TorchBatchGMM", "TorchBatchGMM_fast", "TorchBatchGMM_graph", "BatchBest", "BatchBestAdj"]:
            r = _time_ratio(d, b)
            cells.append(f"${r.median():.2f}\\times$ ({int((r > 1).sum())}/{len(r)})")
        q = pair_stats(d, "EGMM_r0", "TorchBatchGMM")
        cells.append(f"{q['W']}/{q['T']}/{q['L']}")
        cells.append(_signed(q["med_dnll"]))
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    (gen / "tab_speed_next.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

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

    restarts(dev, suites["Dev"][1], "Dev", (2, 3))
    restarts(large, suites["Lg"][1], "Lg", (2, 3))
    n3 = dev[dev["method"] == "EGMM_n3"].groupby("dataset")["fit_time"].mean()
    s3 = dev[dev["method"].isin(["EGMM_r0", "EGMM_r1", "EGMM_r2"])].groupby(["dataset", "method"])["fit_time"].mean().unstack().sum(axis=1)
    macros["XNinitRatio"] = f"{(n3 / s3.loc[n3.index]).median():.2f}"

    # ---------------- large-N table and macros (quality vs float64 TorchBatchGMM; the variants agree exactly here)
    adj = _adjust_check_interval(large)
    g = large.groupby(["dataset", "method"])[["fit_time", "test_nll", "dead_components"]].mean().unstack()
    ga = adj.groupby(["dataset", "method"])["fit_time"].mean().unstack()
    meta = large.groupby("dataset")[["N_train", "D", "K", "cov_type"]].first()
    best_lg, best_adj_lg = suites["Lg"][1], ga[BATCH].idxmin(axis=1)
    t_rows, dq = [], {}
    order = meta.assign(real=[0 if d.startswith("syn") else 1 for d in meta.index]).sort_values(["real", "cov_type", "N_train"],
                                                                                                ascending=[True, False, True])
    for ds in order.index:
        n, dd, k, cov = meta.loc[ds]
        te = g.loc[ds, ("fit_time", "EGMM_r0")]
        tb, tba = g.loc[ds, ("fit_time", best_lg[ds])], ga.loc[ds, best_adj_lg[ds]]
        dn = g.loc[ds, ("test_nll", "EGMM_r0")] - g.loc[ds, ("test_nll", "TorchBatchGMM")]
        dq[ds] = dn
        name = {"higgs": "HIGGS", "susy": "SUSY"}.get(ds, "synthetic")
        times = " & ".join(f"{g.loc[ds, ('fit_time', m)]:.2f}" for m in ["TorchBatchGMM", "TorchBatchGMM_fast", "TorchBatchGMM_graph"])
        t_rows.append(f"{name} & {n / 1e6:.1f} & {dd} & {cov} & {te:.2f} & {times} & {SHORT[best_lg[ds]]} & ${tb / te:.1f}\\times$ & "
                      f"${tba / te:.1f}\\times$ & {_signed(dn)} & {g.loc[ds, ('dead_components', 'TorchBatchGMM')]:.1f} \\\\")
        if not ds.startswith("syn"):  # synthetic names collide once digits are stripped; only real sets get macros
            macros[f"XLg{ds}Spd"] = f"{tb / te:.0f}" if tb / te >= 10 else f"{tb / te:.1f}"
            macros[f"XLg{ds}dNLL"] = f"{dn:+.2f}"
            macros[f"XLg{ds}DeadTB"] = f"{g.loc[ds, ('dead_components', 'TorchBatchGMM')]:.1f}"
    (gen / "tab_large.tex").write_text("\n".join(t_rows) + "\n", encoding="utf-8")
    dq = pd.Series(dq)
    q = pair_stats(large, "EGMM_r0", "TorchBatchGMM")
    macros["XLgTBWTL"], macros["XLgTBdNLL"] = f"{q['W']}/{q['T']}/{q['L']}", f"{q['med_dnll']:+.2f}"
    macros["XLgNBetter"], macros["XLgNEqual"] = str(int((dq < -0.1).sum())), str(int((dq.abs() < 0.01).sum()))
    macros["XLgBetterMin"] = f"{-dq[dq < -0.1].max():.2f}"
    macros["XLgDeadEGMMMax"] = f"{g['dead_components']['EGMM_r0'].max():.1f}"
    it = large.groupby(["dataset", "method"])["n_iter"].mean().unstack()
    syn_ds = [d for d in it.index if d.startswith("syn")]
    syn_seed = large[(large["method"] == "TorchBatchGMM") & large["dataset"].isin(syn_ds)]["n_iter"]
    macros["XLgTBIterSynMin"], macros["XLgTBIterSynMax"] = f"{syn_seed.min():.0f}", f"{syn_seed.max():.0f}"
    hig = large[(large["method"] == "TorchBatchGMM") & (large["dataset"] == "higgs")]["n_iter"]
    macros["XLgTBIterHiggsMin"], macros["XLgTBIterHiggsMax"] = f"{hig.min():.0f}", f"{hig.max():.0f}"
    macros["XLgTBIterSusy"] = f"{it.loc['susy', 'TorchBatchGMM']:.0f}"
    macros["XLgEGMMPasses"] = f"{it['EGMM_r0'].median():.0f}"
    lines.append(f"large N: NLL vs TorchBatchGMM {dq.round(4).to_dict()}; TorchBatchGMM iterations per run synthetic {syn_seed.min():.0f}-"
                 f"{syn_seed.max():.0f}, HIGGS {hig.min():.0f}-{hig.max():.0f}, SUSY {it.loc['susy', 'TorchBatchGMM']:.0f}")
    syn = [d for d in lg_spd.index if d.startswith("syn")]
    for cov in ("full", "diag"):
        sel = [d for d in syn if meta.loc[d, "cov_type"] == cov]
        macros[f"XLgSyn{cov.capitalize()}Min"], macros[f"XLgSyn{cov.capitalize()}Max"] = f"{lg_spd[sel].min():.1f}", f"{lg_spd[sel].max():.1f}"
        macros[f"XLgSyn{cov.capitalize()}AdjMin"], macros[f"XLgSyn{cov.capitalize()}AdjMax"] = f"{lg_adj[sel].min():.1f}", f"{lg_adj[sel].max():.1f}"
    sk = large[large["method"] == "StdSklearnGMM"]
    for ds in sk["dataset"].unique():
        cov = meta.loc[ds, "cov_type"].capitalize()
        ts = sk[sk["dataset"] == ds]["fit_time"].mean()
        eg = large[(large["dataset"] == ds) & (large["method"] == "EGMM_r0")]
        macros[f"XSk{cov}Time"] = f"{ts:.0f}"
        macros[f"XSk{cov}Ratio"] = f"{ts / eg[eg['seed'] == 0]['fit_time'].mean():.0f}"  # same seed as the single sklearn fit
        macros[f"XSk{cov}dNLL"] = f"{eg['test_nll'].mean() - sk[sk['dataset'] == ds]['test_nll'].mean():+.2f}"  # e-GMM 3-seed mean

    # ---------------- diagonal-covariance diagnostics (appendix table)
    if diag is not None:
        gd = diag.groupby(["dataset", "method"])["test_nll"].mean().unstack()
        md = diag.groupby("dataset")[["N_train", "D", "regime"]].first()
        cols = ["EGMM_polish0", "EGMM_a085", "EGMM_ep2x", "EGMM_notol", "TorchBatchGMM"]
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
        macros["XDiagNotolMax"] = f"{(gd['EGMM_notol'] - gd['EGMM_r0']).abs().max():.3f}"
        lines.append(f"diag: alpha0=0.85 vs deployed: improves >1 nat on {len(big)}/{len(da)} ({', '.join(big.index)}); worst {da.idxmax()} "
                     f"{da.max():+.2f} (its e-GMM seed SD {sd[da.idxmax()]:.1f}); polish without early stop max |d| "
                     f"{(gd['EGMM_notol'] - gd['EGMM_r0']).abs().max():.4f}")

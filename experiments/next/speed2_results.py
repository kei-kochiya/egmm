"""Revised-paper speed macros, tables and quality-at-equal-time results from the speed2 re-timing.

Reads docs/experiments/next/speed2_{dev,holdout,large}.csv (one interleaved session per suite; run by
``experiments/next/run_next.py --stages speed2``) and is called by ``paper_results.add_next_results``.

* Time reference: per dataset, the fastest of TorchBatchGMM (eager), TorchBatchGMM_fast / _fast1 (sync-light,
  convergence read every 5 / every iteration) and TorchBatchGMM_graph / _graph1 (CUDA graph, every 5 / every
  iteration), all with tol = 1e-3 and max_iter = 50.
* Quality reference: the float64 eager TorchBatchGMM.
* Large N: full-data-seeded batch EM (speed2_large) and, from speed2b_large, batch EM with fixed budgets of
  2-20 iterations (each budget timed in the graph and the sync-light implementation; the faster one counts)
  and e-GMM with 1-3 epochs, for comparisons at equal quality and at equal passes. Times of runs from
  different sessions are only compared as ratios to e-GMM's time in their own session.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.next.paper_results import LARGE_N, _load, _median_ci, _signed, _time_ratio

TIMED = ["TorchBatchGMM", "TorchBatchGMM_fast", "TorchBatchGMM_fast1", "TorchBatchGMM_graph", "TorchBatchGMM_graph1"]
SHORT = {"TorchBatchGMM": "eager", "TorchBatchGMM_fast": "light-5", "TorchBatchGMM_fast1": "light-1",
         "TorchBatchGMM_graph": "graph-5", "TorchBatchGMM_graph1": "graph-1"}
BUDGETS = (2, 3, 5, 10, 20)
EQ_TOL = 0.01  # nats: "equal quality" band for the time-to-quality comparison


def _add_best(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    t = df[df["method"].isin(TIMED)].groupby(["dataset", "method"])["fit_time"].mean().unstack()
    best = t.idxmin(axis=1)
    rows = [df[(df["dataset"] == ds) & (df["method"] == m)].assign(method="BatchBest") for ds, m in best.items()]
    return pd.concat([df] + rows, ignore_index=True), best


def budget_points(sb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per dataset: time ratio to e-GMM (same session) and test NLL of each fixed batch-EM budget (faster of the
    graph and sync-light implementations) and of e-GMM at 1, 2 (default) and 3 epochs."""
    g = sb.groupby(["dataset", "method"])[["fit_time", "test_nll"]].mean().unstack()
    t, nll = g["fit_time"], g["test_nll"]
    ratio, q = {}, {}
    for b in BUDGETS:
        tb = pd.concat([t[f"TorchBatchGMM_graph1_it{b}"], t[f"TorchBatchGMM_fast1_it{b}"]], axis=1).min(axis=1)
        ratio[f"it{b}"], q[f"it{b}"] = tb / t["EGMM_r0"], nll[f"TorchBatchGMM_fast1_it{b}"]
        gap = (nll[f"TorchBatchGMM_fast1_it{b}"] - nll[f"TorchBatchGMM_graph1_it{b}"]).abs().max()
        assert gap < 1e-3, f"graph and sync-light budget runs disagree by {gap} nats at {b} iterations"
    for m, key in [("EGMM_e1", "e1"), ("EGMM_r0", "e2"), ("EGMM_e3", "e3")]:
        ratio[key], q[key] = t[m] / t["EGMM_r0"], nll[m]
    return pd.DataFrame(ratio), pd.DataFrame(q)


def add_speed2_results(exp: Path, gen: Path, macros: dict, lines: list[str]) -> None:
    from experiments.analyze_results import fmt_p, fmt_p_txt, pair_stats, wtl

    sd, sh, sl = (_load(exp, f"speed2_{s}") for s in ("dev", "holdout", "large"))
    sb = _load(exp, "speed2b_large")
    if sd is None or sh is None or sl is None or sb is None:
        lines.append("\n(speed2 runs not found; speed macros skipped)")
        return
    raw = [pd.read_csv(exp / "next" / f"speed2_{x}.csv") for x in ("dev", "holdout", "large")]
    macros["SNFits"] = f"{sum(len(r) for r in raw):,}".replace(",", "{,}")
    macros["SNFail"] = str(sum(int((r["error"].fillna("") != "").sum()) for r in raw))
    macros["SbNFits"] = f"{len(pd.read_csv(exp / 'next' / 'speed2b_large.csv')):,}".replace(",", "{,}")
    lines.append("\n=== Revised paper: speed2 re-timing (every-iteration convergence checks; one session per suite) ===")
    p = lambda v: fmt_p(v).strip("$")  # noqa: E731

    # ---------------- speed against the fastest batch-EM variant
    suites = {}
    for tag, df, label in [("Dev", sd, "development"), ("Ho", sh, "holdout (post hoc)"), ("Lg", sl, "large N")]:
        dfb, best = _add_best(df)
        n_ds = df.groupby("dataset")["N_train"].first()
        spd = _time_ratio(dfb, "BatchBest")
        suites[tag] = (dfb, best, spd, n_ds)
        macros[f"Sn{tag}"] = str(len(spd))
        macros[f"S{tag}BestSpd"] = f"{spd.median():.2f}"
        macros[f"S{tag}GraphFastest"] = str(int(best.isin(["TorchBatchGMM_graph", "TorchBatchGMM_graph1"]).sum()))
        g = df.groupby(["dataset", "method"])["fit_time"].mean().unstack()
        macros[f"S{tag}GraphVsEager"] = f"{(g['TorchBatchGMM'] / g[['TorchBatchGMM_graph', 'TorchBatchGMM_graph1']].min(axis=1)).median():.2f}"
        macros[f"S{tag}CheckOne"] = f"{(g['TorchBatchGMM_graph'] / g['TorchBatchGMM_graph1']).median():.2f}"
        for sub, msk in [("Small", n_ds[spd.index] <= LARGE_N), ("Large", n_ds[spd.index] > LARGE_N)]:
            if not msk.sum():
                continue
            s = spd[msk]
            macros[f"Sn{tag}{sub}"] = str(int(msk.sum()))
            macros[f"S{tag}{sub}Spd"] = f"{s.median():.2f}"
            macros[f"S{tag}{sub}Faster"] = str(int((s > 1).sum()))
            macros[f"S{tag}{sub}OverTwo"] = str(int((s > 2).sum()))
            macros[f"S{tag}{sub}Min"] = f"{s.min():.2f}"
            macros[f"S{tag}{sub}Max"] = f"{s.max():.1f}" if s.max() < 10 else f"{s.max():.0f}"
            if msk.sum() >= 6:
                lo, hi, cov = _median_ci(s)
                macros[f"S{tag}{sub}CI"] = f"{lo:.2f}--{hi:.2f}"  # text mode
                macros[f"S{tag}{sub}Cov"] = f"{cov:.0f}"
        lines.append(f"{label}: e-GMM vs fastest batch EM: median {spd.median():.2f}x (faster on {int((spd > 1).sum())}/{len(spd)}); "
                     f"fastest variant {best.value_counts().to_dict()}; graph-5/graph-1 time {macros[f'S{tag}CheckOne']}")
    lg_spd = suites["Lg"][2]
    macros["SLgMinSpd"] = f"{lg_spd.min():.1f}"
    macros["SLgMaxSpd"] = f"{lg_spd.max():.0f}"
    lines.append(f"large N per-dataset speedups vs fastest: {lg_spd.round(2).to_dict()}")

    # Table: four strata; speed vs eager / graph (check every 5) / graph (check every iteration) / fastest; NLL vs float64 eager
    rows = []
    devb, hob, lab = suites["Dev"][0], suites["Ho"][0], suites["Lg"][0]
    n_dev, n_ho = suites["Dev"][3], suites["Ho"][3]
    groups = [(r"Development, $N_{\mathrm{train}} \le 10^5$", devb[devb["dataset"].isin(n_dev[n_dev <= LARGE_N].index)]),
              (r"Development, $N_{\mathrm{train}} > 10^5$", devb[devb["dataset"].isin(n_dev[n_dev > LARGE_N].index)]),
              (r"Holdout, $N_{\mathrm{train}} \le 10^5$ (post hoc)", hob[hob["dataset"].isin(n_ho[n_ho <= LARGE_N].index)]),
              (r"Large $N$ ($1$--$10$\,M rows)", lab)]
    for name, d in groups:
        cells = []
        for b in ["TorchBatchGMM", "TorchBatchGMM_graph", "TorchBatchGMM_graph1", "BatchBest"]:
            r = _time_ratio(d, b)
            cells.append(f"${r.median():.2f}\\times$ ({int((r > 1).sum())}/{len(r)})")
        q = pair_stats(d, "EGMM_r0", "TorchBatchGMM")
        cells.append(f"{q['W']}/{q['T']}/{q['L']}")
        cells.append(_signed(q["med_dnll"]))
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    (gen / "tab_speed_next.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # executed vs stopping iterations of the 5-iteration-check variants (logged since speed2)
    ex = pd.concat([sd, sh, sl])
    ex = ex[ex["method"].isin(["TorchBatchGMM_fast", "TorchBatchGMM_graph"])]
    extra = ex["n_iter_executed"] - ex["n_iter"] - (ex["method"] == "TorchBatchGMM_graph")
    macros["SExtraItersShare"] = f"{100 * (extra > 0).mean():.0f}"

    # ---------------- large N: speed table, quality, equal-quality and equal-pass comparisons
    g = sl.groupby(["dataset", "method"])[["fit_time", "test_nll", "dead_components"]].mean().unstack()
    t, nll = g["fit_time"], g["test_nll"]
    meta = sl.groupby("dataset")[["N_train", "D", "K", "cov_type"]].first()
    best_lg = suites["Lg"][1]
    order = meta.assign(real=[0 if d.startswith("syn") else 1 for d in meta.index]).sort_values(["real", "cov_type", "N_train"],
                                                                                                ascending=[True, False, True])
    t_rows, q_rows, dq, dq_fi, dq_p2, eq = [], [], {}, {}, {}, {}
    br, bq = budget_points(sb)
    for ds in order.index:
        n, dd, k, cov = meta.loc[ds]
        name = {"higgs": "HIGGS", "susy": "SUSY"}.get(ds, "synthetic")
        te, tb = t.loc[ds, "EGMM_r0"], t.loc[ds, best_lg[ds]]
        times = " & ".join(f"{t.loc[ds, m]:.2f}" for m in ["TorchBatchGMM", "TorchBatchGMM_graph", "TorchBatchGMM_graph1"])
        t_rows.append(f"{name} & {n / 1e6:.1f} & {dd} & {cov} & {te:.2f} & {times} & {SHORT[best_lg[ds]]} & ${tb / te:.1f}\\times$ \\\\")
        target = nll.loc[ds, "EGMM_r0"]
        dq[ds] = target - nll.loc[ds, "TorchBatchGMM"]
        dq_fi[ds] = target - nll.loc[ds, "TorchBatchGMM_fullinit"]
        dq_p2[ds] = target - bq.loc[ds, "it2"]
        # cheapest batch-EM run (fixed budget, or default stopping) whose test NLL is within EQ_TOL of e-GMM's;
        # every candidate is a time ratio to e-GMM in its own session
        cand = [(br.loc[ds, f"it{b}"], bq.loc[ds, f"it{b}"], str(b)) for b in BUDGETS]
        cand.append((tb / te, nll.loc[ds, "TorchBatchGMM"], "default"))
        ok = [c for c in cand if c[1] <= target + EQ_TOL]
        if ok:
            rr, _, lab_it = min(ok)
            eq[ds] = (rr, lab_it)
            eq_cell = f"${rr:.1f}\\times$ ({lab_it})"
        else:
            eq[ds] = (np.nan, "not reached")
            eq_cell = f"$>{tb / te:.1f}\\times$ (n.r.)"
        dead_tb = g.loc[ds, ("dead_components", "TorchBatchGMM")]
        q_rows.append(f"{name} & {n / 1e6:.1f} & {cov} & {_signed(dq[ds])} & {_signed(dq_fi[ds])} & {_signed(dq_p2[ds])} & "
                      f"${br.loc[ds, 'it2']:.1f}\\times$ & {eq_cell} & {dead_tb:.1f} \\\\")
        if not ds.startswith("syn"):
            macros[f"SLg{ds}Spd"] = f"{tb / te:.0f}" if tb / te >= 10 else f"{tb / te:.1f}"
            macros[f"SLg{ds}dNLL"] = f"{dq[ds]:+.2f}"
            macros[f"SLg{ds}dNLLFi"] = f"{dq_fi[ds]:+.2f}"
            macros[f"SLg{ds}DeadTB"] = f"{dead_tb:.1f}"
            macros[f"SLg{ds}Eq"] = "n.r." if np.isnan(eq[ds][0]) else f"{eq[ds][0]:.0f}" if eq[ds][0] >= 10 else f"{eq[ds][0]:.1f}"
    (gen / "tab_large.tex").write_text("\n".join(t_rows) + "\n", encoding="utf-8")
    (gen / "tab_large_quality.tex").write_text("\n".join(q_rows) + "\n", encoding="utf-8")

    dq, dq_fi, dq_p2 = pd.Series(dq), pd.Series(dq_fi), pd.Series(dq_p2)
    macros["SLgNBetter"], macros["SLgNEqual"] = str(int((dq < -0.1).sum())), str(int((dq.abs() < 0.01).sum()))
    macros["SLgNWorse"] = str(int((dq > 0.01).sum()))
    if (dq < -0.1).any():
        macros["SLgBetterMin"] = f"{-dq[dq < -0.1].max():.2f}"
    w, ti, l_ = wtl(dq_fi)
    macros["SLgFiWTL"], macros["SLgFidNLL"] = f"{w}/{ti}/{l_}", f"{dq_fi.median():+.2f}"
    w, ti, l_ = wtl(dq_p2)
    macros["SLgPassWTL"], macros["SLgPassdNLL"] = f"{w}/{ti}/{l_}", f"{dq_p2.median():+.2f}"
    p2_time = br["it2"]
    macros["SLgPassTime"] = f"{p2_time.median():.1f}"
    macros["SLgPassMin"] = f"{-dq_p2.max():.2f}"
    macros["SLgPassMax"] = f"{-dq_p2.min():.1f}"
    eqs = pd.Series({k: v[0] for k, v in eq.items()})
    macros["SLgEqN"] = str(int(eqs.notna().sum()))
    macros["SLgEqNotReached"] = str(int(eqs.isna().sum()))
    if eqs.notna().any():
        macros["SLgEqMin"], macros["SLgEqMax"] = f"{eqs.min():.1f}", f"{eqs.max():.1f}" if eqs.max() < 10 else f"{eqs.max():.0f}"
        macros["SLgEqMed"] = f"{eqs.dropna().median():.1f}"
    lines.append(f"large N, e-GMM minus: TorchBatchGMM {dq.round(3).to_dict()}; full-data-seeded {dq_fi.round(3).to_dict()}; "
                 f"2-iteration batch EM {dq_p2.round(3).to_dict()} (time ratio {p2_time.round(2).to_dict()})")
    e3_worse = (bq["e3"] - bq["e2"]) > 0.01
    macros["SLgEthreeWorse"] = str(int(e3_worse.sum()))
    macros["SLgEoneWorse"] = str(int(((bq["e1"] - bq["e2"]) > 0.01).sum()))
    macros["SLgEoneTime"] = f"{br['e1'].median():.2f}"
    lines.append(f"large N, e-GMM NLL at 1/3 epochs minus 2 epochs: {(bq['e1'] - bq['e2']).round(3).to_dict()} / "
                 f"{(bq['e3'] - bq['e2']).round(3).to_dict()}; time ratios e1 {br['e1'].round(2).to_dict()}")
    lines.append(f"large N, equal-quality speedup (cheapest batch run within {EQ_TOL} nats of e-GMM): {eq}")
    eg = sl[sl["method"] == "EGMM_r0"]["dead_components"]
    macros["SLgDeadEGMMMax"] = f"{sl[sl['method'] == 'EGMM_r0'].groupby('dataset')['dead_components'].mean().max():.1f}"
    del eg
    it = sl.groupby(["dataset", "method"])["n_iter"].mean().unstack()
    syn_ds = [d for d in it.index if d.startswith("syn")]
    syn_seed = sl[(sl["method"] == "TorchBatchGMM") & sl["dataset"].isin(syn_ds)]["n_iter"]
    macros["SLgTBIterSynMin"], macros["SLgTBIterSynMax"] = f"{syn_seed.min():.0f}", f"{syn_seed.max():.0f}"
    hig = sl[(sl["method"] == "TorchBatchGMM") & (sl["dataset"] == "higgs")]["n_iter"]
    macros["SLgTBIterHiggsMin"] = f"{hig.min():.0f}"
    macros["SLgTBIterSusy"] = f"{it.loc['susy', 'TorchBatchGMM']:.0f}"
    real_runs = sl[(sl["method"] == "TorchBatchGMM") & sl["dataset"].isin(["higgs", "susy"])]
    macros["SLgCapRuns"] = f"{int((real_runs['n_iter'] >= 50).sum())}"
    macros["SLgRealRuns"] = f"{len(real_runs)}"
    macros["SLgEGMMPasses"] = f"{it['EGMM_r0'].median():.0f}"
    syn = [d for d in lg_spd.index if d.startswith("syn")]
    for cov in ("full", "diag"):
        sel = [d for d in syn if meta.loc[d, "cov_type"] == cov]
        macros[f"SLgSyn{cov.capitalize()}Min"], macros[f"SLgSyn{cov.capitalize()}Max"] = f"{lg_spd[sel].min():.1f}", f"{lg_spd[sel].max():.1f}"

    # ---------------- large N: per-seed signs, best-of-3 batch EM, timing spread, per-pass cost
    full_ds = [d for d in meta.index if meta.loc[d, "cov_type"] == "full"]
    macros["SLgnFull"], macros["SLgnDiag"] = str(len(full_ds)), str(len(meta) - len(full_ds))
    ps = sl.pivot_table(index=["dataset", "seed"], columns="method", values="test_nll")
    pf = ps[ps.index.get_level_values(0).isin(full_ds)]
    macros["SLgSeedWins"] = f"{int((pf['EGMM_r0'] < pf['TorchBatchGMM'] - 1e-3).sum())}/{len(pf)}"
    def best_of_seeds(m):  # the seed with the lowest validation NLL (best of 3 starts)
        r = sl[sl["method"] == m]
        return r.loc[r.groupby("dataset")["val_nll"].idxmin()].set_index("dataset")["test_nll"]

    best3_tb, best3_eg = best_of_seeds("TorchBatchGMM"), best_of_seeds("EGMM_r0")
    d3 = nll["EGMM_r0"] - best3_tb          # single e-GMM start (3-seed mean) vs best of 3 batch-EM starts
    d33 = best3_eg - best3_tb               # best of 3 starts on both sides
    macros["SLgBestThreeWTL"] = "/".join(str(x) for x in wtl(d3))
    macros["SLgBestThreeFullWTL"] = "/".join(str(x) for x in wtl(d3[full_ds]))
    macros["SLgBothThreeWTL"] = "/".join(str(x) for x in wtl(d33))
    lines.append(f"large N, e-GMM (3-seed mean) minus the best of batch EM's 3 seeds by validation NLL: {d3.round(3).to_dict()}")
    lines.append(f"large N, best of 3 e-GMM seeds minus best of 3 batch-EM seeds (both by validation NLL): {d33.round(3).to_dict()}")
    lg_t = sl[sl["method"].isin(["EGMM_r0"] + TIMED)]
    spread = lg_t["fit_time_max"] / lg_t["fit_time_min"]
    macros["SLgSpreadShare"] = f"{100 * (spread > 1.5).mean():.0f}"
    lines.append(f"large N: timed cells with max/min repeat time > 1.5: {100 * (spread > 1.5).mean():.1f}% (max {spread.max():.1f})")
    gb = sb.groupby(["dataset", "method"])["fit_time"].mean().unstack()
    per_it = pd.concat([(gb[f"TorchBatchGMM_{v}_it20"] - gb[f"TorchBatchGMM_{v}_it10"]) / 10 for v in ("graph1", "fast1")], axis=1).min(axis=1)
    fit_iters = gb["EGMM_r0"] / per_it  # a whole e-GMM fit in units of batch EM's marginal time per iteration (same session)
    for cov in ("full", "diag"):
        sel = [d for d in fit_iters.index if meta.loc[d, "cov_type"] == cov]
        macros[f"SLgFitIters{cov.capitalize()}"] = f"{fit_iters[sel].median():.1f}"
    lines.append(f"large N: an e-GMM fit costs this many marginal batch-EM iterations: {fit_iters.round(2).to_dict()}")

    # ---------------- e-GMM float64 vs float32 training on the full-covariance development datasets
    f64 = sd[sd["method"].isin(["EGMM_r0", "EGMM_f64"])]
    if (f64["method"] == "EGMM_f64").any():
        sub = f64[f64["dataset"].isin(f64[f64["method"] == "EGMM_f64"]["dataset"].unique())]
        s = pair_stats(sub, "EGMM_r0", "EGMM_f64")
        dd = sub.groupby(["dataset", "method"])["test_nll"].mean().unstack()
        d64 = (dd["EGMM_r0"] - dd["EGMM_f64"])
        macros["SFsixtyfourN"] = str(s["n"])
        macros["SFsixtyfourWTL"] = f"{s['W']}/{s['T']}/{s['L']}"
        macros["SFsixtyfourdNLL"] = f"{s['med_dnll']:+.3f}"
        macros["SFsixtyfourP"] = p(s["p_nll"])
        macros["SFsixtyfourMaxAbs"] = f"{d64.abs().max():.2f}"
        macros["SFsixtyfourMaxName"] = d64.abs().idxmax().replace("_", r"\_")
        macros["SFsixtyfourTime"] = f"{s['med_spd']:.2f}"  # float64 time / float32 time
        lines.append(f"e-GMM float32 minus float64 training (full covariance, n={s['n']}): W/T/L {s['W']}/{s['T']}/{s['L']}, median "
                     f"{s['med_dnll']:+.4f}, raw p {fmt_p_txt(s['p_nll'])}, max |d| {d64.abs().max():.3f} ({d64.abs().idxmax()}); "
                     f"float64 time / float32 time {s['med_spd']:.2f}")

    # ---------------- GPU state during the large-N session (logged per dataset and seed, outside timed regions)
    if "gpu_sm_clock" in sl.columns and sl["gpu_sm_clock"].notna().any():
        st = sl.groupby(["dataset", "seed"])[["gpu_sm_clock", "gpu_temp"]].first()
        macros["SClockMin"], macros["SClockMax"] = f"{st['gpu_sm_clock'].min():.0f}", f"{st['gpu_sm_clock'].max():.0f}"
        macros["STempMin"], macros["STempMax"] = f"{st['gpu_temp'].min():.0f}", f"{st['gpu_temp'].max():.0f}"
        lines.append(f"large-N session GPU state: SM clock {st['gpu_sm_clock'].min():.0f}-{st['gpu_sm_clock'].max():.0f} MHz, "
                     f"temperature {st['gpu_temp'].min():.0f}-{st['gpu_temp'].max():.0f} C")

    # ---------------- cost decomposition at large N (speed2b): fixed cost + cost per iteration / per epoch
    def fit_line(x, y):
        a, b = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
        return b, a  # intercept (fixed), slope (per unit)

    c_rows, ratio_ep = [], {}
    for ds in order.index:
        n, dd, k, cov = meta.loc[ds]
        cells = []
        for v in ("graph1", "fast1"):
            f0, per = fit_line(BUDGETS, [gb.loc[ds, f"TorchBatchGMM_{v}_it{b}"] for b in BUDGETS])
            cells += [f"{f0:.2f}", f"{per:.3f}"]
            if v == "graph1":
                per_graph = per
        e0, per_ep = fit_line([1, 2, 3], [gb.loc[ds, m] for m in ("EGMM_e1", "EGMM_r0", "EGMM_e3")])
        cells += [f"{e0:.2f}", f"{per_ep:.3f}"]
        ratio_ep[ds] = per_ep / per_graph
        name = {"higgs": "HIGGS", "susy": "SUSY"}.get(ds, "synthetic")
        c_rows.append(f"{name} & {n / 1e6:.1f} & {cov} & " + " & ".join(cells) + f" & {it.loc[ds, 'TorchBatchGMM']:.1f} \\\\")
    (gen / "tab_cost.tex").write_text("\n".join(c_rows) + "\n", encoding="utf-8")
    ratio_ep = pd.Series(ratio_ep)
    for cov in ("full", "diag"):
        sel = [d for d in ratio_ep.index if meta.loc[d, "cov_type"] == cov]
        macros[f"SLgEpochPerIt{cov.capitalize()}"] = f"{ratio_ep[sel].median():.1f}"
    lines.append(f"large N: e-GMM cost per epoch / graph batch-EM cost per iteration: {ratio_ep.round(2).to_dict()}")

    # ---------------- long batch-EM runs without early stopping (speed2c; seed 0, one timed fit)
    sc = _load(exp, "speed2c_large")
    if sc is not None:
        gc_ = sc.groupby(["dataset", "method"])[["fit_time", "test_nll"]].mean().unstack()
        long_gap, long_rows = {}, {}
        for ds in gc_.index:
            e = gc_.loc[ds, ("test_nll", "EGMM_r0")]
            for it_ in (100, 500):
                long_gap[(ds, it_)] = e - gc_.loc[ds, ("test_nll", f"TorchBatchGMM_graph1_it{it_}")]
            long_rows[ds] = gc_.loc[ds, ("fit_time", "TorchBatchGMM_graph1_it500")] / gc_.loc[ds, ("fit_time", "EGMM_r0")]
        g500 = pd.Series({d: v for (d, i), v in long_gap.items() if i == 500})
        macros["SLongN"] = str(len(g500))
        macros["SLongReached"] = str(int((g500 >= -EQ_TOL).sum()))
        macros["SLongMinGap"] = f"{-g500.max():.2f}"
        macros["SLongMaxGap"] = f"{-g500.min():.2f}"
        macros["SLongTime"] = f"{pd.Series(long_rows).median():.0f}"
        if "susy" in g500.index:
            macros["SLongSusyGap"] = f"{g500['susy']:+.2f}"
        if "higgs" in g500.index:
            macros["SLongHiggsGap"] = f"{g500['higgs']:+.2f}"
        lines.append(f"long batch EM (seed 0), e-GMM minus batch EM after 100 / 500 iterations: "
                     f"{ {k: round(v, 3) for k, v in long_gap.items()} }; 500-iteration time / e-GMM time {pd.Series(long_rows).round(1).to_dict()}")

    # ---------------- e-GMM without revival at large N, and HIGGS without b-tag features (speed2d)
    sdl = _load(exp, "speed2d_large")
    if sdl is not None:
        nr = pair_stats(sdl, "EGMM_r0", "EGMM_no_revival")
        macros["SLgNoRevWTL"], macros["SLgNoRevdNLL"] = f"{nr['W']}/{nr['T']}/{nr['L']}", f"{nr['med_dnll']:+.2f}"
        gn = sdl.groupby(["dataset", "method"])["test_nll"].mean().unstack()
        dnr_tb = (gn["EGMM_no_revival"] - nll["TorchBatchGMM"].reindex(gn.index))  # NLL is deterministic per seed across sessions
        fsel = [d for d in dnr_tb.index if meta.loc[d, "cov_type"] == "full"]
        macros["SLgNoRevVsTBFullWTL"] = "/".join(str(x) for x in wtl(dnr_tb[fsel]))
        macros["SLgNoRevHiggs"] = f"{gn.loc['higgs', 'EGMM_no_revival'] - nll.loc['higgs', 'TorchBatchGMM']:+.2f}" if "higgs" in gn.index else "--"
        lines.append(f"large N: e-GMM vs e-GMM without revival {nr['W']}/{nr['T']}/{nr['L']} (median {nr['med_dnll']:+.3f}); "
                     f"no-revival minus TorchBatchGMM {dnr_tb.round(3).to_dict()}")
    shc = _load(exp, "speed2d_higgscont")
    if shc is not None and len(shc):
        gh = shc.groupby("method")[["test_nll", "fit_time", "dead_components"]].mean()
        e = gh.loc["EGMM_r0", "test_nll"]
        macros["SHcD"] = str(int(shc["D"].iloc[0]))
        macros["SHcdNLL"] = f"{e - gh.loc['TorchBatchGMM', 'test_nll']:+.2f}"
        macros["SHcdNLLFi"] = f"{e - gh.loc['TorchBatchGMM_fullinit', 'test_nll']:+.2f}"
        macros["SHcdNLLNoRev"] = f"{gh.loc['EGMM_no_revival', 'test_nll'] - gh.loc['TorchBatchGMM', 'test_nll']:+.2f}"
        macros["SHcDeadTB"] = f"{gh.loc['TorchBatchGMM', 'dead_components']:.1f}"
        macros["SHcSpd"] = f"{gh.loc['TorchBatchGMM_fast1', 'fit_time'] / gh.loc['EGMM_r0', 'fit_time']:.0f}"
        lines.append(f"HIGGS without b-tags: {gh.round(3).to_dict()}")

"""Single source of truth for every statistic and table in the paper.

Reads ``docs/experiments/*.csv`` and writes
  * ``docs/experiments/stats_summary.txt``      (human-readable statistics)
  * ``docs/experiments/small_dataset_stats.txt``
  * ``paper/generated/*.tex``                   (LaTeX table bodies + ``numbers.tex`` macros)

Conventions (identical everywhere): per-(dataset, method) means over seeds; W/T/L on dataset means with a
1e-3 nat tie band; two-sided Wilcoxon signed-rank (``zero_method="zsplit"``) on dataset means; speedup =
median over datasets of ``time_base / time_method``; seed dispersion = sample std (ddof=1) over seeds.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "docs" / "experiments"
GEN = ROOT / "paper" / "generated"

PRIMARY = [
    "EGMM",
    "TorchBatchGMM",
    "OnlineGMM_tuned",
    "NaiveMinibatchGMM",
    "EMeansInitGMM",
    "StandardizedSklearnGMM",
    "SklearnGMM",
]
TEX_NAME = {
    "EGMM": r"\egmm{}",
    "TorchBatchGMM": r"\texttt{TorchBatchGMM}",
    "OnlineGMM_tuned": r"\texttt{OnlineGMM\_tuned}",
    "NaiveMinibatchGMM": r"\texttt{NaiveMinibatchGMM}",
    "EMeansInitGMM": r"\texttt{EMeansInitGMM}",
    "StandardizedSklearnGMM": r"\texttt{StdSklearnGMM} (CPU)",
    "SklearnGMM": r"\texttt{SklearnGMM} (Raw CPU)",
    "EGMM_polish0": r"\texttt{EGMM\_polish0}",
    "TorchBatchGMM_fullinit": r"\texttt{TorchBatch\_fullinit}",
    "EGMM_no_init_scale": r"\texttt{EGMM\_no\_init\_scale}",
    "EGMM_no_revival": r"\texttt{EGMM\_no\_revival}",
    "EGMM_bracket_law": r"\texttt{EGMM\_bracket\_law}",
    "OnlineGMM_matched_polish": r"\texttt{OnlineGMM\_matched\_pol}",
    "EGMM_eager": r"\texttt{EGMM\_eager}",
}
TIE = 1e-3
LARGE_N = 100_000
EXTRA_BASELINES = ["TorchBatchGMM_fullinit"]  # sensitivity baselines (not part of the Friedman set)
# The 10 primary datasets added after the bracket/polish rules were frozen on the other 23 during
# development (they never informed a design decision of the e-GMM recipe before this revision).
HELDOUT_DEV = [
    "jannis",
    "numerai28.6",
    "higgs_98k",
    "philippine",
    "gas_drift",
    "har",
    "skin_nonskin",
    "creditcard",
    "syn_scale300k_k64_full",
    "syn_scale400k_d256_diag",
]


# ----------------------------------------------------------------------------- helpers
def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    if p >= 0.1:
        return f"${p:.3f}$"
    e = int(math.floor(math.log10(p)))
    m = p / 10**e
    if m >= 9.995:
        m, e = 1.0, e + 1
    return f"${m:.2f}\\times 10^{{{e}}}$"


def fmt_p_txt(p: float) -> str:
    return f"{p:.4g}"


def wtl(diff: pd.Series) -> tuple[int, int, int]:
    return int((diff < -TIE).sum()), int((diff.abs() <= TIE).sum()), int((diff > TIE).sum())


def wilcoxon_p(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or np.allclose(a.values, b.values):
        return float("nan")
    return float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)


def pair_stats(df: pd.DataFrame, a: str, b: str, datasets=None) -> dict:
    """Paired comparison of method a vs method b (a first: dNLL < 0 and speedup > 1 favour a)."""
    agg = df.groupby(["dataset", "method"])[["test_nll", "fit_time"]].mean().reset_index()
    pn = agg.pivot(index="dataset", columns="method", values="test_nll")
    pt = agg.pivot(index="dataset", columns="method", values="fit_time")
    pair = pn[[a, b]].dropna()
    if datasets is not None:
        pair = pair.loc[pair.index.intersection(datasets)]
    tt = pt.loc[pair.index, [a, b]]
    d = pair[a] - pair[b]
    w, t, l = wtl(d)
    # seed-level (dataset, seed) paired outcomes
    ps = df.pivot_table(index=["dataset", "seed"], columns="method", values="test_nll")
    ps = ps[[a, b]].dropna()
    ps = ps[ps.index.get_level_values(0).isin(pair.index)]
    sw, st, sl = wtl(ps[a] - ps[b])
    pts = df.pivot_table(index=["dataset", "seed"], columns="method", values="fit_time")[[a, b]].dropna()
    pts = pts[pts.index.get_level_values(0).isin(pair.index)]
    seed_spd = (pts[b] / pts[a]).groupby(level=0)
    n_faster_all = int((seed_spd.min() > 1.0).sum())
    n_slower_all = int((seed_spd.max() < 1.0).sum())
    return {
        "n_faster_all": n_faster_all,
        "n_slower_all": n_slower_all,
        "n": len(pair),
        "W": w,
        "T": t,
        "L": l,
        "med_dnll": float(d.median()),
        "q1_dnll": float(d.quantile(0.25)),
        "q3_dnll": float(d.quantile(0.75)),
        "p_nll": wilcoxon_p(pair[a], pair[b]),
        "med_spd": float((tt[b] / tt[a]).median()),
        "max_spd": float((tt[b] / tt[a]).max()),
        "n_faster": int((tt[b] > tt[a]).sum()),
        "p_time": wilcoxon_p(tt[a], tt[b]),
        "seed_W": sw,
        "seed_T": st,
        "seed_L": sl,
        "seed_n": len(ps),
    }


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaNs are left out of the family and returned as NaN)."""
    idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    m = len(idx)
    out = [float("nan")] * len(pvals)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: pvals[j])):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i] = running
    return out


def apply_holm(rows: list) -> list:
    """Adjust p_nll and p_time separately within one family of (a, b, stats) rows (keeps raw copies)."""
    for key in ("p_nll", "p_time"):
        raw = [r[2][key] for r in rows]
        adj = holm_adjust(raw)
        for r, pr, pa in zip(rows, raw, adj):
            r[2][key + "_raw"] = pr
            r[2][key] = pa
    return rows


def data_passes(df: pd.DataFrame) -> pd.Series:
    """Hardware-independent cost per run: full-data passes (E-step evaluations of N points).

    EGMM-family: T_steps * B / N EMA passes + executed polish steps; batch EM: EM iterations;
    online EM: epochs (+ polish); EMeansInitGMM: 2 k-means epochs + EM iterations; scikit-learn: EM
    iterations only (its KMeans initialization is not counted, so its cost is a lower bound).
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from egmm.scaling import auto_hyperparameters

    out = []
    for r in df.itertuples():
        m, n_it = r.method, float(r.n_iter)
        if m in ("EGMM", "EGMM_eager", "EGMM_polish0", "EGMM_no_init_scale", "EGMM_no_revival", "EGMM_bracket_law", "NaiveMinibatchGMM"):
            law = "bracket" if m == "EGMM_bracket_law" else "smooth"
            hp = auto_hyperparameters(int(r.N_train), int(r.K), int(r.D), r.cov_type, schedule_law=law,
                                      polish_steps=0 if m in ("EGMM_polish0", "NaiveMinibatchGMM") else "auto")
            out.append(hp.n_steps * hp.batch_size / float(r.N_train) + max(0.0, n_it - hp.max_epochs))
        elif m == "EMeansInitGMM":
            out.append(2.0 + n_it)
        else:
            out.append(n_it)
    return pd.Series(out, index=df.index)


def bold_if(val: str, cond: bool) -> str:
    return f"\\textbf{{{val}}}" if cond else val


# ----------------------------------------------------------------------------- main analysis
def main(out_exp: Path | None = None, out_gen: Path | None = None, in_exp: Path | None = None) -> None:
    global GEN, EXP
    EXP = in_exp or EXP
    out_exp = out_exp or EXP
    GEN = out_gen or GEN
    GEN.mkdir(parents=True, exist_ok=True)
    out_exp.mkdir(parents=True, exist_ok=True)
    h2h = pd.read_csv(EXP / "benchmark_head_to_head.csv")
    abl = pd.read_csv(EXP / "benchmark_ablations.csv")
    phase = pd.read_csv(EXP / "benchmark_phase_breakdown.csv")
    small = pd.read_csv(EXP / "small_dataset_eval.csv")
    pareto = pd.read_csv(EXP / "benchmark_pareto.csv") if (EXP / "benchmark_pareto.csv").exists() else None

    macros: dict[str, str] = {}
    lines: list[str] = []
    datasets = sorted(h2h["dataset"].unique())
    n_ds = len(datasets)
    large = sorted(h2h.loc[h2h["N_train"] > LARGE_N, "dataset"].unique())
    seeds = sorted(h2h["seed"].unique())
    macros["nDatasets"] = str(n_ds)
    macros["nLarge"] = str(len(large))
    macros["nSeeds"] = str(len(seeds))

    # completeness check: every primary method on every dataset and seed
    cnt = h2h.groupby(["dataset", "method"]).size().unstack()
    missing = cnt[PRIMARY].isna().sum().sum() + (cnt[PRIMARY] != len(seeds)).sum().sum()
    lines.append(f"Runs: {len(h2h)} rows, {n_ds} datasets x {len(seeds)} seeds; incomplete (dataset, method) cells: {int(missing)}")

    # ---------------- Panel A / B
    def panel(df, first, bases, ds_subset, key_prefix, title):
        out = [(first, b, pair_stats(df, first, b, ds_subset)) for b in bases if b in df["method"].unique()]
        apply_holm(out)
        lines.append(f"\n=== {title} (p-values Holm-adjusted within this panel; raw in brackets) ===")
        for _, b, s in out:
            lines.append(
                f"{first} vs {b:24s} (n={s['n']:2d}): W/T/L={s['W']}/{s['T']}/{s['L']}, med_dNLL={s['med_dnll']:+.4f} "
                f"[IQR {s['q1_dnll']:+.3f},{s['q3_dnll']:+.3f}], p_nll={fmt_p_txt(s['p_nll'])} [{fmt_p_txt(s['p_nll_raw'])}] | med_spd={s['med_spd']:.2f}x "
                f"(max {s['max_spd']:.2f}x, faster on {s['n_faster']}/{s['n']}; on all seeds {s['n_faster_all']}, slower on all seeds {s['n_slower_all']}), p_time={fmt_p_txt(s['p_time'])} [{fmt_p_txt(s['p_time_raw'])}] | "
                f"seed-level W/T/L={s['seed_W']}/{s['seed_T']}/{s['seed_L']} of {s['seed_n']}"
            )
            tag = key_prefix + "".join(ch for ch in b if ch.isalpha())
            macros[tag + "WTL"] = f"{s['W']}/{s['T']}/{s['L']}"
            macros[tag + "W"] = str(s["W"])
            macros[tag + "dNLL"] = f"{s['med_dnll']:+.4f}".replace("+", "+")
            macros[tag + "pNLL"] = fmt_p(s["p_nll"]).strip("$")
            macros[tag + "pNLLraw"] = fmt_p(s["p_nll_raw"]).strip("$")
            macros[tag + "Spd"] = f"{s['med_spd']:.2f}"
            macros[tag + "MaxSpd"] = f"{s['max_spd']:.2f}"
            macros[tag + "Faster"] = str(s["n_faster"])
            macros[tag + "FasterAll"] = str(s["n_faster_all"])
            macros[tag + "SlowerAll"] = str(s["n_slower_all"])
            macros[tag + "pTime"] = fmt_p(s["p_time"]).strip("$")
            macros[tag + "SeedWTL"] = f"{s['seed_W']}/{s['seed_T']}/{s['seed_L']}"
        return out

    bases = [m for m in PRIMARY if m != "EGMM"] + [m for m in EXTRA_BASELINES if m in set(h2h["method"])]
    panel_a = panel(h2h, "EGMM", bases, None, "A", f"All {n_ds} primary datasets (3-seed means)")
    panel_b = panel(h2h, "EGMM", bases, large, "B", f"Large-scale subset (N_train > 1e5, n={len(large)})")
    held = [d for d in HELDOUT_DEV if d in set(datasets)]
    panel_h = panel(
        h2h, "EGMM", ["TorchBatchGMM", "OnlineGMM_tuned", "StandardizedSklearnGMM"], held, "H",
        f"Held-out-from-development subset (n={len(held)})",
    )
    macros["nHeld"] = str(len(held))

    # Friedman / Nemenyi on the primary methods
    agg = h2h.groupby(["dataset", "method"])[["test_nll", "fit_time", "dead_components", "ari", "nmi", "n_iter", "revivals"]].mean().reset_index()
    pn = agg.pivot(index="dataset", columns="method", values="test_nll")
    comp = pn[PRIMARY].dropna()
    ranks = comp.rank(axis=1, ascending=True).mean(axis=0).sort_values()
    f_stat, f_p = stats.friedmanchisquare(*[comp[c].values for c in PRIMARY])
    k_m, n_d = len(PRIMARY), len(comp)
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}
    cd = q_table[k_m] * math.sqrt(k_m * (k_m + 1) / (6.0 * n_d))
    lines.append(f"\nFriedman (n={n_d}, k={k_m}): chi2={f_stat:.3f}, p={fmt_p_txt(f_p)}, Nemenyi CD(0.05)={cd:.3f}")
    lines.append("Mean NLL ranks (1=best): " + ", ".join(f"{m}={r:.3f}" for m, r in ranks.items()))
    macros["FriedmanChi"] = f"{f_stat:.2f}"
    macros["FriedmanP"] = fmt_p(f_p).strip("$")
    macros["NemenyiCD"] = f"{cd:.3f}"
    for m, r in ranks.items():
        macros["Rank" + "".join(ch for ch in m if ch.isalpha())] = f"{r:.3f}"
    macros["RankEGMMplusCD"] = f"{ranks['EGMM'] + cd:.3f}"
    macros["RankFirst"] = ranks.index[0].replace("_", r"\_")

    # dead components / ARI / NMI / iterations
    lines.append("\nMean dead components (pi_k < 0.05/K), revivals, ARI, NMI, iterations per method:")
    pdead = agg.pivot(index="dataset", columns="method", values="dead_components")
    pari = agg.pivot(index="dataset", columns="method", values="ari")
    pnmi = agg.pivot(index="dataset", columns="method", values="nmi")
    pit = agg.pivot(index="dataset", columns="method", values="n_iter")
    prev = agg.pivot(index="dataset", columns="method", values="revivals")
    for m in PRIMARY + (["EGMM_polish0"] if "EGMM_polish0" in pn.columns else []):
        lines.append(
            f"  {m:24s}: dead={pdead[m].mean():.2f}  ARI={pari[m].mean():.4f}  NMI={pnmi[m].mean():.4f}  "
            f"iters mean={pit[m].mean():.1f} median={pit[m].median():.1f}  revivals={prev[m].mean():.2f}"
        )
        t = "".join(ch for ch in m if ch.isalpha())
        macros["Dead" + t] = f"{pdead[m].mean():.2f}"
        macros["ARI" + t] = f"{pari[m].mean():.4f}"
        macros["NMI" + t] = f"{pnmi[m].mean():.4f}"
        macros["ItMean" + t] = f"{pit[m].mean():.1f}"
        macros["ItMed" + t] = f"{pit[m].median():.1f}"
    macros["RevivalsEGMM"] = f"{prev['EGMM'].mean():.2f}"
    ari_rank = pari[PRIMARY].mean().sort_values(ascending=False)
    macros["ARIRankEGMM"] = str(int(list(ari_rank.index).index("EGMM")) + 1)
    macros["ARIBestName"] = TEX_NAME[ari_rank.index[0]].replace(" (CPU)", "").replace(" (Raw CPU)", "")
    macros["ARIBest"] = f"{ari_rank.iloc[0]:.4f}"
    lines.append("Mean ARI ranking: " + ", ".join(f"{m}={v:.4f}" for m, v in ari_rank.items()))
    ptime = agg.pivot(index="dataset", columns="method", values="fit_time")
    shown = ptime[["EGMM", "TorchBatchGMM", "StandardizedSklearnGMM"]].min(axis=1)
    for m, tag in [("NaiveMinibatchGMM", "Naive"), ("OnlineGMM_tuned", "Online"), ("EMeansInitGMM", "EMeans")]:
        if m in ptime.columns:
            k = int((ptime[m] < shown).sum())
            macros["Unshown" + tag + "Faster"] = str(k)
            lines.append(f"{m} (not shown in Table 2) is faster than every displayed time on {k}/{len(ptime)} datasets")
    macros["nLabeled"] = str(int(pari["EGMM"].notna().sum()))

    # ---------------- Panel D (ablations; same-run comparisons only; one Holm family)
    abl_rows = []
    for v in ["EGMM_no_revival", "OnlineGMM_matched_polish", "EGMM_polish0", "EGMM_no_init_scale", "EGMM_bracket_law"]:
        if v in abl["method"].unique():
            abl_rows.append(("EGMM", v, pair_stats(abl, "EGMM", v)))
    if "EGMM_polish0" in h2h["method"].unique():  # polish0 vs baselines: from the head-to-head run
        for b in ["TorchBatchGMM", "OnlineGMM_tuned"]:
            abl_rows.append(("EGMM_polish0", b, pair_stats(h2h, "EGMM_polish0", b)))
    apply_holm(abl_rows)
    lines.append("\n=== Component & fairness ablations (same-run references; Holm-adjusted p, raw in brackets) ===")
    ad = abl.groupby(["dataset", "method"])["dead_components"].mean().unstack()
    macros["AblDeadEGMM"] = f"{ad['EGMM'].mean():.2f}"
    for a, v, s in abl_rows:
        lines.append(
            f"{a} vs {v:26s} (n={s['n']:2d}): W/T/L={s['W']}/{s['T']}/{s['L']}, med_dNLL={s['med_dnll']:+.4f}, "
            f"p={fmt_p_txt(s['p_nll'])} [{fmt_p_txt(s['p_nll_raw'])}] | time ratio med={s['med_spd']:.2f}x "
            f"p_time={fmt_p_txt(s['p_time'])} [{fmt_p_txt(s['p_time_raw'])}]"
            + (f" | mean_dead={ad['EGMM'].mean():.2f} vs {ad[v].mean():.2f}" if a == "EGMM" else "")
        )
        t = ("C" if a == "EGMM" else "P") + "".join(ch for ch in v if ch.isalpha())
        macros[t + "WTL"] = f"{s['W']}/{s['T']}/{s['L']}"
        macros[t + "W"] = str(s["W"])
        macros[t + "dNLL"] = f"{s['med_dnll']:+.4f}"
        macros[t + "pNLL"] = fmt_p(s["p_nll"]).strip("$")
        macros[t + "pNLLraw"] = fmt_p(s["p_nll_raw"]).strip("$")
        macros[t + "Spd"] = f"{s['med_spd']:.2f}"
        macros[t + "nonTied"] = str(s["W"] + s["L"])
        if a == "EGMM":
            macros["AblDead" + "".join(ch for ch in v if ch.isalpha())] = f"{ad[v].mean():.2f}"
    pa = abl.groupby(["dataset", "method"])[["ari", "nmi"]].mean().reset_index()
    for v in ["EGMM", "EGMM_no_revival"]:
        sub = pa[pa["method"] == v]
        macros["AblARI" + "".join(ch for ch in v if ch.isalpha())] = f"{sub['ari'].mean():.4f}"
        macros["AblNMI" + "".join(ch for ch in v if ch.isalpha())] = f"{sub['nmi'].mean():.4f}"

    # ---------------- Panel E (systems: algorithm vs engineering, one harness pass, one Holm family)
    sys_rows = []
    if (EXP / "benchmark_systems.csv").exists():
        sysdf = pd.read_csv(EXP / "benchmark_systems.csv")
        sys_rows = [(a, b, pair_stats(sysdf, a, b)) for a, b in
                    [("EGMM", "EGMM_eager"), ("EGMM_eager", "TorchBatchGMM"), ("EGMM", "TorchBatchGMM")]]
        apply_holm(sys_rows)
        sysdf = sysdf.assign(passes=data_passes(sysdf))
        lines.append("\n=== Systems ablation (benchmark_systems.csv; Holm-adjusted p, raw in brackets) ===")
        for a, b, st in sys_rows:
            lines.append(
                f"{a} vs {b:16s} (n={st['n']}): W/T/L={st['W']}/{st['T']}/{st['L']}, med_dNLL={st['med_dnll']:+.4f}, "
                f"med_spd={st['med_spd']:.2f}x (faster on {st['n_faster']}; all seeds {st['n_faster_all']}), "
                f"p_time={fmt_p_txt(st['p_time'])} [{fmt_p_txt(st['p_time_raw'])}]"
            )
            t = "E" + "".join(ch for ch in a + "vs" + b if ch.isalpha())
            macros[t + "WTL"] = f"{st['W']}/{st['T']}/{st['L']}"
            macros[t + "Spd"] = f"{st['med_spd']:.2f}"
            macros[t + "Faster"] = str(st["n_faster"])
            macros[t + "FasterAll"] = str(st["n_faster_all"])
            macros[t + "pTime"] = fmt_p(st["p_time"]).strip("$")
            macros[t + "dNLL"] = f"{st['med_dnll']:+.4f}"
        pm = sysdf.groupby(["dataset", "method"])["passes"].mean().unstack()
        ratio = (pm["TorchBatchGMM"] / pm["EGMM"])
        macros["EPassRatio"] = f"{ratio.median():.1f}"
        macros["EPassEGMM"] = f"{pm['EGMM'].median():.1f}"
        macros["EPassTB"] = f"{pm['TorchBatchGMM'].median():.1f}"
        tg = sysdf.groupby(["dataset", "method"])["fit_time"].mean().unstack()
        tpp = (tg["TorchBatchGMM"] / pm["TorchBatchGMM"]) / (tg["EGMM_eager"] / pm["EGMM"])
        macros["ETimePerPassRatio"] = f"{tpp.median():.2f}"
        lines.append(
            f"  median passes EGMM={pm['EGMM'].median():.1f} vs TorchBatch={pm['TorchBatchGMM'].median():.1f} "
            f"(median ratio {ratio.median():.1f}x); time-per-pass TB / EGMM_eager median {tpp.median():.2f}"
        )
        macros["nSys"] = str(sysdf["dataset"].nunique())

    # ---------------- Panel F (pre-registered fresh holdout suite, one Holm family)
    ho_rows = []
    if (EXP / "holdout_head_to_head.csv").exists():
        ho = pd.read_csv(EXP / "holdout_head_to_head.csv")
        ho_bases = [m for m in PRIMARY if m != "EGMM"] + [m for m in EXTRA_BASELINES if m in set(ho["method"])]
        ho_rows = [("EGMM", b, pair_stats(ho, "EGMM", b)) for b in ho_bases if b in set(ho["method"])]
        apply_holm(ho_rows)
        n_ho = ho["dataset"].nunique()
        macros["nHoldout"] = str(n_ho)
        macros["HoldoutNmin"] = f"{int(ho['N_train'].min()):,}".replace(",", "{,}")
        macros["HoldoutNmax"] = f"{int(ho['N_train'].max()):,}".replace(",", "{,}")
        lines.append(f"\n=== Pre-registered fresh holdout (n={n_ho}; Holm-adjusted p, raw in brackets) ===")
        for _, b, st in ho_rows:
            lines.append(
                f"EGMM vs {b:24s} (n={st['n']}): W/T/L={st['W']}/{st['T']}/{st['L']}, med_dNLL={st['med_dnll']:+.4f}, "
                f"p_nll={fmt_p_txt(st['p_nll'])} [{fmt_p_txt(st['p_nll_raw'])}] | med_spd={st['med_spd']:.2f}x "
                f"(faster on {st['n_faster']}, all seeds {st['n_faster_all']}), p_time={fmt_p_txt(st['p_time'])} | "
                f"seed W/T/L={st['seed_W']}/{st['seed_T']}/{st['seed_L']}"
            )
            t = "F" + "".join(ch for ch in b if ch.isalpha())
            macros[t + "WTL"] = f"{st['W']}/{st['T']}/{st['L']}"
            macros[t + "W"] = str(st["W"])
            macros[t + "dNLL"] = f"{st['med_dnll']:+.4f}"
            macros[t + "pNLL"] = fmt_p(st["p_nll"]).strip("$")
            macros[t + "Spd"] = f"{st['med_spd']:.2f}"
            macros[t + "Faster"] = str(st["n_faster"])
            macros[t + "pTime"] = fmt_p(st["p_time"]).strip("$")
        hpn = ho.groupby(["dataset", "method"])["test_nll"].mean().unstack()
        hcomp = hpn[[m for m in PRIMARY if m in hpn.columns]].dropna()
        if len(hcomp) >= 3 and hcomp.shape[1] >= 3:
            hr = hcomp.rank(axis=1).mean().sort_values()
            hf, hfp = stats.friedmanchisquare(*[hcomp[c].values for c in hcomp.columns])
            macros["FHoldChi"], macros["FHoldP"] = f"{hf:.2f}", fmt_p(hfp).strip("$")
            macros["FHoldRankEGMM"] = f"{hr['EGMM']:.2f}"
            macros["FHoldRankPos"] = str(int(list(hr.index).index("EGMM")) + 1)
            macros["FHoldFirst"] = TEX_NAME[hr.index[0]].replace(" (CPU)", "").replace(" (Raw CPU)", "")
            lines.append(f"Holdout Friedman chi2={hf:.2f}, p={fmt_p_txt(hfp)}; ranks: " + ", ".join(f"{m}={v:.2f}" for m, v in hr.items()))
        hmeta = ho.groupby("dataset")[["N_train", "D", "K", "cov_type"]].first()
        hg = ho.groupby(["dataset", "method"])[["test_nll", "fit_time"]].mean()
        hs = ho.groupby(["dataset", "method"])["test_nll"].std(ddof=1)
        t_ho = []
        for dsn in sorted(hmeta.index, key=lambda x: int(hmeta.loc[x, "N_train"])):
            N, D, K, cov = hmeta.loc[dsn]
            cols = [c for c in ["EGMM", "TorchBatchGMM", "TorchBatchGMM_fullinit", "OnlineGMM_tuned", "StandardizedSklearnGMM", "SklearnGMM"] if (dsn, c) in hg.index]
            nl = {c: hg.loc[(dsn, c), "test_nll"] for c in cols}
            best = min(round(v, 3) for v in nl.values())
            cells = [bold_if(f"{nl[c]:.3f}", round(nl[c], 3) == best) + (f"{{\\tiny$\\pm${hs.loc[(dsn, c)]:.2f}}}" if c in ("EGMM", "TorchBatchGMM", "StandardizedSklearnGMM") else "") for c in cols]
            te = hg.loc[(dsn, "EGMM"), "fit_time"]
            cells.append(f"{te * 1e3:.1f}")
            cells.append(f"{hg.loc[(dsn, 'TorchBatchGMM'), 'fit_time'] / te:.2f}$\\times$")
            cells.append(f"{hg.loc[(dsn, 'StandardizedSklearnGMM'), 'fit_time'] / te:.1f}$\\times$")
            t_ho.append(f"\\texttt{{{dsn.replace('_', chr(92) + '_')}}} ({int(N)}, {int(D)}, {int(K)}, {cov}) & " + " & ".join(cells) + " \\\\")
        (GEN / "tab_holdout.tex").write_text("\n".join(t_ho) + "\n", encoding="utf-8")

    # ---------------- Table 1 LaTeX body
    def t1_row(a, b, s, short=None):
        name = f"{TEX_NAME.get(a, a)} vs {short or TEX_NAME.get(b, b)}"
        good_n = s["W"] > s["L"]
        good_t = s["med_spd"] > 1.0
        wtl_s = bold_if(f"{s['W']}/{s['T']}/{s['L']}", good_n)
        spd = bold_if(f"{s['med_spd']:.2f}$\\times$", good_t)
        return f"{name} & {wtl_s} & ${s['med_dnll']:+.4f}$ & {fmt_p(s['p_nll'])} & {spd} & {fmt_p(s['p_time'])} & {s['seed_W']}/{s['seed_T']}/{s['seed_L']} \\\\"

    t1 = [
        f"\\multicolumn{{7}}{{l}}{{\\emph{{A. All {n_ds} primary datasets (3-seed means; Friedman $\\chi^2_F = {f_stat:.2f}$, $p = {fmt_p(f_p).strip('$')}$)}}}} \\\\"
    ]
    t1 += [t1_row(a, b, s) for a, b, s in panel_a]
    t1.append("\\midrule")
    t1.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{B. Large-scale subset ($N_{{\\mathrm{{train}}}} > 10^5$, $n = {len(large)}$)}}}} \\\\")
    t1 += [t1_row(a, b, s) for a, b, s in panel_b]
    t1.append("\\midrule")
    t1.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{C. The {len(held)} primary datasets never used during development}}}} \\\\")
    t1 += [t1_row(a, b, s) for a, b, s in panel_h]
    t1.append("\\midrule")
    t1.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{D. Component \\& fairness ablations across all {n_ds} datasets}}}} \\\\")
    short = {"TorchBatchGMM": r"\texttt{TorchBatch}", "OnlineGMM_tuned": r"\texttt{OnlineGMM}"}
    t1 += [t1_row(a, b, s, short.get(b) if a == "EGMM_polish0" else None) for a, b, s in abl_rows]
    if ho_rows:
        t1.append("\\midrule")
        t1.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{F. Pre-registered fresh holdout ({macros['nHoldout']} new OpenML datasets, code frozen before download)}}}} \\\\")
        t1 += [t1_row(a, b, st) for a, b, st in ho_rows]
    if sys_rows:
        t1.append("\\midrule")
        t1.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{E. Algorithm vs.\\ engineering (separate systems run, {macros['nSys']} datasets)}}}} \\\\")
        t1 += [t1_row(a, b, st) for a, b, st in sys_rows]
    (GEN / "tab_wilcoxon_main.tex").write_text("\n".join(t1) + "\n", encoding="utf-8")

    # ---------------- Table 2 (33 datasets, mean +- std)
    cols_nll = ["EGMM", "EGMM_polish0", "TorchBatchGMM", "OnlineGMM_tuned", "StandardizedSklearnGMM", "SklearnGMM"]
    cols_nll = [c for c in cols_nll if c in pn.columns]
    g_mean = h2h.groupby(["dataset", "method"])[["test_nll", "fit_time", "dead_components"]].mean()
    g_std = h2h.groupby(["dataset", "method"])[["test_nll", "fit_time"]].std(ddof=1)
    meta = h2h.groupby("dataset")[["N_train", "D", "K", "cov_type"]].first()
    t2 = []
    for ds in datasets:
        N, D, K, cov = meta.loc[ds]
        nll = {m: g_mean.loc[(ds, m), "test_nll"] for m in cols_nll}
        sd = {m: g_std.loc[(ds, m), "test_nll"] for m in cols_nll}
        tm = {m: g_mean.loc[(ds, m), "fit_time"] * 1e3 for m in ["EGMM", "TorchBatchGMM", "StandardizedSklearnGMM"]}
        best_n = min(round(v, 3) for v in nll.values())
        best_t = min(round(v, 1) for v in tm.values())
        cells = []
        for m in cols_nll:
            v = f"{nll[m]:.3f}"
            s = f"{{\\tiny$\\pm${sd[m]:.2f}}}" if m in ("EGMM", "TorchBatchGMM", "StandardizedSklearnGMM") else ""
            cells.append(bold_if(v, round(nll[m], 3) == best_n) + s)
        for m in ["EGMM", "TorchBatchGMM", "StandardizedSklearnGMM"]:
            cells.append(bold_if(f"{tm[m]:.1f}", round(tm[m], 1) == best_t))
        cells.append(f"{tm['TorchBatchGMM'] / tm['EGMM']:.2f}$\\times$")
        cells.append(f"{tm['StandardizedSklearnGMM'] / tm['EGMM']:.1f}$\\times$")
        de = g_mean.loc[(ds, "EGMM"), "dead_components"]
        dt = g_mean.loc[(ds, "TorchBatchGMM"), "dead_components"]
        cells.append(f"{de:.2f} / {dt:.2f}")
        name = ds.replace("_", r"\_")
        t2.append(f"\\texttt{{{name}}} ({int(N)}, {int(D)}, {int(K)}, {cov}) & " + " & ".join(cells) + " \\\\")
    (GEN / "tab_full33.tex").write_text("\n".join(t2) + "\n", encoding="utf-8")

    # per-dataset speedup extremes for prose
    pt = agg.pivot(index="dataset", columns="method", values="fit_time")
    spd_tb = (pt["TorchBatchGMM"] / pt["EGMM"]).sort_values(ascending=False)
    spd_sk = (pt["StandardizedSklearnGMM"] / pt["EGMM"]).sort_values(ascending=False)
    for i in range(3):
        macros[f"TopTBName{'abc'[i]}"] = spd_tb.index[i].replace("_", r"\_")
        macros[f"TopTBSpd{'abc'[i]}"] = f"{spd_tb.iloc[i]:.2f}"
    macros["TopSKName"] = spd_sk.index[0].replace("_", r"\_")
    macros["TopSKSpd"] = f"{spd_sk.iloc[0]:.1f}"
    slower_tb = spd_tb[spd_tb < 1.0]
    macros["SlowerTBList"] = ", ".join(f"\\texttt{{{d.replace('_', chr(92) + '_')}}} (${v:.2f}\\times$)" for d, v in slower_tb.items()) or "none"
    lines.append("\nSpeedup vs TorchBatch (top): " + ", ".join(f"{d}={v:.2f}x" for d, v in spd_tb.head(5).items()))
    lines.append("Datasets where EGMM is slower than TorchBatch: " + (", ".join(f"{d}={v:.2f}x" for d, v in slower_tb.items()) or "none"))

    # ---------------- Dispersion table
    lines.append("\n=== Seed dispersion (per-dataset sample std over seeds; medians across datasets) ===")
    disp = []
    h2h = h2h.assign(passes=data_passes(h2h))
    med_pass = h2h.groupby(["dataset", "method"])["passes"].mean().groupby("method").median()
    for m in list(med_pass.index):
        macros["Passes" + "".join(ch for ch in m if ch.isalpha())] = f"{med_pass[m]:.1f}"
    lines.append("Median full-data passes per run: " + ", ".join(f"{m}={v:.1f}" for m, v in med_pass.sort_values().items()))
    for m in PRIMARY:
        sd_nll = g_std.xs(m, level="method")["test_nll"]
        cv_t = (g_std.xs(m, level="method")["fit_time"] / g_mean.xs(m, level="method")["fit_time"])
        row = {"method": m, "med_std_nll": float(sd_nll.median()), "q3_std_nll": float(sd_nll.quantile(0.75)),
               "max_std_nll": float(sd_nll.max()), "med_cv_time": float(cv_t.median()),
               "med_passes": float(med_pass.get(m, float("nan")))}
        if m != "EGMM":
            s = pair_stats(h2h, "EGMM", m)
            row.update({"seed_wtl": f"{s['seed_W']}/{s['seed_T']}/{s['seed_L']}", "iqr": f"[{s['q1_dnll']:+.3f}, {s['q3_dnll']:+.3f}]"})
        else:
            row.update({"seed_wtl": "--", "iqr": "--"})
        disp.append(row)
        lines.append(
            f"  {m:24s}: median std(NLL)={row['med_std_nll']:.3f} (Q3 {row['q3_std_nll']:.3f}, max {row['max_std_nll']:.3f}), "
            f"median CV(time)={row['med_cv_time']:.3f}, EGMM seed-level W/T/L={row['seed_wtl']}, dNLL IQR={row['iqr']}"
        )
    t_disp = []
    for r in disp:
        name = TEX_NAME[r["method"]].replace(" (CPU)", "").replace(" (Raw CPU)", "")
        t_disp.append(
            f"{name} & {r['med_passes']:.1f} & {r['med_std_nll']:.3f} & {r['q3_std_nll']:.3f} & {r['med_cv_time']:.3f} & {r['seed_wtl']} & {r['iqr']} \\\\"
        )
    (GEN / "tab_dispersion.tex").write_text("\n".join(t_disp) + "\n", encoding="utf-8")
    macros["MedStdEGMM"] = f"{disp[0]['med_std_nll']:.3f}"
    macros["MedStdTB"] = f"{disp[1]['med_std_nll']:.3f}"

    # ---------------- Table 3 phase breakdown
    order = phase.drop_duplicates("dataset")["dataset"].tolist()
    order = sorted(order, key=lambda d: int(phase.loc[phase["dataset"] == d, "N_train"].iloc[0]))
    t3 = []
    for ds in order:
        sub = phase[phase["dataset"] == ds]
        r0 = sub.iloc[0]
        nk = f"{r0['N_train'] / 1e3:.1f}k".replace(".0k", "k")
        label = f"\\texttt{{{ds.replace('_', chr(92) + '_')}}} ({nk}, {int(r0['D'])}, {int(r0['K'])}, {r0['cov_type']})"
        for j, mode in enumerate(["cold", "warm"]):
            r = sub[sub["start_mode"] == mode].iloc[0]
            vals = [r["standardize_ms"], r["seeding_ms"], r["graph_capture_ms"], r["ema_replay_ms"], r["polish_ms"], r["export_ms"]]
            tot = sum(vals)
            pre = f"\\multirow{{2}}{{*}}{{{label}}}\n" if j == 0 else ""
            t3.append(pre + f"  & {mode.capitalize()} & " + " & ".join(f"{v:.1f}" for v in vals) + f" & {tot:.1f} \\\\")
    (GEN / "tab_phase.tex").write_text("\n".join(t3) + "\n", encoding="utf-8")
    try:
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from egmm.scaling import auto_hyperparameters

        ph = phase.copy()
        ph["total"] = ph[["standardize_ms", "seeding_ms", "graph_capture_ms", "ema_replay_ms", "polish_ms", "export_ms"]].sum(axis=1)
        ph["T"] = [
            auto_hyperparameters(int(r.N_train), int(r.K), int(r.D), r.cov_type).n_steps for r in ph.itertuples()
        ]
        med = ph[(ph["N_train"] <= LARGE_N) & (ph["start_mode"] == "warm")]
        macros["PhaseGraphMin"] = f"{ph['graph_capture_ms'].min():.1f}"
        macros["PhaseGraphMax"] = f"{ph['graph_capture_ms'].max():.1f}"
        macros["PhaseEMAMin"] = f"{med['ema_replay_ms'].min():.1f}"
        macros["PhaseEMAMax"] = f"{med['ema_replay_ms'].max():.1f}"
        macros["PhaseStepsMin"] = str(int(med["T"].min()))
        macros["PhaseStepsMax"] = str(int(med["T"].max()))
        polf = ph[(ph["polish_ms"] > 0) & (ph["cov_type"] == "full")]
        pold = ph[(ph["polish_ms"] > 0) & (ph["cov_type"] == "diag")]
        if len(polf):
            share = 100 * polf["polish_ms"] / polf["total"]
            macros["PhasePolShareFullMin"] = f"{share.min():.0f}"
            macros["PhasePolShareFullMax"] = f"{share.max():.0f}"
        if len(pold):
            share = 100 * pold["polish_ms"] / pold["total"]
            macros["PhasePolShareDiagMin"] = f"{share.min():.0f}"
            macros["PhasePolShareDiagMax"] = f"{share.max():.0f}"
        for ds, tag in [("covtype_581k", "Cov"), ("syn_scale400k_d256_diag", "Dtwo")]:
            sub = ph[ph["dataset"] == ds]
            if len(sub):
                for mode in ("cold", "warm"):
                    r = sub[sub["start_mode"] == mode].iloc[0]
                    macros[f"Phase{tag}{mode.capitalize()}"] = f"{r['total']:.1f}"
                    macros[f"Phase{tag}Std{mode.capitalize()}"] = f"{r['standardize_ms']:.1f}"
    except Exception as exc:  # pragma: no cover - reporting helper only
        lines.append(f"[phase macros skipped: {exc}]")
    lines.append("\n=== Phase breakdown totals (ms) ===")
    for ds in order:
        sub = phase[phase["dataset"] == ds]
        for _, r in sub.iterrows():
            tot = r[["standardize_ms", "seeding_ms", "graph_capture_ms", "ema_replay_ms", "polish_ms", "export_ms"]].sum()
            lines.append(f"  {ds:24s} {r['start_mode']}: total={tot:.1f} std={r['standardize_ms']:.1f} ema={r['ema_replay_ms']:.1f} pol={r['polish_ms']:.1f} P={int(r['polish_steps'])}")

    # ---------------- Small datasets (Tables 4, 5)
    s_lines = [f"=== Small-Dataset Summary (n={small['dataset'].nunique()}; Holm-adjusted p over the 6 tests, raw in brackets) ==="]
    t4 = []
    s_rows = [(eg, b, pair_stats(small, eg, b)) for eg in ["EGMM_auto", "EGMM_cuda"] for b in ["TorchBatchGMM", "OnlineGMM_tuned", "SklearnGMM"]]
    apply_holm(s_rows)
    for eg, b, s in s_rows:
        s_lines.append(
            f"{eg:10s} vs {b:16s}: NLL W/T/L={s['W']}/{s['T']}/{s['L']}, med_dNLL={s['med_dnll']:+.4f}, "
            f"p_nll={fmt_p_txt(s['p_nll'])} [{fmt_p_txt(s['p_nll_raw'])}] | med_spd={s['med_spd']:.2f}x, "
            f"p_time={fmt_p_txt(s['p_time'])} [{fmt_p_txt(s['p_time_raw'])}]"
        )
        t = "S" + "".join(ch for ch in eg + b if ch.isalpha())
        macros[t + "WTL"] = f"{s['W']}/{s['T']}/{s['L']}"
        macros[t + "W"] = str(s["W"])
        macros[t + "dNLL"] = f"{s['med_dnll']:+.4f}"
        macros[t + "pNLL"] = fmt_p(s["p_nll"]).strip("$")
        macros[t + "Spd"] = f"{s['med_spd']:.2f}"
        macros[t + "pTime"] = fmt_p(s["p_time"]).strip("$")
        name = f"\\texttt{{{eg.replace('_', chr(92) + '_')}}} vs \\texttt{{{b.replace('_', chr(92) + '_')}}}"
        t4.append(
            f"{name} & {bold_if(f'{s['W']}/{s['T']}/{s['L']}', s['W'] > s['L'])} & ${s['med_dnll']:+.4f}$ & {fmt_p(s['p_nll'])} & "
            f"{bold_if(f'{s['med_spd']:.2f}$\\times$', s['med_spd'] > 1)} & {fmt_p(s['p_time'])} \\\\"
        )
    (GEN / "tab_small_wilcoxon.tex").write_text("\n".join(t4) + "\n", encoding="utf-8")
    (out_exp / "small_dataset_stats.txt").write_text("\n".join(s_lines) + "\n", encoding="utf-8")
    sg = small.groupby(["dataset", "method"])[["test_nll", "fit_time"]].mean()
    smeta = small.groupby("dataset")[["n_train", "n_features", "n_components", "cov_type"]].first()
    t5 = []
    s_methods = ["EGMM_auto", "EGMM_cuda", "TorchBatchGMM", "OnlineGMM_tuned", "SklearnGMM"]
    t_methods = ["EGMM_auto", "EGMM_cuda", "TorchBatchGMM", "SklearnGMM"]
    for ds in sorted(small["dataset"].unique()):
        n, d, k, cov = smeta.loc[ds]
        nl = {m: sg.loc[(ds, m), "test_nll"] for m in s_methods}
        tm = {m: sg.loc[(ds, m), "fit_time"] * 1e3 for m in t_methods}
        bn = min(round(v, 3) for v in nl.values())
        bt = min(round(v, 1) for v in tm.values())
        cells = [bold_if(f"{nl[m]:.3f}", round(nl[m], 3) == bn) for m in s_methods]
        cells += [bold_if(f"{tm[m]:.1f}", round(tm[m], 1) == bt) for m in t_methods]
        t5.append(f"\\texttt{{{ds.replace('_', chr(92) + '_')}}} ({int(n)}, {int(d)}, {int(k)}, {cov}) & " + " & ".join(cells) + " \\\\")
    (GEN / "tab_full16.tex").write_text("\n".join(t5) + "\n", encoding="utf-8")
    macros["nSmall"] = str(small["dataset"].nunique())
    st_ = small.groupby(["dataset", "method"])["fit_time"].mean().unstack()
    shown_s = st_[["EGMM_auto", "EGMM_cuda", "TorchBatchGMM", "SklearnGMM"]].min(axis=1)
    macros["SmallOnlineFaster"] = str(int((st_["OnlineGMM_tuned"] < shown_s).sum()))
    macros["SmallNmin"] = str(int(smeta["n_train"].min()))
    macros["SmallNmax"] = str(int(smeta["n_train"].max()))

    # ---------------- Pareto dominance (exact): a baseline budget point is dominated if some EGMM budget
    # point is at least as fast AND at least as good in test NLL.
    if pareto is not None:
        lines.append("\n=== Pareto dominance (benchmark_pareto.csv, seed 0) ===")
        n_pts = n_dom = 0
        n_ds_all = 0
        for ds_name, sub in pareto.groupby("dataset"):
            eg = sub[sub["method"] == "EGMM"][["fit_time", "test_nll"]].to_numpy()
            und = []
            for _, r in sub[sub["method"] != "EGMM"].iterrows():
                dom = bool(((eg[:, 0] <= r["fit_time"]) & (eg[:, 1] <= r["test_nll"])).any())
                n_pts += 1
                n_dom += int(dom)
                if not dom:
                    und.append(f"{r['method']}@{int(r['budget'])}")
            n_ds_all += int(len(und) == 0)
            lines.append(f"  {ds_name:24s}: undominated baseline points = {', '.join(und) if und else 'none'}")
        lines.append(f"  dominated {n_dom}/{n_pts} baseline budget points; fully dominated datasets: {n_ds_all}/{pareto['dataset'].nunique()}")
        macros["ParetoDominated"] = str(n_dom)
        macros["ParetoPoints"] = str(n_pts)
        macros["ParetoFullDom"] = str(n_ds_all)
        macros["ParetoDatasets"] = str(pareto["dataset"].nunique())

    # ---------------- Pareto prose numbers (covtype)
    if pareto is not None and "covtype_581k" in set(pareto["dataset"]):
        pc = pareto[pareto["dataset"] == "covtype_581k"]
        e2 = pc[(pc["method"] == "EGMM") & (pc["budget"] == 2)].iloc[0]
        tb80 = pc[(pc["method"] == "TorchBatchGMM") & (pc["budget"] == pc[pc["method"] == "TorchBatchGMM"]["budget"].max())].iloc[0]
        macros["ParetoCovEGMMNLL"] = f"{e2['test_nll']:.2f}"
        macros["ParetoCovEGMMms"] = f"{e2['fit_time'] * 1e3:.0f}"
        macros["ParetoCovTBNLL"] = f"{tb80['test_nll']:.2f}"
        macros["ParetoCovTBs"] = f"{tb80['fit_time']:.2f}"
        macros["ParetoCovTBit"] = str(int(tb80["budget"]))

    # ---------------- Design studies (validation-split selection; docs/experiments/design_studies/)
    dsd = EXP / "design_studies"
    lines.append("\n=== Design studies (selection on validation NLL) ===")

    def variant_vs(df, a, b, split="val_nll"):
        g = df.groupby(["dataset", "variant"])[[split, "fit_ms"]].mean().reset_index()
        pv = g.pivot(index="dataset", columns="variant", values=split)
        tv = g.pivot(index="dataset", columns="variant", values="fit_ms")
        d = pv[a] - pv[b]
        nmeta = df.groupby("dataset")["N"].first()
        lg = nmeta[pv.index] > LARGE_N
        return {
            "W": int((d < -TIE).sum()), "L": int((d > TIE).sum()), "changed": int((d.abs() > 1e-9).sum()),
            "med": float(d.median()), "med_changed": float(d[d.abs() > 1e-9].median()) if (d.abs() > 1e-9).any() else 0.0,
            "p": wilcoxon_p(pv[a], pv[b]), "lW": int((d[lg] < -TIE).sum()), "lL": int((d[lg] > TIE).sum()),
            "lmed": float(d[lg].median()) if lg.any() else float("nan"),
            "t_ratio": float((tv[a] / tv[b]).median()), "p_t": wilcoxon_p(tv[a], tv[b]), "d": d,
        }

    if (dsd / "seeding_variants.csv").exists():
        sv = pd.read_csv(dsd / "seeding_variants.csv")
        for v in sorted(set(sv["variant"]) - {"base"}):
            r = variant_vs(sv, v, "base")
            lines.append(
                f"  seeding {v:16s} vs base: val W/L={r['W']}/{r['L']} med={r['med']:+.4f} p={fmt_p_txt(r['p'])} | "
                f"large W/L={r['lW']}/{r['lL']} med={r['lmed']:+.4f} | time ratio={r['t_ratio']:.3f}"
            )
            if v == "greedy128x_l10":
                macros["SeedW"], macros["SeedL"] = str(r["W"]), str(r["L"])
                macros["SeedMed"], macros["SeedP"] = f"{r['med']:+.3f}", f"{r['p']:.3f}"
                macros["SeedLW"], macros["SeedLL"], macros["SeedLMed"] = str(r["lW"]), str(r["lL"]), f"{r['lmed']:+.2f}"
                macros["SeedTime"] = f"{r['t_ratio']:.2f}"
    if (dsd / "schedule_law_ab.csv").exists():
        la = pd.read_csv(dsd / "schedule_law_ab.csv")
        r = variant_vs(la, "smooth", "bracket")
        lines.append(
            f"  schedule law smooth vs bracket: changed={r['changed']} val W/L={r['W']}/{r['L']} "
            f"med(changed)={r['med_changed']:+.4f} p={fmt_p_txt(r['p'])} | time ratio={r['t_ratio']:.3f} p_t={fmt_p_txt(r['p_t'])}"
        )
        macros["LawChanged"], macros["LawW"], macros["LawL"] = str(r["changed"]), str(r["W"]), str(r["L"])
        macros["LawMed"], macros["LawP"], macros["LawPt"] = f"{r['med_changed']:+.4f}", f"{r['p']:.2f}", f"{r['p_t']:.2f}"
    if (dsd / "closing_refinement_large_n.csv").exists():
        cr = pd.read_csv(dsd / "closing_refinement_large_n.csv")
        split = "val_nll" if "val_nll" in cr.columns else "test_nll"
        syn = [d for d in ["syn_scale200k", "syn_scale300k_k64_full", "syn_scale400k_d256_diag", "syn_scale500k_diag"]]
        g = cr.groupby(["dataset", "variant"])[split].mean().unstack()
        best_gain = 0.0
        for v in ["pol1", "pol2", "sub64k_r2", "sub128k_r3"]:
            if v in g.columns:
                d = g[v] - g["base"]
                best_gain = min(best_gain, float(d.loc[d.index.intersection(syn)].min()))
                lines.append(f"  closing {v:12s} vs base ({split}): synthetic-large min/max d={d.loc[d.index.intersection(syn)].min():+.3f}/{d.loc[d.index.intersection(syn)].max():+.3f}; creditcard {d.get('creditcard', float('nan')):+.2f}")
        macros["CloseBest"] = f"{-best_gain:.2f}"
        if "sub64k_r2" in g.columns and "creditcard" in g.index:
            macros["CloseCredit"] = f"{(g['sub64k_r2'] - g['base'])['creditcard']:+.1f}"
        for v, tag in [("seed256x_l10", "SeedBig"), ("ep3", "EpThree")]:
            if v in g.columns:
                d = (g[v] - g["base"]).loc[g.index.intersection(syn)]
                macros[tag + "Min"], macros[tag + "Max"] = f"{d.min():+.2f}", f"{d.max():+.2f}"
    if (dsd / "precision_f32_vs_f64.csv").exists():
        pr = pd.read_csv(dsd / "precision_f32_vs_f64.csv")
        if "variant" in pr.columns:
            g = pr.groupby(["dataset", "variant"])["test_nll"].mean().unstack()
            d = (g["float32"] - g["float64"]).abs()
        else:
            g = pr.groupby("dataset").mean(numeric_only=True)
            d = (g["test_float32"] - g["test_float64"]).abs()
        lines.append(f"  float32 vs float64 training (shared RNG): max |d test NLL| = {d.max():.3f} ({d.idxmax()})")
        macros["PrecMax"], macros["PrecMaxName"] = f"{d.max():.2f}", d.idxmax().replace("_", r"\_")
        macros["PrecN"] = str(len(d))

    # ---------------- write outputs
    (out_exp / "stats_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(GEN / "numbers.tex", "w", encoding="utf-8") as f:
        f.write("% Auto-generated by experiments/analyze_results.py -- do not edit by hand.\n")
        for k, v in sorted(macros.items()):
            safe = "".join(ch for ch in k if ch.isalpha())
            f.write(f"\\newcommand{{\\r{safe}}}{{{v}}}\n")
    print("\n".join(lines))
    print("\n".join(s_lines))
    print(f"\nWrote {len(macros)} macros and tables to {GEN}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-exp", type=Path, default=None, help="directory for stats_summary.txt (default docs/experiments)")
    ap.add_argument("--out-gen", type=Path, default=None, help="directory for LaTeX outputs (default paper/generated)")
    ap.add_argument("--in-exp", type=Path, default=None, help="directory with the input CSVs (default docs/experiments)")
    a = ap.parse_args()
    main(a.out_exp, a.out_gen, a.in_exp)

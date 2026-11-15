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
    "TorchBatchGMM_fullinit": r"\texttt{TorchBatchGMM\_fullinit}",
    "EGMM_no_init_scale": r"\texttt{EGMM\_no\_init\_scale}",
    "EGMM_no_revival": r"\texttt{EGMM\_no\_revival}",
    "EGMM_bracket_law": r"\texttt{EGMM\_bracket\_law}",
    "OnlineGMM_matched_polish": r"\texttt{OnlineGMM\_matched\_polish}",
    "EGMM_eager": r"\texttt{EGMM\_eager}",
    "TorchBatchGMM_f32chol": r"\texttt{TorchBatch\_f32chol}",
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


def drop_failed_cells(df: pd.DataFrame, lines: list[str] | None = None, label: str = "") -> pd.DataFrame:
    """Remove every (dataset, method) cell in which any seed failed to fit (NaN test NLL), and log it.

    A failed cell is excluded from that method's comparisons only (it is never imputed or averaged over
    the remaining seeds); the failure itself is reported in stats_summary.txt and the paper.
    """
    bad = df[df["test_nll"].isna()][["dataset", "method"]].drop_duplicates()
    if len(bad) and lines is not None:
        for r in bad.itertuples():
            n_fail = int(df[(df.dataset == r.dataset) & (df.method == r.method)]["test_nll"].isna().sum())
            lines.append(f"[{label}] FAILED FIT: {r.method} on {r.dataset} ({n_fail} seed(s)); cell excluded from comparisons")
    if not len(bad):
        return df
    key = set(map(tuple, bad.to_numpy()))
    return df[[(d, m) not in key for d, m in zip(df["dataset"], df["method"])]]


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


def _signed_rank_null_cdf(n: int) -> np.ndarray:
    """Exact null CDF of the Wilcoxon signed-rank statistic T+ for n untied pairs."""
    max_s = n * (n + 1) // 2
    counts = np.zeros(max_s + 1, dtype=np.float64)
    counts[0] = 1.0
    for r in range(1, n + 1):
        counts[r:] = counts[r:] + counts[: max_s + 1 - r].copy()
    return np.cumsum(counts) / counts.sum()


def hodges_lehmann(d: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Hodges-Lehmann location estimate of paired differences with its exact signed-rank CI."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    walsh = np.sort([(d[i] + d[j]) / 2.0 for i in range(n) for j in range(i, n)])
    est = float(np.median(walsh))
    if n < 6:
        return est, float("nan"), float("nan")
    cdf = _signed_rank_null_cdf(n)
    c = int(np.searchsorted(cdf, alpha / 2.0, side="right")) - 1  # largest t with P(T <= t) <= alpha/2
    c = max(c, 0)
    return est, float(walsh[c]), float(walsh[len(walsh) - 1 - c])


def sign_test_p(d: np.ndarray) -> float:
    d = np.asarray(d)
    w, l_ = int((d < -TIE).sum()), int((d > TIE).sum())
    if w + l_ == 0:
        return float("nan")
    return float(stats.binomtest(w, w + l_, 0.5).pvalue)


def bold_if(val: str, cond: bool) -> str:
    return f"\\textbf{{{val}}}" if cond else val


def sci_tex(x: float, digits: int = 2) -> str:
    """Signed LaTeX scientific notation without surrounding dollars, e.g. '+1.17\\times 10^{6}'."""
    e = int(math.floor(math.log10(abs(x))))
    return f"{x / 10**e:+.{digits}f}\\times 10^{{{e}}}"


def join_and(items) -> str:
    """English list for prose macros: 'a', 'a and b', 'a, b, and c'."""
    items = [str(x) for x in items]
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


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
    # Robustness of Panel A to the scale of the paired difference: dNLL / D weights datasets more evenly
    # (reported alongside the primary raw-nat analysis, never instead of it).
    d_meta = h2h.groupby("dataset")["D"].first().astype(float)
    pn_a = h2h.groupby(["dataset", "method"])["test_nll"].mean().unstack()
    dim_raw = []
    for _, b, _s in panel_a:
        pr = pn_a[["EGMM", b]].dropna()
        dd = (pr["EGMM"] - pr[b]) / d_meta.loc[pr.index]
        dim_raw.append((b, wilcoxon_p(dd, pd.Series(0.0, index=dd.index))))
    dim_adj = holm_adjust([p for _, p in dim_raw])
    lines.append("Panel A on per-dimension differences dNLL/D (Holm within panel; raw in brackets): "
                 + ", ".join(f"{b}={fmt_p_txt(a)} [{fmt_p_txt(p)}]" for (b, p), a in zip(dim_raw, dim_adj)))
    for (b, p_raw), a in zip(dim_raw, dim_adj):
        macros["ADim" + "".join(ch for ch in b if ch.isalpha()) + "p"] = fmt_p(a).strip("$")
        macros["ADim" + "".join(ch for ch in b if ch.isalpha()) + "praw"] = f"{p_raw:.3f}"
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
    pt_all = agg.pivot(index="dataset", columns="method", values="fit_time")
    macros["GPUvsSK"] = f"{(pt_all['StandardizedSklearnGMM'] / pt_all['TorchBatchGMM']).median():.1f}"
    beyond = [m for m, r in ranks.items() if r > ranks["EGMM"] + cd]
    macros["NemenyiBeyond"] = join_and(TEX_NAME[m].replace(" (CPU)", "").replace(" (Raw CPU)", "") for m in beyond) or "none"
    macros["NemenyiBeyondN"] = str(len(beyond))
    capp = h2h[h2h.method.isin(["TorchBatchGMM", "StandardizedSklearnGMM"])].groupby("method")["n_iter"].apply(lambda s: 100 * (s >= 50).mean())
    macros["PCapShareTB"] = f"{capp.get('TorchBatchGMM', float('nan')):.0f}"
    macros["PCapShareSK"] = f"{capp.get('StandardizedSklearnGMM', float('nan')):.0f}"
    macros["RankFirst"] = TEX_NAME[ranks.index[0]].replace(" (CPU)", "").replace(" (Raw CPU)", "")

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
                    [("EGMM", "EGMM_eager"), ("EGMM_eager", "TorchBatchGMM"), ("EGMM", "TorchBatchGMM"),
                     ("EGMM_eager", "TorchBatchGMM_f32chol"), ("EGMM", "TorchBatchGMM_f32chol")]
                    if b in set(sysdf["method"])]
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
        g_sp = float(sys_rows[0][2]["med_spd"]); a_sp = float(sys_rows[1][2]["med_spd"]); t_sp = float(sys_rows[2][2]["med_spd"])
        macros["EGraphShare"] = f"{100 * math.log(g_sp) / math.log(t_sp):.0f}"
        # the decomposition by size: launch/sync-bound small fits vs larger fits
        tgN = sysdf.groupby(["dataset", "method"])["fit_time"].mean().unstack()
        nN = sysdf.groupby("dataset")["N_train"].first()
        for tag, msk in [("Small", nN <= LARGE_N), ("Large", nN > LARGE_N)]:
            macros["E" + tag + "Graph"] = f"{(tgN['EGMM_eager'] / tgN['EGMM'])[msk].median():.2f}"
            macros["E" + tag + "Alg"] = f"{(tgN['TorchBatchGMM'] / tgN['EGMM_eager'])[msk].median():.2f}"
            macros["E" + tag + "Total"] = f"{(tgN['TorchBatchGMM'] / tgN['EGMM'])[msk].median():.2f}"
            macros["nSys" + tag] = str(int(msk.sum()))
        lines.append("  by size (N <= 1e5 / > 1e5): graph " + macros["ESmallGraph"] + " / " + macros["ELargeGraph"]
                     + ", algorithm " + macros["ESmallAlg"] + " / " + macros["ELargeAlg"] + ", total " + macros["ESmallTotal"] + " / " + macros["ELargeTotal"])
        if len(sys_rows) >= 5:  # decomposition against batch EM with e-GMM's float32 inner Cholesky
            a_f = float(sys_rows[3][2]["med_spd"]); t_f = float(sys_rows[4][2]["med_spd"])
            macros["EGraphShareF"] = f"{100 * math.log(g_sp) / math.log(t_f):.0f}" if t_f > 1 else "--"
            macros["EAlgShareF"] = f"{100 * math.log(a_f) / math.log(t_f):.0f}" if t_f > 1 and a_f > 1 else "--"
            tg_ = sysdf.groupby(["dataset", "method"])["fit_time"].mean().unstack()
            f64_cost = (tg_["TorchBatchGMM"] / tg_["TorchBatchGMM_f32chol"])
            macros["EFsixtyfourCost"] = f"{f64_cost.median():.2f}"
            macros["EFsixtyfourCostMax"] = f"{f64_cost.max():.2f}"
            lines.append(f"  float64 vs float32 inner Cholesky in batch EM: median time ratio {f64_cost.median():.2f} (max {f64_cost.max():.2f}); "
                         f"shares vs f32chol batch EM: graph {macros['EGraphShareF']}%, algorithm {macros['EAlgShareF']}%")
        macros["EEagerSlower"] = str(int(sys_rows[1][2]["n"] - sys_rows[1][2]["n_faster"]))
        # session-to-session timing variability: the same deterministic EGMM / TorchBatchGMM fits were
        # timed in two independent sessions (head-to-head run and systems run)
        hh = h2h.groupby(["dataset", "method"])["fit_time"].mean().unstack()
        ss = sysdf.groupby(["dataset", "method"])["fit_time"].mean().unstack()
        sp_hh = hh["TorchBatchGMM"] / hh["EGMM"]
        sp_ss = ss["TorchBatchGMM"] / ss["EGMM"]
        common = sp_hh.index.intersection(sp_ss.index)
        shift = (sp_hh[common] / sp_ss[common])
        macros["SessSpdA"], macros["SessSpdB"] = f"{sp_hh[common].median():.2f}", f"{sp_ss[common].median():.2f}"
        macros["SessFasterA"], macros["SessFasterB"] = str(int((sp_hh[common] > 1).sum())), str(int((sp_ss[common] > 1).sum()))
        macros["SessShiftMax"] = f"{max(shift.max(), 1 / shift.min()):.1f}"
        egr = (hh.loc[common, "EGMM"] / ss.loc[common, "EGMM"])
        macros["SessEGMMRatioMed"] = f"{egr.median():.2f}"
        tbr = (hh.loc[common, "TorchBatchGMM"] / ss.loc[common, "TorchBatchGMM"])
        macros["SessAbsMax"] = f"{max(egr.max(), 1 / egr.min(), tbr.max(), 1 / tbr.min()):.1f}"
        macros["SessTBRatioMed"] = f"{tbr.median():.2f}"
        lines.append(f"  absolute fit-time ratio between sessions (head-to-head / systems): EGMM median {egr.median():.2f} "
                     f"(range {egr.min():.2f}-{egr.max():.2f}), TorchBatchGMM median {tbr.median():.2f} (range {tbr.min():.2f}-{tbr.max():.2f})")
        lines.append(
            f"  timing sessions: median speedup vs TB {sp_hh[common].median():.2f}x (head-to-head) vs {sp_ss[common].median():.2f}x (systems); "
            f"faster on {int((sp_hh[common] > 1).sum())} vs {int((sp_ss[common] > 1).sum())}; max per-dataset shift {max(shift.max(), 1 / shift.min()):.2f}x; "
            f"EGMM time ratio between sessions median {egr.median():.2f} (range {egr.min():.2f}-{egr.max():.2f})"
        )
        macros["EAlgShare"] = f"{100 * math.log(a_sp) / math.log(t_sp):.0f}"
        lines.append(
            f"  median passes EGMM={pm['EGMM'].median():.1f} vs TorchBatch={pm['TorchBatchGMM'].median():.1f} "
            f"(median ratio {ratio.median():.1f}x); time-per-pass TB / EGMM_eager median {tpp.median():.2f}"
        )
        macros["nSys"] = str(sysdf["dataset"].nunique())

    # ---------------- Panel F (pre-registered fresh holdout suite, one Holm family)
    ho_rows = []
    if (EXP / "holdout_head_to_head.csv").exists():
        ho_raw = pd.read_csv(EXP / "holdout_head_to_head.csv")
        n_fail_cells = int(ho_raw[ho_raw["test_nll"].isna()][["dataset", "method"]].drop_duplicates().shape[0])
        fail_desc = sorted({f"{m}/{d}" for d, m in ho_raw[ho_raw["test_nll"].isna()][["dataset", "method"]].itertuples(index=False)})
        macros["HoldoutFailCells"] = str(n_fail_cells)
        macros["HoldoutFailList"] = (join_and(x.replace("_", r"\_") for x in fail_desc)) or "none"
        ho = drop_failed_cells(ho_raw, lines, "holdout")
        ho_bases = [m for m in PRIMARY if m != "EGMM"] + [m for m in EXTRA_BASELINES if m in set(ho["method"])]
        ho_rows = [("EGMM", b, pair_stats(ho, "EGMM", b)) for b in ho_bases if b in set(ho["method"])]
        apply_holm(ho_rows)
        n_ho = ho["dataset"].nunique()
        macros["nHoldout"] = str(n_ho)
        hot = ho.groupby(["dataset", "method"])["fit_time"].mean().unstack()
        macros["HGPUvsSK"] = f"{(hot['StandardizedSklearnGMM'] / hot['TorchBatchGMM']).median():.1f}"
        macros["HMedMsEGMM"] = f"{1e3 * hot['EGMM'].median():.0f}"
        macros["HMedMsTB"] = f"{1e3 * hot['TorchBatchGMM'].median():.0f}"
        macros["HMedSSK"] = f"{hot['StandardizedSklearnGMM'].median():.1f}"
        macros["FTorchBatchGMMSpdR"] = f"{(hot['TorchBatchGMM'] / hot['EGMM']).median():.1f}"
        macros["FStandardizedSklearnGMMSpdR"] = f"{(hot['StandardizedSklearnGMM'] / hot['EGMM']).median():.0f}"
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
            macros[t + "pNLLraw"] = fmt_p(st["p_nll_raw"]).strip("$")
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
            for m in hr.index:
                macros["FHoldRank" + "".join(ch for ch in m if ch.isalpha())] = f"{hr[m]:.2f}"
            # post-hoc rank sensitivity: without the two metric-artefact datasets, and with failed fits
            # counted as the worst rank instead of dropping the dataset
            keep = [x for x in hcomp.index if x not in ("eeg_eye_state", "ailerons")]
            hr2 = hcomp.loc[keep].rank(axis=1).mean().sort_values()
            macros["FHoldRankPosNoArt"] = str(int(list(hr2.index).index("EGMM")) + 1)
            macros["FHoldNNoArt"] = str(len(keep))
            rk = hpn[[m for m in PRIMARY if m in hpn.columns]].rank(axis=1, na_option="bottom")
            hr3 = rk.mean().sort_values()
            macros["FHoldRankPosFailWorst"] = str(int(list(hr3.index).index("EGMM")) + 1)
            macros["FHoldNFailWorst"] = str(len(rk))
            lines.append("  ranks without eeg_eye_state/ailerons (n=%d): " % len(keep) + ", ".join(f"{m}={v:.2f}" for m, v in hr2.items()))
            lines.append("  ranks with failed fits ranked last (n=%d): " % len(rk) + ", ".join(f"{m}={v:.2f}" for m, v in hr3.items()))
        hmeta = ho.groupby("dataset")[["N_train", "D", "K", "cov_type"]].first()
        hg = ho.groupby(["dataset", "method"])[["test_nll", "fit_time"]].mean()
        hs = ho.groupby(["dataset", "method"])["test_nll"].std(ddof=1)
        t_ho = []
        for dsn in sorted(hmeta.index, key=lambda x: int(hmeta.loc[x, "N_train"])):
            N, D, K, cov = hmeta.loc[dsn]
            all_cols = ["EGMM", "TorchBatchGMM", "TorchBatchGMM_fullinit", "OnlineGMM_tuned", "StandardizedSklearnGMM", "SklearnGMM"]
            cols = [c for c in all_cols if (dsn, c) in hg.index]
            nl = {c: hg.loc[(dsn, c), "test_nll"] for c in cols}
            best = min(round(v, 3) for v in nl.values())
            cells = []
            for c in all_cols:  # fixed column layout: failed/missing cells print as "failed"
                if c not in nl:
                    cells.append("failed")
                    continue
                def big(v, nd):  # math-mode body; scientific notation for the metric-artifact rows (|NLL| >= 1e4)
                    if abs(v) < 1e4:
                        return f"{v:.{nd}f}"
                    e = int(math.floor(math.log10(abs(v))))
                    return f"{v / 10**e:.2f}{{\\times}}10^{{{e}}}"

                pm = f"{{\\tiny$\\pm {big(hs.loc[(dsn, c)], 2)}$}}" if c in ("EGMM", "TorchBatchGMM", "StandardizedSklearnGMM") else ""
                val = f"{nl[c]:.3f}" if abs(nl[c]) < 1e4 else f"${big(nl[c], 3)}$"
                cells.append(bold_if(val, round(nl[c], 3) == best) + pm)
            te = hg.loc[(dsn, "EGMM"), "fit_time"]
            cells.append(f"{te * 1e3:.1f}")
            cells.append(f"{hg.loc[(dsn, 'TorchBatchGMM'), 'fit_time'] / te:.2f}$\\times$")
            cells.append(f"{hg.loc[(dsn, 'StandardizedSklearnGMM'), 'fit_time'] / te:.1f}$\\times$")
            t_ho.append(f"\\texttt{{{dsn.replace('_', chr(92) + '_')}}} ({int(N)}, {int(D)}, {int(K)}, {cov}) & " + " & ".join(cells) + " \\\\")
        (GEN / "tab_holdout.tex").write_text("\n".join(t_ho) + "\n", encoding="utf-8")
        macros["FHoldN"] = str(len(hcomp)) if len(hcomp) else "0"

        # --- holdout dead components / polish / revivals (all logged quantities are reported)
        hd = ho.groupby(["dataset", "method"])[["dead_components", "revivals", "test_nll"]].mean().unstack()
        for m in ["EGMM", "EGMM_polish0", "TorchBatchGMM", "StandardizedSklearnGMM", "OnlineGMM_tuned"]:
            if m in hd["dead_components"]:
                macros["HDead" + "".join(ch for ch in m if ch.isalpha())] = f"{hd['dead_components'][m].mean():.2f}"
        macros["HRevivalsEGMM"] = f"{hd['revivals']['EGMM'].mean():.2f}"
        har = ho.groupby(["dataset", "method"])["ari"].mean().unstack().dropna(how="all")
        macros["nHLabeled"] = str(int(har["EGMM"].notna().sum()))
        for m, tg in [("EGMM", "EGMM"), ("TorchBatchGMM", "TB"), ("StandardizedSklearnGMM", "SK"), ("OnlineGMM_tuned", "Online")]:
            if m in har.columns:
                macros["HARI" + tg] = f"{har[m].mean():.3f}"
        lines.append("Holdout mean ARI (labelled datasets): " + ", ".join(f"{m}={har[m].mean():.3f}" for m in har.columns))
        worst = hd["dead_components"]["EGMM"].sort_values(ascending=False)
        macros["HDeadWorst"] = join_and(
            f"\\texttt{{{dn.replace('_', chr(92) + '_')}}} ({worst[dn]:.1f}/{int(hmeta.loc[dn, 'K'])})" for dn in worst.index[:2])
        if "EGMM_polish0" in hd["test_nll"]:
            dpol = (hd["test_nll"]["EGMM"] - hd["test_nll"]["EGMM_polish0"]).dropna()
            w_, t_, l_ = wtl(dpol)
            macros["HPolWTL"] = f"{w_}/{t_}/{l_}"
            macros["HPolP"] = fmt_p(wilcoxon_p(hd["test_nll"]["EGMM"].loc[dpol.index], hd["test_nll"]["EGMM_polish0"].loc[dpol.index])).strip("$")
            worst_pol = dpol.sort_values(ascending=False)
            macros["HPolWorst"] = join_and(
                f"\\texttt{{{dn.replace('_', chr(92) + '_')}}} ($+{worst_pol[dn]:,.0f}$)".replace(",", "{,}") for dn in worst_pol.index[:2])
            lines.append(f"Holdout EGMM vs EGMM_polish0: W/T/L={w_}/{t_}/{l_}; largest polish damage: " + ", ".join(f"{k}={v:+.1f}" for k, v in worst_pol.head(3).items()))
            best_pol = dpol.sort_values()
            macros["HPolBest"] = join_and(
                f"\\texttt{{{dn.replace('_', chr(92) + '_')}}} (${best_pol[dn]:.1f}$)" for dn in best_pol.index[:2])
            macros["HPolWins"] = str(w_)
            # e-GMM without polish vs online EM: does the online-EM advantage need the batch polish?
            if "OnlineGMM_tuned" in hd["test_nll"]:
                dpo = (hd["test_nll"]["EGMM_polish0"] - hd["test_nll"]["OnlineGMM_tuned"]).dropna()
                w2, t2_, l2 = wtl(dpo)
                p2 = wilcoxon_p(hd["test_nll"]["EGMM_polish0"].loc[dpo.index], hd["test_nll"]["OnlineGMM_tuned"].loc[dpo.index])
                macros["HPolZeroOnlineWTL"] = f"{w2}/{t2_}/{l2}"
                macros["HPolZeroOnlineMed"] = f"{dpo.median():+.2f}"
                macros["HEGMMOnlineMed"] = f"{(hd['test_nll']['EGMM'] - hd['test_nll']['OnlineGMM_tuned']).dropna().median():+.2f}"
                macros["HPolZeroOnlineP"] = fmt_p(p2).strip("$")
                lines.append(f"Holdout EGMM_polish0 vs OnlineGMM_tuned: W/T/L={w2}/{t2_}/{l2}, raw p={fmt_p_txt(p2)}")
        hop = ho.assign(passes=data_passes(ho)).groupby(["dataset", "method"])["passes"].mean().unstack()
        macros["HPassesEGMM"] = f"{hop['EGMM'].median():.1f}"
        macros["HPassesTB"] = f"{hop['TorchBatchGMM'].median():.1f}"
        macros["HPassesOnline"] = f"{hop['OnlineGMM_tuned'].median():.0f}"
        lines.append(f"Holdout median full-data passes: EGMM={hop['EGMM'].median():.1f}, TorchBatchGMM={hop['TorchBatchGMM'].median():.1f}")
        lines.append("Holdout mean dead components: " + ", ".join(f"{m}={hd['dead_components'][m].mean():.2f}" for m in hd["dead_components"].columns))

        # --- robustness of the holdout comparisons (post-hoc, reported as such)
        conv_path = EXP / "holdout_converged_em.csv"
        ho_rb = ho
        if conv_path.exists():
            conv = pd.read_csv(conv_path)
            ho_rb = pd.concat([ho, conv], ignore_index=True)
            macros["ConvIterMed"] = f"{conv['n_iter'].median():.0f}"
            macros["ConvCapShare"] = f"{100 * (conv['n_iter'] >= 500).mean():.0f}"
        cap = ho[ho.method.isin(["TorchBatchGMM", "StandardizedSklearnGMM"])].groupby("method")["n_iter"].apply(lambda s: 100 * (s >= 50).mean())
        macros["CapShareTB"] = f"{cap.get('TorchBatchGMM', float('nan')):.0f}"
        macros["CapShareSK"] = f"{cap.get('StandardizedSklearnGMM', float('nan')):.0f}"
        DEGENERATE = ["eeg_eye_state", "ailerons"]   # NLL decided by one outlier row / collapse onto discrete values
        PRECACHED = ["eeg_eye_state", "isolet"]      # present in the local cache before registration
        rb_rows = []
        lines.append("\n=== Holdout robustness (post-hoc; HL = Hodges-Lehmann estimate of EGMM - method, 95% CI) ===")
        hm = ho_rb.groupby(["dataset", "method"])["test_nll"].mean().unstack()
        for b in ["TorchBatchGMM", "TorchBatchGMM_fullinit", "BatchEM_converged", "StandardizedSklearnGMM", "OnlineGMM_tuned"]:
            if b not in hm.columns:
                continue
            pair = hm[["EGMM", b]].dropna()
            d = (pair["EGMM"] - pair[b]).to_numpy()
            dim = (pair["EGMM"] - pair[b]) / hmeta.loc[pair.index, "D"].astype(float)
            nd = [x for x in pair.index if x not in DEGENERATE]
            hl, lo, hi = hodges_lehmann((pair.loc[nd, "EGMM"] - pair.loc[nd, b]).to_numpy())
            hld, lod, hid = hodges_lehmann(dim.loc[nd].to_numpy())
            res = {"b": b, "n": len(pair), "wtl": "/".join(map(str, wtl(pd.Series(d)))), "hl": hl, "lo": lo, "hi": hi,
                   "hld": hld, "lod": lod, "hid": hid,
                   "p_dim": wilcoxon_p(dim, pd.Series(0.0, index=dim.index)), "p_sign": sign_test_p(d)}
            for tag, excl in [("deg", DEGENERATE), ("pre", PRECACHED)]:
                keep = [x for x in pair.index if x not in excl]
                dd = pair.loc[keep, "EGMM"] - pair.loc[keep, b]
                res[tag + "_wtl"] = "/".join(map(str, wtl(dd)))
                res[tag + "_p"] = wilcoxon_p(pair.loc[keep, "EGMM"], pair.loc[keep, b])
                res[tag + "_sign"] = sign_test_p(dd.to_numpy())
            rb_rows.append(res)
            t = "R" + "".join(ch for ch in b if ch.isalpha())
            macros[t + "DegSignP"] = fmt_p(res["deg_sign"]).strip("$")
            macros[t + "SignP"] = fmt_p(res["p_sign"]).strip("$")
            macros[t + "DimCI"] = f"[{lod:+.3f}, {hid:+.3f}]"
            # Tail of the paired differences (non-degenerate datasets): the HL interval assumes symmetric
            # differences, so also report the median with its distribution-free order-statistic CI.
            ddn = (pair.loc[nd, "EGMM"] - pair.loc[nd, b]).sort_values()
            n_nd = len(ddn)
            k = max(j for j in range(1, n_nd // 2 + 1) if stats.binom.cdf(j - 1, n_nd, 0.5) <= 0.025)
            cov_ = 1.0 - 2.0 * stats.binom.cdf(k - 1, n_nd, 0.5)
            macros[t + "Med"] = f"{ddn.median():+.2f}"
            macros[t + "MeanD"] = f"{ddn.mean():+.2f}"
            macros[t + "MedCI"] = f"[{ddn.iloc[k - 1]:+.2f}, {ddn.iloc[n_nd - k]:+.2f}]"
            macros[t + "MedCov"] = f"{100 * cov_:.1f}"
            macros[t + "MaxGain"] = f"{-ddn.min():.2f}"
            losses = ddn[ddn > 1.0].sort_values(ascending=False)
            if b == "TorchBatchGMM" and "EGMM_polish0" in hd["test_nll"]:
                for dn in losses.index:  # what the closing polish does on the datasets e-GMM loses
                    tag_ds = "".join(ch for ch in dn if ch.isalpha())
                    macros["HPolZeroVsTB" + tag_ds] = f"{hd['test_nll']['EGMM_polish0'][dn] - hd['test_nll']['TorchBatchGMM'][dn]:+.2f}"
                    macros["HDeadOn" + tag_ds] = f"{hd['dead_components']['EGMM'][dn]:.1f}"
                    macros["HKOn" + tag_ds] = str(int(hmeta.loc[dn, "K"]))
            macros[t + "NBigLoss"] = str(len(losses))
            if len(losses):
                macros[t + "BigLossMin"] = f"{losses.min():.1f}"
                macros[t + "BigLossMax"] = f"{losses.max():.1f}"
            macros[t + "BigLosses"] = join_and(
                f"\\texttt{{{dn.replace('_', chr(92) + '_')}}} ($+{v:.2f}$)" for dn, v in losses.items()) or "none"
            lines.append(
                f"  tail EGMM vs {b}: median={ddn.median():+.3f} order-stat CI [{ddn.iloc[k - 1]:+.3f}, {ddn.iloc[n_nd - k]:+.3f}] "
                f"(coverage {100 * cov_:.1f}%), largest gain {-ddn.min():.2f}, losses > 1 nat: "
                + (", ".join(f"{dn}={v:+.2f}" for dn, v in losses.items()) or "none")
                + f"; sign test excl. degenerate p={fmt_p_txt(res['deg_sign'])}"
            )
            lines.append(
                f"  EGMM vs {b:24s} n={res['n']} W/T/L={res['wtl']} HL(non-degenerate)={hl:+.3f} [{lo:+.3f}, {hi:+.3f}] nats; per-dim HL={hld:+.4f} "
                f"[{lod:+.4f}, {hid:+.4f}]; p_dim={fmt_p_txt(res['p_dim'])} p_sign={fmt_p_txt(res['p_sign'])}; "
                f"excl. degenerate {res['deg_wtl']} p={fmt_p_txt(res['deg_p'])}; excl. pre-cached {res['pre_wtl']} p={fmt_p_txt(res['pre_p'])}"
            )
            t = "R" + "".join(ch for ch in b if ch.isalpha())
            macros[t + "HL"] = f"{hl:+.2f}"
            macros[t + "CI"] = f"[{lo:+.2f}, {hi:+.2f}]"
            macros[t + "WTL"] = res["wtl"]
            macros[t + "DegWTL"] = res["deg_wtl"]
            macros[t + "DegP"] = fmt_p(res["deg_p"]).strip("$")
        name = {"TorchBatchGMM": r"\texttt{TorchBatchGMM}", "TorchBatchGMM_fullinit": r"\texttt{TorchBatchGMM\_fullinit}",
                "BatchEM_converged": r"Batch EM, full init, converged$^\dagger$", "StandardizedSklearnGMM": r"\texttt{StdSklearnGMM}",
                "OnlineGMM_tuned": r"\texttt{OnlineGMM\_tuned}"}
        t_rb = [
            f"{name[r['b']]} & {r['wtl']} & ${r['hl']:+.2f}$ $[{r['lo']:+.2f}, {r['hi']:+.2f}]$ & ${r['hld']:+.4f}$ & {fmt_p(r['p_dim'])} & "
            f"{fmt_p(r['p_sign'])} & {r['deg_wtl']}, {fmt_p(r['deg_p'])} & {r['pre_wtl']}, {fmt_p(r['pre_p'])} \\\\"
            for r in rb_rows
        ]
        (GEN / "tab_holdout_robust.tex").write_text("\n".join(t_rb) + "\n", encoding="utf-8")

    # ---------------- Table 1 LaTeX body
    def t1_row(a, b, s, short=None):
        name = f"{TEX_NAME.get(a, a)} vs {short or TEX_NAME.get(b, b)}"
        good_n = s["W"] > s["L"]
        good_t = s["med_spd"] > 1.0
        wtl_s = bold_if(f"{s['W']}/{s['T']}/{s['L']}", good_n)
        spd = bold_if(f"{s['med_spd']:.2f}$\\times$", good_t)
        return f"{name} & {wtl_s} & ${s['med_dnll']:+.4f}$ & {fmt_p(s['p_nll'])} & {spd} & {fmt_p(s['p_time'])} & {s['seed_W']}/{s['seed_T']}/{s['seed_L']} \\\\"

    # Main Table 1: A = pre-registered holdout, B = development suite, C = ablations, D = systems run.
    # Appendix table: E = large-N subset, F = later-added development datasets.
    t1 = []
    if ho_rows:
        t1.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{A. Pre-registered holdout ({macros['nHoldout']} new OpenML datasets, code frozen before fitting)}}}} \\\\")
        t1 += [t1_row(a, b, st) for a, b, st in ho_rows]
        t1.append("\\midrule")
    t1.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{B. All {n_ds} development datasets (Friedman $\\chi^2_F = {f_stat:.2f}$, $p = {fmt_p(f_p).strip('$')}$)}}}} \\\\")
    t1 += [t1_row(a, b, s) for a, b, s in panel_a]
    t1.append("\\midrule")
    t1.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{C. Component \\& fairness ablations on the {n_ds} development datasets}}}} \\\\")
    short = {}
    t1 += [t1_row(a, b, s, short.get(b) if a == "EGMM_polish0" else None) for a, b, s in abl_rows]
    if sys_rows:
        t1.append("\\midrule")
        t1.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{D. Algorithm vs.\\ engineering (separate systems run, {macros['nSys']} development datasets)}}}} \\\\")
        t1 += [t1_row(a, b, st) for a, b, st in sys_rows]
    (GEN / "tab_wilcoxon_main.tex").write_text("\n".join(t1) + "\n", encoding="utf-8")
    t1x = [f"\\multicolumn{{7}}{{l}}{{\\emph{{E. Large-scale development subset ($N_{{\\mathrm{{train}}}} > 10^5$, $n = {len(large)}$)}}}} \\\\"]
    t1x += [t1_row(a, b, s) for a, b, s in panel_b]
    t1x.append("\\midrule")
    t1x.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{F. The {len(held)} development datasets added after the original rules were set}}}} \\\\")
    t1x += [t1_row(a, b, s) for a, b, s in panel_h]
    (GEN / "tab_wilcoxon_extra.tex").write_text("\n".join(t1x) + "\n", encoding="utf-8")

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
    macros["SlowerTBList"] = join_and(f"\\texttt{{{d.replace('_', chr(92) + '_')}}} (${v:.2f}\\times$)" for d, v in slower_tb.items()) or "none"
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
    # M-steps (covariance factorizations) per fit: e-GMM runs one per EMA step plus one per executed polish step,
    # batch EM one per iteration. Fewer E-step passes do not imply fewer M-steps.
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from egmm.scaling import auto_hyperparameters as _ahp
    eg_rows = h2h[h2h["method"] == "EGMM"]
    msteps = []
    for r in eg_rows.itertuples():
        hp = _ahp(int(r.N_train), int(r.K), int(r.D), r.cov_type)
        msteps.append(hp.n_steps + max(0.0, float(r.n_iter) - hp.max_epochs))
    ms = pd.Series(msteps, index=eg_rows.index).groupby(eg_rows["dataset"]).mean()
    tb_it = h2h[h2h["method"] == "TorchBatchGMM"].groupby("dataset")["n_iter"].mean()
    macros["MStepsEGMM"] = f"{ms.median():.1f}"
    macros["MStepsMoreN"] = str(int((ms > tb_it.loc[ms.index]).sum()))
    lines.append(f"Median M-steps per fit: EGMM={ms.median():.1f} (T_steps + executed polish) vs TorchBatchGMM iterations "
                 f"{tb_it.median():.1f}; EGMM runs more M-steps on {int((ms > tb_it.loc[ms.index]).sum())}/{len(ms)} datasets")
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
        und_by = {}
        for ds_name, sub in pareto.groupby("dataset"):
            eg = sub[sub["method"] == "EGMM"][["fit_time", "test_nll"]].to_numpy()
            und = []
            for _, r in sub[sub["method"] != "EGMM"].iterrows():
                dom = bool(((eg[:, 0] <= r["fit_time"]) & (eg[:, 1] <= r["test_nll"])).any())
                n_pts += 1
                n_dom += int(dom)
                if not dom:
                    und.append(f"{r['method']}@{int(r['budget'])}")
                    und_by[r["method"]] = und_by.get(r["method"], 0) + 1
            n_ds_all += int(len(und) == 0)
            lines.append(f"  {ds_name:24s}: undominated baseline points = {', '.join(und) if und else 'none'}")
        lines.append(f"  dominated {n_dom}/{n_pts} baseline budget points; fully dominated datasets: {n_ds_all}/{pareto['dataset'].nunique()}")
        macros["ParetoDominated"] = str(n_dom)
        macros["ParetoPoints"] = str(n_pts)
        macros["ParetoFullDom"] = str(n_ds_all)
        macros["ParetoDatasets"] = str(pareto["dataset"].nunique())
        macros["ParetoUndom"] = str(n_pts - n_dom)
        for m, tag in [("OnlineGMM_tuned", "Online"), ("TorchBatchGMM", "TB"), ("SklearnGMM", "SK")]:
            macros["ParetoUndom" + tag] = str(und_by.get(m, 0))
        n_budgets = pareto[pareto["method"] == "TorchBatchGMM"].groupby("dataset")["budget"].nunique().sum()
        macros["ParetoSKSkipped"] = str(int(n_budgets - (pareto["method"] == "SklearnGMM").sum()))

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
            if v == "greedy128x_l20":
                macros["SeedTwentyW"], macros["SeedTwentyL"] = str(r["W"]), str(r["L"])
                macros["SeedTwentyP"], macros["SeedTwentyTime"] = f"{r['p']:.3f}", f"{r['t_ratio']:.2f}"
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
        if "ep3" in g.columns and "creditcard" in g.index:
            macros["EpThreeCredit"] = f"{(g['ep3'] - g['base'])['creditcard']:+.1f}"
        for v, tag in [("seed256x_l10", "SeedBig"), ("ep3", "EpThree")]:
            if v in g.columns:
                d = (g[v] - g["base"]).loc[g.index.intersection(syn)]
                macros[tag + "Min"], macros[tag + "Max"] = f"{d.min():+.2f}", f"{d.max():+.2f}"
        # every dataset x variant (appendix table; nothing is summarized away)
        tm_ = cr.groupby(["dataset", "variant"])["fit_ms"].median().unstack()
        order = [v for v in ["ep3", "pol1", "pol2", "sub64k_r2", "sub128k_r3", "seed64x", "seed256x_l10"] if v in g.columns]
        rows_c = []
        for dsn in sorted(g.index, key=lambda x: int(cr[cr.dataset == x]["N"].iloc[0])):
            cells = []
            for v in order:
                dv = g.loc[dsn, v] - g.loc[dsn, "base"]
                cells.append(f"${sci_tex(dv)}$" if abs(dv) >= 1e3 else f"${dv:+.2f}$")
            rows_c.append(f"\\texttt{{{dsn.replace('_', chr(92) + '_')}}} ({int(cr[cr.dataset == dsn]['N'].iloc[0])}) & " + " & ".join(cells) + " \\\\")
        rows_c.append("\\midrule")
        rows_c.append("Median time ratio & " + " & ".join(f"{(tm_[v] / tm_['base']).median():.2f}$\\times$" for v in order) + " \\\\")
        (GEN / "tab_closing.tex").write_text("\n".join(rows_c) + "\n", encoding="utf-8")
        ep = g["ep3"] - g["base"]
        macros["EpThreeBetter"], macros["nCloseDs"] = str(int((ep < -TIE).sum())), str(len(g))
        macros["EpThreeTime"] = f"{(tm_['ep3'] / tm_['base']).median():.2f}"
        if "covtype_581k" in g.index:
            subc = [g.loc["covtype_581k", v] - g.loc["covtype_581k", "base"] for v in ["sub64k_r2", "sub128k_r3"] if v in g.columns]
            macros["CloseCovSubMin"] = sci_tex(min(subc), 1)
            macros["CloseCovSubMax"] = sci_tex(max(subc), 1)
            gt = cr.groupby(["dataset", "variant"])["test_nll"].mean().unstack()
            subt = [gt.loc["covtype_581k", v] - gt.loc["covtype_581k", "base"] for v in ["sub64k_r2", "sub128k_r3"] if v in gt.columns]
            macros["CloseCovSubTestMin"], macros["CloseCovSubTestMax"] = f"{min(subt):+.2f}", f"{max(subt):+.2f}"
        sb = g["seed256x_l10"] - g["base"]
        macros["SeedBigWorse"] = join_and(f"\\texttt{{{dn.replace('_', chr(92) + '_')}}} (${sb[dn]:+.2f}$)" for dn in sb[sb > 0.5].index)
        lines.append("  closing-refinement table (val dNLL vs base, mean over seeds): " + "; ".join(
            f"{dsn}: " + ", ".join(f"{v}={g.loc[dsn, v] - g.loc[dsn, 'base']:+.3g}" for v in order) for dsn in g.index))
    if (dsd / "precision_f32_vs_f64.csv").exists():
        pr = pd.read_csv(dsd / "precision_f32_vs_f64.csv")
        if "variant" in pr.columns:
            g = pr.groupby(["dataset", "variant"])["test_nll"].mean().unstack()
            d = (g["float32"] - g["float64"]).abs()
        else:
            g = pr.groupby("dataset").mean(numeric_only=True)
            d = (g["test_float32"] - g["test_float64"]).abs()
        lines.append(f"  float32 vs float64 training (shared RNG): max |d mean test NLL| = {d.max():.3f} ({d.idxmax()})")
        macros["PrecMax"], macros["PrecMaxName"] = f"{d.max():.2f}", d.idxmax().replace("_", r"\_")
        if "variant" in pr.columns:
            wide = pr.pivot_table(index=["dataset", "seed"], columns="variant", values="test_nll")
            per_run = (wide["float32"] - wide["float64"]).abs()
            macros["PrecRunMax"] = f"{per_run.max():.2f}"
            macros["PrecRunMaxName"] = per_run.idxmax()[0].replace("_", r"\_")
            lines.append(f"  per-run max |d test NLL| = {per_run.max():.3f} ({per_run.idxmax()})")
        macros["PrecN"] = str(len(d))

    if (dsd / "split_psd_check.csv").exists():
        sp = pd.read_csv(dsd / "split_psd_check.csv")
        macros["SplitTrials"] = f"{len(sp):,}".replace(",", "{,}")
        macros["SplitViolClassical"] = str(int((sp["lmin_classical"] < 0).sum()))
        macros["SplitViolWedderburn"] = str(int((sp["lmin_wedderburn"] < 0).sum()))
        macros["SplitAdvViol"] = str(int((sp["lmin_classical_adversarial"] < 0).sum()))
        lines.append(f"  split PSD check: {len(sp)} trials; classical violations (5-step power iteration) = {int((sp['lmin_classical'] < 0).sum())}; "
                     f"Wedderburn = {int((sp['lmin_wedderburn'] < 0).sum())}; classical with 1-step power iteration = {int((sp['lmin_classical_adversarial'] < 0).sum())}")

    # ---------------- revised paper: stronger batch-EM baselines, large N, ablations (docs/experiments/next)
    from experiments.next.paper_results import add_next_results

    add_next_results(EXP, GEN, macros, lines)

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

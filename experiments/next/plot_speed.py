"""Figures for the revised paper from the speed2 re-timing (docs/experiments/next/speed2_*.csv).

    python experiments/next/plot_speed.py   # -> docs/experiments/next/fig_speed_vs_n.png, fig_large_budget.png
                                            #    (copy both to paper/figures/)

* fig_speed_vs_n: e-GMM's speedup over the fastest batch-EM variant (eager, sync-light or CUDA graph, with a
  convergence check every 5 iterations or every iteration) on each dataset, against training-set size.
* fig_large_budget: test NLL against fit time (as a ratio to e-GMM's time in the same session) at 1-10
  million rows: batch EM with fixed budgets of 2-20 iterations (speed2b) and with its default stopping rule,
  full-data-seeded batch EM (speed2), and e-GMM with 1-3 epochs (speed2b).
All points are 3-seed means.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.next.paper_results import _load, _time_ratio  # noqa: E402
from experiments.next.speed2_results import BUDGETS, _add_best, budget_points  # noqa: E402
from experiments.plot_results import _setup_style  # noqa: E402

EXP = ROOT / "docs" / "experiments"
SUITES = [("speed2_dev", "Development (33)", "o", "#4C72B0"), ("speed2_holdout", "Holdout, post hoc (16)", "^", "#DD8452"),
          ("speed2_large", "1–10 M rows (8)", "s", "#55A868")]
NAMES = {"higgs": "HIGGS", "susy": "SUSY"}


def speed_vs_n() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    for stage, label, marker, color in SUITES:
        df = _load(EXP, stage)
        dfb, _ = _add_best(df)
        n = df.groupby("dataset")["N_train"].first()
        spd = _time_ratio(dfb, "BatchBest")
        ax.scatter(n[spd.index], spd, marker=marker, s=34, color=color, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color="#333333", lw=0.8)
    ax.axvline(1e5, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel(r"Training rows $N_{\mathrm{train}}$")
    ax.set_ylabel("Speedup over fastest batch EM")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, which="major", alpha=0.25)
    out = EXP / "next" / "fig_speed_vs_n.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


def large_budget() -> None:
    sl, sb = _load(EXP, "speed2_large"), _load(EXP, "speed2b_large")
    br, bq = budget_points(sb)
    dfb, _ = _add_best(sl)
    g = dfb.groupby(["dataset", "method"])[["fit_time", "test_nll"]].mean().unstack()
    t, nll = g["fit_time"], g["test_nll"]
    meta = sl.groupby("dataset")[["N_train", "cov_type"]].first()
    order = meta.assign(real=[0 if d.startswith("syn") else 1 for d in meta.index]).sort_values(["real", "cov_type", "N_train"],
                                                                                                ascending=[True, False, True]).index
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.4))
    for ax, ds in zip(axes.ravel(), order):
        ref = nll.loc[ds, "EGMM_r0"]
        bt = [br.loc[ds, f"it{b}"] for b in BUDGETS]
        bn = [bq.loc[ds, f"it{b}"] - ref for b in BUDGETS]
        ax.plot(bt, bn, "-o", color="#4C72B0", ms=4, lw=1.1, label="batch EM, 2–20 iterations")
        for x, y, b in zip(bt, bn, BUDGETS):
            ax.annotate(str(b), (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7, color="#4C72B0")
        d_def = nll.loc[ds, "TorchBatchGMM"] - ref
        d_fi = nll.loc[ds, "TorchBatchGMM_fullinit"] - ref
        ax.plot([t.loc[ds, "BatchBest"] / t.loc[ds, "EGMM_r0"]], [d_def], "D", color="#4C72B0", ms=5, label="batch EM, default stop")
        ax.plot([t.loc[ds, "TorchBatchGMM_fullinit"] / t.loc[ds, "EGMM_r0"]], [d_fi], "v", color="#8172B3", ms=5,
                label="batch EM, full-data seeding")
        et = [br.loc[ds, k] for k in ("e1", "e2", "e3")]
        en = [bq.loc[ds, k] - ref for k in ("e1", "e2", "e3")]
        ax.plot(et, en, "-s", color="#C44E52", ms=4, lw=1.1, label="e-GMM, 1–3 epochs")
        ax.axhline(0.0, color="#999999", lw=0.7, ls="--")
        ax.axvline(1.0, color="#555555", lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{NAMES.get(ds, 'synthetic')} ({meta.loc[ds, 'N_train'] / 1e6:.1f} M, {meta.loc[ds, 'cov_type']})", fontsize=9.5)
        vals = en + bn[2:] + [d_def, d_fi]
        lo, hi = min(vals), max(vals)
        pad = 0.15 * max(hi - lo, 0.05)
        ax.set_ylim(lo - pad, hi + pad)  # 2- and 3-iteration points may lie above the frame
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("Fit time / e-GMM fit time", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Test NLL minus e-GMM", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = EXP / "next" / "fig_large_budget.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    _setup_style()
    speed_vs_n()
    large_budget()


if __name__ == "__main__":
    main()

"""Figure for the revised paper: e-GMM's speedup over the fastest batch-EM variant against training-set size.

    python experiments/next/plot_speed.py   # -> docs/experiments/next/fig_speed_vs_n.png (copy to paper/figures/)

One point per dataset (3-seed means) from the follow-up runs (docs/experiments/next/{dev,holdout,large}.csv).
Filled markers: measured; hollow markers: the conservative estimate for batch EM checking convergence every
iteration (experiments/next/paper_results.py), drawn only where it differs from the measurement by >2%.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.next.paper_results import _add_best, _load, _time_ratio  # noqa: E402
from experiments.plot_results import _setup_style  # noqa: E402

EXP = ROOT / "docs" / "experiments"
SUITES = [("dev", "Development (33)", "o", "#4C72B0"), ("holdout", "Holdout, post hoc (16)", "^", "#DD8452"),
          ("large", "1--10 M rows (8)", "s", "#55A868")]


def main() -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    for stage, label, marker, color in SUITES:
        df = _load(EXP, stage)
        dfb, _ = _add_best(df, "BatchBest")
        dfb, _ = _add_best(dfb, "BatchBestAdj", adjust=True)
        n = df.groupby("dataset")["N_train"].first()
        spd, adj = _time_ratio(dfb, "BatchBest"), _time_ratio(dfb, "BatchBestAdj")
        ax.scatter(n[spd.index], spd, marker=marker, s=34, color=color, label=label.replace("--", "–"), zorder=3)
        moved = (adj / spd - 1).abs() > 0.02
        ax.scatter(n[adj.index][moved], adj[moved], marker=marker, s=34, facecolors="none", edgecolors=color, zorder=3)
        for ds in adj.index[moved]:
            ax.plot([n[ds], n[ds]], [spd[ds], adj[ds]], color=color, lw=0.7, alpha=0.6, zorder=2)
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


if __name__ == "__main__":
    main()

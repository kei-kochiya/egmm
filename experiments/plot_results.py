"""Generate clean, single-panel, publication-quality figures for e-GMM (zero overlapping labels)."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
        }
    )


def plot_scaling_laws_individual(csv_path: Path, out_dir: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    for cov_type in ["diag", "full"]:
        sub = df[df["cov_type"] == cov_type]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
        for i, T in enumerate(sorted(sub["T"].unique())):
            g = sub[sub["T"] == T].groupby("alpha_0")["nll_diff"].median()
            ax.plot(
                g.index,
                g.values,
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=colors[i % len(colors)],
                label=f"E = {T} epochs",
            )
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.2, label="50-iter Batch EM baseline")
        ax.set_xscale("log")
        alpha_ticks = [0.05, 0.12, 0.25, 0.45, 0.7, 0.85]
        ax.xaxis.set_major_locator(mticker.FixedLocator(alpha_ticks))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(["0.05", "0.12", "0.25", "0.45", "0.70", "0.85"]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel(r"Peak EMA step size $\alpha_0$ (log scale)")
        ax.set_ylabel(r"Median Validation $\Delta$NLL (lower is better)")
        cov_title = "Diagonal Covariance" if cov_type == "diag" else "Full Covariance"
        ax.set_title(f"e-GMM Step-Size Sensitivity ({cov_title})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=True, facecolor="white", framealpha=0.92, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_scaling_{cov_type}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_pareto_individual(csv_path: Path, out_dir: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    datasets = sorted(df["dataset"].unique())
    style_map = {
        "EGMM": ("#d62728", "o", "-", 2.4, "e-GMM (Ours)"),
        "TorchBatchGMM": ("#1f77b4", "s", "--", 1.9, "TorchBatchGMM (GPU)"),
        "OnlineGMM_tuned": ("#2ca02c", "^", "-.", 1.8, "OnlineGMM (tuned)"),
        "SklearnGMM": ("#555555", "D", ":", 1.7, "SklearnGMM (CPU)"),
    }
    nice_titles = {
        "covtype_581k": "Covertype (N=406.7k, D=54, K=32, full)",
        "miniboone": "MiniBooNE (N=91.0k, D=50, K=24, diag)",
        "mnist784_pca48": "MNIST-PCA48 (N=49.0k, D=48, K=24, full)",
        "syn_anisotropic_30k": "Syn-Anisotropic (N=21.0k, D=24, K=16, full)",
        "syn_imbalanced_40k": "Syn-Imbalanced (N=28.0k, D=16, K=24, full)",
        "syn_scale200k": "Syn-Scale200k (N=140.0k, D=32, K=32, full)",
    }

    for ds_name in datasets:
        fig, ax = plt.subplots(figsize=(5.4, 4.1))
        sub = df[df["dataset"] == ds_name]
        for mname, (col, mk, ls, lw, lbl) in style_map.items():
            m = sub[sub["method"] == mname].sort_values("fit_time")
            if m.empty:
                continue
            ax.plot(
                m["fit_time"],
                m["test_nll"],
                color=col,
                marker=mk,
                linestyle=ls,
                linewidth=lw,
                markersize=5.5,
                label=lbl,
            )
        others = sub[sub["method"] != "SklearnGMM"]["test_nll"]
        sk = sub[sub["method"] == "SklearnGMM"]["test_nll"]
        lo, hi = float(others.min()), float(others.max())
        span = max(hi - lo, 1e-6)
        if len(sk) and (sk.min() < lo - 1.5 * span or sk.max() > hi + 1.5 * span):
            ax.set_ylim(lo - 0.15 * span, hi + 0.15 * span)
            ax.text(0.98, 0.04, f"raw SklearnGMM off-scale (NLL {sk.min():.1f} to {sk.max():.1f})",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
        ax.set_xscale("log")
        ax.set_title(nice_titles.get(ds_name, ds_name))
        ax.set_xlabel("Wall-Clock Fit Time (s, log scale)")
        ax.set_ylabel("Test NLL (lower = better)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=True, facecolor="white", framealpha=0.92, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_pareto_{ds_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_cd_ranks(csv_path: Path, out_path: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    agg = df.groupby(["dataset", "method"])[["test_nll", "fit_time"]].mean().reset_index()
    piv_nll = agg.pivot(index="dataset", columns="method", values="test_nll")

    methods = [
        m
        for m in [
            "EGMM",
            "TorchBatchGMM",
            "StandardizedSklearnGMM",
            "OnlineGMM_tuned",
            "NaiveMinibatchGMM",
            "EMeansInitGMM",
            "SklearnGMM",
        ]
        if m in piv_nll.columns
    ]
    label_map = {
        "EGMM": "e-GMM (Ours)",
        "StandardizedSklearnGMM": "StdSklearnGMM (CPU)",
        "SklearnGMM": "SklearnGMM (Raw CPU)",
        "TorchBatchGMM": "TorchBatchGMM (GPU)",
        "OnlineGMM_tuned": "OnlineGMM (tuned)",
        "EMeansInitGMM": "EMeansInitGMM",
        "NaiveMinibatchGMM": "NaiveMinibatchGMM",
    }
    comp = piv_nll[methods].dropna()
    ranks = comp.rank(axis=1, ascending=True).mean(axis=0).sort_values()

    k_m = len(methods)
    n_d = len(comp)
    q_table = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}
    cd = q_table.get(k_m, 2.85) * math.sqrt(k_m * (k_m + 1) / (6.0 * max(1, n_d)))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = ["#d62728" if m == "EGMM" else "#4c72b0" for m in ranks.index]
    y_pos = np.arange(len(ranks))
    bars = ax.barh(y_pos, ranks.values, height=0.62, color=colors, alpha=0.90, edgecolor="#222222", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label_map.get(m, m) for m in ranks.index])
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(ranks.max() + 0.90, ranks.iloc[0] + cd + 0.75))
    cd_thresh = ranks.iloc[0] + cd
    ax.axvline(cd_thresh, color="#111111", linestyle="--", linewidth=1.3, zorder=2, label=f"Best + Nemenyi CD ({cd_thresh:.2f})")
    for bar, val in zip(bars, ranks.values):
        ax.text(
            val + 0.09,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=9.2,
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.88},
        )

    ax.set_xlabel(f"Mean Test NLL Rank Across {n_d} Development Datasets (1 = best)")
    ax.set_title(f"Mean Test NLL Ranks & Nemenyi Critical Difference (CD = {cd:.2f}, p < 0.05)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _repel_labels(
    ax: plt.Axes,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    labels: list[str],
    seed_offsets: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Place scatter annotations without overlaps in exact [0, 1] axes coordinates."""
    to_axes = ax.transData + ax.transAxes.inverted()
    to_data = ax.transAxes + ax.transData.inverted()

    pts_disp = to_axes.transform(np.column_stack([pts_x, pts_y]))
    px = pts_disp[:, 0]
    py = pts_disp[:, 1]

    tx = px.copy()
    ty = py.copy()
    for i, name in enumerate(labels):
        dx, dy = seed_offsets.get(name, (0.02, 0.035))
        tx[i] = np.clip(px[i] + dx, 0.06, 0.92)
        ty[i] = np.clip(py[i] + dy, 0.05, 0.95)

    widths = np.array([0.0092 * len(lbl) + 0.018 for lbl in labels])
    heights = np.full(len(labels), 0.042)

    for _ in range(700):
        forces_x = np.zeros(len(labels))
        forces_y = np.zeros(len(labels))

        # Repel label-label overlaps
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                dx = tx[i] - tx[j]
                dy = ty[i] - ty[j]
                req_x = 0.5 * (widths[i] + widths[j]) + 0.012
                req_y = 0.5 * (heights[i] + heights[j]) + 0.009
                if abs(dx) < req_x and abs(dy) < req_y:
                    ov_x = req_x - abs(dx)
                    ov_y = req_y - abs(dy)
                    sx = 1.0 if dx >= 0 else -1.0
                    sy = 1.0 if dy >= 0 else -1.0
                    if abs(dy) < 1e-4:
                        sy = 1.0 if (i % 2 == 0) else -1.0
                    if ov_y < ov_x:
                        forces_y[i] += 0.36 * sy * ov_y
                        forces_y[j] -= 0.36 * sy * ov_y
                        forces_x[i] += 0.08 * sx * ov_x
                        forces_x[j] -= 0.08 * sx * ov_x
                    else:
                        forces_x[i] += 0.32 * sx * ov_x
                        forces_x[j] -= 0.32 * sx * ov_x
                        forces_y[i] += 0.14 * sy * ov_y
                        forces_y[j] -= 0.14 * sy * ov_y

        # Repel label-marker overlaps
        for i in range(len(labels)):
            for j in range(len(labels)):
                dx = tx[i] - px[j]
                dy = ty[i] - py[j]
                req_x = 0.5 * widths[i] + 0.022
                req_y = 0.5 * heights[i] + 0.024
                if abs(dx) < req_x and abs(dy) < req_y:
                    sy = 1.0 if dy >= 0 else -1.0
                    sx = 1.0 if dx >= 0 else -1.0
                    forces_y[i] += 0.28 * sy * (req_y - abs(dy))
                    forces_x[i] += 0.16 * sx * (req_x - abs(dx))

            # Mild anchor tether
            dx0, dy0 = seed_offsets.get(labels[i], (0.02, 0.035))
            forces_x[i] -= 0.028 * (tx[i] - (px[i] + dx0))
            forces_y[i] -= 0.028 * (ty[i] - (py[i] + dy0))

        tx = np.clip(tx + forces_x, 0.04 + 0.5 * widths, 0.96 - 0.5 * widths)
        ty = np.clip(ty + forces_y, 0.05, 0.95)

    out_data = to_data.transform(np.column_stack([tx, ty]))
    return out_data[:, 0], out_data[:, 1]


def plot_speedup_vs_nll(csv_path: Path, out_path: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    agg = df.groupby(["dataset", "method"])[["test_nll", "fit_time"]].mean().reset_index()
    piv_nll = agg.pivot(index="dataset", columns="method", values="test_nll")
    piv_time = agg.pivot(index="dataset", columns="method", values="fit_time")

    pair_nll = piv_nll[["EGMM", "TorchBatchGMM"]].dropna()
    pair_time = piv_time.loc[pair_nll.index, ["EGMM", "TorchBatchGMM"]]
    speedup = pair_time["TorchBatchGMM"] / pair_time["EGMM"]
    dnll = pair_nll["EGMM"] - pair_nll["TorchBatchGMM"]

    diag_sets = set(df[df["cov_type"] == "diag"]["dataset"].unique())
    n_total = len(pair_nll)
    n_diag = sum(1 for n in pair_nll.index if n in diag_sets)
    n_full = n_total - n_diag
    n_qual_win = int((dnll < -1e-3).sum())
    n_speed_win = int((speedup > 1.0).sum())
    n_strict = int(((dnll < -1e-3) & (speedup > 1.0)).sum())

    fig, ax = plt.subplots(figsize=(10.2, 6.2))

    # Data-driven limits (never clip a dataset), with the historical framing as the minimum extent.
    xlim = (min(0.56, float(speedup.min()) * 0.85), max(10.8, float(speedup.max()) * 1.15))
    ylim = (min(-19.0, float(dnll.min()) * 1.25), max(1.15, float(dnll.max()) * 1.25))
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.5, linscale=1.35)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    yticks = [0.5, 0.0, -0.5, -1.0, -1.5, -3.0, -6.0, -10.0, -15.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:g}" for y in yticks])

    # Subtle shading for strictly dominating quadrant (faster & lower NLL)
    y0_axes = (ax.transData + ax.transAxes.inverted()).transform([[1.0, 0.0]])[0, 1]
    ax.axvspan(1.0, xlim[1], ymin=0.0, ymax=float(y0_axes), color="#2ca02c", alpha=0.045)
    ax.axvline(1.0, color="#666666", linestyle="--", linewidth=1.1)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.1)

    for is_diag, marker, color, lbl in [
        (False, "o", "#d62728", f"Full covariance ({n_full} datasets)"),
        (True, "s", "#1f77b4", f"Diagonal covariance ({n_diag} datasets)"),
    ]:
        idxs = [n for n in pair_nll.index if (n in diag_sets) == is_diag]
        ax.scatter(
            speedup.loc[idxs],
            dnll.loc[idxs],
            s=68,
            marker=marker,
            color=color,
            alpha=0.90,
            edgecolor="#111111",
            linewidth=0.7,
            zorder=4,
            label=lbl,
        )

    names = list(pair_nll.index)
    seed_offsets = {
        "syn_scale400k_d256_diag": (0.07, -0.02),
        "syn_highdim128_50k": (-0.02, -0.05),
        "segment": (0.04, -0.04),
        "syn_highk64_60k": (-0.04, -0.06),
        "syn_imbalanced_40k": (0.01, 0.05),
        "syn_overlapping_30k": (0.02, -0.05),
        "syn_anisotropic_30k": (-0.04, -0.04),
        "philippine": (-0.02, -0.05),
        "gas_drift": (-0.03, 0.05),
        "digits_pca32": (0.03, 0.04),
        "optdigits": (0.02, 0.05),
        "syn_scale200k": (-0.04, -0.04),
        "letter": (-0.03, 0.04),
        "mnist784_pca48": (-0.03, -0.05),
        "california_housing": (-0.03, 0.06),
        "satimage": (-0.04, 0.04),
        "shuttle": (0.03, -0.03),
        "fashion_mnist_pca48": (-0.02, -0.05),
        "spambase": (0.05, 0.02),
        "pendigits": (-0.04, -0.04),
        "har": (-0.04, -0.04),
        "magic04": (-0.02, 0.05),
        "waveform5000": (-0.04, 0.04),
        "electricity": (-0.02, 0.05),
        "syn_scale500k_diag": (0.04, 0.03),
        "miniboone": (-0.03, -0.05),
        "syn_scale300k_k64_full": (0.05, -0.04),
        "higgs_98k": (0.04, -0.04),
        "jannis": (0.05, -0.02),
        "numerai28.6": (0.03, 0.05),
        "covtype_581k": (-0.06, -0.02),
        "creditcard": (0.02, -0.05),
        "skin_nonskin": (-0.02, 0.05),
    }
    lx, ly = _repel_labels(
        ax,
        speedup.loc[names].to_numpy(),
        dnll.loc[names].to_numpy(),
        names,
        seed_offsets,
    )
    for i, name in enumerate(names):
        ax.annotate(
            name,
            xy=(speedup.loc[name], dnll.loc[name]),
            xytext=(lx[i], ly[i]),
            fontsize=7.8,
            ha="center",
            va="center",
            color="#222222",
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.65, "shrinkA": 2, "shrinkB": 4},
            zorder=5,
        )

    ax.text(
        xlim[1] * 0.92,
        -13.0,
        f"Faster and lower NLL: {n_strict}/{n_total}\n(NLL wins {n_qual_win}/{n_total}; faster on {n_speed_win}/{n_total})",
        ha="right",
        va="center",
        fontsize=9.0,
        color="#1d6b1d",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef8ee", "edgecolor": "#7ab87a", "alpha": 0.92},
    )
    ax.set_xlabel(r"Wall-Clock Speedup vs TorchBatchGMM (log scale, $>1\times$ = e-GMM faster)")
    ax.set_ylabel(r"Test $\Delta$NLL = e-GMM $-$ Batch ($<0$ = e-GMM better, symlog)")
    ax.set_title(f"e-GMM vs GPU TorchBatchGMM Across {n_total} Benchmark Datasets (3-Seed Mean)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.94, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_small_dataset_generalization(csv_path: Path, out_path: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df["work"] = df["n_train"] * df["n_components"] * df["n_features"]
    agg = df.groupby(["dataset", "work", "method"])[["test_nll", "fit_time"]].mean().reset_index()
    pt = agg.pivot(index=["dataset", "work"], columns="method", values="fit_time").reset_index().sort_values("work")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    style = [
        ("EGMM_auto", "#d62728", "o", "-", 2.2, "e-GMM (device='auto': CPU/CUDA switch)"),
        ("EGMM_cuda", "#ff7f0e", "^", "--", 1.8, "e-GMM (forced CUDA)"),
        ("TorchBatchGMM", "#1f77b4", "s", "-.", 1.8, "TorchBatchGMM (CUDA)"),
        ("SklearnGMM", "#555555", "D", ":", 1.8, "SklearnGMM (CPU)"),
    ]
    for col, c, mk, ls, lw, lbl in style:
        if col not in pt.columns:
            continue
        ax.plot(pt["work"], pt[col] * 1e3, color=c, marker=mk, linestyle="none", markersize=5.6, label=lbl)

    ax.axvline(1.5e5, color="#888888", linestyle="--", linewidth=1.1, label="Full-cov auto CPU/CUDA switch (1.5e5)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    vals = pt[[c[0] for c in style if c[0] in pt.columns]].to_numpy() * 1e3
    ax.set_ylim(0.8 * np.nanmin(vals), 1.6 * np.nanmax(vals))
    ax.set_xlabel(r"Workload Complexity $N_{\mathrm{train}} \cdot K \cdot D$ (log scale; 16 small datasets)")
    ax.set_ylabel("Mean Wall-Clock Fit Time (ms, log scale)")
    ax.set_title("Small Datasets: Adaptive CPU/GPU Routing")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=8.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_holdout_dnll(csv_path: Path, out_path: Path, conv_path: Path | None = None) -> None:
    """Per-dataset holdout differences (e-GMM minus baseline) on the datasets without metric artefacts."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if conv_path is not None and conv_path.exists():
        df = pd.concat([df, pd.read_csv(conv_path)], ignore_index=True)
    m = df.groupby(["dataset", "method"])["test_nll"].mean().unstack()
    meta = df.groupby("dataset")[["D", "cov_type"]].first()
    m = m.drop(index=[d for d in ["eeg_eye_state", "ailerons"] if d in m.index])
    base = [("TorchBatchGMM", "#1f77b4", "s", "TorchBatchGMM"),
            ("StandardizedSklearnGMM", "#555555", "D", "StdSklearnGMM"),
            ("OnlineGMM_tuned", "#2ca02c", "^", "OnlineGMM_tuned")]
    order = (m["EGMM"] - m["TorchBatchGMM"]).sort_values().index
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    y = np.arange(len(order))
    for j, (b, c, mk, lbl) in enumerate(base):
        d = (m.loc[order, "EGMM"] - m.loc[order, b]).to_numpy()
        ax.scatter(d, y + (j - 1) * 0.22, color=c, marker=mk, s=26, label=f"vs {lbl}", zorder=3)
    ax.axvline(0.0, color="#222222", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d} (D={int(meta.loc[d, 'D'])}, {meta.loc[d, 'cov_type']})" for d in order], fontsize=8.2)
    ax.set_xlabel(r"Test $\Delta$NLL = e-GMM $-$ baseline (nats; $<0$ = e-GMM better)")
    ax.set_title(f"Pre-registered holdout: {len(order)} datasets without metric artefacts")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="lower right", fontsize=8.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    _setup_style()
    exp_dir = Path("docs/experiments")
    plot_scaling_laws_individual(exp_dir / "scaling_sweep.csv", exp_dir)
    plot_pareto_individual(exp_dir / "benchmark_pareto.csv", exp_dir)
    plot_cd_ranks(exp_dir / "benchmark_head_to_head.csv", exp_dir / "fig_cd_ranks.png")
    plot_speedup_vs_nll(exp_dir / "benchmark_head_to_head.csv", exp_dir / "fig_speedup_vs_nll.png")
    plot_small_dataset_generalization(exp_dir / "small_dataset_eval.csv", exp_dir / "fig_small_dataset_eval.png")
    plot_holdout_dnll(exp_dir / "holdout_head_to_head.csv", exp_dir / "fig_holdout_dnll.png")
    print("Generated clean single-panel figures in docs/experiments/")

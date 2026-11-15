"""How often would the classical split delta = Delta * sqrt(v' S v) * v violate PSD in practice?

Review question: Theorem 1 guarantees PSD for the Wedderburn split with *any* direction v; does the
classical split actually fail with the deployed direction (5 float32 power-iteration steps from e_{j*})?
We sample covariances from several spectra, run exactly the deployed power iteration, and record the
smallest eigenvalue of S - delta delta' for both splits. A 1-step power iteration is included as a
deliberately poor direction. CPU only.

    python experiments/design_studies/split_psd_check.py  # -> docs/experiments/design_studies/split_psd_check.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parents[2] / "docs" / "experiments" / "design_studies" / "split_psd_check.csv"


def spectrum(kind: str, d: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "wishart":
        a = rng.standard_normal((d, 2 * d))
        return np.linalg.eigvalsh(a @ a.T / (2 * d))
    if kind == "exp_decay":
        return np.exp(-rng.uniform(0.2, 1.0) * np.arange(d))
    if kind == "flat_tiny_tail":
        lam = np.ones(d)
        lam[d // 2 :] = 10.0 ** rng.uniform(-8, -4, size=d - d // 2)
        return lam
    if kind == "near_degenerate":
        lam = 10.0 ** rng.uniform(-7, 0, size=d)
        lam[0] = 1.0
        return lam
    raise ValueError(kind)


def power_dir(S32: torch.Tensor, steps: int) -> torch.Tensor:
    j = int(torch.argmax(torch.diagonal(S32)))
    v = torch.zeros(S32.shape[0], dtype=torch.float32)
    v[j] = 1.0
    for _ in range(steps):
        v = S32 @ v
        v = v / v.norm().clamp_min(1e-12)
    return v.double()


def main(trials_per_cell: int = 250, delta: float = 0.5, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for d in (8, 16, 32, 54, 128):
        for kind in ("wishart", "exp_decay", "flat_tiny_tail", "near_degenerate"):
            for t in range(trials_per_cell):
                lam = spectrum(kind, d, rng)
                q, _ = np.linalg.qr(rng.standard_normal((d, d)))
                S = torch.from_numpy((q * lam) @ q.T)
                S = 0.5 * (S + S.T)
                out = {"D": d, "spectrum": kind, "trial": t}
                for tag, steps in (("", 5), ("_adversarial", 1)):
                    v = power_dir(S.float(), steps)
                    vSv = float(v @ S @ v)
                    cl = delta * np.sqrt(max(vSv, 0.0)) * v
                    lmin_cl = float(torch.linalg.eigvalsh(S - torch.outer(cl, cl))[0])
                    out["lmin_classical" + tag] = lmin_cl / float(lam.max())
                    if not tag:
                        wd = delta * (S @ v) / np.sqrt(max(vSv, 1e-300))
                        out["lmin_wedderburn"] = float(torch.linalg.eigvalsh(S - torch.outer(wd, wd))[0]) / float(lam.max())
                rows.append(out)
    df = pd.DataFrame(rows)
    # float64 eigvalsh round-off: treat |lmin| below 1e-12 (relative to the top eigenvalue) as zero
    for c in ("lmin_classical", "lmin_classical_adversarial", "lmin_wedderburn"):
        df.loc[df[c].abs() < 1e-12, c] = 0.0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} trials; classical violations: {(df.lmin_classical < 0).sum()}, "
          f"wedderburn: {(df.lmin_wedderburn < 0).sum()}, classical (1 step): {(df.lmin_classical_adversarial < 0).sum()}")


if __name__ == "__main__":
    main()

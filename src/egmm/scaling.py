"""Scaling laws and cosine decay schedules for e-GMM."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EGMMHyperparams:
    batch_size: int
    max_epochs: int
    alpha_0: float
    alpha_min: float
    revive_every: int
    init_usage_scale: float = 1.0
    polish_steps: int = 0
    revive_points: int = 0
    steps_per_epoch: int = 1


def compute_cosine_alpha(
    step: int,
    total_steps: int,
    alpha_0: float,
    alpha_min: float | None = None,
) -> float:
    """Cosine-annealed EMA decay rate from alpha_0 down to alpha_min (= 0.1 * alpha_0)."""
    if alpha_min is None:
        alpha_min = 0.1 * alpha_0
    if total_steps <= 1:
        return float(alpha_0)
    t = min(max(step, 0), total_steps - 1)
    cos_term = 0.5 * (1.0 + math.cos(math.pi * t / (total_steps - 1)))
    return float(alpha_min + (alpha_0 - alpha_min) * cos_term)


def auto_hyperparameters(
    n_samples: int,
    n_components: int,
    n_features: int,
    covariance_type: str = "full",
    batch_size: int | str = "auto",
    max_epochs: int | str = "auto",
    alpha: float | str = "auto",
    revive_every: int | str = "auto",
    init_usage_scale: float | str = "auto",
    polish_steps: int | str = "auto",
) -> EGMMHyperparams:
    """Choose batch size, epoch budget, EMA peak decay, initial moment scale, and polish budget.

    Uses smooth regime adaptation via effective batch degrees of freedom:
    - ``full``: ``eff_df = B / (D * K)``
    - ``diag``: ``eff_df = B / (sqrt(D) * K)``
    All parameters (``alpha``, ``max_epochs``, ``init_usage_scale``, ``polish_steps``) are
    decoupled so manual overrides of one hyperparameter never silently disable others.
    """
    N = max(1, int(n_samples))
    K = max(1, int(n_components))
    D = max(1, int(n_features))
    is_auto_alpha = alpha == "auto"
    is_auto_epochs = max_epochs == "auto"
    is_auto_init = init_usage_scale == "auto"
    is_auto_polish = polish_steps == "auto"

    if covariance_type == "diag":
        if batch_size == "auto":
            b_min = min(N, max(256, 16 * K))
            bs_tgt = max(b_min, min(N, 16384, max(256, N // 6)))
            steps_ep = max(1, N // bs_tgt)
            bs = max(1, N // steps_ep)
            if bs > 16384:
                steps_ep += 1
                bs = max(1, N // steps_ep)
        else:
            bs = max(1, min(N, int(batch_size)))
            steps_ep = max(1, math.ceil(N / bs))

        if is_auto_epochs:
            epochs = 3 if N < 30_000 else 2
            if N < 10_000 and steps_ep < 5:
                epochs = max(epochs, min(5, math.ceil(18 / steps_ep)))
        else:
            epochs = max(1, int(max_epochs))

        total_steps = max(1, epochs * steps_ep)
        eff_df_diag = bs / max(1.0, float(K) * math.sqrt(float(D)))
        t_df = max(0.0, min(1.0, (eff_df_diag - 12.0) / 36.0))

        resolved_init_scale = (1.0 - 0.50 * t_df) if is_auto_init else float(init_usage_scale)
        if is_auto_alpha:
            alpha0_base = 0.44 + 0.24 * t_df
            alpha_0 = float(min(0.68, max(0.38, alpha0_base * (total_steps ** -0.03))))
            alpha_min = (0.10 - 0.03 * t_df) * alpha_0
        else:
            alpha_0 = float(alpha)
            alpha_min = 0.10 * alpha_0

        rev_points = 2 if revive_every == "auto" else -1
        rev_every = max(2, steps_ep // 2) if revive_every == "auto" else max(1, int(revive_every))
        if is_auto_polish:
            base_polish = 12 if N <= 100_000 else 0
            resolved_polish = (
                base_polish
                if is_auto_epochs
                else (0 if epochs <= 1 else min(base_polish, 2 * epochs))
            )
        else:
            resolved_polish = max(0, int(polish_steps))
    else:
        if batch_size == "auto":
            b_min = min(N, max(256, 16 * K))
            if N < 3000:
                b_min = min(b_min, max(128, N // 5))
            b_floor = 128 if N < 3000 else 256
            bs_tgt = max(b_min, min(N, 8192, max(b_floor, N // 9)))
            steps_ep = max(1, N // bs_tgt)
            bs = max(1, N // steps_ep)
            if bs > 8192:
                steps_ep += 1
                bs = max(1, N // steps_ep)
        else:
            bs = max(1, min(N, int(batch_size)))
            steps_ep = max(1, math.ceil(N / bs))

        if is_auto_epochs:
            if N < 15_000:
                epochs = max(4, min(7, math.ceil(30 / steps_ep)))
            elif N < 100_000:
                epochs = 4
            else:
                epochs = 2
        else:
            epochs = max(1, int(max_epochs))

        total_steps = max(1, epochs * steps_ep)
        eff_df = bs / max(1.0, float(D * K))
        t_df = max(0.0, min(1.0, (eff_df - 1.5) / 1.5))

        resolved_init_scale = (1.0 - 0.75 * t_df) if is_auto_init else float(init_usage_scale)
        if is_auto_alpha:
            alpha0_base = 0.36 + 0.56 * t_df
            alpha_0 = float(min(0.85, max(0.32, alpha0_base * (total_steps ** -0.04))))
            alpha_min = (0.10 - 0.05 * t_df) * alpha_0
        else:
            alpha_0 = float(alpha)
            alpha_min = 0.10 * alpha_0

        rev_points = 4 if revive_every == "auto" else -1
        rev_every = max(2, steps_ep // 2) if revive_every == "auto" else max(1, int(revive_every))
        if is_auto_polish:
            base_polish = (6 if N <= 25_000 else 5) if N <= 100_000 else 0
            resolved_polish = (
                base_polish
                if is_auto_epochs
                else (0 if epochs <= 1 else min(base_polish, epochs))
            )
        else:
            resolved_polish = max(0, int(polish_steps))

    return EGMMHyperparams(
        batch_size=bs,
        max_epochs=epochs,
        alpha_0=alpha_0,
        alpha_min=alpha_min,
        revive_every=rev_every,
        init_usage_scale=resolved_init_scale,
        polish_steps=resolved_polish,
        revive_points=rev_points,
        steps_per_epoch=steps_ep,
    )

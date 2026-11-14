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
) -> EGMMHyperparams:
    """Choose batch size, epoch budget, EMA peak decay, and revival cadence.

    Default rules scale smoothly with (N, K, D); empirical power-law constants
    are calibrated via ``experiments/fit_scaling_laws.py`` on held-out synthetic regimes.
    """
    N = max(1, int(n_samples))
    K = max(1, int(n_components))
    D = max(1, int(n_features))

    if max_epochs == "auto":
        # Small N needs more epochs to get enough gradient/EMA updates; large N converges in 2-4 epochs.
        if N < 4000:
            epochs = 12
        elif N < 20000:
            epochs = 6
        elif N < 100000:
            epochs = 4
        else:
            epochs = 3
    else:
        epochs = max(1, int(max_epochs))

    if batch_size == "auto":
        # Ensure each mini-batch has enough expected points per component (~32*K) without exceeding memory/steps.
        if covariance_type == "full":
            b_est = int(round(48.0 * (K ** 0.75) * (D ** 0.25) * ((N / 10000.0) ** 0.25)))
            b_cap = 8192 if D <= 128 else 4096
        else:
            b_est = int(round(64.0 * (K ** 0.75) * (D ** 0.15) * ((N / 10000.0) ** 0.25)))
            b_cap = 16384
        b_min = min(N, max(64, 4 * K))
        # Ensure at least ~4 steps per epoch for small/medium N when possible
        b_step_floor = max(64, N // 4) if N >= 256 else N
        bs = max(b_min, min(N, b_cap, b_est, b_step_floor))
    else:
        bs = max(1, min(N, int(batch_size)))

    steps_per_epoch = max(1, math.ceil(N / bs))
    total_steps = max(1, epochs * steps_per_epoch)

    if alpha == "auto":
        # Larger step count -> smaller peak alpha_0 so EMA memory horizon spans ~1/3 epoch to 1 epoch.
        # Ensure alpha_0 stays well-behaved in [0.05, 0.55].
        ratio = bs / max(1.0, float(N))
        a0 = 1.8 * (ratio ** 0.55) * (max(1, epochs) ** -0.25)
        alpha_0 = float(min(0.55, max(0.05, a0)))
    else:
        alpha_0 = float(alpha)

    alpha_min = 0.1 * alpha_0

    if revive_every == "auto":
        rev_every = max(2, steps_per_epoch // 2)
    else:
        rev_every = max(1, int(revive_every))

    return EGMMHyperparams(
        batch_size=bs,
        max_epochs=epochs,
        alpha_0=alpha_0,
        alpha_min=alpha_min,
        revive_every=rev_every,
    )

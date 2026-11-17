"""e-GMM variants for the follow-up runs, defined outside the frozen library.

* ``EagerEGMMHostAlpha``: the eager (no CUDA graph) e-GMM with the step-size schedule held on the host, so the
  eager loop no longer synchronizes with the GPU at every step to read alpha_t. ``EGMM`` vs this variant
  isolates what the CUDA graph saves beyond that avoidable synchronization.
* ``RobbinsMonroEGMM`` / ``ConstantStepEGMM``: single-factor schedule ablations. Everything in the e-GMM
  pipeline (seeding, batch size, epochs, s_init, revival, polish, CUDA graph) is unchanged; only the cosine
  schedule is replaced by the stepwise-EM schedule alpha_t = (t + 2)^(-rho) (rho = 0.7, the middle of the
  online-EM baseline's grid), or by a constant equal to the cosine schedule's mean (same total step size).
"""

from __future__ import annotations

from typing import Any

from egmm.baselines import CappeMoulinesOnlineGMM
from egmm.estimator import EGMM
from experiments.run_benchmark_suite import EagerEGMM


class EagerEGMMHostAlpha(EagerEGMM):
    def _build_schedule_and_revivals(self, total_steps: int, hp: Any, device):
        alphas_t, rev_set = super()._build_schedule_and_revivals(total_steps, hp, device)
        return alphas_t.cpu(), rev_set  # one transfer up front; the eager loop's alpha_t.item() is then host-only


class RobbinsMonroEGMM(EGMM):
    _RHO = 0.7

    def _step_alpha(self, step: int, total_steps: int, hp: Any) -> float:
        return float((step + 2.0) ** (-self._RHO))


class ConstantStepEGMM(EGMM):
    def _step_alpha(self, step: int, total_steps: int, hp: Any) -> float:
        return float(0.5 * (hp.alpha_0 + hp.alpha_min))  # the mean of the cosine schedule from alpha_0 to alpha_min


class OnlineGMMRevival(CappeMoulinesOnlineGMM):
    """Stepwise online EM ((t + 2)^-rho, tuned rho and batch size) with e-GMM's revival checkpoints enabled."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.revive_dead = True

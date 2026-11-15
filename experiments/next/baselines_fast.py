"""Stronger GPU batch-EM baselines for the next version of the paper (defined outside the frozen library).

Both classes run the same algorithm and stopping rule as ``egmm.baselines.TorchBatchGMM`` (full-batch EM,
``max_iter`` / ``tol`` on the per-sample training log-likelihood) and the same kernels, but remove the
implementation overheads that a reviewer could call a straw man:

* ``TorchBatchGMMFast``: float32 inner Cholesky with the branchless diagonal fallback and no ``info``
  synchronization (as e-GMM's EMA steps), the log-likelihood kept on the device, the convergence test run
  every ``check_every`` iterations (one host synchronization), larger chunks, and the log-Jacobian read once.
  It returns the state of the first iteration at which the usual rule fires, so it stops at exactly the
  same iteration as ``TorchBatchGMM`` would (up to float32-vs-float64 Cholesky differences); the extra
  iterations run between checks cost time but do not change the result.
* ``TorchBatchGMMGraph``: the same, with one complete EM iteration (all chunks of the E-step, the
  accumulation, the M-step projection, and the log-likelihood write) captured in a single CUDA graph and
  replayed, mirroring how e-GMM captures its EMA step. Falls back to the eager fast path off CUDA.
"""

from __future__ import annotations

import torch

from egmm.baselines import TorchBatchGMM
from egmm.kernels import e_step_batch


class TorchBatchGMMFast(TorchBatchGMM):
    def __init__(self, *args, check_every: int = 5, chunk_size: int = 32768, **kwargs) -> None:
        super().__init__(*args, chunk_size=chunk_size, **kwargs)
        self.check_every = check_every

    # -- helpers ---------------------------------------------------------------------------------------
    def _proj_fast(self, M, S, Q, dtype):
        return self._moments_to_estep_params(
            M, S, Q, estep_dtype=dtype, allow_sync=False, fast_cholesky_f32=(dtype == torch.float32)
        )

    def _first_converged(self, vals: list[float], start: int, stop: int) -> int | None:
        """Index j in [max(1, start), stop) of the first iteration with |lb_j - lb_{j-1}| < tol."""
        for j in range(max(1, start), stop):
            if abs(vals[j] - vals[j - 1]) < self.tol:
                return j
        return None

    def _finish(self, X_std, state, j_stop, vals, max_iters, converged):
        M, S, Q = state
        weights, means, covs, prec_factor, log_det_half = self._moments_to_estep_params(M, S, Q, estep_dtype=X_std.dtype)
        iters_done = j_stop + 1
        return M, S, Q, weights, means, covs, prec_factor, log_det_half, vals[j_stop], iters_done, converged

    # -- eager, sync-light batch EM ----------------------------------------------------------------------
    def _run_batch_em(self, X_std, M, S, Q, max_iters):
        N = X_std.shape[0]
        dev = X_std.device
        dtype = X_std.dtype
        k = max(1, int(self.check_every))
        log_jac = float(self._log_jac_t.item())
        weights, means, _, prec_factor, log_det_half = self._proj_fast(M, S, Q, dtype)
        lbs = torch.zeros(max_iters, dtype=torch.float64, device=dev)
        states: dict[int, tuple] = {}
        checked = 0
        vals: list[float] = []
        for it in range(max_iters):
            M_acc = torch.zeros_like(M)
            S_acc = torch.zeros_like(S)
            Q_acc = torch.zeros_like(Q)
            ll_sum = torch.zeros((), device=dev, dtype=torch.float64)
            for start in range(0, N, self.chunk_size):
                end = min(N, start + self.chunk_size)
                _, batch_lb, m_bar, s_bar, q_bar = e_step_batch(
                    X_std[start:end], weights, means, covariance_type=self.covariance_type,
                    prec_factor=prec_factor, log_det_half=log_det_half, stats_dtype=M.dtype,
                )
                frac = (end - start) / N
                M_acc += m_bar.to(M.dtype) * frac
                S_acc += s_bar.to(S.dtype) * frac
                Q_acc += q_bar.to(Q.dtype) * frac
                ll_sum = ll_sum + batch_lb.to(torch.float64) * (end - start)
            lbs[it] = ll_sum / N
            M, S, Q = M_acc, S_acc, Q_acc
            states[it] = (M, S, Q)
            weights, means, _, prec_factor, log_det_half = self._proj_fast(M, S, Q, dtype)
            if (it + 1) % k == 0 or it + 1 == max_iters:
                vals = [v + log_jac for v in lbs[: it + 1].cpu().tolist()]  # the only host sync
                j = self._first_converged(vals, checked, it + 1)
                if j is not None:
                    self.n_iter_executed_ = it + 1  # iterations actually computed (>= the stopping iteration)
                    return self._finish(X_std, states[j], j, vals, max_iters, True)
                checked = it + 1
                states = {it: states[it]}
        self.n_iter_executed_ = max_iters
        return self._finish(X_std, states[max_iters - 1], max_iters - 1, vals, max_iters, False)


class TorchBatchGMMGraph(TorchBatchGMMFast):
    def _run_batch_em(self, X_std, M, S, Q, max_iters):
        dev = X_std.device
        if dev.type != "cuda" or max_iters < 2:
            return super()._run_batch_em(X_std, M, S, Q, max_iters)
        N = X_std.shape[0]
        dtype = X_std.dtype
        k = max(1, int(self.check_every))
        log_jac = float(self._log_jac_t.item())
        w0, mu0, _, pf0, ldh0 = self._proj_fast(M, S, Q, dtype)
        M_buf, S_buf, Q_buf = M.clone(), S.clone(), Q.clone()
        w_buf, mu_buf, pf_buf, ldh_buf = w0.clone(), mu0.clone(), pf0.clone(), ldh0.clone()
        lbs = torch.zeros(max_iters, dtype=torch.float64, device=dev)
        pos = torch.zeros((1,), dtype=torch.long, device=dev)
        chunks = [(s, min(N, s + self.chunk_size)) for s in range(0, N, self.chunk_size)]

        def body() -> None:
            M_acc = torch.zeros_like(M_buf)
            S_acc = torch.zeros_like(S_buf)
            Q_acc = torch.zeros_like(Q_buf)
            ll = torch.zeros((1,), device=dev, dtype=torch.float64)
            for s, e in chunks:
                _, blb, mb, sb, qb = e_step_batch(
                    X_std[s:e], w_buf, mu_buf, covariance_type=self.covariance_type,
                    prec_factor=pf_buf, log_det_half=ldh_buf, stats_dtype=M_buf.dtype,
                )
                frac = (e - s) / N
                M_acc.add_(mb.to(M_buf.dtype) * frac)
                S_acc.add_(sb.to(S_buf.dtype) * frac)
                Q_acc.add_(qb.to(Q_buf.dtype) * frac)
                ll.add_(blb.to(torch.float64).reshape(1) * (e - s))
            lbs.index_copy_(0, pos, ll / N)
            M_buf.copy_(M_acc)
            S_buf.copy_(S_acc)
            Q_buf.copy_(Q_acc)
            w2, mu2, _, pf2, ldh2 = self._proj_fast(M_buf, S_buf, Q_buf, dtype)
            w_buf.copy_(w2)
            mu_buf.copy_(mu2)
            pf_buf.copy_(pf2)
            ldh_buf.copy_(ldh2)
            pos.add_(1)

        # warm-up on a side stream (required before capture), then restore the initial state
        stream = torch.cuda.Stream(device=dev)
        stream.wait_stream(torch.cuda.current_stream(device=dev))
        with torch.cuda.stream(stream):
            body()
        torch.cuda.current_stream(device=dev).wait_stream(stream)
        pos.zero_()
        lbs.zero_()
        M_buf.copy_(M)
        S_buf.copy_(S)
        Q_buf.copy_(Q)
        w_buf.copy_(w0)
        mu_buf.copy_(mu0)
        pf_buf.copy_(pf0)
        ldh_buf.copy_(ldh0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            body()

        states: dict[int, tuple] = {}
        checked = 0
        vals: list[float] = []
        result = None
        for it in range(max_iters):
            graph.replay()
            states[it] = (M_buf.clone(), S_buf.clone(), Q_buf.clone())
            if (it + 1) % k == 0 or it + 1 == max_iters:
                vals = [v + log_jac for v in lbs[: it + 1].cpu().tolist()]
                j = self._first_converged(vals, checked, it + 1)
                if j is not None:
                    result = self._finish(X_std, states[j], j, vals, max_iters, True)
                    self.n_iter_executed_ = it + 2  # replayed iterations plus the warm-up iteration
                    break
                checked = it + 1
                states = {it: states[it]}
        del graph
        if result is None:
            result = self._finish(X_std, states[max_iters - 1], max_iters - 1, vals, max_iters, False)
            self.n_iter_executed_ = max_iters + 1
        return result


__all__ = ["TorchBatchGMMFast", "TorchBatchGMMGraph"]

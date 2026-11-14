# Reference Notes: Meta FAIR `e-means` (`arXiv:2611.09501`)

Source code mirrored from `facebookresearch/faiss/contrib/e_means` into this folder:
- `README.md` — Usage, parameters, scaling law summary, backend matrix
- `base.py` — Shared scaling constants (`_BATCH_LAWS`, `_ALPHA_LAW`), cosine decay (`_scheduled_alpha`), pass layout (`_epoch_rows`)
- `e_means_torch.py` — GPU/PyTorch `Emeans`, Gumbel-top-k `_choose_splits`, CUDA Graph `_capture_step`, BF16 cast, readout modes
- `torch_backend.py` — Vectorized GPU split & usage decay ops
- `e_means.py` — NumPy CPU reference implementation

---

## 1. Key Equations & Constants (from `base.py` and `e_means_torch.py`)

- **Two-part EMA update** with near-zero initial mass `_INIT_USAGE = 2.0**-16`:
  $$N_j^{\text{new}} = (1 - \alpha_t) N_j + \alpha_t n_j, \qquad c_j^{\text{new}} = c_j + \frac{\alpha_t n_j}{N_j^{\text{new}}} \left(\frac{s_j}{n_j} - c_j\right)$$
  Empty clusters hold centroid position while usage decays (`all_usage *= (1 - alpha_t)`).
- **Cosine step-size annealing** (`_ALPHA_MIN_FACTOR = 0.1`):
  $$\alpha(t) = \alpha_0 \left(0.1 + 0.9 \cdot \frac{1 + \cos(\pi t / M)}{2}\right)$$
- **Dead-cluster revival** (`_CHECK_UNUSED_EVERY = 5`, `_UNUSED_THRESHOLD = 0.1`, `_SPLIT_EPSILON = 1e-4`, disabled when $d \le 2$ via `_NO_REVIVAL_MAX_D = 2`):
  - Identifies unused clusters (`usage < 0.1 * total_usage / k`).
  - Samples donor clusters without replacement proportional to mass via Gumbel-top-k (`usage.log() - (-uniform.log()).log()`) on CUDA with fixed shapes (zero `.nonzero()` host compaction).
  - Splits donors via sign-alternating coordinate perturbation (`donors * (1 +/- 1e-4 * sign)`), $O(K+D)$ without any eigendecomposition.
- **Zero-cost final pass readout** (`_epoch_rows`):
  - Accumulates unweighted `(_pass_sums, _pass_counts)` during the **final** training pass (0 extra search FLOPs), or maintains a `slow EMA` regularizer for low $n/k$ out-of-sample test MSE.
- **Scaling laws** (`_AUTO_QUALITY_MAX_D = 16`):
  - Two batch laws: `"quality"` (8× smaller batches, for $d \le 16$) and `"throughput"` (for $d > 16$).
  - `_ALPHA_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)` coupling $\alpha_0$ to $(k, B, n, T)$.

---

## 2. Empirical Paper Takeaways (`arXiv:2611.09501`)

- **Revival dominates accuracy (Table 1 & Table 14):**
  - Turning revival off degrades train MSE by `+2.97%` and test MSE by `+2.24%` (`DEEP* 1M` dead clusters jump from `0.3%` to `26.3%`).
  - Resampling from batch degrades train MSE by `+2.06%`, test MSE by `+1.15%` (`14.3%` dead on `DEEP* 1M`).
- **Random init vs `k-means++` (Appendix G.1):**
  - On `DEEP* 1M, k=16,384`: uniform random row init takes `0.1s` vs `2,852s` for greedy `k-means++`, and still achieves `0.15%` better test MSE after 5 epochs (`22.3s`).
- **Wall-clock speedup vs Flash-KMeans@25 (Table 4):**
  - Matches Flash-KMeans@25 test MSE in `1.4`–`2.6` epochs (`9.5×`–`12.9×` faster at `1M, k=16,384`; `19.2×`–`45.5×` faster at `10M`–`1B`).
- **GPU systems stack (`e_means_torch.py` + Section 4.3 / Eq. 9–10):**
  - Automatic CUDA `BF16` casting for assignment GEMM with centroid norm $\|\hat{c}_j\|_2^2$ computed from the casted centroid $\hat{c}_j$ (cancels quantization bias).
  - Tiled SRAM CuTe/Triton assignment avoiding `B × K` HBM materialization.
  - `torch.cuda.CUDAGraph()` step capture (`_capture_step`) when `steps_per_pass >= 3` + sync-free CUDA ops.

---

## 3. Verified Baseline Findings in `e-GMM` Before Pausing (`task-843`)

Prototyping four `e-means`-inspired fixes in `V1` against `TorchBatchGMM(max_iter=50, tol=1e-3)` across the 23-dataset suite already achieved dual significance for the first time:
- **Prototype `V1_s28_lastpass_p2`**: **`15/23` NLL wins (`p = 0.0459`)** and **`17/23` Speed wins (`p = 0.00121`, median `1.52×`, max `12.17×`)**.
- **`covtype_581k` breakthrough**: Centered float32→float64 `e_step_batch` + sync-free M-step eliminated the 2,992 offset-cancellation `eigh` repairs, swinging `covtype_581k` from `-289.428 (627ms)` (`TorchBatchGMM`) to **`-299.919 (261ms)`** (`-10.490` NLL, `2.41×` faster).
- **Remaining gap identified (to close next):** Replacing cosine-annealed EMA with `last_pass` unweighted readout + lowering polish cutoff hurt `mnist784_pca48`, `shuttle`, `miniboone`, and `spambase`, while small-$N$ datasets (`digits_pca32`, `spambase`, `syn_anisotropic_30k`) spent too many small Cholesky M-steps without early stopping (`CandD`/`CandE` already showed `18/23` NLL wins with `p = 0.00028` when retaining cosine EMA + light small-$N$ polish).

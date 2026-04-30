"""
Flash-GMM v8: Atomic-free Pass 2 kernel.

Key insight: Pass 2 (stats accumulation) has 256 N-blocks all doing
atomic_add to the same K×d output slots → massive contention.

Fix: Reorganize grid so each block OWNS a K-tile and streams ALL N.
     Grid: (K/BK,) instead of (K/BK, N/BLOCK_N).
     Each block is the sole writer → ZERO atomics.

Trade-off: each block reads X[N,d] in full (more HBM reads),
but eliminates ~500K atomics that were the bottleneck.
"""
import torch
import math
import triton
import triton.language as tl
from flash_gmm_v3 import _flash_logsumexp_bf16_kernel


# ============================================================================
# Pass 2 v2: Atomic-free stats accumulation
#
# Grid: (K/BK,) — each block owns centroids [k_start, k_start+BK)
# Each block streams ALL N points in tiles of BLOCK_N:
#   for each N-tile:
#     load X[BLOCK_N, d], compute γ[BLOCK_N, BK]
#     local_nk += γ.sum(0)
#     local_sk += γ^T @ X
#     local_sqk += γ^T @ X²
# Then writes final local_nk, local_sk, local_sqk to global (no atomics!)
# ============================================================================

@triton.jit
def _atomic_free_accum_kernel(
    X_ptr, X_sq_ptr, inv_var_ptr, mu_iv_ptr,
    quad_mu_ptr, log_coeff_ptr, ln_ptr,
    nk_ptr, sk_ptr, sqk_ptr,
    N, K, d: tl.constexpr,
    BK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Each block owns BK centroids, streams all N points. Zero atomics."""
    pid_k = tl.program_id(0)
    k_start = pid_k * BK
    k_offs = k_start + tl.arange(0, BK)
    d_offs = tl.arange(0, d)
    k_mask = k_offs < K

    # Load centroid tile (stays constant for this block)
    iv = tl.load(inv_var_ptr + k_offs[:, None] * d + d_offs[None, :],
                  mask=k_mask[:, None], other=0.0).to(tl.bfloat16)
    mi = tl.load(mu_iv_ptr + k_offs[:, None] * d + d_offs[None, :],
                  mask=k_mask[:, None], other=0.0).to(tl.bfloat16)
    qm = tl.load(quad_mu_ptr + k_offs, mask=k_mask, other=0.0)
    lc = tl.load(log_coeff_ptr + k_offs, mask=k_mask, other=-float('inf'))

    # Block-local accumulators (in registers!)
    local_nk = tl.zeros([BK], dtype=tl.float32)
    local_sk = tl.zeros([BK, d], dtype=tl.float32)
    local_sqk = tl.zeros([BK, d], dtype=tl.float32)

    # Stream all N points
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # Load X tile
        x_fp32 = tl.load(X_ptr + n_offs[:, None] * d + d_offs[None, :],
                          mask=n_mask[:, None], other=0.0)
        xsq_fp32 = tl.load(X_sq_ptr + n_offs[:, None] * d + d_offs[None, :],
                             mask=n_mask[:, None], other=0.0)
        x_bf16 = x_fp32.to(tl.bfloat16)
        xsq_bf16 = xsq_fp32.to(tl.bfloat16)

        # Distance via BF16 dots
        L = tl.dot(xsq_bf16, tl.trans(iv)).to(tl.float32)
        L += tl.dot(x_bf16, tl.trans(mi)).to(tl.float32) * (-2.0)
        L += qm[None, :]
        L = L * (-0.5) + lc[None, :]

        # γ = exp(L - log_normalizer)
        ln_vals = tl.load(ln_ptr + n_offs, mask=n_mask, other=0.0)
        gamma = tl.exp(L - ln_vals[:, None])
        gamma = tl.where(k_mask[None, :] & n_mask[:, None], gamma, 0.0)

        # Accumulate (no atomics — this block is sole owner!)
        local_nk += tl.sum(gamma, axis=0)

        gamma_bf16 = gamma.to(tl.bfloat16)
        local_sk += tl.dot(tl.trans(gamma_bf16), x_bf16).to(tl.float32)
        local_sqk += tl.dot(tl.trans(gamma_bf16), xsq_bf16).to(tl.float32)

    # Write results (each block writes non-overlapping slots)
    tl.store(nk_ptr + k_offs, local_nk, mask=k_mask)
    tl.store(sk_ptr + k_offs[:, None] * d + d_offs[None, :], local_sk, mask=k_mask[:, None])
    tl.store(sqk_ptr + k_offs[:, None] * d + d_offs[None, :], local_sqk, mask=k_mask[:, None])


class AtomicFreeFlashGMM:
    """Flash GMM with atomic-free Pass 2 for large K, atomic for small K."""

    def __init__(self, BLOCK_N=None, BK=None):
        self._BLOCK_N = BLOCK_N
        self._BK = BK

    def _auto_tile(self, d, K):
        if d <= 64:
            return (self._BLOCK_N or 128), min(self._BK or 32, K)
        else:
            return (self._BLOCK_N or 32), min(self._BK or 16, K)

    def em_step(self, X, mu, var, log_pi):
        N, d = X.shape
        K = mu.shape[0]
        BLOCK_N, BK = self._auto_tile(d, K)

        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)
        X_sq = X * X

        log_normalizer = torch.empty(N, device=X.device, dtype=torch.float32)
        n_k = torch.zeros(K, device=X.device, dtype=torch.float32) if K < 256 else torch.empty(K, device=X.device, dtype=torch.float32)
        s_k = torch.zeros(K, d, device=X.device, dtype=torch.float32) if K < 256 else torch.empty(K, d, device=X.device, dtype=torch.float32)
        sq_k = torch.zeros(K, d, device=X.device, dtype=torch.float32) if K < 256 else torch.empty(K, d, device=X.device, dtype=torch.float32)

        # Pass 1: logsumexp
        grid1 = ((N + BLOCK_N - 1) // BLOCK_N,)
        _flash_logsumexp_bf16_kernel[grid1](
            X, X_sq, inv_var, mu_iv, quad_mu, log_coeff, log_normalizer,
            N, K, d, BK, BLOCK_N,
        )

        # Pass 2: choose strategy based on K
        n_k_blocks = (K + BK - 1) // BK
        if n_k_blocks >= 8:  # enough K-blocks to fill SMs → atomic-free
            grid2 = (n_k_blocks,)
            _atomic_free_accum_kernel[grid2](
                X, X_sq, inv_var, mu_iv, quad_mu, log_coeff, log_normalizer,
                n_k, s_k, sq_k,
                N, K, d, BK, BLOCK_N,
            )
        else:  # small K → use atomic version for parallelism
            from flash_gmm_v3 import _flash_accum_stats_bf16_kernel
            n_k.zero_(); s_k.zero_(); sq_k.zero_()
            grid2 = (n_k_blocks, (N + BLOCK_N - 1) // BLOCK_N)
            _flash_accum_stats_bf16_kernel[grid2](
                X, X_sq, inv_var, mu_iv, quad_mu, log_coeff, log_normalizer,
                n_k, s_k, sq_k,
                N, K, d, BK, BLOCK_N,
            )

        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer

"""
Flash-GMM v3: Optimized variants for maximum GPU throughput.

Variants:
  1. GemmFlashBF16   — BF16 distance GEMMs (3× Tensor Core throughput) + FP32 accum
  2. TritonFlashGMM  — Triton fused kernel: tl.dot distance + online logsumexp in one launch
  3. GemmStdBF16     — BF16 Standard GMM (best absolute speed)
  4. CompiledFlash   — torch.compile'd GEMM Flash
"""
import torch
import math
import triton
import triton.language as tl


# ============================================================================
# 1. BF16 Mixed-Precision GEMM GMM
# ============================================================================

class GemmFlashBF16:
    """GEMM Flash with BF16 distance computation, FP32 accumulators."""

    def __init__(self, BK=None):
        self.BK = BK

    def em_step(self, X, mu, var, log_pi):
        N, d = X.shape
        K = mu.shape[0]
        BK = self.BK or min(K, 128)

        # Precompute in FP32
        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)

        # BF16 copies for distance GEMMs
        X_bf = X.bfloat16()
        X_sq_bf = (X * X).bfloat16()  # compute in FP32, cast

        # ---- Pass 1: online logsumexp ----
        running_max = torch.full((N,), -float('inf'), device=X.device)
        running_sum = torch.zeros(N, device=X.device)

        for t in range(0, K, BK):
            tk = min(BK, K - t)
            inv_var_bf = inv_var[t:t+tk].bfloat16()
            mu_iv_bf = mu_iv[t:t+tk].bfloat16()

            # BF16 GEMMs → FP32 accumulation
            L_tile = torch.mm(X_sq_bf, inv_var_bf.T).float()  # BF16 GEMM → FP32
            L_tile -= 2.0 * torch.mm(X_bf, mu_iv_bf.T).float()
            L_tile.add_(quad_mu[t:t+tk])
            L_tile.mul_(-0.5)
            L_tile.add_(log_coeff[t:t+tk])

            tile_max = L_tile.max(dim=1).values
            new_max = torch.max(running_max, tile_max)
            running_sum = (running_sum * (running_max - new_max).exp()
                          + (L_tile - new_max.unsqueeze(1)).exp().sum(1))
            running_max = new_max

        log_normalizer = running_max + running_sum.log()

        # ---- Pass 2: stats accumulation ----
        n_k = torch.zeros(K, device=X.device)
        s_k = torch.zeros(K, d, device=X.device)
        sq_k = torch.zeros(K, d, device=X.device)

        for t in range(0, K, BK):
            tk = min(BK, K - t)
            inv_var_bf = inv_var[t:t+tk].bfloat16()
            mu_iv_bf = mu_iv[t:t+tk].bfloat16()

            L_tile = torch.mm(X_sq_bf, inv_var_bf.T).float()
            L_tile -= 2.0 * torch.mm(X_bf, mu_iv_bf.T).float()
            L_tile.add_(quad_mu[t:t+tk])
            L_tile.mul_(-0.5)
            L_tile.add_(log_coeff[t:t+tk])

            gamma_tile = (L_tile - log_normalizer.unsqueeze(1)).exp()
            n_k[t:t+tk] = gamma_tile.sum(0)
            # M-step GEMMs in FP32
            s_k[t:t+tk] = torch.mm(gamma_tile.T, X)
            sq_k[t:t+tk] = torch.mm(gamma_tile.T, X * X)

        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer


class GemmStdBF16:
    """Standard GEMM GMM with BF16 distance computation."""

    def em_step(self, X, mu, var, log_pi):
        N, d = X.shape
        K = mu.shape[0]

        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)

        X_bf = X.bfloat16()
        X_sq_bf = (X * X).bfloat16()

        # BF16 distance GEMM → FP32
        L = torch.mm(X_sq_bf, inv_var.bfloat16().T).float()
        L -= 2.0 * torch.mm(X_bf, mu_iv.bfloat16().T).float()
        L.add_(quad_mu)
        L.mul_(-0.5)
        L.add_(log_coeff)

        log_normalizer = torch.logsumexp(L, dim=1)
        gamma = (L - log_normalizer.unsqueeze(1)).exp()

        n_k = gamma.sum(0)
        new_mu = torch.mm(gamma.T, X) / n_k.unsqueeze(1)
        new_var = (torch.mm(gamma.T, X * X) / n_k.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer


# ============================================================================
# 2. Triton Fused Kernel: distance GEMM + online logsumexp in one launch
# ============================================================================

@triton.jit
def _flash_logsumexp_kernel(
    X_ptr, X_sq_ptr, inv_var_ptr, mu_iv_ptr,
    quad_mu_ptr, log_coeff_ptr, out_ptr,
    N, K, d: tl.constexpr,
    BK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused distance computation + online logsumexp.

    Each program handles BLOCK_N data points, streams ALL K centroids.
    Distance computed via two tl.dot() calls (Tensor Cores).
    """
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N

    n_offs = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    n_mask = n_offs < N

    # Load X and X² blocks: [BLOCK_N, d]
    d_offs = tl.arange(0, d)
    x_ptrs = X_ptr + n_offs[:, None] * d + d_offs[None, :]      # [BLOCK_N, d]
    xsq_ptrs = X_sq_ptr + n_offs[:, None] * d + d_offs[None, :]
    x_block = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)       # [BLOCK_N, d]
    xsq_block = tl.load(xsq_ptrs, mask=n_mask[:, None], other=0.0)   # [BLOCK_N, d]

    # Online logsumexp state
    running_max = tl.full([BLOCK_N], value=-float('inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Stream centroid tiles
    for t in range(0, K, BK):
        k_offs = t + tl.arange(0, BK)
        k_mask = k_offs < K

        # Load centroid tile: inv_var[BK, d], mu_iv[BK, d], quad_mu[BK], log_coeff[BK]
        iv_ptrs = inv_var_ptr + k_offs[:, None] * d + d_offs[None, :]
        mi_ptrs = mu_iv_ptr + k_offs[:, None] * d + d_offs[None, :]
        iv_tile = tl.load(iv_ptrs, mask=k_mask[:, None], other=0.0)    # [BK, d]
        mi_tile = tl.load(mi_ptrs, mask=k_mask[:, None], other=0.0)    # [BK, d]
        qm_tile = tl.load(quad_mu_ptr + k_offs, mask=k_mask, other=0.0)  # [BK]
        lc_tile = tl.load(log_coeff_ptr + k_offs, mask=k_mask, other=-float('inf'))  # [BK]

        # Distance via two dot products (Tensor Cores!)
        # L = X² @ iv^T - 2·X @ mi^T + quad_mu
        L_tile = tl.dot(xsq_block, tl.trans(iv_tile))      # [BLOCK_N, BK]
        L_tile += tl.dot(x_block, tl.trans(mi_tile)) * (-2.0)
        L_tile += qm_tile[None, :]
        L_tile = L_tile * (-0.5) + lc_tile[None, :]

        # Mask out-of-range centroids
        L_tile = tl.where(k_mask[None, :], L_tile, -float('inf'))

        # Online logsumexp update
        tile_max = tl.max(L_tile, axis=1)  # [BLOCK_N]
        new_max = tl.maximum(running_max, tile_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + \
                      tl.sum(tl.exp(L_tile - new_max[:, None]), axis=1)
        running_max = new_max

    # Write log_normalizer
    result = running_max + tl.log(running_sum)
    tl.store(out_ptr + n_offs, result, mask=n_mask)


@triton.jit
def _flash_accum_stats_kernel(
    X_ptr, X_sq_ptr, inv_var_ptr, mu_iv_ptr,
    quad_mu_ptr, log_coeff_ptr, log_norm_ptr,
    nk_ptr, sk_ptr, sqk_ptr,
    N, K, d: tl.constexpr,
    BK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused distance recompute + γ + weighted accumulation.

    Each program handles one K-tile across ALL N blocks.
    Reduces over N using atomic adds to global accumulators.
    """
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    k_start = pid_k * BK
    n_start = pid_n * BLOCK_N

    k_offs = k_start + tl.arange(0, BK)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, d)

    k_mask = k_offs < K
    n_mask = n_offs < N

    # Load X block
    x_ptrs = X_ptr + n_offs[:, None] * d + d_offs[None, :]
    xsq_ptrs = X_sq_ptr + n_offs[:, None] * d + d_offs[None, :]
    x_block = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    xsq_block = tl.load(xsq_ptrs, mask=n_mask[:, None], other=0.0)

    # Load centroid tile
    iv_ptrs = inv_var_ptr + k_offs[:, None] * d + d_offs[None, :]
    mi_ptrs = mu_iv_ptr + k_offs[:, None] * d + d_offs[None, :]
    iv_tile = tl.load(iv_ptrs, mask=k_mask[:, None], other=0.0)
    mi_tile = tl.load(mi_ptrs, mask=k_mask[:, None], other=0.0)
    qm_tile = tl.load(quad_mu_ptr + k_offs, mask=k_mask, other=0.0)
    lc_tile = tl.load(log_coeff_ptr + k_offs, mask=k_mask, other=-float('inf'))

    # Recompute L_tile via GEMM
    L_tile = tl.dot(xsq_block, tl.trans(iv_tile))
    L_tile += tl.dot(x_block, tl.trans(mi_tile)) * (-2.0)
    L_tile += qm_tile[None, :]
    L_tile = L_tile * (-0.5) + lc_tile[None, :]

    # γ = exp(L - log_normalizer)
    ln_vals = tl.load(log_norm_ptr + n_offs, mask=n_mask, other=0.0)
    gamma_tile = tl.exp(L_tile - ln_vals[:, None])  # [BLOCK_N, BK]
    gamma_tile = tl.where(k_mask[None, :] & n_mask[:, None], gamma_tile, 0.0)

    # Accumulate n_k: sum over N  [BK]
    nk_local = tl.sum(gamma_tile, axis=0)
    tl.atomic_add(nk_ptr + k_offs, nk_local, mask=k_mask)

    # Accumulate s_k = γ^T @ X → [BK, d]
    sk_local = tl.dot(tl.trans(gamma_tile), x_block)  # [BK, BLOCK_N] @ [BLOCK_N, d] → [BK, d]
    sk_ptrs = sk_ptr + k_offs[:, None] * d + d_offs[None, :]
    tl.atomic_add(sk_ptrs, sk_local, mask=k_mask[:, None])

    # Accumulate sq_k = γ^T @ X² → [BK, d]
    sqk_local = tl.dot(tl.trans(gamma_tile), xsq_block)
    sqk_ptrs = sqk_ptr + k_offs[:, None] * d + d_offs[None, :]
    tl.atomic_add(sqk_ptrs, sqk_local, mask=k_mask[:, None])


class TritonFlashGMM:
    """Flash-GMM using Triton fused kernels with tl.dot (Tensor Cores).

    Pass 1: fused distance GEMM + online logsumexp → single kernel launch
    Pass 2: fused distance GEMM + γ + GEMM accumulation → single kernel launch
    """

    def __init__(self, BLOCK_N=None, BK=None):
        self._BLOCK_N = BLOCK_N
        self._BK = BK

    def _auto_tile(self, d, K):
        """Auto-select tile sizes based on dimension."""
        if self._BLOCK_N and self._BK:
            return self._BLOCK_N, self._BK
        if d <= 64:
            return 128, min(32, K)
        elif d <= 128:
            return 64, min(16, K)
        else:
            return 32, min(16, K)

    def em_step(self, X, mu, var, log_pi):
        N, d = X.shape
        K = mu.shape[0]
        BLOCK_N, BK = self._auto_tile(d, K)

        # Precompute
        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)
        X_sq = X * X

        log_normalizer = torch.empty(N, device=X.device, dtype=torch.float32)

        # Pass 1: fused logsumexp
        grid_p1 = ((N + BLOCK_N - 1) // BLOCK_N,)
        _flash_logsumexp_kernel[grid_p1](
            X, X_sq, inv_var, mu_iv,
            quad_mu, log_coeff, log_normalizer,
            N, K, d, BK, BLOCK_N,
        )

        # Pass 2: fused stats accumulation
        n_k = torch.zeros(K, device=X.device, dtype=torch.float32)
        s_k = torch.zeros(K, d, device=X.device, dtype=torch.float32)
        sq_k = torch.zeros(K, d, device=X.device, dtype=torch.float32)

        grid_p2 = ((K + BK - 1) // BK, (N + BLOCK_N - 1) // BLOCK_N)
        _flash_accum_stats_kernel[grid_p2](
            X, X_sq, inv_var, mu_iv,
            quad_mu, log_coeff, log_normalizer,
            n_k, s_k, sq_k,
            N, K, d, BK, BLOCK_N,
        )

        # Parameter update
        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer


# ============================================================================
# 3. Triton + BF16 combined
# ============================================================================

@triton.jit
def _flash_logsumexp_bf16_kernel(
    X_ptr, X_sq_ptr, inv_var_ptr, mu_iv_ptr,
    quad_mu_ptr, log_coeff_ptr, out_ptr,
    N, K, d: tl.constexpr,
    BK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Same as _flash_logsumexp_kernel but with BF16 dot products."""
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N

    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    d_offs = tl.arange(0, d)

    x_ptrs = X_ptr + n_offs[:, None] * d + d_offs[None, :]
    xsq_ptrs = X_sq_ptr + n_offs[:, None] * d + d_offs[None, :]
    x_block = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0).to(tl.bfloat16)
    xsq_block = tl.load(xsq_ptrs, mask=n_mask[:, None], other=0.0).to(tl.bfloat16)

    running_max = tl.full([BLOCK_N], value=-float('inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    for t in range(0, K, BK):
        k_offs = t + tl.arange(0, BK)
        k_mask = k_offs < K

        iv_ptrs = inv_var_ptr + k_offs[:, None] * d + d_offs[None, :]
        mi_ptrs = mu_iv_ptr + k_offs[:, None] * d + d_offs[None, :]
        iv_tile = tl.load(iv_ptrs, mask=k_mask[:, None], other=0.0).to(tl.bfloat16)
        mi_tile = tl.load(mi_ptrs, mask=k_mask[:, None], other=0.0).to(tl.bfloat16)
        qm_tile = tl.load(quad_mu_ptr + k_offs, mask=k_mask, other=0.0)
        lc_tile = tl.load(log_coeff_ptr + k_offs, mask=k_mask, other=-float('inf'))

        # BF16 dots → FP32 accumulator (Tensor Cores)
        L_tile = tl.dot(xsq_block, tl.trans(iv_tile)).to(tl.float32)
        L_tile += tl.dot(x_block, tl.trans(mi_tile)).to(tl.float32) * (-2.0)
        L_tile += qm_tile[None, :]
        L_tile = L_tile * (-0.5) + lc_tile[None, :]
        L_tile = tl.where(k_mask[None, :], L_tile, -float('inf'))

        tile_max = tl.max(L_tile, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + \
                      tl.sum(tl.exp(L_tile - new_max[:, None]), axis=1)
        running_max = new_max

    result = running_max + tl.log(running_sum)
    tl.store(out_ptr + n_offs, result, mask=n_mask)


@triton.jit
def _flash_accum_stats_bf16_kernel(
    X_ptr, X_sq_ptr, inv_var_ptr, mu_iv_ptr,
    quad_mu_ptr, log_coeff_ptr, log_norm_ptr,
    nk_ptr, sk_ptr, sqk_ptr,
    N, K, d: tl.constexpr,
    BK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Same as _flash_accum_stats_kernel but with BF16 distance computation."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    k_start = pid_k * BK
    n_start = pid_n * BLOCK_N

    k_offs = k_start + tl.arange(0, BK)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, d)

    k_mask = k_offs < K
    n_mask = n_offs < N

    x_ptrs = X_ptr + n_offs[:, None] * d + d_offs[None, :]
    xsq_ptrs = X_sq_ptr + n_offs[:, None] * d + d_offs[None, :]
    x_block_fp32 = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    xsq_block_fp32 = tl.load(xsq_ptrs, mask=n_mask[:, None], other=0.0)
    x_block_bf16 = x_block_fp32.to(tl.bfloat16)
    xsq_block_bf16 = xsq_block_fp32.to(tl.bfloat16)

    iv_ptrs = inv_var_ptr + k_offs[:, None] * d + d_offs[None, :]
    mi_ptrs = mu_iv_ptr + k_offs[:, None] * d + d_offs[None, :]
    iv_tile = tl.load(iv_ptrs, mask=k_mask[:, None], other=0.0).to(tl.bfloat16)
    mi_tile = tl.load(mi_ptrs, mask=k_mask[:, None], other=0.0).to(tl.bfloat16)
    qm_tile = tl.load(quad_mu_ptr + k_offs, mask=k_mask, other=0.0)
    lc_tile = tl.load(log_coeff_ptr + k_offs, mask=k_mask, other=-float('inf'))

    L_tile = tl.dot(xsq_block_bf16, tl.trans(iv_tile)).to(tl.float32)
    L_tile += tl.dot(x_block_bf16, tl.trans(mi_tile)).to(tl.float32) * (-2.0)
    L_tile += qm_tile[None, :]
    L_tile = L_tile * (-0.5) + lc_tile[None, :]

    ln_vals = tl.load(log_norm_ptr + n_offs, mask=n_mask, other=0.0)
    gamma_tile = tl.exp(L_tile - ln_vals[:, None])
    gamma_tile = tl.where(k_mask[None, :] & n_mask[:, None], gamma_tile, 0.0)

    nk_local = tl.sum(gamma_tile, axis=0)
    tl.atomic_add(nk_ptr + k_offs, nk_local, mask=k_mask)

    # FP32 accumulation GEMMs for numerical stability
    gamma_bf16 = gamma_tile.to(tl.bfloat16)
    sk_local = tl.dot(tl.trans(gamma_bf16), x_block_bf16).to(tl.float32)
    sk_ptrs = sk_ptr + k_offs[:, None] * d + d_offs[None, :]
    tl.atomic_add(sk_ptrs, sk_local, mask=k_mask[:, None])

    sqk_local = tl.dot(tl.trans(gamma_bf16), xsq_block_bf16).to(tl.float32)
    sqk_ptrs = sqk_ptr + k_offs[:, None] * d + d_offs[None, :]
    tl.atomic_add(sqk_ptrs, sqk_local, mask=k_mask[:, None])


class TritonFlashBF16GMM:
    """Triton Flash with BF16 dot products — maximum Tensor Core utilization."""

    def __init__(self, BLOCK_N=None, BK=None):
        self._BLOCK_N = BLOCK_N
        self._BK = BK

    def _auto_tile(self, d, K):
        if self._BLOCK_N and self._BK:
            return self._BLOCK_N, self._BK
        if d <= 64:
            return 128, min(32, K)
        elif d <= 128:
            return 32, min(16, K)
        else:
            return 16, min(16, K)

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

        grid_p1 = ((N + BLOCK_N - 1) // BLOCK_N,)
        _flash_logsumexp_bf16_kernel[grid_p1](
            X, X_sq, inv_var, mu_iv,
            quad_mu, log_coeff, log_normalizer,
            N, K, d, BK, BLOCK_N,
        )

        n_k = torch.zeros(K, device=X.device, dtype=torch.float32)
        s_k = torch.zeros(K, d, device=X.device, dtype=torch.float32)
        sq_k = torch.zeros(K, d, device=X.device, dtype=torch.float32)

        grid_p2 = ((K + BK - 1) // BK, (N + BLOCK_N - 1) // BLOCK_N)
        _flash_accum_stats_bf16_kernel[grid_p2](
            X, X_sq, inv_var, mu_iv,
            quad_mu, log_coeff, log_normalizer,
            n_k, s_k, sq_k,
            N, K, d, BK, BLOCK_N,
        )

        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer

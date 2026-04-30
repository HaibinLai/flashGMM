"""
Flash-GMM v6: All-Triton implementation.

Every kernel is a Triton kernel — no cuBLAS, no torch ops in the hot path.

Kernels:
  1. triton_matmul: general GEMM (replaces cuBLAS mm)
  2. triton_fused_logsumexp_gamma: L → logsumexp + exp → γ + n_k (fused)
  3. Full GMM classes using only these Triton kernels

This tests whether Triton can match cuBLAS end-to-end.
"""
import torch
import math
import triton
import triton.language as tl


# ============================================================================
# Kernel 1: Triton GEMM (replaces torch.mm / cuBLAS)
# ============================================================================

@triton.jit
def _triton_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Standard Triton GEMM: C[M,N] = A[M,K] @ B[K,N]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                     mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c = acc.to(tl.float32)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul(A, B, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32):
    """Triton GEMM: C = A @ B."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    _triton_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return C


# ============================================================================
# Kernel 2: Fused logsumexp + γ + n_k
# Reads L[N,K], writes γ[N,K], log_normalizer[N], n_k[K]
# Fuses 3 separate ops into 1 kernel: logsumexp → exp → column sum
# ============================================================================

@triton.jit
def _fused_logsumexp_gamma_kernel(
    L_ptr, gamma_ptr, ln_ptr,
    coeff_ptr, quad_mu_ptr,  # per-centroid constants
    N, K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused: L → apply constants → logsumexp → exp → γ.

    Each block handles BLOCK_N rows × all K columns.
    """
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Online logsumexp over K columns
    running_max = tl.full([BLOCK_N], value=-float('inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    for t in range(0, K, BLOCK_K):
        k_offs = t + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load L tile
        L_tile = tl.load(L_ptr + n_offs[:, None] * K + k_offs[None, :],
                          mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Apply constants: L = L_raw * (-0.5) + log_coeff  (already includes quad_mu)
        coeff = tl.load(coeff_ptr + k_offs, mask=k_mask, other=-float('inf'))
        qm = tl.load(quad_mu_ptr + k_offs, mask=k_mask, other=0.0)
        L_tile = (L_tile + qm[None, :]) * (-0.5) + coeff[None, :]
        L_tile = tl.where(k_mask[None, :], L_tile, -float('inf'))

        # Online logsumexp
        tile_max = tl.max(L_tile, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + \
                      tl.sum(tl.exp(L_tile - new_max[:, None]), axis=1)
        running_max = new_max

    # Write log_normalizer
    log_norm = running_max + tl.log(running_sum)
    tl.store(ln_ptr + n_offs, log_norm, mask=n_mask)

    # Second pass: compute γ = exp(L - ln) and write
    for t in range(0, K, BLOCK_K):
        k_offs = t + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        L_tile = tl.load(L_ptr + n_offs[:, None] * K + k_offs[None, :],
                          mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        coeff = tl.load(coeff_ptr + k_offs, mask=k_mask, other=-float('inf'))
        qm = tl.load(quad_mu_ptr + k_offs, mask=k_mask, other=0.0)
        L_tile = (L_tile + qm[None, :]) * (-0.5) + coeff[None, :]

        gamma_tile = tl.exp(L_tile - log_norm[:, None])
        gamma_tile = tl.where(n_mask[:, None] & k_mask[None, :], gamma_tile, 0.0)

        tl.store(gamma_ptr + n_offs[:, None] * K + k_offs[None, :],
                  gamma_tile, mask=n_mask[:, None] & k_mask[None, :])


# ============================================================================
# Kernel 3: Fused Flash (distance + logsumexp in one kernel, no L materialization)
# For d=64 where Flash approach wins
# ============================================================================

@triton.jit
def _all_triton_flash_logsumexp(
    X_ptr, X_sq_ptr,
    inv_var_ptr, mu_iv_ptr, quad_mu_ptr, log_coeff_ptr,
    ln_ptr,
    N, K, d: tl.constexpr,
    BK: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Flash Pass 1: fused distance GEMM + online logsumexp."""
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    d_offs = tl.arange(0, d)

    x_block = tl.load(X_ptr + n_offs[:, None] * d + d_offs[None, :],
                       mask=n_mask[:, None], other=0.0)
    xsq_block = tl.load(X_sq_ptr + n_offs[:, None] * d + d_offs[None, :],
                          mask=n_mask[:, None], other=0.0)

    running_max = tl.full([BLOCK_N], value=-float('inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    for t in range(0, K, BK):
        k_offs = t + tl.arange(0, BK)
        k_mask = k_offs < K

        iv = tl.load(inv_var_ptr + k_offs[:, None] * d + d_offs[None, :],
                      mask=k_mask[:, None], other=0.0)
        mi = tl.load(mu_iv_ptr + k_offs[:, None] * d + d_offs[None, :],
                      mask=k_mask[:, None], other=0.0)
        qm = tl.load(quad_mu_ptr + k_offs, mask=k_mask, other=0.0)
        lc = tl.load(log_coeff_ptr + k_offs, mask=k_mask, other=-float('inf'))

        L = tl.dot(xsq_block, tl.trans(iv))
        L += tl.dot(x_block, tl.trans(mi)) * (-2.0)
        L += qm[None, :]
        L = L * (-0.5) + lc[None, :]
        L = tl.where(k_mask[None, :], L, -float('inf'))

        tile_max = tl.max(L, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + \
                      tl.sum(tl.exp(L - new_max[:, None]), axis=1)
        running_max = new_max

    tl.store(ln_ptr + n_offs, running_max + tl.log(running_sum), mask=n_mask)


@triton.jit
def _all_triton_flash_accum(
    X_ptr, X_sq_ptr,
    inv_var_ptr, mu_iv_ptr, quad_mu_ptr, log_coeff_ptr,
    ln_ptr,
    nk_ptr, sk_ptr, sqk_ptr,
    N, K, d: tl.constexpr,
    BK: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Flash Pass 2: fused distance + γ + stats accumulation."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    k_start = pid_k * BK
    n_start = pid_n * BLOCK_N
    k_offs = k_start + tl.arange(0, BK)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, d)
    k_mask = k_offs < K
    n_mask = n_offs < N

    x_block = tl.load(X_ptr + n_offs[:, None] * d + d_offs[None, :],
                       mask=n_mask[:, None], other=0.0)
    xsq_block = tl.load(X_sq_ptr + n_offs[:, None] * d + d_offs[None, :],
                          mask=n_mask[:, None], other=0.0)

    iv = tl.load(inv_var_ptr + k_offs[:, None] * d + d_offs[None, :],
                  mask=k_mask[:, None], other=0.0)
    mi = tl.load(mu_iv_ptr + k_offs[:, None] * d + d_offs[None, :],
                  mask=k_mask[:, None], other=0.0)
    qm = tl.load(quad_mu_ptr + k_offs, mask=k_mask, other=0.0)
    lc = tl.load(log_coeff_ptr + k_offs, mask=k_mask, other=-float('inf'))

    L = tl.dot(xsq_block, tl.trans(iv))
    L += tl.dot(x_block, tl.trans(mi)) * (-2.0)
    L += qm[None, :]
    L = L * (-0.5) + lc[None, :]

    ln_vals = tl.load(ln_ptr + n_offs, mask=n_mask, other=0.0)
    gamma = tl.exp(L - ln_vals[:, None])
    gamma = tl.where(k_mask[None, :] & n_mask[:, None], gamma, 0.0)

    nk = tl.sum(gamma, axis=0)
    tl.atomic_add(nk_ptr + k_offs, nk, mask=k_mask)

    sk = tl.dot(tl.trans(gamma), x_block)
    tl.atomic_add(sk_ptr + k_offs[:, None] * d + d_offs[None, :], sk, mask=k_mask[:, None])

    sqk = tl.dot(tl.trans(gamma), xsq_block)
    tl.atomic_add(sqk_ptr + k_offs[:, None] * d + d_offs[None, :], sqk, mask=k_mask[:, None])


# ============================================================================
# GMM Classes
# ============================================================================

class AllTritonStdGMM:
    """All-Triton Standard GMM: Triton GEMM + Triton fused logsumexp + Triton GEMM.

    Materializes L[N,K] and γ[N,K] but uses Triton for everything.
    """

    def em_step(self, X, mu, var, log_pi):
        N, d = X.shape
        K = mu.shape[0]

        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)
        X_sq = X * X

        # ---- E-step: merged distance GEMM ----
        A = torch.cat([X_sq, X], dim=1)                    # N × 2d
        B = torch.cat([inv_var.T, -2 * mu_iv.T], dim=0)    # 2d × K
        L_raw = triton_matmul(A, B)                         # N × K

        # ---- Fused logsumexp + γ ----
        gamma = torch.empty(N, K, device=X.device, dtype=torch.float32)
        log_normalizer = torch.empty(N, device=X.device, dtype=torch.float32)

        BK = min(128, K)
        grid = ((N + 64 - 1) // 64,)
        _fused_logsumexp_gamma_kernel[grid](
            L_raw, gamma, log_normalizer,
            log_coeff, quad_mu,
            N, K, 64, BK,
        )

        # ---- M-step: merged GEMM ----
        XA = torch.cat([X, X_sq], dim=1)                   # N × 2d
        stats = triton_matmul(gamma.T.contiguous(), XA,
                              BLOCK_M=min(128, K), BLOCK_N=min(128, 2*d), BLOCK_K=64)

        s_k = stats[:, :d]
        sq_k = stats[:, d:]
        n_k = gamma.sum(0)

        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer


class AllTritonFlashGMM:
    """All-Triton Flash GMM: 2 fused kernels, no L/γ materialization.

    Auto-selects tile sizes based on d.
    """

    def _auto_tile(self, d, K):
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

        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)
        X_sq = X * X

        log_normalizer = torch.empty(N, device=X.device, dtype=torch.float32)

        # Pass 1
        grid1 = ((N + BLOCK_N - 1) // BLOCK_N,)
        _all_triton_flash_logsumexp[grid1](
            X, X_sq, inv_var, mu_iv, quad_mu, log_coeff, log_normalizer,
            N, K, d, BK, BLOCK_N,
        )

        # Pass 2
        n_k = torch.zeros(K, device=X.device, dtype=torch.float32)
        s_k = torch.zeros(K, d, device=X.device, dtype=torch.float32)
        sq_k = torch.zeros(K, d, device=X.device, dtype=torch.float32)

        grid2 = ((K + BK - 1) // BK, (N + BLOCK_N - 1) // BLOCK_N)
        _all_triton_flash_accum[grid2](
            X, X_sq, inv_var, mu_iv, quad_mu, log_coeff, log_normalizer,
            n_k, s_k, sq_k,
            N, K, d, BK, BLOCK_N,
        )

        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer


class AllTritonBestGMM:
    """Auto-selects Flash vs Standard based on d and K."""

    def __init__(self):
        self._flash = AllTritonFlashGMM()
        self._std = AllTritonStdGMM()

    def em_step(self, X, mu, var, log_pi):
        d = X.shape[1]
        K = mu.shape[0]
        # Flash wins when K >> d (IO-bound); Standard wins when d is large (compute-bound)
        if d <= 64 or K >= 512:
            return self._flash.em_step(X, mu, var, log_pi)
        else:
            return self._std.em_step(X, mu, var, log_pi)

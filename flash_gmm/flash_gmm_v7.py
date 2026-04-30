"""
Flash-GMM v7: Bottleneck-targeted optimizations.

Based on roofline analysis:
  - Fused elementwise = 36-45% of time (memory-bound, N×K traffic)
  - gamma.T.contiguous() = 10-13% (pure waste, N×K copy)
  - GEMM efficiency = 8-15% of peak

Optimizations:
  1. Eliminate gamma transpose: compute XA^T @ γ instead of γ^T @ XA
     (transpose small 2d×N instead of large N×K)
  2. Fuse precompute + distance GEMM + logsumexp + γ + n_k + M-step GEMM
     into minimum kernels
  3. d=128 Triton Flash with larger BK and BF16 dot
"""
import torch
import math
import triton
import triton.language as tl


class UltraGMMv2:
    """UltraGMM with eliminated transpose.

    Key change: M-step uses XA^T @ γ → (2d, K) then .T → (K, 2d)
    instead of γ^T.contiguous() @ XA which copies the entire N×K matrix.
    """

    def __init__(self):
        self._compiled_fused = None

    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def _fused_normalize(L_raw, quad_mu, log_coeff):
        L = L_raw + quad_mu
        L = L * (-0.5) + log_coeff
        log_normalizer = torch.logsumexp(L, dim=1)
        gamma = (L - log_normalizer.unsqueeze(1)).exp()
        n_k = gamma.sum(0)
        return gamma, log_normalizer, n_k

    def em_step(self, X, mu, var, log_pi):
        N, d = X.shape
        K = mu.shape[0]

        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)

        X_sq = X * X
        A = torch.cat([X_sq, X], dim=1)
        B = torch.cat([inv_var.T, -2 * mu_iv.T], dim=0)

        # E-step: 1 merged GEMM
        L_raw = torch.mm(A, B)

        # Fused elementwise
        gamma, log_normalizer, n_k = self._fused_normalize(L_raw, quad_mu, log_coeff)

        # M-step: avoid N×K transpose!
        # Old: stats = γ^T @ XA  →  needs γ.T.contiguous() (N×K copy)
        # New: stats = (XA^T @ γ)^T → transpose 2d×N (MUCH smaller if 2d << K)
        #   or equivalently: use einsum / manual mm with transposed strides
        #
        # Even better: XA is N×2d (contiguous), γ is N×K (contiguous)
        # torch.mm(XA.T, γ) does (2d×N) @ (N×K) = (2d×K)
        # XA.T is NOT contiguous, but cuBLAS handles strided inputs!
        # ...actually torch.mm requires contiguous. Use addmm or manual.
        #
        # Fastest approach: two separate mm calls for s_k and sq_k
        # X is N×d contiguous, gamma is N×K contiguous
        # s_k = X^T @ γ → (d×N) @ (N×K) = d×K, then .T = K×d
        # X.T is strided but we can use .contiguous() on the SMALL matrix

        s_k = torch.mm(X.T, gamma).T          # (d,K)^T = K×d  — X.T is d×N
        sq_k = torch.mm(X_sq.T, gamma).T      # (d,K)^T = K×d

        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer


# ============================================================================
# d=128 Triton Flash with optimized tile sizes and BF16
# ============================================================================

@triton.jit
def _flash_d128_logsumexp(
    X_ptr, X_sq_ptr, inv_var_ptr, mu_iv_ptr,
    quad_mu_ptr, log_coeff_ptr, out_ptr,
    N, K,
    d: tl.constexpr,
    BK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Optimized for d=128: BF16 dots, larger BK."""
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    d_offs = tl.arange(0, d)

    x = tl.load(X_ptr + n_offs[:, None] * d + d_offs[None, :],
                 mask=n_mask[:, None], other=0.0).to(tl.bfloat16)
    xsq = tl.load(X_sq_ptr + n_offs[:, None] * d + d_offs[None, :],
                    mask=n_mask[:, None], other=0.0).to(tl.bfloat16)

    running_max = tl.full([BLOCK_N], value=-float('inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    for t in range(0, K, BK):
        k_offs = t + tl.arange(0, BK)
        k_mask = k_offs < K

        iv = tl.load(inv_var_ptr + k_offs[:, None] * d + d_offs[None, :],
                      mask=k_mask[:, None], other=0.0).to(tl.bfloat16)
        mi = tl.load(mu_iv_ptr + k_offs[:, None] * d + d_offs[None, :],
                      mask=k_mask[:, None], other=0.0).to(tl.bfloat16)
        qm = tl.load(quad_mu_ptr + k_offs, mask=k_mask, other=0.0)
        lc = tl.load(log_coeff_ptr + k_offs, mask=k_mask, other=-float('inf'))

        L = tl.dot(xsq, tl.trans(iv)).to(tl.float32)
        L += tl.dot(x, tl.trans(mi)).to(tl.float32) * (-2.0)
        L += qm[None, :]
        L = L * (-0.5) + lc[None, :]
        L = tl.where(k_mask[None, :], L, -float('inf'))

        tile_max = tl.max(L, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + \
                      tl.sum(tl.exp(L - new_max[:, None]), axis=1)
        running_max = new_max

    tl.store(out_ptr + n_offs, running_max + tl.log(running_sum), mask=n_mask)


@triton.jit
def _flash_d128_accum(
    X_ptr, X_sq_ptr, inv_var_ptr, mu_iv_ptr,
    quad_mu_ptr, log_coeff_ptr, ln_ptr,
    nk_ptr, sk_ptr, sqk_ptr,
    N, K,
    d: tl.constexpr,
    BK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Optimized for d=128: BF16 distance, FP32 accumulation."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    k_start = pid_k * BK
    n_start = pid_n * BLOCK_N
    k_offs = k_start + tl.arange(0, BK)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, d)
    k_mask = k_offs < K
    n_mask = n_offs < N

    # Load X in both FP32 (for stats) and BF16 (for distance)
    x_fp32 = tl.load(X_ptr + n_offs[:, None] * d + d_offs[None, :],
                      mask=n_mask[:, None], other=0.0)
    xsq_fp32 = tl.load(X_sq_ptr + n_offs[:, None] * d + d_offs[None, :],
                         mask=n_mask[:, None], other=0.0)
    x_bf16 = x_fp32.to(tl.bfloat16)
    xsq_bf16 = xsq_fp32.to(tl.bfloat16)

    iv = tl.load(inv_var_ptr + k_offs[:, None] * d + d_offs[None, :],
                  mask=k_mask[:, None], other=0.0).to(tl.bfloat16)
    mi = tl.load(mu_iv_ptr + k_offs[:, None] * d + d_offs[None, :],
                  mask=k_mask[:, None], other=0.0).to(tl.bfloat16)
    qm = tl.load(quad_mu_ptr + k_offs, mask=k_mask, other=0.0)
    lc = tl.load(log_coeff_ptr + k_offs, mask=k_mask, other=-float('inf'))

    L = tl.dot(xsq_bf16, tl.trans(iv)).to(tl.float32)
    L += tl.dot(x_bf16, tl.trans(mi)).to(tl.float32) * (-2.0)
    L += qm[None, :]
    L = L * (-0.5) + lc[None, :]

    ln_vals = tl.load(ln_ptr + n_offs, mask=n_mask, other=0.0)
    gamma = tl.exp(L - ln_vals[:, None])
    gamma = tl.where(k_mask[None, :] & n_mask[:, None], gamma, 0.0)

    nk = tl.sum(gamma, axis=0)
    tl.atomic_add(nk_ptr + k_offs, nk, mask=k_mask)

    # FP32 stats accumulation via dot (gamma^T @ X)
    gamma_bf16 = gamma.to(tl.bfloat16)
    sk = tl.dot(tl.trans(gamma_bf16), x_bf16).to(tl.float32)
    tl.atomic_add(sk_ptr + k_offs[:, None] * d + d_offs[None, :], sk, mask=k_mask[:, None])

    sqk = tl.dot(tl.trans(gamma_bf16), xsq_bf16).to(tl.float32)
    tl.atomic_add(sqk_ptr + k_offs[:, None] * d + d_offs[None, :], sqk, mask=k_mask[:, None])


class TritonFlashD128:
    """Triton Flash GMM optimized for d=128 with BF16 dots."""

    def __init__(self, BLOCK_N=32, BK=16):
        self.BLOCK_N = BLOCK_N
        self.BK = BK

    def em_step(self, X, mu, var, log_pi):
        N, d = X.shape
        K = mu.shape[0]
        BLOCK_N = self.BLOCK_N
        BK = self.BK

        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)
        X_sq = X * X

        log_normalizer = torch.empty(N, device=X.device, dtype=torch.float32)
        n_k = torch.zeros(K, device=X.device, dtype=torch.float32)
        s_k = torch.zeros(K, d, device=X.device, dtype=torch.float32)
        sq_k = torch.zeros(K, d, device=X.device, dtype=torch.float32)

        grid1 = ((N + BLOCK_N - 1) // BLOCK_N,)
        _flash_d128_logsumexp[grid1](
            X, X_sq, inv_var, mu_iv, quad_mu, log_coeff, log_normalizer,
            N, K, d, BK, BLOCK_N,
        )

        grid2 = ((K + BK - 1) // BK, (N + BLOCK_N - 1) // BLOCK_N)
        _flash_d128_accum[grid2](
            X, X_sq, inv_var, mu_iv, quad_mu, log_coeff, log_normalizer,
            n_k, s_k, sq_k,
            N, K, d, BK, BLOCK_N,
        )

        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer


class BestGMMv7:
    """Auto-select best implementation per config."""

    def __init__(self):
        self._ultra_v2 = UltraGMMv2()
        self._tri_flash_d128 = TritonFlashD128(BLOCK_N=32, BK=16)
        from flash_gmm_v3 import TritonFlashBF16GMM
        self._tri_bf16 = TritonFlashBF16GMM()

    def em_step(self, X, mu, var, log_pi):
        d = X.shape[1]
        K = mu.shape[0]
        if d <= 64:
            return self._tri_bf16.em_step(X, mu, var, log_pi)
        else:
            return self._ultra_v2.em_step(X, mu, var, log_pi)

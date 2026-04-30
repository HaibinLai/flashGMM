"""
Flash-GMM v5: Maximum throughput single-iteration GMM.

Key optimizations over v2 GemmStandardGMM:
  1. Merged distance GEMM: [X², X] @ [inv_var; -2·mu_iv]^T → single GEMM (4 GEMMs → 2)
  2. torch.compile fused elementwise: add + scale + logsumexp + exp + sum in one fused kernel
  3. Merged M-step GEMM: γ^T @ [X, X²] → single GEMM
  4. Pre-allocated X² avoids recomputation

Total: 2 GEMMs + 1 compiled kernel per EM step (was: 4 GEMMs + 4 elementwise)
"""
import torch
import math


class UltraGMM:
    """Maximum-throughput GMM: 2 merged GEMMs + compiled fused elementwise."""

    def __init__(self):
        self._compiled_fused = None

    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def _fused_normalize(L_raw, quad_mu, log_coeff):
        """Fused: add constants + scale + logsumexp + exp-normalize + sum."""
        L = L_raw + quad_mu
        L = L * (-0.5) + log_coeff
        log_normalizer = torch.logsumexp(L, dim=1)
        gamma = (L - log_normalizer.unsqueeze(1)).exp()
        n_k = gamma.sum(0)
        return gamma, log_normalizer, n_k

    def em_step(self, X, mu, var, log_pi):
        N, d = X.shape
        K = mu.shape[0]

        # Precompute (O(Kd) — negligible)
        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)

        X_sq = X * X

        # ---- E-step: 1 merged GEMM + 1 fused kernel ----
        # Distance: [X², X]_{N×2d} @ [inv_var^T; -2·mu_iv^T]_{2d×K} = N×K
        A = torch.cat([X_sq, X], dim=1)                    # N × 2d
        B = torch.cat([inv_var.T, -2 * mu_iv.T], dim=0)    # 2d × K
        L_raw = torch.mm(A, B)                              # N × K  (one GEMM!)

        # Fused: L_raw + quad_mu → scale → logsumexp → exp → γ + n_k
        gamma, log_normalizer, n_k = self._fused_normalize(L_raw, quad_mu, log_coeff)

        # ---- M-step: 1 merged GEMM ----
        XA = torch.cat([X, X_sq], dim=1)                   # N × 2d
        stats = torch.mm(gamma.T, XA)                      # K × 2d  (one GEMM!)
        s_k = stats[:, :d]                                  # K × d
        sq_k = stats[:, d:]                                 # K × d

        # Update
        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer


class UltraFlashGMM:
    """Flash variant of UltraGMM: tiled, no L[N,K] materialization.

    Same merged-GEMM trick but with K-tiling + online logsumexp.
    Saves VRAM for large K at cost of 2x distance computation.
    """

    def __init__(self, BK=None):
        self.BK = BK

    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def _fused_update(s_k, sq_k, n_k, N):
        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()
        return new_mu, new_var, new_log_pi

    def em_step(self, X, mu, var, log_pi):
        N, d = X.shape
        K = mu.shape[0]
        BK = self.BK or min(K, 128)

        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)

        X_sq = X * X
        A = torch.cat([X_sq, X], dim=1)     # N × 2d (reused both passes)
        XA = torch.cat([X, X_sq], dim=1)     # N × 2d (for M-step)

        # ---- Pass 1: tiled online logsumexp ----
        running_max = torch.full((N,), -float('inf'), device=X.device)
        running_sum = torch.zeros(N, device=X.device)

        for t in range(0, K, BK):
            tk = min(BK, K - t)
            B_tile = torch.cat([inv_var[t:t+tk].T, -2 * mu_iv[t:t+tk].T], dim=0)
            L_tile = torch.mm(A, B_tile)  # N × tk
            L_tile.add_(quad_mu[t:t+tk])
            L_tile.mul_(-0.5)
            L_tile.add_(log_coeff[t:t+tk])

            tile_max = L_tile.max(dim=1).values
            new_max = torch.max(running_max, tile_max)
            running_sum = (running_sum * (running_max - new_max).exp()
                          + (L_tile - new_max.unsqueeze(1)).exp().sum(1))
            running_max = new_max

        log_normalizer = running_max + running_sum.log()

        # ---- Pass 2: recompute γ, accumulate stats ----
        n_k = torch.zeros(K, device=X.device)
        s_k = torch.zeros(K, d, device=X.device)
        sq_k = torch.zeros(K, d, device=X.device)

        for t in range(0, K, BK):
            tk = min(BK, K - t)
            B_tile = torch.cat([inv_var[t:t+tk].T, -2 * mu_iv[t:t+tk].T], dim=0)
            L_tile = torch.mm(A, B_tile)
            L_tile.add_(quad_mu[t:t+tk])
            L_tile.mul_(-0.5)
            L_tile.add_(log_coeff[t:t+tk])

            gamma_tile = (L_tile - log_normalizer.unsqueeze(1)).exp()
            n_k[t:t+tk] = gamma_tile.sum(0)

            # Merged M-step GEMM: γ_tile^T @ [X, X²]
            stats = torch.mm(gamma_tile.T, XA)  # tk × 2d
            s_k[t:t+tk] = stats[:, :d]
            sq_k[t:t+tk] = stats[:, d:]

        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer

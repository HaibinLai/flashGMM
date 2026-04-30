"""
Flash-GMM v2: GEMM-tiled approach using cuBLAS (Tensor Cores).

Key insight: Mahalanobis distance with diagonal covariance decomposes into GEMMs:
    dist[n,k] = (X²)@(1/σ²)ᵀ - 2·X@(μ/σ²)ᵀ + (μ²/σ²).sum(1)
Two GEMMs + per-k constant, leveraging cuBLAS Tensor Cores.

This approach is especially effective for d=64 (compute-bound regime) where
the custom CUDA kernel can't match cuBLAS's Tensor Core throughput.
"""
import torch
import math


class GemmFlashGMM:
    """Flash-GMM using cuBLAS GEMM for distance computation + M-step."""

    def __init__(self, BK=None):
        """BK: centroid tile size. None = auto (K if fits, else 64)."""
        self.BK = BK

    def em_step(self, X, mu, var, log_pi):
        """One full E+M iteration. Returns (new_mu, new_var, new_log_pi, log_normalizer)."""
        N, d = X.shape
        K = mu.shape[0]
        BK = self.BK or min(K, 64)

        # Precompute (small, O(Kd))
        inv_var = 1.0 / var                         # K×d
        mu_iv = mu * inv_var                         # K×d
        quad_mu = (mu * mu_iv).sum(1)                # K
        log_det = var.log().sum(1)                   # K
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)  # K
        X_sq = X * X                                 # N×d

        # ---- Pass 1: online logsumexp → log_normalizer ----
        running_max = torch.full((N,), -float('inf'), device=X.device)
        running_sum = torch.zeros(N, device=X.device)

        for t in range(0, K, BK):
            tk = min(BK, K - t)
            # GEMM-based Mahalanobis distance: O(N·tk·d) FLOPs via Tensor Cores
            # dist = X² @ inv_var[tile]ᵀ - 2·X @ mu_iv[tile]ᵀ + quad_mu[tile]
            L_tile = torch.mm(X_sq, inv_var[t:t+tk].T)       # N×tk
            L_tile.addmm_(X, mu_iv[t:t+tk].T, alpha=-2.0)    # fused: L_tile -= 2·X@mu_ivᵀ
            L_tile.add_(quad_mu[t:t+tk])                       # broadcast add
            L_tile.mul_(-0.5)
            L_tile.add_(log_coeff[t:t+tk])                     # L_tile = log_coeff - 0.5*dist

            # Online logsumexp update
            tile_max = L_tile.max(dim=1).values                # N
            new_max = torch.max(running_max, tile_max)
            running_sum = (running_sum * (running_max - new_max).exp()
                          + (L_tile - new_max.unsqueeze(1)).exp().sum(1))
            running_max = new_max

        log_normalizer = running_max + running_sum.log()

        # ---- Pass 2: recompute γ, accumulate stats via GEMM ----
        n_k = torch.zeros(K, device=X.device)
        s_k = torch.zeros(K, d, device=X.device)
        sq_k = torch.zeros(K, d, device=X.device)

        for t in range(0, K, BK):
            tk = min(BK, K - t)
            # Recompute L_tile
            L_tile = torch.mm(X_sq, inv_var[t:t+tk].T)
            L_tile.addmm_(X, mu_iv[t:t+tk].T, alpha=-2.0)
            L_tile.add_(quad_mu[t:t+tk])
            L_tile.mul_(-0.5)
            L_tile.add_(log_coeff[t:t+tk])

            # γ_tile = exp(L_tile - log_normalizer)
            gamma_tile = (L_tile - log_normalizer.unsqueeze(1)).exp()  # N×tk

            # Accumulate M-step stats via GEMM
            n_k[t:t+tk] = gamma_tile.sum(0)
            s_k[t:t+tk] = torch.mm(gamma_tile.T, X)       # tk×d GEMM
            sq_k[t:t+tk] = torch.mm(gamma_tile.T, X_sq)   # tk×d GEMM

        # ---- Parameter update ----
        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer


class GemmStandardGMM:
    """Standard GMM using cuBLAS GEMM (no tiling, materializes full L[N,K]).

    Serves as an improved standard baseline: replaces per-element logf()/division
    in the custom CUDA E-step kernel with cuBLAS GEMM for distance computation.
    """

    def em_step(self, X, mu, var, log_pi):
        N, d = X.shape
        K = mu.shape[0]

        # Precompute
        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)
        X_sq = X * X

        # E-step: GEMM-based distance → full L[N,K]
        L = torch.mm(X_sq, inv_var.T)                # N×K
        L.addmm_(X, mu_iv.T, alpha=-2.0)
        L.add_(quad_mu)
        L.mul_(-0.5)
        L.add_(log_coeff)

        # Logsumexp normalize
        log_normalizer = torch.logsumexp(L, dim=1)   # N
        gamma = (L - log_normalizer.unsqueeze(1)).exp()  # N×K

        # M-step via GEMM
        n_k = gamma.sum(0)
        new_mu = torch.mm(gamma.T, X) / n_k.unsqueeze(1)
        new_var = (torch.mm(gamma.T, X_sq) / n_k.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer

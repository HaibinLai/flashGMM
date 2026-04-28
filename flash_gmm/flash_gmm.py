"""
Flash-GMM: E+M fully fused GMM with diagonal covariance.

Eliminates ALL N×K matrix materialization by:
  1. Pass 1: Online log-sum-exp over centroid tiles → log_normalizer (O(N))
  2. Pass 2: Recompute γ_tile on-chip, immediately accumulate M-step statistics
  3. Update parameters from sufficient statistics (O(Kd))

Total IO: O(Nd + Kd) per iteration — same level as FlashAssign in Flash-KMeans.
Trade-off: 2× FLOPs for distance computation, but eliminates dominant HBM traffic.
"""

import torch
import time
from standard_gmm import IOCounter, generate_gmm_data


class FlashGMM:
    """
    Flash-GMM with diagonal covariance.
    E+M fully fused: no N×K matrix ever touches main memory.
    """

    def __init__(self, K: int, d: int, BK: int = 4, max_iter: int = 100,
                 tol: float = 1e-4, seed: int = 42):
        self.K = K
        self.d = d
        self.BK = BK
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self.mu = None
        self.var = None
        self.log_pi = None
        self.io_counter = IOCounter()

    def _init_params(self, X: torch.Tensor):
        N = X.shape[0]
        gen = torch.Generator().manual_seed(self.seed)
        idx = torch.randperm(N, generator=gen)[:self.K]
        self.mu = X[idx].clone()
        self.var = torch.ones(self.K, self.d) * X.var(dim=0, keepdim=True).mean()
        self.log_pi = torch.full((self.K,), -torch.log(torch.tensor(float(self.K))))

    def _fused_em_step(self, X: torch.Tensor) -> torch.Tensor:
        """
        Single fused E+M step.

        Pass 1 (online log-sum-exp):
          For each centroid tile:
            - Read tile params from HBM
            - Compute L_tile on-chip
            - Update running (max, sum_exp)
          → Output: log_normalizer ∈ R^N

        Pass 2 (accumulate sufficient statistics):
          For each centroid tile:
            - Read tile params from HBM (again)
            - Recompute L_tile on-chip
            - Compute γ_tile = exp(L_tile - log_normalizer) on-chip
            - Accumulate: n_k += Σ_n γ_nk, s_k += γ^T X, sq_k += γ^T X²
          → Output: n_k, s_k, sq_k (all O(Kd))

        Update: μ, σ², π from sufficient statistics.

        Returns: log_normalizer (for log-likelihood tracking)
        """
        N, d = X.shape
        K = self.K
        BK = self.BK
        io = self.io_counter
        num_tiles = (K + BK - 1) // BK
        log_2pi_d = d * torch.log(torch.tensor(2 * torch.pi))

        # ============ PASS 1: Online log-sum-exp ============
        io.record("X_pass1", X, "read")

        running_max = torch.full((N,), -float("inf"))
        running_sum_exp = torch.zeros(N)

        for t in range(num_tiles):
            k_start = t * BK
            k_end = min(k_start + BK, K)

            mu_tile = self.mu[k_start:k_end]
            var_tile = self.var[k_start:k_end]
            lp_tile = self.log_pi[k_start:k_end]
            io.record(f"params_p1_{t}", mu_tile, "read")
            io.record(f"params_p1_{t}", var_tile, "read")
            io.record(f"params_p1_{t}", lp_tile, "read")

            # On-chip computation
            log_det_tile = var_tile.log().sum(dim=1)
            diff = X.unsqueeze(1) - mu_tile.unsqueeze(0)
            mahal = (diff ** 2 / var_tile.unsqueeze(0)).sum(dim=2)
            L_tile = lp_tile.unsqueeze(0) - 0.5 * (log_2pi_d + log_det_tile.unsqueeze(0) + mahal)

            tile_max = L_tile.max(dim=1).values
            new_max = torch.maximum(running_max, tile_max)
            running_sum_exp = running_sum_exp * (running_max - new_max).exp()
            running_sum_exp += (L_tile - new_max.unsqueeze(1)).exp().sum(dim=1)
            running_max = new_max

        log_normalizer = running_max + running_sum_exp.log()

        # ============ PASS 2: Recompute + accumulate M-step stats ============
        io.record("X_pass2", X, "read")

        n_k = torch.zeros(K)
        s_k = torch.zeros(K, d)
        sq_k = torch.zeros(K, d)

        for t in range(num_tiles):
            k_start = t * BK
            k_end = min(k_start + BK, K)

            mu_tile = self.mu[k_start:k_end]
            var_tile = self.var[k_start:k_end]
            lp_tile = self.log_pi[k_start:k_end]
            io.record(f"params_p2_{t}", mu_tile, "read")
            io.record(f"params_p2_{t}", var_tile, "read")
            io.record(f"params_p2_{t}", lp_tile, "read")

            # Recompute L_tile (2× FLOPs, 0× HBM for L)
            log_det_tile = var_tile.log().sum(dim=1)
            diff = X.unsqueeze(1) - mu_tile.unsqueeze(0)
            mahal = (diff ** 2 / var_tile.unsqueeze(0)).sum(dim=2)
            L_tile = lp_tile.unsqueeze(0) - 0.5 * (log_2pi_d + log_det_tile.unsqueeze(0) + mahal)

            # γ_tile on-chip
            gamma_tile = (L_tile - log_normalizer.unsqueeze(1)).exp()

            # Accumulate sufficient statistics on-chip
            n_k[k_start:k_end] = gamma_tile.sum(dim=0)
            s_k[k_start:k_end] = gamma_tile.T @ X
            sq_k[k_start:k_end] = gamma_tile.T @ (X ** 2)

        # ============ Update parameters from sufficient statistics ============
        # μ_k = s_k / n_k
        self.mu = s_k / n_k.unsqueeze(1)
        # σ²_k = sq_k / n_k - μ_k²
        self.var = (sq_k / n_k.unsqueeze(1) - self.mu ** 2).clamp(min=1e-6)
        # π_k = n_k / N
        self.log_pi = (n_k / N).log()

        # Write updated params (tiny: O(Kd))
        io.record("mu_new", self.mu, "write")
        io.record("var_new", self.var, "write")
        io.record("log_pi_new", self.log_pi, "write")

        return log_normalizer

    def fit(self, X: torch.Tensor, verbose: bool = False):
        self._init_params(X)
        prev_ll = -float("inf")
        history = []

        for it in range(self.max_iter):
            self.io_counter.reset()
            t0 = time.time()

            log_normalizer = self._fused_em_step(X)
            avg_ll = log_normalizer.mean().item()

            elapsed = time.time() - t0
            history.append({
                "iter": it,
                "log_likelihood": avg_ll,
                "time_ms": elapsed * 1000,
                "io_bytes": self.io_counter.total(),
            })

            if verbose and (it % 10 == 0 or it < 5):
                print(f"  iter {it:3d}: LL={avg_ll:.4f}  time={elapsed*1000:.1f}ms  IO={self.io_counter.total()/1e6:.1f}MB")

            if abs(avg_ll - prev_ll) < self.tol:
                if verbose:
                    print(f"  Converged at iter {it}")
                break
            prev_ll = avg_ll

        return history

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Compute γ (requires materializing it since caller needs it)."""
        from flash_e_step import flash_e_step
        io = IOCounter()
        gamma, _ = flash_e_step(X, self.mu, self.var, self.log_pi, BK=self.BK, io_counter=io)
        return gamma

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(X).argmax(dim=1)


if __name__ == "__main__":
    print("=== Flash-GMM (E+M Fused) ===")
    N, K, d = 4096, 8, 32
    X, labels, _, _, _ = generate_gmm_data(N, K, d)
    print(f"Data: N={N}, K={K}, d={d}")

    model = FlashGMM(K=K, d=d, BK=4, max_iter=50, tol=1e-4)
    history = model.fit(X, verbose=True)
    print(f"\nFinal IO per iteration: {history[-1]['io_bytes']/1e6:.2f} MB")
    print(f"IO breakdown (last iter):\n{model.io_counter.summary()}")

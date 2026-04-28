"""
FlashE-step: Fused E-step that eliminates log-likelihood matrix materialization.

Simulates the FlashAssign pattern from Flash-KMeans:
  - Stream centroid tiles sequentially
  - Maintain online log-sum-exp running state (m_i, sum_exp_i) per point
  - Never materialize the full N×K log-likelihood matrix L

IO savings: eliminates 2×Θ(NK) HBM traffic from L materialization.
Still outputs γ ∈ R^{N×K} for the standard M-step.
"""

import torch
from standard_gmm import IOCounter


def flash_e_step(
    X: torch.Tensor,        # (N, d)
    mu: torch.Tensor,       # (K, d)
    var: torch.Tensor,       # (K, d) diagonal variances
    log_pi: torch.Tensor,   # (K,) log mixing weights
    BK: int = 4,            # centroid tile size
    io_counter: IOCounter | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused E-step using online log-sum-exp (analogous to FlashAssign's online argmin).

    Instead of:
      1. Compute full L ∈ R^{N×K} → write to HBM
      2. Read L → log-sum-exp → write γ

    We do:
      For each centroid tile of size BK:
        - Compute local log-likelihoods on-chip
        - Update running (max, sum_exp) state using online log-sum-exp
        - Accumulate partial γ columns
      Final: normalize to get γ

    This eliminates the L matrix entirely.
    """
    N, d = X.shape
    K = mu.shape[0]
    io = io_counter or IOCounter()

    # --- Read X once ---
    io.record("X", X, "read")

    # Running states per point: (N,)
    running_max = torch.full((N,), -float("inf"))  # m_i: running max of log-likelihoods
    running_sum_exp = torch.zeros(N)                # Σ exp(ℓ_{ik} - m_i)

    # We need to store γ columns as we go (since M-step needs full γ)
    # But we avoid the L matrix
    gamma = torch.empty(N, K)

    # Precompute constant
    log_2pi_d = d * torch.log(torch.tensor(2 * torch.pi))

    # --- Stream centroid tiles ---
    num_tiles = (K + BK - 1) // BK
    for t in range(num_tiles):
        k_start = t * BK
        k_end = min(k_start + BK, K)
        tile_size = k_end - k_start

        # --- Read centroid tile from "HBM" ---
        mu_tile = mu[k_start:k_end]     # (tile_size, d)
        var_tile = var[k_start:k_end]   # (tile_size, d)
        lp_tile = log_pi[k_start:k_end] # (tile_size,)
        io.record(f"mu_tile_{t}", mu_tile, "read")
        io.record(f"var_tile_{t}", var_tile, "read")
        io.record(f"logpi_tile_{t}", lp_tile, "read")

        # --- Compute local log-likelihoods ON CHIP ---
        # log N(x|μ,σ²) = -d/2 log(2π) - 1/2 Σ log(σ²) - 1/2 Σ (x-μ)²/σ²
        log_det_tile = var_tile.log().sum(dim=1)  # (tile_size,)
        diff = X.unsqueeze(1) - mu_tile.unsqueeze(0)  # (N, tile_size, d)
        mahal = (diff ** 2 / var_tile.unsqueeze(0)).sum(dim=2)  # (N, tile_size)
        L_tile = lp_tile.unsqueeze(0) - 0.5 * (log_2pi_d + log_det_tile.unsqueeze(0) + mahal)
        # L_tile: (N, tile_size) — ON CHIP, never written to HBM

        # --- Online log-sum-exp update ---
        # For each point i, we maintain (m_i, s_i) where:
        #   m_i = max over all seen ℓ_{ik}
        #   s_i = Σ_{k seen} exp(ℓ_{ik} - m_i)
        #
        # When processing a new tile with values ℓ_new:
        #   m_new = max(ℓ_new) per point
        #   m_combined = max(m_i, m_new)
        #   s_i = s_i * exp(m_i - m_combined) + Σ exp(ℓ_new - m_combined)

        tile_max = L_tile.max(dim=1).values  # (N,)
        new_max = torch.maximum(running_max, tile_max)  # (N,)

        # Rescale previous sum_exp
        running_sum_exp = running_sum_exp * (running_max - new_max).exp()
        # Add current tile contribution
        running_sum_exp += (L_tile - new_max.unsqueeze(1)).exp().sum(dim=1)
        running_max = new_max

        # Store unnormalized log-gamma for this tile (will normalize at end)
        gamma[:, k_start:k_end] = L_tile

    # --- Final normalization ---
    # log_sum_exp = running_max + log(running_sum_exp)
    log_normalizer = running_max + running_sum_exp.log()  # (N,)
    gamma = (gamma - log_normalizer.unsqueeze(1)).exp()

    # --- Write γ to HBM ---
    io.record("gamma", gamma, "write")

    return gamma, log_normalizer


def flash_e_step_no_gamma(
    X: torch.Tensor,        # (N, d)
    mu: torch.Tensor,       # (K, d)
    var: torch.Tensor,       # (K, d)
    log_pi: torch.Tensor,   # (K,)
    BK: int = 4,
    io_counter: IOCounter | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flash E+M fused variant: computes E-step statistics WITHOUT materializing γ.

    Instead of outputting γ, accumulates M-step sufficient statistics on-chip:
      - n_k = Σ_n γ_nk                    (K,)
      - s_k = Σ_n γ_nk * x_n              (K, d)
      - sq_k = Σ_n γ_nk * x_n²            (K, d)

    This is the key innovation for diagonal covariance:
    per-component on-chip storage is only (1 + d + d) = (2d+1) floats.

    Returns: (n_k, s_k, sq_k, log_likelihood)
    """
    N, d = X.shape
    K = mu.shape[0]
    io = io_counter or IOCounter()

    io.record("X", X, "read")

    log_2pi_d = d * torch.log(torch.tensor(2 * torch.pi))

    # === PASS 1: Compute log-normalizer via online log-sum-exp ===
    running_max = torch.full((N,), -float("inf"))
    running_sum_exp = torch.zeros(N)

    # We also need to store the raw log-likelihoods per tile to compute γ in pass 2.
    # But that defeats the purpose! Instead, we do TWO passes over centroids:
    #   Pass 1: compute log_normalizer
    #   Pass 2: recompute L_tile, compute γ_tile = exp(L_tile - log_normalizer), accumulate stats
    # This doubles compute but eliminates ALL N×K memory traffic.

    num_tiles = (K + BK - 1) // BK
    for t in range(num_tiles):
        k_start = t * BK
        k_end = min(k_start + BK, K)

        mu_tile = mu[k_start:k_end]
        var_tile = var[k_start:k_end]
        lp_tile = log_pi[k_start:k_end]
        io.record(f"mu_tile_p1_{t}", mu_tile, "read")
        io.record(f"var_tile_p1_{t}", var_tile, "read")
        io.record(f"logpi_tile_p1_{t}", lp_tile, "read")

        log_det_tile = var_tile.log().sum(dim=1)
        diff = X.unsqueeze(1) - mu_tile.unsqueeze(0)
        mahal = (diff ** 2 / var_tile.unsqueeze(0)).sum(dim=2)
        L_tile = lp_tile.unsqueeze(0) - 0.5 * (log_2pi_d + log_det_tile.unsqueeze(0) + mahal)

        tile_max = L_tile.max(dim=1).values
        new_max = torch.maximum(running_max, tile_max)
        running_sum_exp = running_sum_exp * (running_max - new_max).exp()
        running_sum_exp += (L_tile - new_max.unsqueeze(1)).exp().sum(dim=1)
        running_max = new_max

    log_normalizer = running_max + running_sum_exp.log()  # (N,)

    # === PASS 2: Recompute L_tile, get γ_tile, accumulate M-step stats ===
    n_k = torch.zeros(K)
    s_k = torch.zeros(K, d)
    sq_k = torch.zeros(K, d)

    io.record("X", X, "read")  # read X again in pass 2

    for t in range(num_tiles):
        k_start = t * BK
        k_end = min(k_start + BK, K)

        mu_tile = mu[k_start:k_end]
        var_tile = var[k_start:k_end]
        lp_tile = log_pi[k_start:k_end]
        io.record(f"mu_tile_p2_{t}", mu_tile, "read")
        io.record(f"var_tile_p2_{t}", var_tile, "read")
        io.record(f"logpi_tile_p2_{t}", lp_tile, "read")

        # Recompute local log-likelihoods (extra FLOPs, but no HBM traffic for L)
        log_det_tile = var_tile.log().sum(dim=1)
        diff = X.unsqueeze(1) - mu_tile.unsqueeze(0)
        mahal = (diff ** 2 / var_tile.unsqueeze(0)).sum(dim=2)
        L_tile = lp_tile.unsqueeze(0) - 0.5 * (log_2pi_d + log_det_tile.unsqueeze(0) + mahal)

        # Compute γ for this tile ON CHIP
        gamma_tile = (L_tile - log_normalizer.unsqueeze(1)).exp()  # (N, tile_size)

        # Accumulate M-step sufficient statistics ON CHIP
        # n_k[k_start:k_end] += γ_tile.sum(dim=0)
        n_k[k_start:k_end] = gamma_tile.sum(dim=0)
        # s_k = γ^T X  for this tile's components
        s_k[k_start:k_end] = gamma_tile.T @ X        # (tile_size, d)
        # sq_k = γ^T (X²) for this tile's components
        sq_k[k_start:k_end] = gamma_tile.T @ (X ** 2)  # (tile_size, d)

    # Write only the sufficient statistics (tiny: O(Kd))
    io.record("n_k", n_k, "write")
    io.record("s_k", s_k, "write")
    io.record("sq_k", sq_k, "write")

    return n_k, s_k, sq_k, log_normalizer


if __name__ == "__main__":
    from standard_gmm import StandardGMM, generate_gmm_data

    print("=== Flash E-step Validation ===")
    N, K, d = 1024, 8, 16
    X, _, _, _, _ = generate_gmm_data(N, K, d)

    # Initialize a standard GMM and run one E-step
    model = StandardGMM(K=K, d=d)
    model._init_params(X)

    # Standard E-step
    model.io_counter.reset()
    gamma_std, ll_std = model.e_step(X)
    print(f"Standard E-step IO: {model.io_counter.total()/1e6:.4f} MB")

    # Flash E-step
    io_flash = IOCounter()
    gamma_flash, ll_flash = flash_e_step(
        X, model.mu, model.var, model.log_pi, BK=4, io_counter=io_flash
    )
    print(f"Flash E-step IO:    {io_flash.total()/1e6:.4f} MB")

    # Compare
    max_diff_gamma = (gamma_std - gamma_flash).abs().max().item()
    max_diff_ll = (ll_std - ll_flash).abs().max().item()
    print(f"\nMax |γ_std - γ_flash|:  {max_diff_gamma:.2e}")
    print(f"Max |LL_std - LL_flash|: {max_diff_ll:.2e}")
    print(f"γ match: {'PASS' if max_diff_gamma < 1e-5 else 'FAIL'}")
    print(f"LL match: {'PASS' if max_diff_ll < 1e-4 else 'FAIL'}")

    # IO savings
    saved = model.io_counter.total() - io_flash.total()
    print(f"\nIO saved: {saved/1e6:.4f} MB ({saved/model.io_counter.total()*100:.1f}%)")
    print(f"  = 2×N×K×4 bytes = {2*N*K*4/1e6:.4f} MB (theoretical L materialization)")

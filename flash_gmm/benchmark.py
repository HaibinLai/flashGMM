"""
Benchmark: IO analysis across Standard GMM, Flash E-step, and Flash E+M.

Compares simulated HBM traffic under varying (N, K, d) configurations.
"""

import torch
import sys
from standard_gmm import StandardGMM, IOCounter, generate_gmm_data
from flash_e_step import flash_e_step
from flash_gmm import FlashGMM


def benchmark_io(N: int, K: int, d: int, seed: int = 42):
    """Run one iteration of each variant and return IO stats."""
    X, _, _, _, _ = generate_gmm_data(N, K, d, seed=seed)
    elem = 4  # float32 bytes

    # --- Standard GMM ---
    std = StandardGMM(K=K, d=d, seed=seed)
    std._init_params(X)
    # Save initial params for flash comparison
    init_mu = std.mu.clone()
    init_var = std.var.clone()
    init_log_pi = std.log_pi.clone()
    std.io_counter.reset()
    gamma_std, ll_std = std.e_step(X)
    std.m_step(X, gamma_std)
    io_standard = std.io_counter.total()

    # --- Flash E-step (L eliminated, γ still materialized) ---
    flash_io = IOCounter()
    gamma_flash, ll_flash = flash_e_step(X, init_mu, init_var, init_log_pi, BK=4, io_counter=flash_io)
    # Simulate M-step IO (same as standard since γ is materialized)
    flash_io.record("gamma_mstep", gamma_flash, "read")
    flash_io.record("X_mstep", X, "read")
    flash_io.record("params_mstep", std.mu, "write")
    flash_io.record("params_mstep", std.var, "write")
    flash_io.record("params_mstep", std.log_pi, "write")
    io_flash_e = flash_io.total()

    # --- Flash E+M (fully fused) ---
    fused = FlashGMM(K=K, d=d, BK=4, seed=seed)
    fused._init_params(X)
    fused.io_counter.reset()
    fused._fused_em_step(X)
    io_flash_em = fused.io_counter.total()

    # --- Theoretical bounds ---
    # Standard: L write + L read + γ write + γ read + X read×2 + params read×2 + params write
    theo_standard = (2 * N * K + 2 * N * K + 2 * N * d + 2 * K * d + K) * elem
    # Flash E: γ write + γ read + X read×1 + params read×2 + params write
    # (L eliminated, but γ still round-trips)
    theo_flash_e = (2 * N * K + N * d + 2 * K * d + K + N * d + K * d) * elem
    # Flash E+M: X read×2 + params read×2 + stats write (O(Kd))
    theo_flash_em = (2 * N * d + 4 * K * d + 2 * K) * elem

    return {
        "N": N, "K": K, "d": d,
        "io_standard_MB": io_standard / 1e6,
        "io_flash_e_MB": io_flash_e / 1e6,
        "io_flash_em_MB": io_flash_em / 1e6,
        "theo_standard_MB": theo_standard / 1e6,
        "theo_flash_em_MB": theo_flash_em / 1e6,
        "speedup_flash_e": io_standard / io_flash_e if io_flash_e > 0 else 0,
        "speedup_flash_em": io_standard / io_flash_em if io_flash_em > 0 else 0,
        "gamma_correct": (gamma_std - gamma_flash).abs().max().item() < 1e-4,
    }


def run_benchmark():
    configs = [
        # (N, K, d) — sweep different regimes
        # Small scale
        (1024,    4,   16),
        (1024,    8,   16),
        (1024,   16,   16),
        # Medium scale
        (4096,    8,   32),
        (4096,   16,   32),
        (4096,   64,   32),
        # Large N, small K (compute-intensive M-step)
        (16384,   8,   64),
        (16384,  16,   64),
        # Large K (IO-dominated E-step)
        (4096,  128,   32),
        (4096,  256,   32),
        # High dimension
        (4096,   16,  128),
        (4096,   16,  256),
        # Extreme IO ratio
        (8192,  512,   64),
        (16384, 256,  128),
    ]

    print(f"{'N':>7s} {'K':>5s} {'d':>5s} | {'Std IO':>10s} {'FlashE IO':>10s} {'FlashEM IO':>11s} | {'E speedup':>10s} {'EM speedup':>10s} | {'NK/Nd':>6s} {'correct':>7s}")
    print("-" * 110)

    for N, K, d in configs:
        try:
            r = benchmark_io(N, K, d)
            nk_nd = (N * K) / (N * d)
            print(f"{N:7d} {K:5d} {d:5d} | "
                  f"{r['io_standard_MB']:9.2f}M {r['io_flash_e_MB']:9.2f}M {r['io_flash_em_MB']:10.2f}M | "
                  f"{r['speedup_flash_e']:9.2f}x {r['speedup_flash_em']:9.2f}x | "
                  f"{nk_nd:5.1f}x {'PASS' if r['gamma_correct'] else 'FAIL':>7s}")
        except Exception as e:
            print(f"{N:7d} {K:5d} {d:5d} | ERROR: {e}")

    # Print IO model explanation
    print("\n=== IO Model Summary ===")
    print("Standard GMM:  4×Θ(NK) + 2×Θ(Nd) + 2×Θ(Kd)  [L write/read + γ write/read + X read + params]")
    print("Flash E-step:  2×Θ(NK) + 2×Θ(Nd) + 2×Θ(Kd)  [γ write/read only, L eliminated]")
    print("Flash E+M:     2×Θ(Nd) + 4×Θ(Kd)             [X read×2 + params read×2 + stats write]")
    print("\nKey insight: When K >> d, the N×K matrices dominate IO.")
    print("Flash E+M eliminates ALL N×K traffic, achieving IO proportional only to input size.")


if __name__ == "__main__":
    run_benchmark()

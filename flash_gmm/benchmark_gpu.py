"""
GPU benchmark: Standard GMM vs Flash-GMM on CUDA.
Designed for ~4GB free VRAM.
"""
import torch
import time
import sys
import os

# Setup library path
torch_lib = os.path.join(torch.__path__[0], "lib")
if torch_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import flash_gmm_native as _C


def warmup_gpu():
    """Warm up GPU with a small allocation."""
    x = torch.randn(256, 256, device="cuda")
    _ = x @ x.T
    torch.cuda.synchronize()


def benchmark_one(name, fn, n_warmup=3, n_repeat=10):
    """Benchmark a function with warmup and timing."""
    for _ in range(n_warmup):
        fn()
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg = sum(times) / len(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5
    return avg, std, result


def run_gpu_benchmark():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
    print()

    configs = [
        # (N, K, d) — fit within ~4GB
        # Small (warm-up level)
        (4096,     8,   32),
        (4096,    16,   32),
        # Medium
        (16384,   16,   64),
        (16384,   64,   64),
        (16384,  128,   64),
        # Large N, large K (IO-dominated — Flash should shine)
        (65536,   64,  128),
        (65536,  256,  128),
        (65536,  512,  128),
        # Very large N
        (262144,  64,   64),
        (262144, 128,   64),
        # Extreme K (maximum IO savings)
        (32768,  1024,  64),
        (65536,  1024,  64),
    ]

    print(f"{'N':>8s} {'K':>6s} {'d':>4s} | {'Std E-step':>12s} {'Flash E+M':>12s} | {'Speedup':>8s} | {'VRAM(std)':>10s} {'VRAM(flash)':>11s} | {'Correct':>7s}")
    print("-" * 105)

    for N, K, d in configs:
        # Check memory fits
        X_bytes = N * d * 4
        L_bytes = N * K * 4  # standard needs this
        total_std = X_bytes + L_bytes * 2 + K * d * 4 * 2
        total_flash = X_bytes + K * d * 4 * 2 + K * 4 * 3  # n_k, s_k, sq_k

        free_mem = torch.cuda.mem_get_info()[0]
        if total_std > free_mem * 0.8:
            print(f"{N:8d} {K:6d} {d:4d} | {'OOM':>12s} {'--':>12s} | {'--':>8s} | {total_std/1e6:>9.1f}M {'--':>11s} | {'SKIP':>7s}")
            continue

        try:
            # Generate data on GPU
            torch.manual_seed(42)
            X = torch.randn(N, d, device="cuda")
            gen = torch.Generator(device="cuda").manual_seed(42)
            idx = torch.randperm(N, device="cuda", generator=gen)[:K]
            mu = X[idx].clone()
            var = torch.ones(K, d, device="cuda") * X.var(dim=0, keepdim=True).mean()
            log_pi = torch.full((K,), -torch.log(torch.tensor(float(K))), device="cuda")

            # Benchmark standard E-step + M-step
            def run_standard():
                gamma, ln = _C.standard_e_step(X, mu, var, log_pi)
                new_mu, new_var, new_lp = _C.standard_m_step(X, gamma)
                return gamma, ln, new_mu

            # Benchmark Flash E+M fused
            BN = min(256, N)
            BK = min(8, K)
            def run_flash():
                return _C.flash_em_fused(X, mu, var, log_pi, BN, BK)

            # Run benchmarks
            t_std, s_std, res_std = benchmark_one("std", run_standard)
            t_flash, s_flash, res_flash = benchmark_one("flash", run_flash)

            # Correctness check
            gamma_std, ln_std, mu_std = res_std
            mu_flash, var_flash, lp_flash, ln_flash = res_flash
            ln_diff = (ln_std - ln_flash).abs().max().item()
            correct = ln_diff < 0.1

            speedup = t_std / t_flash if t_flash > 0 else 0

            print(f"{N:8d} {K:6d} {d:4d} | "
                  f"{t_std:10.2f}ms {t_flash:10.2f}ms | "
                  f"{speedup:7.2f}x | "
                  f"{total_std/1e6:>9.1f}M {total_flash/1e6:>10.1f}M | "
                  f"{'PASS' if correct else 'FAIL':>7s}")

            # Cleanup
            del X, mu, var, log_pi, res_std, res_flash
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{N:8d} {K:6d} {d:4d} | {'OOM':>12s} {'--':>12s} | {'--':>8s} | {'--':>10s} {'--':>11s} | {'OOM':>7s}")
                torch.cuda.empty_cache()
            else:
                print(f"{N:8d} {K:6d} {d:4d} | ERROR: {e}")


def run_convergence_test():
    """Test multi-iteration convergence on GPU."""
    print("\n=== GPU Convergence Test ===")
    N, K, d = 8192, 8, 32
    torch.manual_seed(0)
    X = torch.randn(N, d, device="cuda")

    # Standard GMM
    gen = torch.Generator(device="cuda").manual_seed(42)
    idx = torch.randperm(N, device="cuda", generator=gen)[:K]
    mu = X[idx].clone()
    var = torch.ones(K, d, device="cuda") * X.var(dim=0, keepdim=True).mean()
    log_pi = torch.full((K,), -torch.log(torch.tensor(float(K))), device="cuda")

    # Flash GMM
    mu_f, var_f, lp_f = mu.clone(), var.clone(), log_pi.clone()

    n_iters = 20
    std_lls = []
    flash_lls = []

    for it in range(n_iters):
        # Standard
        gamma, ln = _C.standard_e_step(X, mu, var, log_pi)
        mu, var, log_pi = _C.standard_m_step(X, gamma)
        std_lls.append(ln.mean().item())

        # Flash
        mu_f, var_f, lp_f, ln_f = _C.flash_em_fused(X, mu_f, var_f, lp_f, 256, 8)
        flash_lls.append(ln_f.mean().item())

    print(f"{'Iter':>4s} {'Std LL':>12s} {'Flash LL':>12s} {'Diff':>12s}")
    for i in range(n_iters):
        diff = abs(std_lls[i] - flash_lls[i])
        if i < 5 or i == n_iters - 1:
            print(f"{i:4d} {std_lls[i]:12.4f} {flash_lls[i]:12.4f} {diff:12.2e}")

    final_diff = abs(std_lls[-1] - flash_lls[-1])
    print(f"Final LL diff: {final_diff:.2e} {'PASS' if final_diff < 1.0 else 'FAIL'}")


if __name__ == "__main__":
    warmup_gpu()
    run_gpu_benchmark()
    run_convergence_test()

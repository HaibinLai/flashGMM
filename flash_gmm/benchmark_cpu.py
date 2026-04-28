"""
CPU Multi-thread Benchmark: PyTorch GMM vs C++ Flash-GMM

Compares:
  1. PyTorch standard GMM (torch ops, controlled thread count)
  2. C++ Standard GMM (native extension)
  3. C++ Flash GMM (native extension)

Thread configs: 1, 4, 8 cores
"""
import torch
import time
import os
import sys

# Set library path for native extension
torch_lib = os.path.join(torch.__path__[0], "lib")
os.environ["LD_LIBRARY_PATH"] = torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standard_gmm import StandardGMM, generate_gmm_data

try:
    import flash_gmm_native as _C
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False
    print("[WARN] Native extension not found, skipping C++ benchmarks")


def pytorch_gmm_one_iter(X, mu, var, log_pi):
    """One EM iteration using pure PyTorch ops (same as StandardGMM but timed)."""
    N, d = X.shape
    K = mu.shape[0]

    # E-step: materialize L then normalize
    log_det = var.log().sum(dim=1)  # (K,)
    diff = X.unsqueeze(1) - mu.unsqueeze(0)  # (N, K, d)
    mahal = (diff ** 2 / var.unsqueeze(0)).sum(dim=2)  # (N, K)
    L = log_pi.unsqueeze(0) - 0.5 * (d * torch.log(torch.tensor(2 * torch.pi)) + log_det.unsqueeze(0) + mahal)
    log_sum = torch.logsumexp(L, dim=1, keepdim=True)
    gamma = (L - log_sum).exp()

    # M-step
    n_k = gamma.sum(dim=0)
    new_mu = (gamma.T @ X) / n_k.unsqueeze(1)
    diff2 = X.unsqueeze(1) - new_mu.unsqueeze(0)
    new_var = (gamma.unsqueeze(2) * diff2 ** 2).sum(dim=0) / n_k.unsqueeze(1)
    new_var = new_var.clamp(min=1e-6)
    new_log_pi = (n_k / N).log()

    return new_mu, new_var, new_log_pi, log_sum.squeeze(1)


def benchmark_fn(fn, n_warmup=3, n_repeat=10):
    """Benchmark a function, return (avg_ms, std_ms)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5
    return avg, std


def run_benchmark():
    configs = [
        # (N, K, d)
        (4096,    8,   32),
        (4096,   64,   32),
        (16384,  64,   64),
        (16384, 128,   64),
        (65536, 256,  128),
    ]

    thread_counts = [1, 4, 8]

    print(f"CPU: {os.cpu_count()} cores available")
    print(f"PyTorch: {torch.__version__}")
    print(f"Native extension: {'loaded' if HAS_NATIVE else 'NOT FOUND'}")
    print()

    for N, K, d in configs:
        X, _, _, _, _ = generate_gmm_data(N, K, d)

        # Init params (shared across all methods)
        gen = torch.Generator().manual_seed(42)
        idx = torch.randperm(N, generator=gen)[:K]
        mu = X[idx].clone()
        var = torch.ones(K, d) * X.var(dim=0, keepdim=True).mean()
        log_pi = torch.full((K,), -torch.log(torch.tensor(float(K))))

        print(f"{'='*90}")
        print(f"N={N:>6d}  K={K:>4d}  d={d:>4d}  |  NK={N*K:>10d}  Nd={N*d:>10d}  K/d={K/d:.1f}")
        print(f"{'─'*90}")
        print(f"  {'Method':<35s} {'1 core':>10s} {'4 cores':>10s} {'8 cores':>10s}")
        print(f"{'─'*90}")

        # --- PyTorch standard GMM ---
        results_pytorch = {}
        for n_threads in thread_counts:
            torch.set_num_threads(n_threads)
            avg, std = benchmark_fn(
                lambda: pytorch_gmm_one_iter(X, mu, var, log_pi))
            results_pytorch[n_threads] = avg

        s1 = results_pytorch[1]
        print(f"  {'PyTorch Standard GMM':<35s} "
              f"{results_pytorch[1]:8.2f}ms "
              f"{results_pytorch[4]:8.2f}ms ({s1/results_pytorch[4]:.1f}x) "
              f"{results_pytorch[8]:8.2f}ms ({s1/results_pytorch[8]:.1f}x)")

        # --- C++ Standard GMM ---
        if HAS_NATIVE:
            results_cpp_std = {}
            for n_threads in thread_counts:
                torch.set_num_threads(n_threads)
                def run_cpp_std():
                    g, ln = _C.standard_e_step(X, mu, var, log_pi)
                    _C.standard_m_step(X, g)
                avg, std = benchmark_fn(run_cpp_std)
                results_cpp_std[n_threads] = avg

            s1 = results_cpp_std[1]
            print(f"  {'C++ Standard GMM':<35s} "
                  f"{results_cpp_std[1]:8.2f}ms "
                  f"{results_cpp_std[4]:8.2f}ms ({s1/results_cpp_std[4]:.1f}x) "
                  f"{results_cpp_std[8]:8.2f}ms ({s1/results_cpp_std[8]:.1f}x)")

        # --- C++ Flash E-step only ---
        if HAS_NATIVE:
            results_cpp_flash_e = {}
            for n_threads in thread_counts:
                torch.set_num_threads(n_threads)
                def run_cpp_flash_e():
                    g, ln = _C.flash_e_step(X, mu, var, log_pi, 4)
                    _C.standard_m_step(X, g)
                avg, std = benchmark_fn(run_cpp_flash_e)
                results_cpp_flash_e[n_threads] = avg

            s1 = results_cpp_flash_e[1]
            print(f"  {'C++ Flash E-step + Std M-step':<35s} "
                  f"{results_cpp_flash_e[1]:8.2f}ms "
                  f"{results_cpp_flash_e[4]:8.2f}ms ({s1/results_cpp_flash_e[4]:.1f}x) "
                  f"{results_cpp_flash_e[8]:8.2f}ms ({s1/results_cpp_flash_e[8]:.1f}x)")

        # --- C++ Flash E+M Fused ---
        if HAS_NATIVE:
            results_cpp_flash_em = {}
            for n_threads in thread_counts:
                torch.set_num_threads(n_threads)
                def run_cpp_flash_em():
                    _C.flash_em_fused(X, mu, var, log_pi, 64, 4)
                avg, std = benchmark_fn(run_cpp_flash_em)
                results_cpp_flash_em[n_threads] = avg

            s1 = results_cpp_flash_em[1]
            print(f"  {'C++ Flash E+M Fused':<35s} "
                  f"{results_cpp_flash_em[1]:8.2f}ms "
                  f"{results_cpp_flash_em[4]:8.2f}ms ({s1/results_cpp_flash_em[4]:.1f}x) "
                  f"{results_cpp_flash_em[8]:8.2f}ms ({s1/results_cpp_flash_em[8]:.1f}x)")

        # --- Speedup summary ---
        if HAS_NATIVE:
            print(f"{'─'*90}")
            for n_threads in thread_counts:
                pt = results_pytorch[n_threads]
                cs = results_cpp_std[n_threads]
                cf = results_cpp_flash_em[n_threads]
                print(f"  @{n_threads} cores: PyTorch={pt:.1f}ms  C++Std={cs:.1f}ms({pt/cs:.1f}x)  "
                      f"FlashEM={cf:.1f}ms({pt/cf:.1f}x vs PT, {cs/cf:.1f}x vs C++Std)")
        print()

    # Restore default
    torch.set_num_threads(torch.get_num_threads())


if __name__ == "__main__":
    run_benchmark()

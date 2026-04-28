"""
Python wrapper for Flash-GMM native (C++/CUDA) extension.

Provides the same interface as the pure-Python FlashGMM class,
but dispatches to compiled C++/CUDA kernels.

Supports two build modes:
  1. Pre-built: `python setup.py build_ext --inplace` then import
  2. JIT: automatically compiles on first import (slower first run)
"""

import torch
import os
import time

# Try to import pre-built extension, fall back to JIT compilation
try:
    import flash_gmm_native as _C
except ImportError:
    print("[Flash-GMM] Pre-built extension not found, JIT compiling...")
    from torch.utils.cpp_extension import load

    csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
    use_cuda = torch.cuda.is_available()

    sources = [
        os.path.join(csrc_dir, "binding.cpp"),
        os.path.join(csrc_dir, "flash_gmm_cpu.cpp"),
    ]
    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = []
    define_macros = []

    if use_cuda:
        sources.append(os.path.join(csrc_dir, "flash_gmm_cuda.cu"))
        define_macros = [("WITH_CUDA", None)]
        extra_cuda_cflags = ["-O3", "--use_fast_math"]

    _C = load(
        name="flash_gmm_native",
        sources=sources,
        extra_cflags=extra_cflags + [f"-D{m[0]}" for m in define_macros],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )


class NativeStandardGMM:
    """Standard GMM using C++/CUDA kernels (materializes L and γ)."""

    def __init__(self, K: int, d: int, max_iter: int = 100, tol: float = 1e-4, seed: int = 42):
        self.K = K
        self.d = d
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.mu = None
        self.var = None
        self.log_pi = None

    def _init_params(self, X: torch.Tensor):
        N = X.shape[0]
        gen = torch.Generator(device=X.device).manual_seed(self.seed)
        idx = torch.randperm(N, generator=gen, device=X.device)[:self.K]
        self.mu = X[idx].clone()
        self.var = torch.ones(self.K, self.d, device=X.device) * X.var(dim=0, keepdim=True).mean()
        self.log_pi = torch.full((self.K,), -torch.log(torch.tensor(float(self.K))), device=X.device)

    def fit(self, X: torch.Tensor, verbose: bool = False):
        self._init_params(X)
        prev_ll = -float("inf")
        history = []

        for it in range(self.max_iter):
            t0 = time.time()

            # E-step (C++/CUDA)
            gamma, log_norm = _C.standard_e_step(X, self.mu, self.var, self.log_pi)
            avg_ll = log_norm.mean().item()

            # M-step (C++/CUDA)
            self.mu, self.var, self.log_pi = _C.standard_m_step(X, gamma)

            elapsed = time.time() - t0
            history.append({"iter": it, "log_likelihood": avg_ll, "time_ms": elapsed * 1000})

            if verbose and (it % 10 == 0 or it < 5):
                print(f"  iter {it:3d}: LL={avg_ll:.4f}  time={elapsed*1000:.1f}ms")

            if abs(avg_ll - prev_ll) < self.tol:
                if verbose:
                    print(f"  Converged at iter {it}")
                break
            prev_ll = avg_ll

        return history


class NativeFlashGMM:
    """Flash-GMM using C++/CUDA kernels (ZERO N×K materialization)."""

    def __init__(self, K: int, d: int, BN: int = 64, BK: int = 8,
                 max_iter: int = 100, tol: float = 1e-4, seed: int = 42):
        self.K = K
        self.d = d
        self.BN = BN
        self.BK = BK
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.mu = None
        self.var = None
        self.log_pi = None

    def _init_params(self, X: torch.Tensor):
        N = X.shape[0]
        gen = torch.Generator(device=X.device).manual_seed(self.seed)
        idx = torch.randperm(N, generator=gen, device=X.device)[:self.K]
        self.mu = X[idx].clone()
        self.var = torch.ones(self.K, self.d, device=X.device) * X.var(dim=0, keepdim=True).mean()
        self.log_pi = torch.full((self.K,), -torch.log(torch.tensor(float(self.K))), device=X.device)

    def fit(self, X: torch.Tensor, verbose: bool = False):
        self._init_params(X)
        prev_ll = -float("inf")
        history = []

        for it in range(self.max_iter):
            t0 = time.time()

            # Fused E+M step (C++/CUDA) — no N×K matrix ever materializes
            self.mu, self.var, self.log_pi, log_norm = _C.flash_em_fused(
                X, self.mu, self.var, self.log_pi, self.BN, self.BK)
            avg_ll = log_norm.mean().item()

            elapsed = time.time() - t0
            history.append({"iter": it, "log_likelihood": avg_ll, "time_ms": elapsed * 1000})

            if verbose and (it % 10 == 0 or it < 5):
                print(f"  iter {it:3d}: LL={avg_ll:.4f}  time={elapsed*1000:.1f}ms")

            if abs(avg_ll - prev_ll) < self.tol:
                if verbose:
                    print(f"  Converged at iter {it}")
                break
            prev_ll = avg_ll

        return history


if __name__ == "__main__":
    from standard_gmm import generate_gmm_data

    print("=== Native C++ Flash-GMM Test ===")
    N, K, d = 4096, 8, 32
    X, labels, _, _, _ = generate_gmm_data(N, K, d)

    # Test standard
    print("\n--- Standard GMM (C++) ---")
    std = NativeStandardGMM(K=K, d=d, max_iter=50, tol=1e-4)
    std_hist = std.fit(X, verbose=True)

    # Test flash
    print("\n--- Flash GMM (C++) ---")
    flash = NativeFlashGMM(K=K, d=d, BK=4, max_iter=50, tol=1e-4)
    flash_hist = flash.fit(X, verbose=True)

    # Compare
    if len(std_hist) > 0 and len(flash_hist) > 0:
        diff = abs(std_hist[-1]["log_likelihood"] - flash_hist[-1]["log_likelihood"])
        print(f"\nFinal LL diff (std vs flash): {diff:.6f}")
        print(f"Avg time std:   {sum(h['time_ms'] for h in std_hist)/len(std_hist):.1f} ms/iter")
        print(f"Avg time flash: {sum(h['time_ms'] for h in flash_hist)/len(flash_hist):.1f} ms/iter")

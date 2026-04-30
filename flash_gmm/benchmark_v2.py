"""
Comprehensive GPU benchmark: all GMM variants.

Compares:
  1. Custom CUDA Standard E+M (baseline)
  2. Custom CUDA Flash E+M (v2 kernel, register-cached X)
  3. GEMM Standard E+M (cuBLAS Tensor Cores, materializes L[N,K])
  4. GEMM Flash E+M (cuBLAS Tensor Cores, tiled, no L[N,K])
"""
import torch
import sys
import os

torch_lib = os.path.join(torch.__path__[0], "lib")
if torch_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flash_gmm_native as _C
from flash_gmm_v2 import GemmFlashGMM, GemmStandardGMM


def warmup_gpu():
    x = torch.randn(1024, 1024, device="cuda")
    for _ in range(10):
        _ = x @ x.T
    torch.cuda.synchronize()


def bench(fn, n_warmup=5, n_repeat=20):
    for _ in range(n_warmup):
        fn()
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
    return avg, result


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print()

    gemm_flash = GemmFlashGMM()
    gemm_std = GemmStandardGMM()

    configs = [
        (4096,     8,   32),
        (4096,    16,   32),
        (16384,   16,   64),
        (16384,   64,   64),
        (16384,  128,   64),
        (65536,   64,  128),
        (65536,  256,  128),
        (65536,  512,  128),
        (262144,  64,   64),
        (262144, 128,   64),
        (32768,  1024,  64),
        (65536,  1024,  64),
    ]

    hdr = (f"{'N':>8s} {'K':>6s} {'d':>4s} | "
           f"{'CUDA Std':>10s} {'CUDA Flash':>11s} {'GEMM Std':>10s} {'GEMM Flash':>11s} | "
           f"{'Best':>6s} {'vs CUDAStd':>10s} | {'Correct':>7s}")
    print(hdr)
    print("-" * len(hdr))

    for N, K, d in configs:
        torch.manual_seed(42)
        X = torch.randn(N, d, device="cuda")
        gen = torch.Generator(device="cuda").manual_seed(42)
        idx = torch.randperm(N, device="cuda", generator=gen)[:K]
        mu = X[idx].clone()
        var = torch.ones(K, d, device="cuda") * X.var(dim=0, keepdim=True).mean()
        log_pi = torch.full((K,), -torch.log(torch.tensor(float(K))), device="cuda")

        BN = min(256, N)

        try:
            # 1. CUDA Standard
            def run_cuda_std():
                g, ln = _C.standard_e_step(X, mu, var, log_pi)
                m, v, lp = _C.standard_m_step(X, g)
                return ln

            # 2. CUDA Flash
            def run_cuda_flash():
                m, v, lp, ln = _C.flash_em_fused(X, mu, var, log_pi, BN, min(8, K))
                return ln

            # 3. GEMM Standard
            def run_gemm_std():
                m, v, lp, ln = gemm_std.em_step(X, mu, var, log_pi)
                return ln

            # 4. GEMM Flash
            def run_gemm_flash():
                m, v, lp, ln = gemm_flash.em_step(X, mu, var, log_pi)
                return ln

            t1, r1 = bench(run_cuda_std)
            t2, r2 = bench(run_cuda_flash)
            t3, r3 = bench(run_gemm_std)
            t4, r4 = bench(run_gemm_flash)

            # Correctness: compare log_normalizer
            ln_diff = max(
                (r1 - r2).abs().max().item(),
                (r1 - r3).abs().max().item(),
                (r1 - r4).abs().max().item(),
            )
            correct = ln_diff < 0.5

            results = {"CUDA_S": t1, "CUDA_F": t2, "GEMM_S": t3, "GEMM_F": t4}
            best_name = min(results, key=results.get)
            best_time = results[best_name]
            speedup = t1 / best_time

            print(f"{N:8d} {K:6d} {d:4d} | "
                  f"{t1:8.2f}ms {t2:9.2f}ms {t3:8.2f}ms {t4:9.2f}ms | "
                  f"{best_name:>6s} {speedup:8.2f}x | "
                  f"{'PASS' if correct else 'FAIL':>7s}")

            del X, mu, var, log_pi
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{N:8d} {K:6d} {d:4d} | OOM")
                torch.cuda.empty_cache()
            else:
                print(f"{N:8d} {K:6d} {d:4d} | ERROR: {e}")

    print()
    print("Legend: CUDA_S=Custom CUDA Standard, CUDA_F=Custom CUDA Flash,")
    print("        GEMM_S=cuBLAS GEMM Standard, GEMM_F=cuBLAS GEMM Flash (tiled)")


if __name__ == "__main__":
    warmup_gpu()
    main()

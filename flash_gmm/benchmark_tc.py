"""
Benchmark: CUDA Tensor Core (WMMA) vs Triton vs cuBLAS GEMM
JIT compiles the WMMA kernel and compares.
"""
import torch
import torch.utils.cpp_extension
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup
torch_lib = os.path.join(torch.__path__[0], "lib")
os.environ["LD_LIBRARY_PATH"] = torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

print("Compiling WMMA Tensor Core kernel (JIT)...")
try:
    torch.utils.cpp_extension._check_cuda_version = lambda *a, **kw: None
    tc_module = torch.utils.cpp_extension.load(
        name="flash_gmm_tc",
        sources=[os.path.join(csrc_dir, "flash_gmm_tc_v2.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17",
                           "-gencode=arch=compute_80,code=sm_80"],
        verbose=False,
    )
    print("  Compiled OK ✓")
except Exception as e:
    print(f"  Compilation FAILED: {e}")
    tc_module = None

from flash_gmm_v2 import GemmStandardGMM
from flash_gmm_v3 import TritonFlashBF16GMM
from flash_gmm_v5 import UltraGMM


def bench(fn, nw=5, nr=20):
    for _ in range(nw):
        fn()
        torch.cuda.synchronize()
    ts = []
    for _ in range(nr):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        r = fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sum(ts) / len(ts), r


def main():
    x = torch.randn(1024, 1024, device="cuda")
    for _ in range(10):
        _ = x @ x.T
    torch.cuda.synchronize()
    del x

    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print()

    gemm_s = GemmStandardGMM()
    ultra = UltraGMM()
    tri_bf = TritonFlashBF16GMM()

    configs = [
        (16384, 64, 64),
        (16384, 128, 64),
        (32768, 1024, 64),
        (65536, 256, 128),
        (262144, 128, 64),
    ]

    names = ["GEMM Std", "Ultra", "Tri BF16", "CUDA TC"]
    hdr = f"{'N':>7} {'K':>5} {'d':>3} | " + "  ".join(f"{n:>9}" for n in names) + " | Best"
    print(hdr)
    print("-" * len(hdr))

    for N, K, d in configs:
        torch.manual_seed(42)
        X = torch.randn(N, d, device="cuda")
        mu = torch.randn(K, d, device="cuda")
        var = torch.ones(K, d, device="cuda")
        lp = torch.full((K,), -math.log(K), device="cuda")

        R = {}

        try:
            t, _ = bench(lambda: gemm_s.em_step(X, mu, var, lp))
            R["GEMM Std"] = t
        except Exception as e:
            R["GEMM Std"] = float("inf")

        try:
            t, _ = bench(lambda: ultra.em_step(X, mu, var, lp))
            R["Ultra"] = t
        except:
            R["Ultra"] = float("inf")

        try:
            t, _ = bench(lambda: tri_bf.em_step(X, mu, var, lp))
            R["Tri BF16"] = t
        except:
            R["Tri BF16"] = float("inf")

        if tc_module and d in (64, 128):
            try:
                # Correctness check first
                ref_mu, ref_v, ref_lp, ref_ln = gemm_s.em_step(X, mu, var, lp)
                tc_mu, tc_v, tc_lp, tc_ln = tc_module.wmma_flash_em(X, mu, var, lp)
                ln_diff = (ref_ln - tc_ln).abs().max().item()
                correct = ln_diff < 1.0

                t, _ = bench(lambda: tc_module.wmma_flash_em(X, mu, var, lp))
                R["CUDA TC"] = t

                if not correct:
                    R["CUDA TC"] = float("inf")  # mark as failed
            except Exception as e:
                print(f"  CUDA TC error: {e}")
                R["CUDA TC"] = float("inf")
        else:
            R["CUDA TC"] = float("inf")

        bn = min(R, key=R.get)
        vals = "  ".join(
            f"{R[n]:7.2f}ms" if R[n] < float("inf") else f"{'ERR':>9}"
            for n in names
        )
        print(f"{N:7d} {K:5d} {d:3d} | {vals} | {bn}")

        del X, mu, var, lp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

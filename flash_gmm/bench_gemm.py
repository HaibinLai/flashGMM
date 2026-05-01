"""
Benchmark: autotuned Triton GEMM vs cuBLAS on GMM-typical shapes.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _tuned_matmul(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


def triton_mm(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _tuned_matmul[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
    )
    return C


if __name__ == "__main__":
    def bench(fn, nw=5, nr=20):
        for _ in range(nw): fn(); torch.cuda.synchronize()
        ts = []
        for _ in range(nr):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return sum(ts) / len(ts)

    w = torch.randn(1024, 1024, device='cuda')
    for _ in range(10): _ = w @ w.T
    torch.cuda.synchronize(); del w

    print(f'GPU: {torch.cuda.get_device_name()}')
    print()
    print(f"{'Shape':>30} | {'cuBLAS':>8} {'Triton':>8} {'Tri/cuB':>7} | {'TFLOPS(cuB)':>11} {'TFLOPS(Tri)':>11}")
    print('─' * 90)

    shapes = [
        (65536, 256, 128, "dist GEMM d=128 K=128"),
        (65536, 256, 256, "dist GEMM d=128 K=256"),
        (65536, 128, 256, "Mstep X.T@γ d=128 K=256"),
        (1000000, 128, 64, "N=1M dist d=64 K=64"),
        (1000000, 128, 128, "N=1M dist d=64 K=128"),
        (32768, 128, 1024, "N=32K K=1024 d=64"),
        (10000000, 128, 64, "N=10M d=64 K=64"),
        (4096, 4096, 4096, "square 4K (cuBLAS best)"),
    ]

    for M, K, N, desc in shapes:
        A = torch.randn(M, K, device='cuda')
        B = torch.randn(K, N, device='cuda')

        t_c = bench(lambda: torch.mm(A, B))
        triton_mm(A, B)  # autotune warmup
        t_t = bench(lambda: triton_mm(A, B))

        flops = 2 * M * N * K
        tf_c = flops / t_c / 1e9
        tf_t = flops / t_t / 1e9
        ratio = t_t / t_c

        label = f'[{M},{K}]@[{K},{N}]'
        print(f'{desc:>30} | {t_c:6.2f}ms {t_t:6.2f}ms {ratio:5.2f}x | {tf_c:9.1f}TF {tf_t:9.1f}TF')

        del A, B; torch.cuda.empty_cache()

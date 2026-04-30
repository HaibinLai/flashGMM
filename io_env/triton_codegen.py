#!/usr/bin/env python3
"""
Triton Kernel Codegen + Compile + Test for the ReAct Environment.

Provides tools for the agent to:
1. generate_kernel: Generate a Triton kernel from the optimized computation graph
2. compile_and_test: Compile the kernel, run correctness tests against PyTorch reference
3. benchmark_kernel: Time the generated kernel vs baseline

The agent can also write raw Triton code via write_kernel for full control.
"""

from __future__ import annotations
import os, sys, tempfile, importlib, traceback, time
from pathlib import Path

# ============================================================================
# Triton kernel templates for common Flash patterns
# ============================================================================

TRITON_TEMPLATES = {}

TRITON_TEMPLATES["cross_entropy"] = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _flash_cross_entropy_kernel(
    LOGITS,      # (N, V)
    LABELS,      # (N,)
    LOSS,        # (N,)
    N: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,  # vocab tile size
):
    """
    Flash Cross-Entropy: single-pass online logsumexp + gather.
    Zero intermediate materialization (no exp_logits, no log_sum_exp in HBM).
    """
    row = tl.program_id(0)
    if row >= N:
        return

    label = tl.load(LABELS + row)

    # Online logsumexp state (in registers)
    m = float("-inf")   # running max
    s = 0.0             # running sum of exp(x - m)
    target_logit = 0.0  # logit at the label position

    # Stream vocab tiles
    for v_start in range(0, V, BV):
        v_offsets = v_start + tl.arange(0, BV)
        mask = v_offsets < V

        # Load tile from HBM (only read, no write!)
        x = tl.load(LOGITS + row * V + v_offsets, mask=mask, other=float("-inf"))

        # Gather target logit
        is_target = (v_offsets == label)
        target_logit += tl.sum(tl.where(is_target, x, 0.0))

        # Online logsumexp update
        tile_max = tl.max(x)
        new_m = tl.maximum(m, tile_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m))
        m = new_m

    # loss = -(target_logit - log_sum_exp)
    log_sum_exp = m + tl.log(s)
    loss = -(target_logit - log_sum_exp)
    tl.store(LOSS + row, loss)


def flash_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Launch the Flash Cross-Entropy Triton kernel."""
    N, V = logits.shape
    loss = torch.empty(N, device=logits.device, dtype=logits.dtype)
    BV = min(1024, triton.next_power_of_2(V))
    grid = (N,)
    _flash_cross_entropy_kernel[grid](logits, labels, loss, N, V, BV)
    return loss


def reference_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """PyTorch reference (materialized baseline)."""
    import torch.nn.functional as F
    return F.cross_entropy(logits, labels, reduction="none")
'''

TRITON_TEMPLATES["kmeans"] = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _flash_kmeans_assign_kernel(
    X,           # (N, d)
    C,           # (K, d)
    ASSIGN,      # (N,)  output: cluster assignments
    N: tl.constexpr,
    K: tl.constexpr,
    d: tl.constexpr,
    BK: tl.constexpr,   # centroid tile size
    BD: tl.constexpr,   # dimension block size
):
    """
    Flash-KMeans Assignment: online argmin over centroid tiles.
    Zero distance matrix materialization.
    """
    row = tl.program_id(0)
    if row >= N:
        return

    # Load point x into registers (stays for all centroid tiles)
    d_offsets = tl.arange(0, BD)
    d_mask = d_offsets < d
    x = tl.load(X + row * d + d_offsets, mask=d_mask, other=0.0)

    # Online argmin state
    best_dist = float("inf")
    best_idx = 0

    # Stream centroid tiles — one centroid at a time
    for k in range(K):
        # Load centroid
        c = tl.load(C + k * d + d_offsets, mask=d_mask, other=0.0)
        diff = x - c
        dist = tl.sum(diff * diff)

        # Online argmin update
        is_better = dist < best_dist
        best_dist = tl.where(is_better, dist, best_dist)
        best_idx = tl.where(is_better, k, best_idx)

    tl.store(ASSIGN + row, best_idx)


def flash_kmeans_assign(X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    N, d = X.shape
    K = C.shape[0]
    assignments = torch.empty(N, device=X.device, dtype=torch.int32)
    BD = triton.next_power_of_2(d)
    BK = min(32, K)
    grid = (N,)
    _flash_kmeans_assign_kernel[grid](X, C, assignments, N, K, d, BK, BD)
    return assignments


def reference_kmeans_assign(X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    D = torch.cdist(X, C, p=2.0)
    return D.argmin(dim=1).to(torch.int32)
'''

TRITON_TEMPLATES["softmax"] = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _flash_softmax_kernel(
    X,           # (N, M) input
    OUT,         # (N, M) output
    M: tl.constexpr,
    BM: tl.constexpr,
):
    """
    Flash row-wise softmax: 2-pass online softmax, no intermediate materialization.
    Pass 1: streaming online max + sum_exp (reads X once)
    Pass 2: normalize and write output (reads X again, writes OUT)
    Total: 2 reads of X + 1 write of OUT. No intermediate in HBM.
    """
    row = tl.program_id(0)

    # Pass 1: online max + sum_exp (single streaming pass)
    m = float("-inf")
    s = 0.0
    for col_start in range(0, M, BM):
        cols = col_start + tl.arange(0, BM)
        mask = cols < M
        x = tl.load(X + row * M + cols, mask=mask, other=float("-inf"))
        tile_max = tl.max(x)
        new_m = tl.maximum(m, tile_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m))
        m = new_m

    # Pass 2: normalize (recompute exp, write output)
    inv_s = 1.0 / s
    for col_start in range(0, M, BM):
        cols = col_start + tl.arange(0, BM)
        mask = cols < M
        x = tl.load(X + row * M + cols, mask=mask, other=float("-inf"))
        out = tl.exp(x - m) * inv_s
        tl.store(OUT + row * M + cols, out, mask=mask)


def flash_softmax(X: torch.Tensor) -> torch.Tensor:
    N, M = X.shape
    OUT = torch.empty_like(X)
    # Use large block for better memory throughput
    BM = min(1024, triton.next_power_of_2(M))
    grid = (N,)
    _flash_softmax_kernel[grid](X, OUT, M, BM, num_warps=8)
    return OUT


def reference_softmax(X: torch.Tensor) -> torch.Tensor:
    return torch.softmax(X, dim=1)
'''


# ============================================================================
# Codegen + Compile + Test API
# ============================================================================

_KERNEL_DIR = os.path.join(tempfile.gettempdir(), "io_env_kernels")
os.makedirs(_KERNEL_DIR, exist_ok=True)

# ============================================================================
# CUDA kernel templates
# ============================================================================

CUDA_TEMPLATES = {}

CUDA_TEMPLATES["gmm_estep"] = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

static constexpr float LOG_2PI = 1.8378770664093453f;

// Flash Log-Normalizer: online log-sum-exp, zero N×K materialization
template <int BK>
__global__ void flash_log_normalizer_kernel(
    const float* __restrict__ X,          // (N, d)
    const float* __restrict__ mu,         // (K, d)
    const float* __restrict__ inv_var,    // (K, d) = 1/var
    const float* __restrict__ half_const, // (K,) = log_pi - 0.5*(d*LOG_2PI + sum_log_var)
    float* __restrict__ log_normalizer,   // (N,)
    int N, int K, int d, int BN
) {
    extern __shared__ float smem[];
    float* mu_tile = smem;
    float* iv_tile = mu_tile + BK * d;
    float* hc_tile = iv_tile + BK * d;

    int block_start = blockIdx.x * BN;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int valid_n = min(BN, N - block_start);

    // Per-point running state
    constexpr int MAX_PTS = 4;
    float running_max[MAX_PTS], running_sum_exp[MAX_PTS];
    for (int i = 0; i < MAX_PTS; i++) {
        running_max[i] = -FLT_MAX;
        running_sum_exp[i] = 0.0f;
    }

    for (int t = 0; t < K; t += BK) {
        int tile_k = min(BK, K - t);
        __syncthreads();
        for (int idx = tid; idx < tile_k * d; idx += nthreads) {
            mu_tile[idx] = mu[(t + idx / d) * d + idx % d];
            iv_tile[idx] = inv_var[(t + idx / d) * d + idx % d];
        }
        if (tid < tile_k) hc_tile[tid] = half_const[t + tid];
        __syncthreads();

        int pt = 0;
        for (int ln = tid; ln < valid_n; ln += nthreads) {
            int n = block_start + ln;
            for (int kk = 0; kk < tile_k; kk++) {
                float mahal = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = X[n * d + j] - mu_tile[kk * d + j];
                    mahal += diff * diff * iv_tile[kk * d + j];
                }
                float ll = hc_tile[kk] - 0.5f * mahal;
                if (ll > running_max[pt]) {
                    running_sum_exp[pt] = running_sum_exp[pt] * expf(running_max[pt] - ll) + 1.0f;
                    running_max[pt] = ll;
                } else {
                    running_sum_exp[pt] += expf(ll - running_max[pt]);
                }
            }
            pt++;
        }
    }

    int pt = 0;
    for (int ln = tid; ln < valid_n; ln += nthreads) {
        log_normalizer[block_start + ln] = running_max[pt] + logf(running_sum_exp[pt]);
        pt++;
    }
}

// Python-callable wrapper
torch::Tensor flash_gmm_log_normalizer(
    torch::Tensor X, torch::Tensor mu, torch::Tensor var, torch::Tensor log_pi
) {
    int N = X.size(0), d = X.size(1), K = mu.size(0);
    auto inv_var = var.reciprocal();
    auto log_var_sum = var.log().sum(1);
    auto half_const = log_pi - 0.5f * (d * LOG_2PI + log_var_sum);
    auto log_norm = torch::empty({N}, X.options());

    int BN = 128, BK = 32, threads = 256;
    int grid = (N + BN - 1) / BN;
    size_t smem = (2 * BK * d + BK) * sizeof(float);
    flash_log_normalizer_kernel<32><<<grid, threads, smem>>>(
        X.data_ptr<float>(), mu.data_ptr<float>(), inv_var.data_ptr<float>(),
        half_const.data_ptr<float>(), log_norm.data_ptr<float>(), N, K, d, BN);
    return log_norm;
}

// Standard baseline: materialize L
torch::Tensor standard_gmm_log_normalizer(
    torch::Tensor X, torch::Tensor mu, torch::Tensor var, torch::Tensor log_pi
) {
    int N = X.size(0), d = X.size(1), K = mu.size(0);
    auto log_det = var.log().sum(1);
    auto diff = X.unsqueeze(1) - mu.unsqueeze(0);
    auto mahal = (diff * diff / var.unsqueeze(0)).sum(2);
    auto L = log_pi.unsqueeze(0) - 0.5f * (d * LOG_2PI + log_det.unsqueeze(0) + mahal);
    return torch::logsumexp(L, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_gmm_estep", &flash_gmm_log_normalizer, "Flash GMM log-normalizer");
    m.def("reference_gmm_estep", &standard_gmm_log_normalizer, "Standard GMM log-normalizer");
}
'''

TRITON_TEMPLATES["layernorm"] = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _flash_layernorm_kernel(
    X, Y, GAMMA, BETA,
    N: tl.constexpr, d: tl.constexpr, BD: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Flash LayerNorm: 2-pass Welford mean+var, fused normalize.
    Pass 1: online Welford for mean and variance (1 read of X)
    Pass 2: normalize with affine transform (1 read of X, 1 write of Y)
    No mean/var materialized to HBM.
    """
    row = tl.program_id(0)

    # Pass 1: Welford online mean + variance
    mean = 0.0
    M2 = 0.0
    count = 0.0
    for start in range(0, d, BD):
        cols = start + tl.arange(0, BD)
        mask = cols < d
        x = tl.load(X + row * d + cols, mask=mask, other=0.0)
        # Vectorized Welford update
        count += tl.sum(mask.to(tl.float32))
        delta = x - mean
        mean += tl.sum(tl.where(mask, delta / count, 0.0))
        delta2 = x - mean
        M2 += tl.sum(tl.where(mask, delta * delta2, 0.0))

    var = M2 / count
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize + affine
    for start in range(0, d, BD):
        cols = start + tl.arange(0, BD)
        mask = cols < d
        x = tl.load(X + row * d + cols, mask=mask, other=0.0)
        g = tl.load(GAMMA + cols, mask=mask, other=1.0)
        b = tl.load(BETA + cols, mask=mask, other=0.0)
        y = (x - mean) * rstd * g + b
        tl.store(Y + row * d + cols, y, mask=mask)


def flash_layernorm(X, gamma, beta, eps=1e-5):
    N, d = X.shape
    Y = torch.empty_like(X)
    BD = min(1024, triton.next_power_of_2(d))
    _flash_layernorm_kernel[(N,)](X, Y, gamma, beta, N, d, BD, eps, num_warps=8)
    return Y


def reference_layernorm(X, gamma, beta, eps=1e-5):
    mean = X.mean(dim=1, keepdim=True)
    var = X.var(dim=1, keepdim=True, unbiased=False)
    return (X - mean) / torch.sqrt(var + eps) * gamma + beta
'''


def generate_kernel(task: str, custom_code: str | None = None, lang: str = "triton") -> tuple[str, str]:
    """
    Generate a Triton or CUDA kernel for the given task.

    Args:
        task: operator name
        custom_code: custom kernel code (Triton Python or CUDA C++)
        lang: "triton" or "cuda"

    Returns (code, filepath)
    """
    if custom_code:
        code = custom_code
    elif task in TRITON_TEMPLATES:
        code = TRITON_TEMPLATES[task]
    elif task in CUDA_TEMPLATES:
        code = CUDA_TEMPLATES[task]
        lang = "cuda"
    else:
        return "", f"No template for '{task}'. Available Triton: {list(TRITON_TEMPLATES.keys())}, CUDA: {list(CUDA_TEMPLATES.keys())}"

    if lang == "cuda":
        # CUDA: save .cu + binding .cpp, compile with torch cpp_extension
        cu_path = os.path.join(_KERNEL_DIR, f"flash_{task}.cu")
        with open(cu_path, "w") as f:
            f.write(code)

        # Auto-generate a minimal Python wrapper that loads the compiled module
        wrapper_code = _make_cuda_wrapper(task, cu_path)
        py_path = os.path.join(_KERNEL_DIR, f"flash_{task}.py")
        with open(py_path, "w") as f:
            f.write(wrapper_code)
        return code, py_path
    else:
        filepath = os.path.join(_KERNEL_DIR, f"flash_{task}.py")
        with open(filepath, "w") as f:
            f.write(code)
        return code, filepath


def _make_cuda_wrapper(task: str, cu_path: str) -> str:
    """Generate a Python wrapper that JIT-compiles a CUDA kernel."""
    return f'''
import torch
from torch.utils.cpp_extension import load
import os

# JIT compile the CUDA kernel
_module = load(
    name="flash_{task}_cuda",
    sources=["{cu_path}"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

# Re-export functions - the CUDA module must define flash_{task} and reference_{task}
def flash_{task}(*args, **kwargs):
    return _module.flash_{task}(*args, **kwargs)

def reference_{task}(*args, **kwargs):
    """PyTorch reference."""
    # Fallback: import from the module if available, else use PyTorch
    if hasattr(_module, "reference_{task}"):
        return _module.reference_{task}(*args, **kwargs)
    raise NotImplementedError("No reference implementation in CUDA module")
'''


def compile_and_test(task: str, params: dict, filepath: str | None = None,
                     atol: float = 1e-4, rtol: float = 1e-3) -> str:
    """
    Compile the Triton kernel and run correctness tests.

    Returns observation string.
    """
    import torch

    if filepath is None:
        filepath = os.path.join(_KERNEL_DIR, f"flash_{task}.py")

    if not os.path.exists(filepath):
        return f"ERROR: Kernel file not found: {filepath}. Run generate_kernel first."

    # Load module
    spec = importlib.util.spec_from_file_location(f"flash_{task}", filepath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return f"COMPILE ERROR: {e}\n{traceback.format_exc()}"

    # Get flash and reference functions
    flash_fn = None
    ref_fn = None
    for name in dir(mod):
        if name.startswith("flash_"):
            flash_fn = getattr(mod, name)
        if name.startswith("reference_"):
            ref_fn = getattr(mod, name)

    if not flash_fn or not ref_fn:
        return f"ERROR: Module must define flash_* and reference_* functions."

    # Generate test data
    results = []
    try:
        torch.manual_seed(42)

        if task == "cross_entropy":
            N, V = params.get("N", 4096), params.get("V", 32000)
            logits = torch.randn(N, V, device="cuda")
            labels = torch.randint(0, V, (N,), device="cuda")
            ref_out = ref_fn(logits, labels)
            flash_out = flash_fn(logits, labels)
            max_err = (ref_out - flash_out).abs().max().item()
            mean_err = (ref_out - flash_out).abs().mean().item()

        elif task == "kmeans":
            N, K, d = params.get("N", 65536), params.get("K", 1024), params.get("d", 128)
            X = torch.randn(N, d, device="cuda")
            C = torch.randn(K, d, device="cuda")
            ref_out = ref_fn(X, C)
            flash_out = flash_fn(X, C)
            # For argmin, check agreement rate
            agree = (ref_out == flash_out).float().mean().item()
            max_err = 1.0 - agree
            mean_err = max_err

        elif task == "softmax":
            N = params.get("N", 4096)
            M = params.get("V", params.get("d", 128)) if "V" in params else 4096
            X = torch.randn(N, M, device="cuda")
            ref_out = ref_fn(X)
            flash_out = flash_fn(X)
            max_err = (ref_out - flash_out).abs().max().item()
            mean_err = (ref_out - flash_out).abs().mean().item()

        elif task == "layernorm":
            N, d = params.get("N", 16384), params.get("d", 4096)
            X = torch.randn(N, d, device="cuda")
            gamma = torch.ones(d, device="cuda")
            beta = torch.zeros(d, device="cuda")
            ref_out = ref_fn(X, gamma, beta)
            flash_out = flash_fn(X, gamma, beta)
            max_err = (ref_out - flash_out).abs().max().item()
            mean_err = (ref_out - flash_out).abs().mean().item()

        elif task in ("cosine_similarity", "contrastive_loss"):
            N, d = params.get("N", 4096), params.get("d", 256)
            X = torch.randn(N, d, device="cuda")
            # Try calling with X only; fall back to (X, labels) for contrastive
            try:
                ref_out = ref_fn(X)
                flash_out = flash_fn(X)
            except TypeError:
                labels = torch.randint(0, N, (N,), device="cuda")
                ref_out = ref_fn(X, labels)
                flash_out = flash_fn(X, labels)
            max_err = (ref_out - flash_out).abs().max().item()
            mean_err = (ref_out - flash_out).abs().mean().item()

        elif task in ("gmm_estep", "gmm_em_fused"):
            N, K, d = params.get("N", 16384), params.get("K", 64), params.get("d", 64)
            X = torch.randn(N, d, device="cuda")
            mu = torch.randn(K, d, device="cuda")
            var = torch.ones(K, d, device="cuda")
            log_pi = torch.full((K,), -torch.log(torch.tensor(float(K))), device="cuda")
            ref_out = ref_fn(X, mu, var, log_pi)
            flash_out = flash_fn(X, mu, var, log_pi)
            max_err = (ref_out - flash_out).abs().max().item()
            mean_err = (ref_out - flash_out).abs().mean().item()

        else:
            return f"No test harness for '{task}'"

        # Report
        passed = max_err < atol or (task == "kmeans" and max_err < 0.01)
        results.append(f"Correctness Test: {'PASS' if passed else 'FAIL'}")
        if task == "kmeans":
            results.append(f"  Agreement rate: {(1-max_err)*100:.2f}%")
        else:
            results.append(f"  Max error:  {max_err:.2e}")
            results.append(f"  Mean error: {mean_err:.2e}")
        results.append(f"  Tolerance:  atol={atol}, rtol={rtol}")

        torch.cuda.empty_cache()
        return "\n".join(results)

    except Exception as e:
        return f"TEST ERROR: {e}\n{traceback.format_exc()}"


def benchmark_kernel(task: str, params: dict, filepath: str | None = None,
                     n_warmup: int = 10, n_iter: int = 50) -> str:
    """
    Benchmark the generated Triton kernel vs baseline.
    """
    import torch

    if filepath is None:
        filepath = os.path.join(_KERNEL_DIR, f"flash_{task}.py")

    spec = importlib.util.spec_from_file_location(f"flash_{task}", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    flash_fn = None
    ref_fn = None
    for name in dir(mod):
        if name.startswith("flash_"):
            flash_fn = getattr(mod, name)
        if name.startswith("reference_"):
            ref_fn = getattr(mod, name)

    torch.manual_seed(42)

    # Setup inputs
    if task == "cross_entropy":
        N, V = params.get("N", 4096), params.get("V", 32000)
        inputs_flash = (torch.randn(N, V, device="cuda"), torch.randint(0, V, (N,), device="cuda"))
        inputs_ref = inputs_flash

        # Also benchmark the naive materialized baseline
        def naive_baseline(logits, labels):
            max_vals = logits.max(dim=1, keepdim=True).values
            exp_logits = (logits - max_vals).exp()  # MATERIALIZED!
            log_sum_exp = exp_logits.sum(dim=1).log()
            gathered = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            return -(gathered - max_vals.squeeze(1) - log_sum_exp)

        ref_fn_actual = naive_baseline  # Use the naive version to show IO savings
    elif task == "kmeans":
        N, K, d = params.get("N", 65536), params.get("K", 1024), params.get("d", 128)
        X = torch.randn(N, d, device="cuda")
        C = torch.randn(K, d, device="cuda")
        inputs_flash = (X, C)
        inputs_ref = (X, C)
        ref_fn_actual = ref_fn
    elif task == "softmax":
        N = params.get("N", 4096)
        M = params.get("V", params.get("d", 128)) if "V" in params else 4096
        X = torch.randn(N, M, device="cuda")
        inputs_flash = (X,)
        inputs_ref = (X,)
        ref_fn_actual = ref_fn
    elif task == "layernorm":
        N, d = params.get("N", 16384), params.get("d", 4096)
        X = torch.randn(N, d, device="cuda")
        gamma = torch.ones(d, device="cuda")
        beta = torch.zeros(d, device="cuda")
        inputs_flash = (X, gamma, beta)
        inputs_ref = (X, gamma, beta)

        def naive_layernorm(X, gamma, beta):
            mean = X.mean(dim=1, keepdim=True)
            var = X.var(dim=1, keepdim=True, unbiased=False)
            return (X - mean) / torch.sqrt(var + 1e-5) * gamma + beta

        ref_fn_actual = naive_layernorm
    elif task in ("cosine_similarity", "contrastive_loss"):
        N, d = params.get("N", 4096), params.get("d", 256)
        X = torch.randn(N, d, device="cuda")
        try:
            # Test which signature works
            _ = ref_fn(X)
            inputs_flash = (X,)
            inputs_ref = (X,)

            def naive_cosine(X):
                Xn = X / X.norm(dim=1, keepdim=True)
                return Xn @ Xn.T

            ref_fn_actual = naive_cosine
        except TypeError:
            labels = torch.randint(0, N, (N,), device="cuda")
            inputs_flash = (X, labels)
            inputs_ref = (X, labels)
            ref_fn_actual = ref_fn
    elif task in ("gmm_estep", "gmm_em_fused"):
        N, K, d = params.get("N", 16384), params.get("K", 64), params.get("d", 64)
        X = torch.randn(N, d, device="cuda")
        mu = torch.randn(K, d, device="cuda")
        var = torch.ones(K, d, device="cuda")
        log_pi = torch.full((K,), -torch.log(torch.tensor(float(K))), device="cuda")
        inputs_flash = (X, mu, var, log_pi)
        inputs_ref = (X, mu, var, log_pi)

        def naive_gmm_logsumexp(X, mu, var, log_pi):
            d = X.shape[1]
            log_det = var.log().sum(1)
            diff = X.unsqueeze(1) - mu.unsqueeze(0)
            mahal = (diff ** 2 / var.unsqueeze(0)).sum(2)
            L = log_pi.unsqueeze(0) - 0.5 * (d * 1.8379 + log_det.unsqueeze(0) + mahal)
            return torch.logsumexp(L, 1)

        ref_fn_actual = naive_gmm_logsumexp
    else:
        return f"No benchmark for '{task}'"

    # Warmup
    for _ in range(n_warmup):
        ref_fn_actual(*inputs_ref)
        flash_fn(*inputs_flash)
    torch.cuda.synchronize()

    # Benchmark baseline (naive materialized)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        ref_fn_actual(*inputs_ref)
    e.record()
    torch.cuda.synchronize()
    t_baseline = s.elapsed_time(e) / n_iter

    # Benchmark flash (Triton kernel)
    s.record()
    for _ in range(n_iter):
        flash_fn(*inputs_flash)
    e.record()
    torch.cuda.synchronize()
    t_flash = s.elapsed_time(e) / n_iter

    speedup = t_baseline / t_flash if t_flash > 0 else 0

    lines = [
        f"Kernel Benchmark: {task} ({torch.cuda.get_device_name()})",
        f"  Params: { {k:v for k,v in params.items() if isinstance(v, int)} }",
        f"  Baseline (materialized PyTorch):  {t_baseline:.3f} ms",
        f"  Flash (Triton kernel):            {t_flash:.3f} ms",
        f"  Speedup:                          {speedup:.2f}×",
    ]

    if speedup > 1.05:
        lines.append(f"  ✓ Triton kernel is {speedup:.2f}× faster!")
    elif speedup > 0.95:
        lines.append(f"  ~ Performance similar (within 5%)")
    else:
        lines.append(f"  ⚠ Triton kernel is slower ({speedup:.2f}×)")

    torch.cuda.empty_cache()
    return "\n".join(lines)


# ============================================================================
# CLI: standalone test
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("task", default="cross_entropy", nargs="?")
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--V", type=int, default=32000)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--d", type=int, default=128)
    args = parser.parse_args()

    params = {"N": args.N, "V": args.V, "K": args.K, "d": args.d}

    print(f"=== {args.task} ===")
    code, filepath = generate_kernel(args.task)
    print(f"Generated: {filepath} ({len(code)} chars)")
    print()
    print(compile_and_test(args.task, params, filepath))
    print()
    print(benchmark_kernel(args.task, params, filepath))

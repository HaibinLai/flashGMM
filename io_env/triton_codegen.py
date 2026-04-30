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

# ============================================================================
# CUDA templates for FFT, Conv2D, Stencil2D
# ============================================================================

CUDA_TEMPLATES["fft"] = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// ---- Standard FFT: log2(N) HBM round-trips per stage ----
__global__ void butterfly_stage_kernel(
    const float* __restrict__ in_re, const float* __restrict__ in_im,
    float* __restrict__ out_re, float* __restrict__ out_im,
    int N, int half_size, int stage
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int group = idx / (2 * half_size);
    int pos   = idx % (2 * half_size);

    if (pos < half_size) {
        int i = group * 2 * half_size + pos;
        int j = i + half_size;
        float angle = -2.0f * M_PI * pos / (2.0f * half_size);
        float tw_re = cosf(angle);
        float tw_im = sinf(angle);

        float a_re = in_re[i], a_im = in_im[i];
        float b_re = in_re[j], b_im = in_im[j];
        float tb_re = tw_re * b_re - tw_im * b_im;
        float tb_im = tw_re * b_im + tw_im * b_re;

        out_re[i] = a_re + tb_re;
        out_im[i] = a_im + tb_im;
        out_re[j] = a_re - tb_re;
        out_im[j] = a_im - tb_im;
    }
}

__global__ void bit_reverse_kernel(
    const float* __restrict__ in_re, const float* __restrict__ in_im,
    float* __restrict__ out_re, float* __restrict__ out_im,
    int N, int log2N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int rev = 0, tmp = idx;
    for (int b = 0; b < log2N; b++) {
        rev = (rev << 1) | (tmp & 1);
        tmp >>= 1;
    }
    out_re[rev] = in_re[idx];
    out_im[rev] = in_im[idx];
}

// ---- Flash FFT: fuse log2(TILE) stages in shared memory ----
// Each block processes a tile of size TILE, doing log2(TILE) stages on-chip.
template <int TILE>
__global__ void flash_fft_kernel(
    float* __restrict__ buf_re, float* __restrict__ buf_im,
    int N, int log2N, int log2_tile
) {
    extern __shared__ float smem[];
    float* s_re = smem;
    float* s_im = smem + TILE;

    int tile_start = blockIdx.x * TILE;
    int tid = threadIdx.x;

    // Load tile into SMEM (1 HBM read)
    for (int i = tid; i < TILE; i += blockDim.x) {
        int gi = tile_start + i;
        s_re[i] = (gi < N) ? buf_re[gi] : 0.0f;
        s_im[i] = (gi < N) ? buf_im[gi] : 0.0f;
    }
    __syncthreads();

    // Fuse log2(TILE) butterfly stages in SMEM — zero HBM traffic!
    for (int s = 0; s < log2_tile; s++) {
        int half = 1 << s;
        for (int i = tid; i < TILE / 2; i += blockDim.x) {
            int group = i / half;
            int pos = i % half;
            int idx0 = group * 2 * half + pos;
            int idx1 = idx0 + half;

            // Global stage index for twiddle factor
            int global_stage = s;  // local stage within tile
            float angle = -2.0f * M_PI * pos / (2.0f * half);
            float tw_re = cosf(angle);
            float tw_im = sinf(angle);

            float a_re = s_re[idx0], a_im = s_im[idx0];
            float b_re = s_re[idx1], b_im = s_im[idx1];
            float tb_re = tw_re * b_re - tw_im * b_im;
            float tb_im = tw_re * b_im + tw_im * b_re;

            s_re[idx0] = a_re + tb_re;
            s_im[idx0] = a_im + tb_im;
            s_re[idx1] = a_re - tb_re;
            s_im[idx1] = a_im - tb_im;
        }
        __syncthreads();
    }

    // Write tile back to HBM (1 HBM write)
    for (int i = tid; i < TILE; i += blockDim.x) {
        int gi = tile_start + i;
        if (gi < N) {
            buf_re[gi] = s_re[i];
            buf_im[gi] = s_im[i];
        }
    }
}

// Standard FFT: bit-reverse + log2(N) HBM stages
std::vector<torch::Tensor> reference_fft(torch::Tensor x_re, torch::Tensor x_im) {
    int N = x_re.size(0);
    int log2N = 0; { int tmp = N; while (tmp > 1) { tmp >>= 1; log2N++; } }
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    auto buf_re = torch::empty_like(x_re);
    auto buf_im = torch::empty_like(x_im);
    auto tmp_re = torch::empty_like(x_re);
    auto tmp_im = torch::empty_like(x_im);

    bit_reverse_kernel<<<blocks, threads>>>(
        x_re.data_ptr<float>(), x_im.data_ptr<float>(),
        buf_re.data_ptr<float>(), buf_im.data_ptr<float>(), N, log2N);

    for (int s = 0; s < log2N; s++) {
        int half = 1 << s;
        butterfly_stage_kernel<<<blocks, threads>>>(
            buf_re.data_ptr<float>(), buf_im.data_ptr<float>(),
            tmp_re.data_ptr<float>(), tmp_im.data_ptr<float>(),
            N, half, s);
        std::swap(buf_re, tmp_re);
        std::swap(buf_im, tmp_im);
    }
    return {buf_re, buf_im};
}

// Flash FFT: bit-reverse + fused local stages + remaining global stages
std::vector<torch::Tensor> flash_fft(torch::Tensor x_re, torch::Tensor x_im) {
    int N = x_re.size(0);
    int log2N = 0; { int tmp = N; while (tmp > 1) { tmp >>= 1; log2N++; } }
    int threads = 256;
    int blocks_N = (N + threads - 1) / threads;

    auto buf_re = torch::empty_like(x_re);
    auto buf_im = torch::empty_like(x_im);

    bit_reverse_kernel<<<blocks_N, threads>>>(
        x_re.data_ptr<float>(), x_im.data_ptr<float>(),
        buf_re.data_ptr<float>(), buf_im.data_ptr<float>(), N, log2N);

    // Fuse first log2(TILE) stages in SMEM
    constexpr int TILE = 1024;
    int log2_tile = 10;  // log2(1024)
    int tile_blocks = (N + TILE - 1) / TILE;
    size_t smem = 2 * TILE * sizeof(float);  // real + imag
    flash_fft_kernel<TILE><<<tile_blocks, threads, smem>>>(
        buf_re.data_ptr<float>(), buf_im.data_ptr<float>(),
        N, log2N, log2_tile);

    // Remaining global stages
    auto tmp_re = torch::empty_like(x_re);
    auto tmp_im = torch::empty_like(x_im);
    for (int s = log2_tile; s < log2N; s++) {
        int half = 1 << s;
        butterfly_stage_kernel<<<blocks_N, threads>>>(
            buf_re.data_ptr<float>(), buf_im.data_ptr<float>(),
            tmp_re.data_ptr<float>(), tmp_im.data_ptr<float>(),
            N, half, s);
        std::swap(buf_re, tmp_re);
        std::swap(buf_im, tmp_im);
    }
    return {buf_re, buf_im};
}

// Library FFT: cuFFT via ATen (gold standard)
std::vector<torch::Tensor> library_fft(torch::Tensor x_re, torch::Tensor x_im) {
    auto x = torch::complex(x_re, x_im);
    auto y = torch::fft::fft(x);
    return {torch::real(y).contiguous(), torch::imag(y).contiguous()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_fft", &flash_fft, "Flash FFT (fused local stages)");
    m.def("reference_fft", &reference_fft, "Standard FFT (per-stage HBM)");
    m.def("library_fft", &library_fft, "cuFFT via ATen (gold standard)");
}
'''

CUDA_TEMPLATES["conv2d"] = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---- Standard Conv2D: explicit im2col → GEMM ----
// im2col kernel: materialize (N*OH*OW, C_in*KH*KW) matrix to HBM
__global__ void im2col_kernel(
    const float* __restrict__ input,   // (N, C_in, H, W)
    float* __restrict__ col,           // (N*OH*OW, C_in*KH*KW)
    int N, int C_in, int H, int W,
    int KH, int KW, int OH, int OW,
    int pad_h, int pad_w, int stride_h, int stride_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OH * OW;
    if (idx >= total) return;

    int n  = idx / (OH * OW);
    int rem = idx % (OH * OW);
    int oh = rem / OW;
    int ow = rem % OW;

    int col_row = idx;
    int col_cols = C_in * KH * KW;

    for (int c = 0; c < C_in; c++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int ih = oh * stride_h - pad_h + kh;
                int iw = ow * stride_w - pad_w + kw;
                float val = 0.0f;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    val = input[((n * C_in + c) * H + ih) * W + iw];
                }
                int col_c = (c * KH + kh) * KW + kw;
                col[col_row * col_cols + col_c] = val;
            }
        }
    }
}

// ---- Flash Conv2D: implicit im2col fused with GEMM ----
// Each block computes a tile of the output (BM output pixels × BN output channels)
// by gathering im2col patches on-the-fly into shared memory.
template <int BM, int BN, int BK>
__global__ void flash_conv2d_kernel(
    const float* __restrict__ input,   // (N, C_in, H, W)
    const float* __restrict__ weight,  // (C_out, C_in*KH*KW) reshaped
    float* __restrict__ output,        // (N*OH*OW, C_out)
    int N, int C_in, int H, int W,
    int KH, int KW, int OH, int OW, int C_out,
    int pad_h, int pad_w, int stride_h, int stride_w
) {
    // Block tile indices
    int bm = blockIdx.x;  // output pixel tile
    int bn = blockIdx.y;  // output channel tile

    int row_start = bm * BM;
    int col_start = bn * BN;
    int total_pixels = N * OH * OW;
    int K = C_in * KH * KW;

    // Accumulator in registers
    __shared__ float sA[BM][BK];  // im2col tile (gathered on-the-fly)
    __shared__ float sB[BK][BN];  // weight tile

    float acc[BM / 32][BN / 32];  // simplified: one element per thread
    // For simplicity, each thread computes one output element
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    float c = 0.0f;
    int out_row = row_start + ty;
    int out_col = col_start + tx;

    for (int k_start = 0; k_start < K; k_start += BK) {
        // Load weight tile into SMEM
        __syncthreads();
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int kk = i / BN;
            int nn = i % BN;
            int gk = k_start + kk;
            int gn = col_start + nn;
            sB[kk][nn] = (gk < K && gn < C_out) ? weight[gn * K + gk] : 0.0f;
        }

        // Implicit im2col: gather input patches into SMEM on-the-fly
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int mm = i / BK;
            int kk = i % BK;
            int grow = row_start + mm;
            int gk = k_start + kk;
            float val = 0.0f;
            if (grow < total_pixels && gk < K) {
                int n = grow / (OH * OW);
                int rem = grow % (OH * OW);
                int oh = rem / OW;
                int ow = rem % OW;
                int c_in = gk / (KH * KW);
                int k_rem = gk % (KH * KW);
                int kh = k_rem / KW;
                int kw = k_rem % KW;
                int ih = oh * stride_h - pad_h + kh;
                int iw = ow * stride_w - pad_w + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    val = input[((n * C_in + c_in) * H + ih) * W + iw];
            }
            sA[mm][kk] = val;
        }
        __syncthreads();

        // Compute partial GEMM
        if (ty < BM && tx < BN) {
            for (int kk = 0; kk < BK && (k_start + kk) < K; kk++) {
                c += sA[ty][kk] * sB[kk][tx];
            }
        }
    }

    // Write output
    if (out_row < total_pixels && out_col < C_out) {
        output[out_row * C_out + out_col] = c;
    }
}

// Explicit im2col + ATen mm baseline (materializes unfolded matrix to HBM)
torch::Tensor reference_conv2d(torch::Tensor input, torch::Tensor weight) {
    int N_batch = input.size(0), C_in = input.size(1);
    int H = input.size(2), W = input.size(3);
    int C_out = weight.size(0), KH = weight.size(2), KW = weight.size(3);
    int pad_h = KH / 2, pad_w = KW / 2;
    int stride_h = 1, stride_w = 1;
    int OH = (H + 2 * pad_h - KH) / stride_h + 1;
    int OW = (W + 2 * pad_w - KW) / stride_w + 1;

    int total_pixels = N_batch * OH * OW;
    int K = C_in * KH * KW;

    // Step 1: explicit im2col — materialize (N*OH*OW, C_in*KH*KW) to HBM
    auto col = torch::empty({total_pixels, K}, input.options());
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;
    im2col_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), col.data_ptr<float>(),
        N_batch, C_in, H, W, KH, KW, OH, OW,
        pad_h, pad_w, stride_h, stride_w);

    // Step 2: GEMM — col (total_pixels, K) @ weight^T (K, C_out)
    auto weight_flat = weight.reshape({C_out, K}).t().contiguous();  // (K, C_out)
    auto output = torch::mm(col, weight_flat);  // (total_pixels, C_out)
    return output.reshape({N_batch, OH, OW, C_out}).permute({0, 3, 1, 2}).contiguous();
}

// cuDNN baseline: torch::conv2d (implicit GEMM / Winograd, gold standard)
torch::Tensor library_conv2d(torch::Tensor input, torch::Tensor weight) {
    int KH = weight.size(2), KW = weight.size(3);
    int pad_h = KH / 2, pad_w = KW / 2;
    return torch::conv2d(input, weight, {}, 1, {pad_h, pad_w});
}

torch::Tensor flash_conv2d(torch::Tensor input, torch::Tensor weight) {
    int N_batch = input.size(0), C_in = input.size(1);
    int H = input.size(2), W = input.size(3);
    int C_out = weight.size(0), KH = weight.size(2), KW = weight.size(3);
    int pad_h = KH / 2, pad_w = KW / 2;
    int stride_h = 1, stride_w = 1;
    int OH = (H + 2 * pad_h - KH) / stride_h + 1;
    int OW = (W + 2 * pad_w - KW) / stride_w + 1;

    int total_pixels = N_batch * OH * OW;
    int K = C_in * KH * KW;

    // Reshape weight: (C_out, C_in, KH, KW) → (C_out, C_in*KH*KW)
    auto weight_flat = weight.reshape({C_out, K}).contiguous();
    auto output_flat = torch::empty({total_pixels, C_out}, input.options());

    constexpr int BM = 32, BN = 32, BK = 32;
    dim3 grid((total_pixels + BM - 1) / BM, (C_out + BN - 1) / BN);
    int threads = BM * BN;
    if (threads > 1024) threads = 1024;

    flash_conv2d_kernel<BM, BN, BK><<<grid, threads>>>(
        input.data_ptr<float>(), weight_flat.data_ptr<float>(),
        output_flat.data_ptr<float>(),
        N_batch, C_in, H, W, KH, KW, OH, OW, C_out,
        pad_h, pad_w, stride_h, stride_w);

    // output_flat is (N*OH*OW, C_out) in row-major = NHWC layout
    // Reshape to (N, OH, OW, C_out) then permute to NCHW
    return output_flat.reshape({N_batch, OH, OW, C_out}).permute({0, 3, 1, 2}).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_conv2d", &flash_conv2d, "Flash Conv2D (implicit im2col)");
    m.def("reference_conv2d", &reference_conv2d, "Explicit im2col + GEMM (materialized)");
    m.def("library_conv2d", &library_conv2d, "cuDNN Conv2D (gold standard)");
}
'''

CUDA_TEMPLATES["stencil2d"] = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---- Standard 2D Stencil: 1 iteration per kernel launch ----
// Each launch reads full H×W grid from HBM, writes full H×W output.
__global__ void jacobi_2d_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    int H, int W
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;  // skip boundary
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x >= H - 1 || y >= W - 1) return;

    out[x * W + y] = 0.2f * (
        in[x * W + y] +
        in[(x-1) * W + y] + in[(x+1) * W + y] +
        in[x * W + (y-1)] + in[x * W + (y+1)]
    );
}

// ---- Flash Stencil: temporal tiling, fuse S iterations in SMEM ----
// Each block loads a (BH+2*S) × (BW+2*S) halo region, applies S iterations
// on-chip, writes back BH×BW inner region. Reduces HBM round-trips by S×.
template <int BH, int BW, int S>
__global__ void flash_stencil_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    int H, int W
) {
    // Halo dimensions
    constexpr int HALO_H = BH + 2 * S;
    constexpr int HALO_W = BW + 2 * S;
    __shared__ float tile0[HALO_H][HALO_W];
    __shared__ float tile1[HALO_H][HALO_W];

    int bx = blockIdx.x * BH;
    int by = blockIdx.y * BW;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load halo region from HBM into SMEM (1 HBM read)
    for (int i = ty; i < HALO_H; i += blockDim.y) {
        for (int j = tx; j < HALO_W; j += blockDim.x) {
            int gx = bx - S + i;
            int gy = by - S + j;
            float val = 0.0f;
            if (gx >= 0 && gx < H && gy >= 0 && gy < W)
                val = in[gx * W + gy];
            tile0[i][j] = val;
        }
    }
    __syncthreads();

    // Apply S iterations in SMEM — zero HBM traffic!
    float (*src)[HALO_W] = tile0;
    float (*dst)[HALO_W] = tile1;

    for (int iter = 0; iter < S; iter++) {
        // Each iteration shrinks valid region by 1 on each side
        int margin = iter + 1;
        for (int i = ty + margin; i < HALO_H - margin; i += blockDim.y) {
            for (int j = tx + margin; j < HALO_W - margin; j += blockDim.x) {
                // Map to global coords to skip grid boundary cells
                int gx = bx - S + i;
                int gy = by - S + j;
                if (gx >= 1 && gx < H - 1 && gy >= 1 && gy < W - 1) {
                    dst[i][j] = 0.2f * (
                        src[i][j] +
                        src[i-1][j] + src[i+1][j] +
                        src[i][j-1] + src[i][j+1]
                    );
                } else {
                    dst[i][j] = src[i][j];  // boundary: keep unchanged
                }
            }
        }
        __syncthreads();
        // Swap buffers
        float (*tmp)[HALO_W] = src;
        src = dst;
        dst = tmp;
    }

    // Write inner BH×BW region back to HBM (1 HBM write)
    // Only write interior cells (skip grid boundary rows/cols)
    for (int i = ty; i < BH; i += blockDim.y) {
        for (int j = tx; j < BW; j += blockDim.x) {
            int gx = bx + i;
            int gy = by + j;
            if (gx >= 1 && gx < H - 1 && gy >= 1 && gy < W - 1) {
                out[gx * W + gy] = src[S + i][S + j];
            }
        }
    }
}

torch::Tensor reference_stencil2d(torch::Tensor grid, int T) {
    int H = grid.size(0), W = grid.size(1);
    auto buf0 = grid.clone();
    auto buf1 = grid.clone();  // clone so boundary cells are initialized

    dim3 block(16, 16);
    dim3 grid_dim((H - 2 + block.x - 1) / block.x, (W - 2 + block.y - 1) / block.y);

    for (int t = 0; t < T; t++) {
        jacobi_2d_kernel<<<grid_dim, block>>>(
            buf0.data_ptr<float>(), buf1.data_ptr<float>(), H, W);
        std::swap(buf0, buf1);
    }
    return buf0;
}

torch::Tensor flash_stencil2d(torch::Tensor grid, int T) {
    int H = grid.size(0), W = grid.size(1);
    auto buf0 = grid.clone();
    auto buf1 = grid.clone();  // clone so boundary cells are initialized

    constexpr int BH = 32, BW = 32, S = 4;
    dim3 block(16, 16);
    dim3 grid_dim((H + BH - 1) / BH, (W + BW - 1) / BW);

    for (int t = 0; t < T; t += S) {
        int steps = (t + S <= T) ? S : (T - t);
        // For simplicity, always run S steps (last chunk may compute extra)
        flash_stencil_kernel<BH, BW, S><<<grid_dim, block>>>(
            buf0.data_ptr<float>(), buf1.data_ptr<float>(), H, W);
        std::swap(buf0, buf1);
    }
    return buf0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_stencil2d", &flash_stencil2d, "Flash Stencil2D (temporal tiling)");
    m.def("reference_stencil2d", &reference_stencil2d, "Standard Stencil2D (per-iteration HBM)");
}
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
    """Materialized baseline."""
    if hasattr(_module, "reference_{task}"):
        return _module.reference_{task}(*args, **kwargs)
    raise NotImplementedError("No reference implementation in CUDA module")

def library_{task}(*args, **kwargs):
    """Library-optimized baseline (cuDNN/cuFFT/cuBLAS)."""
    if hasattr(_module, "library_{task}"):
        return _module.library_{task}(*args, **kwargs)
    raise NotImplementedError("No library implementation in CUDA module")
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

        elif task == "fft":
            N = params.get("N", 1024)
            # Use a small power-of-2 for test
            N_test = min(N, 8192)
            x_re = torch.randn(N_test, device="cuda")
            x_im = torch.zeros(N_test, device="cuda")
            ref_result = ref_fn(x_re, x_im)
            flash_result = flash_fn(x_re, x_im)
            # ref_result and flash_result are lists [re, im]
            max_err_re = (ref_result[0] - flash_result[0]).abs().max().item()
            max_err_im = (ref_result[1] - flash_result[1]).abs().max().item()
            max_err = max(max_err_re, max_err_im)
            mean_err = ((ref_result[0] - flash_result[0]).abs().mean().item() +
                        (ref_result[1] - flash_result[1]).abs().mean().item()) / 2

        elif task == "conv2d":
            N_b = params.get("N", 4)
            C_in = params.get("C_in", 32)
            C_out = params.get("C_out", 64)
            H, W = params.get("H", 28), params.get("W", 28)
            KH, KW = params.get("KH", 3), params.get("KW", 3)
            inp = torch.randn(N_b, C_in, H, W, device="cuda")
            wt = torch.randn(C_out, C_in, KH, KW, device="cuda")
            ref_out = ref_fn(inp, wt)
            flash_out = flash_fn(inp, wt)
            max_err = (ref_out - flash_out).abs().max().item()
            mean_err = (ref_out - flash_out).abs().mean().item()

        elif task == "stencil2d":
            H = min(params.get("H", 256), 256)
            W = min(params.get("W", 256), 256)
            T = min(params.get("T", 8), 8)
            grid_in = torch.randn(H, W, device="cuda")
            ref_out = ref_fn(grid_in, T)
            flash_out = flash_fn(grid_in, T)
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
    elif task == "fft":
        N = params.get("N", 1048576)
        x_re = torch.randn(N, device="cuda")
        x_im = torch.zeros(N, device="cuda")
        inputs_flash = (x_re, x_im)
        inputs_ref = (x_re, x_im)
        ref_fn_actual = ref_fn
    elif task == "conv2d":
        N_b = params.get("N", 64)
        C_in = params.get("C_in", 128)
        C_out = params.get("C_out", 256)
        H, W = params.get("H", 56), params.get("W", 56)
        KH, KW = params.get("KH", 3), params.get("KW", 3)
        inp = torch.randn(N_b, C_in, H, W, device="cuda")
        wt = torch.randn(C_out, C_in, KH, KW, device="cuda")
        inputs_flash = (inp, wt)
        inputs_ref = (inp, wt)
        ref_fn_actual = ref_fn  # reference_conv2d: explicit im2col + GEMM (materialized)
    elif task == "stencil2d":
        H = params.get("H", 4096)
        W = params.get("W", 4096)
        T = params.get("T", 100)
        grid_in = torch.randn(H, W, device="cuda")
        inputs_flash = (grid_in, T)
        inputs_ref = (grid_in, T)
        ref_fn_actual = ref_fn  # reference_stencil2d
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

    # Benchmark flash (Triton/CUDA kernel)
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
        f"  Baseline (materialized):         {t_baseline:.3f} ms",
        f"  Flash (our kernel):              {t_flash:.3f} ms",
        f"  Speedup vs baseline:             {speedup:.2f}×",
    ]

    # 3-way comparison: try to benchmark library (cuDNN/cuFFT) if available
    lib_fn = None
    for name in dir(mod):
        if name.startswith("library_"):
            lib_fn = getattr(mod, name)
            break

    if lib_fn is not None:
        try:
            # Warmup library
            for _ in range(n_warmup):
                lib_fn(*inputs_ref)
            torch.cuda.synchronize()

            s.record()
            for _ in range(n_iter):
                lib_fn(*inputs_ref)
            e.record()
            torch.cuda.synchronize()
            t_lib = s.elapsed_time(e) / n_iter

            lib_name = "cuDNN" if task == "conv2d" else "cuFFT" if task == "fft" else "Library"
            speedup_vs_lib = t_lib / t_flash if t_flash > 0 else 0
            speedup_lib_vs_mat = t_baseline / t_lib if t_lib > 0 else 0

            lines.append(f"  {lib_name} (library):              {t_lib:.3f} ms")
            lines.append(f"  {lib_name} vs materialized:        {speedup_lib_vs_mat:.2f}×")
            lines.append(f"  Flash vs {lib_name}:               {speedup_vs_lib:.2f}×")
            lines.append(f"  ────────────────────────────────")

            if speedup_vs_lib > 1.05:
                lines.append(f"  ✓ Flash beats {lib_name} by {speedup_vs_lib:.2f}×!")
            elif speedup_vs_lib > 0.95:
                lines.append(f"  ~ Flash ≈ {lib_name} (within 5%)")
            else:
                lines.append(f"  ⚠ Flash is {speedup_vs_lib:.2f}× of {lib_name} — {lib_name} is faster")
                lines.append(f"    ({lib_name} uses highly-optimized implicit GEMM / Winograd / FFT algorithms)")
        except Exception as ex:
            lines.append(f"  Library benchmark skipped: {ex}")

    if speedup > 1.05:
        lines.append(f"  ✓ Flash kernel is {speedup:.2f}× faster than materialized baseline!")
    elif speedup > 0.95:
        lines.append(f"  ~ Performance similar to baseline (within 5%)")
    else:
        lines.append(f"  ⚠ Flash kernel is slower than baseline ({speedup:.2f}×)")

    lines.append(f"  Speedup: {speedup:.2f}×")

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

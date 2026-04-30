/*
 * Flash-GMM: CUDA GPU Kernels (Optimized v2)
 *
 * Optimizations:
 *   1. Pre-computed inv_var and log_coeff — no logf() in hot kernels
 *   2. Warp-collaborative distance — coalesced memory access via lane-striped dims
 *   3. Per-warp accumulators (Kernel 2) — eliminates all shared memory atomicAdd
 *   4. Online log-sum-exp state in shared memory
 *
 * Note: Double-buffered cp.async was tested (v3) but degraded performance by ~13%
 * because centroid tiles are small enough to fit in L2 cache, and the extra shared
 * memory usage halves occupancy. Reverted to single-buffer approach.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <algorithm>

static constexpr float LOG_2PI = 1.8378770664093453f;
static constexpr int MAX_DIMS_PER_LANE = 8;  // supports d up to 256
static constexpr int BLOCK_THREADS = 256;

// ============================================================================
// Warp-level primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================================
// Kernel 0: Pre-compute inv_var and log_coeff
// ============================================================================

__global__ void precompute_params_kernel(
    const float* __restrict__ var,
    const float* __restrict__ log_pi,
    float* __restrict__ inv_var,
    float* __restrict__ log_coeff,
    int K, int d
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    float log_det = 0.0f;
    for (int j = 0; j < d; j++) {
        float v = var[k * d + j];
        inv_var[k * d + j] = 1.0f / v;
        log_det += logf(v);
    }
    log_coeff[k] = log_pi[k] - 0.5f * (d * LOG_2PI + log_det);
}

// ============================================================================
// Kernel 1 v2: Flash Log-Normalizer
//
// Grid:  ceil(N / BN)
// Block: BLOCK_THREADS (256 = 8 warps)
//
// Each warp collaboratively handles one point at a time:
//   - 32 lanes handle different dimensions (coalesced X reads)
//   - Warp-reduce for full Mahalanobis distance
//
// Shared memory: mu_tile[BK*d] + iv_tile[BK*d] + lc_tile[BK] + max[BN] + sum[BN]
// ============================================================================

template <int BK_COMPILE>
__global__ void flash_log_normalizer_v2_kernel(
    const float* __restrict__ X,
    const float* __restrict__ mu,
    const float* __restrict__ inv_var,
    const float* __restrict__ log_coeff,
    float* __restrict__ log_normalizer,
    int N, int K, int d, int BN
) {
    extern __shared__ float smem[];
    float* mu_tile  = smem;
    float* iv_tile  = mu_tile + BK_COMPILE * d;
    float* lc_tile  = iv_tile + BK_COMPILE * d;
    float* max_smem = lc_tile + BK_COMPILE;
    float* sum_smem = max_smem + BN;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int n_warps = blockDim.x >> 5;
    const int block_start = blockIdx.x * BN;
    const int BN_actual = min(BN, N - block_start);

    // Initialize logsumexp state
    for (int i = tid; i < BN_actual; i += blockDim.x) {
        max_smem[i] = -FLT_MAX;
        sum_smem[i] = 0.0f;
    }
    __syncthreads();

    // Stream centroid tiles
    for (int t = 0; t < K; t += BK_COMPILE) {
        int tile_k = min(BK_COMPILE, K - t);

        // Cooperative load of centroid tile
        __syncthreads();
        for (int idx = tid; idx < tile_k * d; idx += blockDim.x) {
            int kk = idx / d, jj = idx % d;
            mu_tile[kk * d + jj] = mu[(t + kk) * d + jj];
            iv_tile[kk * d + jj] = inv_var[(t + kk) * d + jj];
        }
        for (int idx = tid; idx < tile_k; idx += blockDim.x) {
            lc_tile[idx] = log_coeff[t + idx];
        }
        __syncthreads();

        // Each warp processes its subset of points
        for (int pt = warp_id; pt < BN_actual; pt += n_warps) {
            int n = block_start + pt;

            // Cache X in registers — avoid repeated global memory reads
            float x_local[MAX_DIMS_PER_LANE];
            #pragma unroll
            for (int i = 0; i < MAX_DIMS_PER_LANE; i++) {
                int j = lane_id + i * 32;
                x_local[i] = (j < d) ? X[n * d + j] : 0.0f;
            }

            float rm = __shfl_sync(0xffffffff,
                           (lane_id == 0) ? max_smem[pt] : 0.0f, 0);
            float rs = __shfl_sync(0xffffffff,
                           (lane_id == 0) ? sum_smem[pt] : 0.0f, 0);

            for (int kk = 0; kk < tile_k; kk++) {
                float partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < MAX_DIMS_PER_LANE; i++) {
                    int j = lane_id + i * 32;
                    if (j < d) {
                        float diff = x_local[i] - mu_tile[kk * d + j];
                        partial += diff * diff * iv_tile[kk * d + j];
                    }
                }
                float mahal = warp_reduce_sum(partial);
                mahal = __shfl_sync(0xffffffff, mahal, 0);

                float ll = lc_tile[kk] - 0.5f * mahal;

                if (ll > rm) {
                    rs = rs * __expf(rm - ll) + 1.0f;
                    rm = ll;
                } else {
                    rs += __expf(ll - rm);
                }
            }

            if (lane_id == 0) {
                max_smem[pt] = rm;
                sum_smem[pt] = rs;
            }
        }
    }

    __syncthreads();

    for (int pt = tid; pt < BN_actual; pt += blockDim.x) {
        log_normalizer[block_start + pt] = max_smem[pt] + logf(sum_smem[pt]);
    }
}

// ============================================================================
// Kernel 2 v2: Flash Accumulate Statistics
//
// Single centroid tile per block (grid.y splits K) → no double-buffering needed.
// Per-warp accumulators → zero atomics in hot loop.
// ============================================================================

template <int BK_COMPILE>
__global__ void flash_accumulate_stats_v2_kernel(
    const float* __restrict__ X,
    const float* __restrict__ mu,
    const float* __restrict__ inv_var,
    const float* __restrict__ log_coeff,
    const float* __restrict__ log_normalizer,
    float* __restrict__ global_n_k,
    float* __restrict__ global_s_k,
    float* __restrict__ global_sq_k,
    int N, int K, int d, int BN
) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int n_warps = blockDim.x >> 5;
    const int k_start = blockIdx.y * BK_COMPILE;
    const int block_start = blockIdx.x * BN;
    const int tile_k = min(BK_COMPILE, K - k_start);
    const int BN_actual = min(BN, N - block_start);

    float* mu_tile  = smem;
    float* iv_tile  = mu_tile  + BK_COMPILE * d;
    float* lc_tile  = iv_tile  + BK_COMPILE * d;
    float* warp_nk  = lc_tile  + BK_COMPILE;
    float* warp_sk  = warp_nk  + n_warps * BK_COMPILE;
    float* warp_sqk = warp_sk  + n_warps * BK_COMPILE * d;

    for (int idx = tid; idx < tile_k * d; idx += blockDim.x) {
        int kk = idx / d, jj = idx % d;
        mu_tile[kk * d + jj] = mu[(k_start + kk) * d + jj];
        iv_tile[kk * d + jj] = inv_var[(k_start + kk) * d + jj];
    }
    for (int idx = tid; idx < tile_k; idx += blockDim.x) {
        lc_tile[idx] = log_coeff[k_start + idx];
    }

    for (int idx = tid; idx < n_warps * BK_COMPILE; idx += blockDim.x)
        warp_nk[idx] = 0.0f;
    for (int idx = tid; idx < n_warps * BK_COMPILE * d; idx += blockDim.x) {
        warp_sk[idx] = 0.0f;
        warp_sqk[idx] = 0.0f;
    }
    __syncthreads();

    for (int pt = warp_id; pt < BN_actual; pt += n_warps) {
        int n = block_start + pt;
        float ln_val = log_normalizer[n];

        float x_local[MAX_DIMS_PER_LANE];
        #pragma unroll
        for (int i = 0; i < MAX_DIMS_PER_LANE; i++) {
            int j = lane_id + i * 32;
            x_local[i] = (j < d) ? X[n * d + j] : 0.0f;
        }

        for (int kk = 0; kk < tile_k; kk++) {
            float partial = 0.0f;
            #pragma unroll
            for (int i = 0; i < MAX_DIMS_PER_LANE; i++) {
                int j = lane_id + i * 32;
                if (j < d) {
                    float diff = x_local[i] - mu_tile[kk * d + j];
                    partial += diff * diff * iv_tile[kk * d + j];
                }
            }
            float mahal = warp_reduce_sum(partial);
            mahal = __shfl_sync(0xffffffff, mahal, 0);

            float ll = lc_tile[kk] - 0.5f * mahal;
            float gamma_nk = __expf(ll - ln_val);

            if (lane_id == 0)
                warp_nk[warp_id * BK_COMPILE + kk] += gamma_nk;

            #pragma unroll
            for (int i = 0; i < MAX_DIMS_PER_LANE; i++) {
                int j = lane_id + i * 32;
                if (j < d) {
                    int base = warp_id * BK_COMPILE * d + kk * d;
                    warp_sk[base + j]  += gamma_nk * x_local[i];
                    warp_sqk[base + j] += gamma_nk * x_local[i] * x_local[i];
                }
            }
        }
    }

    __syncthreads();

    for (int stride = n_warps >> 1; stride > 0; stride >>= 1) {
        if (warp_id < stride) {
            for (int idx = lane_id; idx < tile_k; idx += 32)
                warp_nk[warp_id * BK_COMPILE + idx] +=
                    warp_nk[(warp_id + stride) * BK_COMPILE + idx];
            for (int idx = lane_id; idx < tile_k * d; idx += 32) {
                warp_sk[warp_id * BK_COMPILE * d + idx] +=
                    warp_sk[(warp_id + stride) * BK_COMPILE * d + idx];
                warp_sqk[warp_id * BK_COMPILE * d + idx] +=
                    warp_sqk[(warp_id + stride) * BK_COMPILE * d + idx];
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        for (int idx = lane_id; idx < tile_k; idx += 32)
            atomicAdd(&global_n_k[k_start + idx], warp_nk[idx]);
        for (int idx = lane_id; idx < tile_k * d; idx += 32) {
            atomicAdd(&global_s_k[k_start * d + idx], warp_sk[idx]);
            atomicAdd(&global_sq_k[k_start * d + idx], warp_sqk[idx]);
        }
    }
}

// ============================================================================
// Kernel 3: Update parameters
// ============================================================================

__global__ void update_params_kernel(
    const float* __restrict__ n_k,
    const float* __restrict__ s_k,
    const float* __restrict__ sq_k,
    float* __restrict__ new_mu,
    float* __restrict__ new_var,
    float* __restrict__ new_log_pi,
    int K, int d, int N
) {
    int k = blockIdx.x;
    if (k >= K) return;

    float inv_nk = 1.0f / fmaxf(n_k[k], 1e-8f);

    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float mu_kj = s_k[k * d + j] * inv_nk;
        new_mu[k * d + j] = mu_kj;
        float var_kj = sq_k[k * d + j] * inv_nk - mu_kj * mu_kj;
        new_var[k * d + j] = fmaxf(var_kj, 1e-6f);
    }

    if (threadIdx.x == 0) {
        new_log_pi[k] = logf(fmaxf(n_k[k] / N, 1e-8f));
    }
}

// ============================================================================
// Standard E-step baseline
// ============================================================================

__global__ void standard_e_step_kernel(
    const float* __restrict__ X,
    const float* __restrict__ mu,
    const float* __restrict__ var,
    const float* __restrict__ log_pi,
    float* __restrict__ L,
    int N, int K, int d
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    for (int k = 0; k < K; k++) {
        float log_det = 0.0f;
        float mahal = 0.0f;
        for (int j = 0; j < d; j++) {
            float v = var[k * d + j];
            log_det += logf(v);
            float diff = X[n * d + j] - mu[k * d + j];
            mahal += diff * diff / v;
        }
        L[n * K + k] = log_pi[k] - 0.5f * (d * LOG_2PI + log_det + mahal);
    }
}

__global__ void logsumexp_normalize_kernel(
    const float* __restrict__ L,
    float* __restrict__ gamma,
    float* __restrict__ log_normalizer,
    int N, int K
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float max_val = -FLT_MAX;
    for (int k = 0; k < K; k++) {
        max_val = fmaxf(max_val, L[n * K + k]);
    }
    float sum_exp = 0.0f;
    for (int k = 0; k < K; k++) {
        sum_exp += expf(L[n * K + k] - max_val);
    }
    float ln = max_val + logf(sum_exp);
    log_normalizer[n] = ln;
    for (int k = 0; k < K; k++) {
        gamma[n * K + k] = expf(L[n * K + k] - ln);
    }
}

// ============================================================================
// ATen Launchers
// ============================================================================

std::vector<torch::Tensor> standard_e_step_cuda(
    torch::Tensor X, torch::Tensor mu, torch::Tensor var, torch::Tensor log_pi
) {
    int N = X.size(0), d = X.size(1), K = mu.size(0);
    auto L = torch::empty({N, K}, X.options());
    auto gamma = torch::empty({N, K}, X.options());
    auto log_normalizer = torch::empty({N}, X.options());

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    standard_e_step_kernel<<<blocks, threads>>>(
        X.data_ptr<float>(), mu.data_ptr<float>(), var.data_ptr<float>(),
        log_pi.data_ptr<float>(), L.data_ptr<float>(), N, K, d);

    logsumexp_normalize_kernel<<<blocks, threads>>>(
        L.data_ptr<float>(), gamma.data_ptr<float>(),
        log_normalizer.data_ptr<float>(), N, K);

    return {gamma, log_normalizer};
}

std::vector<torch::Tensor> flash_em_fused_cuda(
    torch::Tensor X, torch::Tensor mu, torch::Tensor var, torch::Tensor log_pi,
    int BN, int BK
) {
    int N = X.size(0), d = X.size(1), K = mu.size(0);
    auto log_normalizer = torch::empty({N}, X.options());
    auto n_k  = torch::zeros({K}, X.options());
    auto s_k  = torch::zeros({K, d}, X.options());
    auto sq_k = torch::zeros({K, d}, X.options());
    auto new_mu     = torch::empty({K, d}, X.options());
    auto new_var    = torch::empty({K, d}, X.options());
    auto new_log_pi = torch::empty({K}, X.options());

    auto inv_var  = torch::empty({K, d}, X.options());
    auto lcoeff   = torch::empty({K}, X.options());

    // ---- Kernel 0: Pre-compute ----
    {
        int threads_pc = std::min(K, 256);
        int blocks_pc = (K + threads_pc - 1) / threads_pc;
        precompute_params_kernel<<<blocks_pc, threads_pc>>>(
            var.data_ptr<float>(), log_pi.data_ptr<float>(),
            inv_var.data_ptr<float>(), lcoeff.data_ptr<float>(),
            K, d);
    }

    const int n_warps = BLOCK_THREADS / 32;

    int BK1 = std::min(32, K);
    if (BK1 >= 32) BK1 = 32;
    else if (BK1 >= 16) BK1 = 16;
    else if (BK1 >= 8) BK1 = 8;
    else BK1 = 4;

    int BK2;
    {
        int budget_floats = 100 * 1024 / 4;
        int factor = (2 * d + 1) * (1 + n_warps);
        int max_bk = (factor > 0) ? budget_floats / factor : 4;
        BK2 = std::min({max_bk, K, 32});
        if (BK2 >= 32) BK2 = 32;
        else if (BK2 >= 16) BK2 = 16;
        else if (BK2 >= 8) BK2 = 8;
        else BK2 = 4;
        BK2 = std::min(BK2, K);
    }

    // ---- Pass 1: log_normalizer ----
    {
        int grid_n = (N + BN - 1) / BN;
        size_t smem_pass1 = (size_t)(BK1 * (2 * d + 1) + 2 * BN) * sizeof(float);

        #define LAUNCH_K1(BK_VAL)                                                              \
            cudaFuncSetAttribute(                                                               \
                flash_log_normalizer_v2_kernel<BK_VAL>,                                         \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_pass1);                   \
            flash_log_normalizer_v2_kernel<BK_VAL><<<grid_n, BLOCK_THREADS, smem_pass1>>>(      \
                X.data_ptr<float>(), mu.data_ptr<float>(),                                      \
                inv_var.data_ptr<float>(), lcoeff.data_ptr<float>(),                             \
                log_normalizer.data_ptr<float>(), N, K, d, BN);

        if (BK1 <= 4)       { LAUNCH_K1(4)  }
        else if (BK1 <= 8)  { LAUNCH_K1(8)  }
        else if (BK1 <= 16) { LAUNCH_K1(16) }
        else                { LAUNCH_K1(32) }

        #undef LAUNCH_K1
    }

    // ---- Pass 2: sufficient statistics ----
    {
        dim3 grid((N + BN - 1) / BN, (K + BK2 - 1) / BK2);
        size_t smem_pass2 = (size_t)BK2 * (2 * d + 1) * (1 + n_warps) * sizeof(float);

        #define LAUNCH_K2(BK_VAL)                                                              \
            cudaFuncSetAttribute(                                                               \
                flash_accumulate_stats_v2_kernel<BK_VAL>,                                       \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_pass2);                   \
            flash_accumulate_stats_v2_kernel<BK_VAL><<<grid, BLOCK_THREADS, smem_pass2>>>(      \
                X.data_ptr<float>(), mu.data_ptr<float>(),                                      \
                inv_var.data_ptr<float>(), lcoeff.data_ptr<float>(),                             \
                log_normalizer.data_ptr<float>(),                                               \
                n_k.data_ptr<float>(), s_k.data_ptr<float>(),                                   \
                sq_k.data_ptr<float>(), N, K, d, BN);

        if (BK2 <= 4)       { LAUNCH_K2(4)  }
        else if (BK2 <= 8)  { LAUNCH_K2(8)  }
        else if (BK2 <= 16) { LAUNCH_K2(16) }
        else                { LAUNCH_K2(32) }

        #undef LAUNCH_K2
    }

    // ---- Update parameters ----
    {
        update_params_kernel<<<K, std::min(d, 256)>>>(
            n_k.data_ptr<float>(), s_k.data_ptr<float>(), sq_k.data_ptr<float>(),
            new_mu.data_ptr<float>(), new_var.data_ptr<float>(),
            new_log_pi.data_ptr<float>(), K, d, N);
    }

    return {new_mu, new_var, new_log_pi, log_normalizer};
}

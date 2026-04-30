/*
 * Flash-GMM Tensor Core v2: WMMA + warp-private logsumexp
 *
 * Architecture:
 *   Block: 128 threads = 4 warps, handles BN=64 points × ALL K centroids
 *   Each warp owns 16 consecutive rows → warp-private logsumexp (no atomics)
 *
 * Distance = X²@inv_var^T - 2·X@mu_iv^T + quad_mu  (two WMMAs per K-tile)
 * Online logsumexp in warp-private registers after each K-tile
 *
 * WMMA config: m16n16k16, bf16 input, float accumulator (A100 sm_80)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;
static constexpr int NUM_WARPS = 4;
static constexpr int BLOCK_THREADS = NUM_WARPS * 32;
static constexpr int BN = NUM_WARPS * WMMA_M;  // 64
static constexpr int BK = WMMA_N;               // 16

__device__ __forceinline__ __nv_bfloat16 f2bf(float x) {
    return __float2bfloat16(x);
}

// ============================================================================
// Kernel 1: WMMA Flash Log-Normalizer (v2 — no atomics)
//
// After WMMA mma_sync, the accumulator fragment stores 16×16 results.
// We store to a warp-private shared buffer, then do logsumexp row-by-row.
// Each warp exclusively owns its 16 rows → zero contention.
// ============================================================================

template <int D>
__global__ void wmma_flash_logsumexp_v2(
    const float* __restrict__ X,
    const float* __restrict__ X_sq,
    const float* __restrict__ inv_var,
    const float* __restrict__ mu_iv,
    const float* __restrict__ quad_mu_g,
    const float* __restrict__ log_coeff_g,
    float* __restrict__ log_normalizer,
    int N, int K
) {
    // Shared memory layout:
    //   x_bf[BN * D]  + xsq_bf[BN * D]       ← data (loaded once)
    //   iv_bf[BK * D] + mi_bf[BK * D]         ← centroids (per K-tile)
    //   qm[BK] + lc[BK]                       ← per-tile constants
    //   L_buf[BN * BK]                         ← WMMA output staging
    extern __shared__ char smem[];

    __nv_bfloat16* x_bf   = (__nv_bfloat16*)smem;
    __nv_bfloat16* xsq_bf = x_bf + BN * D;
    __nv_bfloat16* iv_bf  = xsq_bf + BN * D;
    __nv_bfloat16* mi_bf  = iv_bf + BK * D;
    float* qm_buf = (float*)(mi_bf + BK * D);
    float* lc_buf = qm_buf + BK;
    float* L_buf   = lc_buf + BK;        // [BN * BK]
    float* dot2_buf = L_buf + BN * BK;   // [BN * BK]  ← separate buffer!

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int block_start = blockIdx.x * BN;

    // ---- Load X, X² → shared memory (float → bf16) ----
    for (int idx = tid; idx < BN * D; idx += BLOCK_THREADS) {
        int r = idx / D, c = idx % D;
        int n = block_start + r;
        float xv  = (n < N) ? X[n * D + c] : 0.0f;
        float xsv = (n < N) ? X_sq[n * D + c] : 0.0f;
        x_bf[r * D + c] = f2bf(xv);
        xsq_bf[r * D + c] = f2bf(xsv);
    }
    __syncthreads();

    // Warp-private logsumexp state (in registers, per row in this warp's tile)
    float row_max[WMMA_M];
    float row_sum[WMMA_M];
    #pragma unroll
    for (int i = 0; i < WMMA_M; i++) {
        row_max[i] = -FLT_MAX;
        row_sum[i] = 0.0f;
    }

    const int warp_row_start = warp_id * WMMA_M;

    // ---- Stream K-tiles ----
    for (int t = 0; t < K; t += BK) {
        int tile_k = min(BK, K - t);

        // Cooperatively load centroid tile
        __syncthreads();
        for (int idx = tid; idx < BK * D; idx += BLOCK_THREADS) {
            int kk = idx / D, c = idx % D;
            float iv_val = (t + kk < K) ? inv_var[(t + kk) * D + c] : 0.0f;
            float mi_val = (t + kk < K) ? mu_iv[(t + kk) * D + c] : 0.0f;
            iv_bf[kk * D + c] = f2bf(iv_val);
            mi_bf[kk * D + c] = f2bf(mi_val);
        }
        for (int idx = tid; idx < BK; idx += BLOCK_THREADS) {
            qm_buf[idx] = (t + idx < K) ? quad_mu_g[t + idx] : 0.0f;
            lc_buf[idx] = (t + idx < K) ? log_coeff_g[t + idx] : -FLT_MAX;
        }
        __syncthreads();

        // ---- WMMA #1: acc1 = X²[16,D] @ inv_var[D,BK] ----
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc1;
        wmma::fill_fragment(acc1, 0.0f);

        #pragma unroll
        for (int kk = 0; kk < D; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::col_major> b_frag;

            // A = xsq_bf[warp_row_start*D + kk], leading dim = D
            wmma::load_matrix_sync(a_frag, xsq_bf + warp_row_start * D + kk, D);
            // B = iv_bf[kk], leading dim = D  (col_major view of [BK,D] row-major)
            wmma::load_matrix_sync(b_frag, iv_bf + kk, D);

            wmma::mma_sync(acc1, a_frag, b_frag, acc1);
        }

        // ---- WMMA #2: acc2 = X[16,D] @ mu_iv[D,BK] ----
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc2;
        wmma::fill_fragment(acc2, 0.0f);

        #pragma unroll
        for (int kk = 0; kk < D; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::col_major> b_frag;

            wmma::load_matrix_sync(a_frag, x_bf + warp_row_start * D + kk, D);
            wmma::load_matrix_sync(b_frag, mi_bf + kk, D);

            wmma::mma_sync(acc2, a_frag, b_frag, acc2);
        }

        // ---- Store WMMA results ----
        wmma::store_matrix_sync(L_buf + warp_row_start * BK, acc1, BK, wmma::mem_row_major);
        wmma::store_matrix_sync(dot2_buf + warp_row_start * BK, acc2, BK, wmma::mem_row_major);

        __syncwarp();  // ensure warp's stores are visible

        // ---- Compute L values and update online logsumexp ----
        // BK=16, 32 lanes: each lane handles at most 1 column per row
        // Strategy: each lane computes its ll, then warp-reduce to combine
        // all BK columns into one (max, sum) pair, then lane 0 updates
        // the running logsumexp state.

        for (int r = 0; r < WMMA_M; r++) {
            int global_row = block_start + warp_row_start + r;
            if (global_row >= N) continue;

            // Each lane computes ll for one column (if valid)
            float ll = -FLT_MAX;
            if (lane_id < tile_k) {
                float d1 = L_buf[(warp_row_start + r) * BK + lane_id];
                float d2 = dot2_buf[(warp_row_start + r) * BK + lane_id];
                ll = (d1 - 2.0f * d2 + qm_buf[lane_id]) * (-0.5f) + lc_buf[lane_id];
            }

            // Warp-reduce: combine BK ll values into one (max, sum) pair
            // Step 1: find max across all lanes
            float tile_max = ll;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
            tile_max = __shfl_sync(0xffffffff, tile_max, 0);  // broadcast

            // Step 2: sum exp(ll - tile_max) across lanes
            float tile_exp = (lane_id < tile_k) ? __expf(ll - tile_max) : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                tile_exp += __shfl_down_sync(0xffffffff, tile_exp, offset);

            // Lane 0: update running logsumexp with this tile's (max, sum)
            if (lane_id == 0) {
                if (tile_max > row_max[r]) {
                    row_sum[r] = row_sum[r] * __expf(row_max[r] - tile_max) + tile_exp;
                    row_max[r] = tile_max;
                } else {
                    row_sum[r] += tile_exp * __expf(tile_max - row_max[r]);
                }
            }
        }
    }

    // ---- Write log_normalizer ----
    if (lane_id == 0) {
        for (int r = 0; r < WMMA_M; r++) {
            int global_row = block_start + warp_row_start + r;
            if (global_row < N) {
                log_normalizer[global_row] = row_max[r] + logf(row_sum[r]);
            }
        }
    }
}


// ============================================================================
// Kernel 2: WMMA Flash Accumulate Stats
//
// Same structure but Pass 2: recompute γ, accumulate n_k, s_k, sq_k
// Each block handles BN points × one K-tile (grid.y splits K)
// ============================================================================

template <int D>
__global__ void wmma_flash_accum_stats_v2(
    const float* __restrict__ X,
    const float* __restrict__ X_sq,
    const float* __restrict__ inv_var,
    const float* __restrict__ mu_iv,
    const float* __restrict__ quad_mu_g,
    const float* __restrict__ log_coeff_g,
    const float* __restrict__ log_normalizer,
    float* __restrict__ global_n_k,
    float* __restrict__ global_s_k,
    float* __restrict__ global_sq_k,
    int N, int K
) {
    extern __shared__ char smem[];

    __nv_bfloat16* x_bf   = (__nv_bfloat16*)smem;
    __nv_bfloat16* xsq_bf = x_bf + BN * D;
    __nv_bfloat16* iv_bf  = xsq_bf + BN * D;
    __nv_bfloat16* mi_bf  = iv_bf + BK * D;
    float* qm_buf  = (float*)(mi_bf + BK * D);
    float* lc_buf  = qm_buf + BK;
    float* L_buf   = lc_buf + BK;
    float* dot2_tmp = L_buf + BN * BK;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int block_start_n = blockIdx.x * BN;
    const int k_start = blockIdx.y * BK;
    int tile_k = min(BK, K - k_start);
    if (tile_k <= 0) return;

    // Load X, X²
    for (int idx = tid; idx < BN * D; idx += BLOCK_THREADS) {
        int r = idx / D, c = idx % D;
        int n = block_start_n + r;
        x_bf[r * D + c] = f2bf((n < N) ? X[n * D + c] : 0.0f);
        xsq_bf[r * D + c] = f2bf((n < N) ? X_sq[n * D + c] : 0.0f);
    }

    // Load centroid tile
    for (int idx = tid; idx < BK * D; idx += BLOCK_THREADS) {
        int kk = idx / D, c = idx % D;
        iv_bf[kk * D + c] = f2bf((k_start + kk < K) ? inv_var[(k_start + kk) * D + c] : 0.0f);
        mi_bf[kk * D + c] = f2bf((k_start + kk < K) ? mu_iv[(k_start + kk) * D + c] : 0.0f);
    }
    for (int idx = tid; idx < BK; idx += BLOCK_THREADS) {
        qm_buf[idx] = (k_start + idx < K) ? quad_mu_g[k_start + idx] : 0.0f;
        lc_buf[idx] = (k_start + idx < K) ? log_coeff_g[k_start + idx] : -FLT_MAX;
    }
    __syncthreads();

    const int warp_row_start = warp_id * WMMA_M;

    // WMMA distance
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc1, acc2;
    wmma::fill_fragment(acc1, 0.0f);
    wmma::fill_fragment(acc2, 0.0f);

    #pragma unroll
    for (int kk = 0; kk < D; kk += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> a1, a2;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::col_major> b1, b2;

        wmma::load_matrix_sync(a1, xsq_bf + warp_row_start * D + kk, D);
        wmma::load_matrix_sync(b1, iv_bf + kk, D);
        wmma::mma_sync(acc1, a1, b1, acc1);

        wmma::load_matrix_sync(a2, x_bf + warp_row_start * D + kk, D);
        wmma::load_matrix_sync(b2, mi_bf + kk, D);
        wmma::mma_sync(acc2, a2, b2, acc2);
    }

    wmma::store_matrix_sync(L_buf + warp_row_start * BK, acc1, BK, wmma::mem_row_major);
    wmma::store_matrix_sync(dot2_tmp + warp_row_start * BK, acc2, BK, wmma::mem_row_major);
    __syncthreads();

    // Compute γ and accumulate stats
    // Per-warp: local accumulators for n_k, s_k, sq_k
    float local_nk[BK];
    for (int i = 0; i < BK; i++) local_nk[i] = 0.0f;

    for (int r = 0; r < WMMA_M; r++) {
        int n = block_start_n + warp_row_start + r;
        if (n >= N) continue;

        float ln_val = log_normalizer[n];

        for (int c = lane_id; c < tile_k; c += 32) {
            float d1 = L_buf[(warp_row_start + r) * BK + c];
            float d2 = dot2_tmp[(warp_row_start + r) * BK + c];
            float ll = (d1 - 2.0f * d2 + qm_buf[c]) * (-0.5f) + lc_buf[c];
            float gamma = __expf(ll - ln_val);

            // Accumulate stats
            atomicAdd(&global_n_k[k_start + c], gamma);
            for (int j = 0; j < D; j++) {
                float x_val = X[n * D + j];
                atomicAdd(&global_s_k[(k_start + c) * D + j], gamma * x_val);
                atomicAdd(&global_sq_k[(k_start + c) * D + j], gamma * x_val * x_val);
            }
        }
    }
}


// ============================================================================
// Host launcher
// ============================================================================

std::vector<torch::Tensor> wmma_flash_em(
    torch::Tensor X, torch::Tensor mu, torch::Tensor var, torch::Tensor log_pi
) {
    int N = X.size(0), D = X.size(1), K = mu.size(0);
    TORCH_CHECK(D == 64 || D == 128, "WMMA kernel only supports d=64 or d=128");

    auto X_sq = X * X;
    auto inv_var = 1.0f / var;
    auto mu_iv = mu * inv_var;
    auto quad_mu = (mu * mu_iv).sum(1);
    auto log_det = var.log().sum(1);
    auto log_coeff = log_pi - 0.5f * (D * 1.8378770664093453f + log_det);

    auto log_normalizer = torch::empty({N}, X.options());
    auto n_k  = torch::zeros({K}, X.options());
    auto s_k  = torch::zeros({K, D}, X.options());
    auto sq_k = torch::zeros({K, D}, X.options());

    // Shared memory size
    size_t smem_bytes = (BN * D + BN * D) * sizeof(__nv_bfloat16)  // x, xsq
                      + (BK * D + BK * D) * sizeof(__nv_bfloat16)  // iv, mi
                      + (BK + BK) * sizeof(float)                  // qm, lc
                      + (BN * BK + BN * BK) * sizeof(float);       // L_buf, dot2_buf

    int grid_n = (N + BN - 1) / BN;

    // Pass 1: logsumexp
    if (D == 64) {
        cudaFuncSetAttribute(wmma_flash_logsumexp_v2<64>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        wmma_flash_logsumexp_v2<64><<<grid_n, BLOCK_THREADS, smem_bytes>>>(
            X.data_ptr<float>(), X_sq.data_ptr<float>(),
            inv_var.data_ptr<float>(), mu_iv.data_ptr<float>(),
            quad_mu.data_ptr<float>(), log_coeff.data_ptr<float>(),
            log_normalizer.data_ptr<float>(), N, K);
    } else {
        cudaFuncSetAttribute(wmma_flash_logsumexp_v2<128>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        wmma_flash_logsumexp_v2<128><<<grid_n, BLOCK_THREADS, smem_bytes>>>(
            X.data_ptr<float>(), X_sq.data_ptr<float>(),
            inv_var.data_ptr<float>(), mu_iv.data_ptr<float>(),
            quad_mu.data_ptr<float>(), log_coeff.data_ptr<float>(),
            log_normalizer.data_ptr<float>(), N, K);
    }

    // Pass 2: accumulate stats
    dim3 grid2(grid_n, (K + BK - 1) / BK);
    if (D == 64) {
        cudaFuncSetAttribute(wmma_flash_accum_stats_v2<64>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        wmma_flash_accum_stats_v2<64><<<grid2, BLOCK_THREADS, smem_bytes>>>(
            X.data_ptr<float>(), X_sq.data_ptr<float>(),
            inv_var.data_ptr<float>(), mu_iv.data_ptr<float>(),
            quad_mu.data_ptr<float>(), log_coeff.data_ptr<float>(),
            log_normalizer.data_ptr<float>(),
            n_k.data_ptr<float>(), s_k.data_ptr<float>(), sq_k.data_ptr<float>(),
            N, K);
    } else {
        cudaFuncSetAttribute(wmma_flash_accum_stats_v2<128>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        wmma_flash_accum_stats_v2<128><<<grid2, BLOCK_THREADS, smem_bytes>>>(
            X.data_ptr<float>(), X_sq.data_ptr<float>(),
            inv_var.data_ptr<float>(), mu_iv.data_ptr<float>(),
            quad_mu.data_ptr<float>(), log_coeff.data_ptr<float>(),
            log_normalizer.data_ptr<float>(),
            n_k.data_ptr<float>(), s_k.data_ptr<float>(), sq_k.data_ptr<float>(),
            N, K);
    }

    // Parameter update
    auto inv_nk = 1.0f / n_k.clamp_min(1e-8f);
    auto new_mu = s_k * inv_nk.unsqueeze(1);
    auto new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp_min(1e-6f);
    auto new_log_pi = (n_k / N).clamp_min(1e-8f).log();

    return {new_mu, new_var, new_log_pi, log_normalizer};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wmma_flash_em", &wmma_flash_em, "WMMA Flash EM step");
}

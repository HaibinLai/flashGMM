/*
 * Flash-GMM: CUDA Tensor Core Kernels (WMMA)
 *
 * Equivalent to Triton flash_gmm_v3.py but in CUDA, using WMMA API
 * to access Tensor Cores for the distance GEMM.
 *
 * Key difference vs flash_gmm_cuda.cu (v2):
 *   v2: scalar FMA per lane → CUDA Core (19.5 TFLOPS)
 *   TC: wmma::mma_sync → Tensor Core (312 TFLOPS BF16)
 *
 * Kernel design (matching Triton):
 *   - Block handles BN data points × ALL K centroids (streaming)
 *   - 4 warps per block, each warp handles 16 rows of BN
 *   - Distance = X²@inv_var^T - 2·X@mu_iv^T + quad_mu (two WMMAs)
 *   - Online logsumexp in registers (no N×K materialization)
 *
 * WMMA tile: m16n16k16 with __nv_bfloat16 inputs, float accumulator
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

// ============================================================================
// Constants
// ============================================================================
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// Block config: 4 warps = 128 threads, BN=64 (4 warps × 16 rows)
static constexpr int NUM_WARPS = 4;
static constexpr int BLOCK_THREADS = NUM_WARPS * 32;  // 128
static constexpr int BN = NUM_WARPS * WMMA_M;         // 64
static constexpr int BK = WMMA_N;                      // 16

// ============================================================================
// Helper: convert float → bf16 during shared memory load
// ============================================================================
__device__ __forceinline__ __nv_bfloat16 f2bf(float x) {
    return __float2bfloat16(x);
}

// ============================================================================
// Kernel 1: WMMA Flash Log-Normalizer
//
// Grid:  ceil(N / BN)
// Block: 128 threads (4 warps)
//
// Each warp owns a 16×BK WMMA output tile.
// For each K-tile (BK centroids):
//   1. Cooperatively load centroid tile to shared memory (float → bf16)
//   2. WMMA: dist_tile = X²_bf16 @ inv_var_bf16^T  (16×d @ d×BK)
//   3. WMMA: dist_tile -= 2 * X_bf16 @ mu_iv_bf16^T
//   4. dist_tile = dist_tile * (-0.5) + log_coeff + quad_mu*(-0.5)
//   5. Online logsumexp update per row
//
// Shared memory layout:
//   X_bf16[BN][d], X²_bf16[BN][d],   ← loaded once
//   iv_bf16[BK][d], mi_bf16[BK][d],  ← reloaded per K-tile
//   quad_mu[BK], log_coeff[BK],
//   running_max[BN], running_sum[BN]
// ============================================================================

template <int D>  // compile-time dimension
__global__ void wmma_flash_logsumexp_kernel(
    const float* __restrict__ X,        // [N, D]
    const float* __restrict__ X_sq,     // [N, D]
    const float* __restrict__ inv_var,  // [K, D]
    const float* __restrict__ mu_iv,    // [K, D]
    const float* __restrict__ quad_mu,  // [K]
    const float* __restrict__ log_coeff,// [K]
    float* __restrict__ log_normalizer, // [N]
    int N, int K
) {
    // Shared memory
    extern __shared__ char smem_raw[];
    // Layout: x_bf16[BN*D] + xsq_bf16[BN*D] + iv_bf16[BK*D] + mi_bf16[BK*D]
    //       + qm[BK] + lc[BK] + rmax[BN] + rsum[BN]
    __nv_bfloat16* x_smem   = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* xsq_smem = x_smem + BN * D;
    __nv_bfloat16* iv_smem  = xsq_smem + BN * D;
    __nv_bfloat16* mi_smem  = iv_smem + BK * D;
    float* qm_smem = (float*)(mi_smem + BK * D);
    float* lc_smem = qm_smem + BK;
    float* rmax    = lc_smem + BK;
    float* rsum    = rmax + BN;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int block_start = blockIdx.x * BN;
    const int BN_actual = min(BN, N - block_start);

    // ---- Load X and X² to shared memory (float → bf16) ----
    // Each thread loads multiple elements
    for (int idx = tid; idx < BN * D; idx += BLOCK_THREADS) {
        int row = idx / D, col = idx % D;
        int n = block_start + row;
        if (n < N) {
            x_smem[row * D + col] = f2bf(X[n * D + col]);
            xsq_smem[row * D + col] = f2bf(X_sq[n * D + col]);
        } else {
            x_smem[row * D + col] = f2bf(0.0f);
            xsq_smem[row * D + col] = f2bf(0.0f);
        }
    }

    // Init logsumexp state
    for (int i = tid; i < BN; i += BLOCK_THREADS) {
        rmax[i] = -FLT_MAX;
        rsum[i] = 0.0f;
    }
    __syncthreads();

    // ---- Stream centroid tiles ----
    // Each warp handles rows [warp_id*16, (warp_id+1)*16) of BN
    const int row_start = warp_id * WMMA_M;  // 0, 16, 32, 48

    for (int t = 0; t < K; t += BK) {
        int tile_k = min(BK, K - t);

        // Cooperatively load centroid tile: inv_var, mu_iv, quad_mu, log_coeff
        __syncthreads();
        for (int idx = tid; idx < BK * D; idx += BLOCK_THREADS) {
            int kk = idx / D, col = idx % D;
            if (t + kk < K) {
                iv_smem[kk * D + col] = f2bf(inv_var[(t + kk) * D + col]);
                mi_smem[kk * D + col] = f2bf(mu_iv[(t + kk) * D + col]);
            } else {
                iv_smem[kk * D + col] = f2bf(0.0f);
                mi_smem[kk * D + col] = f2bf(0.0f);
            }
        }
        for (int idx = tid; idx < BK; idx += BLOCK_THREADS) {
            qm_smem[idx] = (t + idx < K) ? quad_mu[t + idx] : 0.0f;
            lc_smem[idx] = (t + idx < K) ? log_coeff[t + idx] : -FLT_MAX;
        }
        __syncthreads();

        // ---- WMMA: dist = X² @ inv_var^T ----
        // A = xsq_smem[row_start : row_start+16, :], shape 16×D, row_major
        // B = iv_smem[0:BK, :]^T, i.e. we want iv_smem as col_major: shape D×BK
        //   but iv_smem is stored row_major [BK, D], which is col_major [D, BK]^T
        //   So B in col_major = iv_smem interpreted as col_major
        // C = accumulator 16×16

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dist_frag;
        wmma::fill_fragment(dist_frag, 0.0f);

        // Loop over D in chunks of 16
        #pragma unroll
        for (int kk = 0; kk < D; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::col_major> b_frag;

            // Load A: 16 rows of X², 16 columns starting at kk
            // A is stored row_major with leading dimension D
            wmma::load_matrix_sync(a_frag, xsq_smem + row_start * D + kk, D);

            // Load B: inv_var tile, we want [kk:kk+16, 0:BK]
            // iv_smem is [BK, D] row_major. We want columns kk..kk+15 of BK rows.
            // In col_major view: B[k][n] = iv_smem[n * D + k]
            // So load with leading dim = D
            wmma::load_matrix_sync(b_frag, iv_smem + kk, D);

            wmma::mma_sync(dist_frag, a_frag, b_frag, dist_frag);
        }

        // ---- WMMA: dist -= 2 * X @ mu_iv^T ----
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dot2_frag;
        wmma::fill_fragment(dot2_frag, 0.0f);

        #pragma unroll
        for (int kk = 0; kk < D; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::col_major> b_frag;

            wmma::load_matrix_sync(a_frag, x_smem + row_start * D + kk, D);
            wmma::load_matrix_sync(b_frag, mi_smem + kk, D);

            wmma::mma_sync(dot2_frag, a_frag, b_frag, dot2_frag);
        }

        // ---- Combine: L = (dist - 2*dot2 + quad_mu) * (-0.5) + log_coeff ----
        // Store dist_frag and dot2_frag to shared memory to extract per-element
        // (WMMA fragments don't support direct element access portably)
        __shared__ float dist_buf[BN * BK];  // reuse-safe since BN*BK ≤ shared budget
        __shared__ float dot2_buf[BN * BK];

        wmma::store_matrix_sync(dist_buf + row_start * BK, dist_frag, BK, wmma::mem_row_major);
        wmma::store_matrix_sync(dot2_buf + row_start * BK, dot2_frag, BK, wmma::mem_row_major);
        __syncthreads();

        // ---- Online logsumexp update ----
        // Each thread handles several (row, col) pairs
        for (int row = warp_id * WMMA_M; row < warp_id * WMMA_M + WMMA_M; row++) {
            if (row >= BN_actual) continue;

            for (int col = lane_id; col < tile_k; col += 32) {
                float d_val = dist_buf[row * BK + col];
                float d2_val = dot2_buf[row * BK + col];
                float ll = (d_val - 2.0f * d2_val + qm_smem[col]) * (-0.5f) + lc_smem[col];

                // Atomic-free online logsumexp (each thread does independent row)
                // But multiple threads may handle same row → need reduction
                // For simplicity in v1: serialize per row with atomics on shared mem
                float old_max = atomicMax_float(&rmax[row], ll);
                float actual_max = fmaxf(old_max, ll);
                if (ll >= actual_max) {
                    // This is the new max
                    atomicAdd(&rsum[row], 1.0f);
                } else {
                    atomicAdd(&rsum[row], __expf(ll - actual_max));
                }
            }
        }
        __syncthreads();
    }

    // ---- Write log_normalizer ----
    for (int i = tid; i < BN_actual; i += BLOCK_THREADS) {
        log_normalizer[block_start + i] = rmax[i] + logf(rsum[i]);
    }
}

// atomicMax for float (not natively supported)
__device__ __forceinline__ float atomicMax_float(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

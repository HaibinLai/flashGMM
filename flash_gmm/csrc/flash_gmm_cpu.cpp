/*
 * Flash-GMM: C++ CPU Implementation
 *
 * Implements three variants of GMM E-step / EM iteration:
 *   1. standard_e_step:  Materializes full N×K log-likelihood matrix L (baseline)
 *   2. flash_e_step:     Online log-sum-exp, eliminates L materialization (2-pass)
 *   3. flash_em_fused:   E+M fully fused, eliminates ALL N×K matrices (2-pass)
 *
 * Uses raw C++ loops to demonstrate the tiling algorithm.
 * ATen tensors used only at the interface boundary.
 */

#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <float.h>
#include <omp.h>

static constexpr float LOG_2PI = 1.8378770664093453f;

// Get thread count from PyTorch's setting (respects torch.set_num_threads)
static inline int get_num_threads() {
    return at::get_num_threads();
}

// ============================================================================
// Helper: compute log-likelihood for a single (point, centroid) pair
// log N(x|μ,σ²) = log_pi - 0.5*(d*log(2π) + Σ log(σ²_j) + Σ (x_j-μ_j)²/σ²_j)
// ============================================================================
static inline float compute_log_likelihood(
    const float* x,       // point: (d,)
    const float* mu,      // centroid mean: (d,)
    const float* var,     // centroid diagonal variance: (d,)
    float log_pi,         // log mixing weight
    int d
) {
    float log_det = 0.0f;
    float mahal = 0.0f;
    for (int j = 0; j < d; j++) {
        log_det += logf(var[j]);
        float diff = x[j] - mu[j];
        mahal += diff * diff / var[j];
    }
    return log_pi - 0.5f * (d * LOG_2PI + log_det + mahal);
}

// ============================================================================
// 1. Standard E-step: materializes L ∈ R^{N×K} then normalizes
//    IO model: write L (NK) + read L (NK) + write γ (NK) = 3×Θ(NK)
// ============================================================================
std::vector<torch::Tensor> standard_e_step_cpu(
    torch::Tensor X,        // (N, d)
    torch::Tensor mu,       // (K, d)
    torch::Tensor var,      // (K, d)
    torch::Tensor log_pi    // (K,)
) {
    int N = X.size(0);
    int d = X.size(1);
    int K = mu.size(0);

    auto L = torch::empty({N, K}, X.options());
    auto gamma = torch::empty({N, K}, X.options());
    auto log_normalizer = torch::empty({N}, X.options());

    const float* X_ptr = X.data_ptr<float>();
    const float* mu_ptr = mu.data_ptr<float>();
    const float* var_ptr = var.data_ptr<float>();
    const float* lp_ptr = log_pi.data_ptr<float>();
    float* L_ptr = L.data_ptr<float>();
    float* g_ptr = gamma.data_ptr<float>();
    float* ln_ptr = log_normalizer.data_ptr<float>();

    // Step 1: Compute and MATERIALIZE full L matrix
    #pragma omp parallel for num_threads(get_num_threads()) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            L_ptr[n * K + k] = compute_log_likelihood(
                X_ptr + n * d, mu_ptr + k * d, var_ptr + k * d, lp_ptr[k], d);
        }
    }

    // Step 2: Row-wise log-sum-exp on L (reads L from memory again)
    #pragma omp parallel for num_threads(get_num_threads()) schedule(static)
    for (int n = 0; n < N; n++) {
        float max_val = -FLT_MAX;
        for (int k = 0; k < K; k++) {
            max_val = std::max(max_val, L_ptr[n * K + k]);
        }
        float sum_exp = 0.0f;
        for (int k = 0; k < K; k++) {
            sum_exp += expf(L_ptr[n * K + k] - max_val);
        }
        ln_ptr[n] = max_val + logf(sum_exp);
        for (int k = 0; k < K; k++) {
            g_ptr[n * K + k] = expf(L_ptr[n * K + k] - ln_ptr[n]);
        }
    }

    return {gamma, log_normalizer};
}

// ============================================================================
// 2. Flash E-step: 2-pass, no L materialization
//    Pass 1: online log-sum-exp → log_normalizer (O(N) output)
//    Pass 2: recompute log-likelihoods, normalize → γ
//    IO model: eliminates 2×Θ(NK) from L, but still writes γ (NK)
// ============================================================================
std::vector<torch::Tensor> flash_e_step_cpu(
    torch::Tensor X,
    torch::Tensor mu,
    torch::Tensor var,
    torch::Tensor log_pi,
    int BK   // centroid tile size
) {
    int N = X.size(0);
    int d = X.size(1);
    int K = mu.size(0);

    auto gamma = torch::empty({N, K}, X.options());
    auto log_normalizer = torch::empty({N}, X.options());

    const float* X_ptr = X.data_ptr<float>();
    const float* mu_ptr = mu.data_ptr<float>();
    const float* var_ptr = var.data_ptr<float>();
    const float* lp_ptr = log_pi.data_ptr<float>();
    float* g_ptr = gamma.data_ptr<float>();
    float* ln_ptr = log_normalizer.data_ptr<float>();

    // ---- PASS 1: Online log-sum-exp (no L materialization) ----
    #pragma omp parallel for num_threads(get_num_threads()) schedule(static)
    for (int n = 0; n < N; n++) {
        float running_max = -FLT_MAX;
        float running_sum_exp = 0.0f;

        // Stream centroid tiles
        for (int t = 0; t < K; t += BK) {
            int tile_end = std::min(t + BK, K);
            for (int k = t; k < tile_end; k++) {
                float ll = compute_log_likelihood(
                    X_ptr + n * d, mu_ptr + k * d, var_ptr + k * d, lp_ptr[k], d);

                // Online log-sum-exp update (same as FlashAttention's online softmax)
                if (ll > running_max) {
                    running_sum_exp = running_sum_exp * expf(running_max - ll) + 1.0f;
                    running_max = ll;
                } else {
                    running_sum_exp += expf(ll - running_max);
                }
            }
        }
        ln_ptr[n] = running_max + logf(running_sum_exp);
    }

    // ---- PASS 2: Recompute + normalize → γ (2× FLOPs, 0× L HBM) ----
    #pragma omp parallel for num_threads(get_num_threads()) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < K; t += BK) {
            int tile_end = std::min(t + BK, K);
            for (int k = t; k < tile_end; k++) {
                float ll = compute_log_likelihood(
                    X_ptr + n * d, mu_ptr + k * d, var_ptr + k * d, lp_ptr[k], d);
                g_ptr[n * K + k] = expf(ll - ln_ptr[n]);
            }
        }
    }

    return {gamma, log_normalizer};
}

// ============================================================================
// 3. Flash E+M Fused: 2-pass, ZERO N×K materialization
//    Pass 1: online log-sum-exp → log_normalizer (O(N))
//    Pass 2: recompute γ on-the-fly, accumulate sufficient statistics
//    Output: n_k (K), s_k (K,d), sq_k (K,d) — only O(Kd)
//    Then: μ = s_k/n_k, σ² = sq_k/n_k - μ², π = n_k/N
// ============================================================================
std::vector<torch::Tensor> flash_em_fused_cpu(
    torch::Tensor X,
    torch::Tensor mu,
    torch::Tensor var,
    torch::Tensor log_pi,
    int BK
) {
    int N = X.size(0);
    int d = X.size(1);
    int K = mu.size(0);

    auto log_normalizer = torch::empty({N}, X.options());
    auto new_mu = torch::empty({K, d}, X.options());
    auto new_var = torch::empty({K, d}, X.options());
    auto new_log_pi = torch::empty({K}, X.options());

    const float* X_ptr = X.data_ptr<float>();
    const float* mu_ptr = mu.data_ptr<float>();
    const float* var_ptr = var.data_ptr<float>();
    const float* lp_ptr = log_pi.data_ptr<float>();
    float* ln_ptr = log_normalizer.data_ptr<float>();

    // ---- PASS 1: Online log-sum-exp → log_normalizer ----
    #pragma omp parallel for num_threads(get_num_threads()) schedule(static)
    for (int n = 0; n < N; n++) {
        float running_max = -FLT_MAX;
        float running_sum_exp = 0.0f;

        for (int t = 0; t < K; t += BK) {
            int tile_end = std::min(t + BK, K);
            for (int k = t; k < tile_end; k++) {
                float ll = compute_log_likelihood(
                    X_ptr + n * d, mu_ptr + k * d, var_ptr + k * d, lp_ptr[k], d);
                if (ll > running_max) {
                    running_sum_exp = running_sum_exp * expf(running_max - ll) + 1.0f;
                    running_max = ll;
                } else {
                    running_sum_exp += expf(ll - running_max);
                }
            }
        }
        ln_ptr[n] = running_max + logf(running_sum_exp);
    }

    // ---- PASS 2: Recompute γ on-the-fly, accumulate M-step statistics ----
    // Per-thread local accumulators to avoid contention, then reduce
    int num_threads = get_num_threads();
    // Allocate per-thread buffers: n_k(K) + s_k(K*d) + sq_k(K*d) per thread
    std::vector<float> all_n_k(num_threads * K, 0.0f);
    std::vector<float> all_s_k(num_threads * K * d, 0.0f);
    std::vector<float> all_sq_k(num_threads * K * d, 0.0f);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        float* local_n_k = all_n_k.data() + tid * K;
        float* local_s_k = all_s_k.data() + tid * K * d;
        float* local_sq_k = all_sq_k.data() + tid * K * d;

        #pragma omp for schedule(static)
        for (int n = 0; n < N; n++) {
            for (int t = 0; t < K; t += BK) {
                int tile_end = std::min(t + BK, K);
                for (int k = t; k < tile_end; k++) {
                    float ll = compute_log_likelihood(
                        X_ptr + n * d, mu_ptr + k * d, var_ptr + k * d, lp_ptr[k], d);
                    float gamma_nk = expf(ll - ln_ptr[n]);

                    local_n_k[k] += gamma_nk;
                    for (int j = 0; j < d; j++) {
                        float x_val = X_ptr[n * d + j];
                        local_s_k[k * d + j] += gamma_nk * x_val;
                        local_sq_k[k * d + j] += gamma_nk * x_val * x_val;
                    }
                }
            }
        }
    }

    // Reduce per-thread accumulators
    std::vector<float> n_k(K, 0.0f);
    std::vector<float> s_k(K * d, 0.0f);
    std::vector<float> sq_k(K * d, 0.0f);
    for (int t = 0; t < num_threads; t++) {
        for (int k = 0; k < K; k++) {
            n_k[k] += all_n_k[t * K + k];
        }
        for (int i = 0; i < K * d; i++) {
            s_k[i] += all_s_k[t * K * d + i];
            sq_k[i] += all_sq_k[t * K * d + i];
        }
    }

    // ---- Update parameters from sufficient statistics ----
    float* new_mu_ptr = new_mu.data_ptr<float>();
    float* new_var_ptr = new_var.data_ptr<float>();
    float* new_lp_ptr = new_log_pi.data_ptr<float>();

    for (int k = 0; k < K; k++) {
        float inv_nk = 1.0f / std::max(n_k[k], 1e-8f);
        for (int j = 0; j < d; j++) {
            float mu_kj = s_k[k * d + j] * inv_nk;
            new_mu_ptr[k * d + j] = mu_kj;
            float var_kj = sq_k[k * d + j] * inv_nk - mu_kj * mu_kj;
            new_var_ptr[k * d + j] = std::max(var_kj, 1e-6f);
        }
        new_lp_ptr[k] = logf(std::max(n_k[k] / N, 1e-8f));
    }

    return {new_mu, new_var, new_log_pi, log_normalizer};
}

// ============================================================================
// Standard M-step (for use with standard/flash E-step)
// ============================================================================
std::vector<torch::Tensor> standard_m_step_cpu(
    torch::Tensor X,       // (N, d)
    torch::Tensor gamma    // (N, K)
) {
    int N = X.size(0);
    int d = X.size(1);
    int K = gamma.size(1);

    // n_k = γ.sum(dim=0)
    auto n_k = gamma.sum(0);
    // mu = γ^T X / n_k
    auto new_mu = gamma.t().mm(X) / n_k.unsqueeze(1);
    // var = γ^T (X²) / n_k - mu²
    auto X_sq = X * X;
    auto new_var = (gamma.t().mm(X_sq) / n_k.unsqueeze(1) - new_mu * new_mu).clamp_min(1e-6);
    // log_pi = log(n_k / N)
    auto new_log_pi = (n_k / N).log();

    return {new_mu, new_var, new_log_pi};
}

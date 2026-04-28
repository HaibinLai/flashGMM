/*
 * Flash-GMM PyTorch Extension: pybind11 bindings
 *
 * Exposes C++ CPU and CUDA GPU implementations to Python.
 * Dispatch based on tensor device.
 */

#include <torch/extension.h>
#include <vector>

// ---- CPU declarations (flash_gmm_cpu.cpp) ----
std::vector<torch::Tensor> standard_e_step_cpu(
    torch::Tensor X, torch::Tensor mu, torch::Tensor var, torch::Tensor log_pi);

std::vector<torch::Tensor> flash_e_step_cpu(
    torch::Tensor X, torch::Tensor mu, torch::Tensor var, torch::Tensor log_pi, int BK);

std::vector<torch::Tensor> flash_em_fused_cpu(
    torch::Tensor X, torch::Tensor mu, torch::Tensor var, torch::Tensor log_pi, int BK);

std::vector<torch::Tensor> standard_m_step_cpu(
    torch::Tensor X, torch::Tensor gamma);

// ---- CUDA declarations (flash_gmm_cuda.cu) ----
#ifdef WITH_CUDA
std::vector<torch::Tensor> standard_e_step_cuda(
    torch::Tensor X, torch::Tensor mu, torch::Tensor var, torch::Tensor log_pi);

std::vector<torch::Tensor> flash_em_fused_cuda(
    torch::Tensor X, torch::Tensor mu, torch::Tensor var, torch::Tensor log_pi,
    int BN, int BK);
#endif

// ---- Dispatch functions ----
std::vector<torch::Tensor> standard_e_step(
    torch::Tensor X, torch::Tensor mu, torch::Tensor var, torch::Tensor log_pi
) {
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(X.dim() == 2, "X must be 2D");

    if (X.is_cuda()) {
#ifdef WITH_CUDA
        return standard_e_step_cuda(X, mu, var, log_pi);
#else
        TORCH_CHECK(false, "CUDA not compiled");
#endif
    }
    return standard_e_step_cpu(X, mu, var, log_pi);
}

std::vector<torch::Tensor> flash_e_step(
    torch::Tensor X, torch::Tensor mu, torch::Tensor var, torch::Tensor log_pi,
    int BK
) {
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be float32");
    if (X.is_cuda()) {
        TORCH_CHECK(false, "flash_e_step CUDA: use flash_em_fused instead");
    }
    return flash_e_step_cpu(X, mu, var, log_pi, BK);
}

std::vector<torch::Tensor> flash_em_fused(
    torch::Tensor X, torch::Tensor mu, torch::Tensor var, torch::Tensor log_pi,
    int BN, int BK
) {
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be float32");

    if (X.is_cuda()) {
#ifdef WITH_CUDA
        return flash_em_fused_cuda(X, mu, var, log_pi, BN, BK);
#else
        TORCH_CHECK(false, "CUDA not compiled");
#endif
    }
    return flash_em_fused_cpu(X, mu, var, log_pi, BK);
}

std::vector<torch::Tensor> standard_m_step(
    torch::Tensor X, torch::Tensor gamma
) {
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be float32");
    if (X.is_cuda()) {
        // M-step is just GEMMs — use ATen on GPU directly
        auto n_k = gamma.sum(0);
        auto new_mu = gamma.t().mm(X) / n_k.unsqueeze(1);
        auto X_sq = X * X;
        auto new_var = (gamma.t().mm(X_sq) / n_k.unsqueeze(1) - new_mu * new_mu).clamp_min(1e-6);
        auto new_log_pi = (n_k / X.size(0)).log();
        return {new_mu, new_var, new_log_pi};
    }
    return standard_m_step_cpu(X, gamma);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Flash-GMM: IO-aware GMM with C++/CUDA kernels";
    m.def("standard_e_step", &standard_e_step,
          "Standard E-step (materializes N×K log-likelihood matrix)",
          py::arg("X"), py::arg("mu"), py::arg("var"), py::arg("log_pi"));
    m.def("flash_e_step", &flash_e_step,
          "Flash E-step (online log-sum-exp, no L materialization)",
          py::arg("X"), py::arg("mu"), py::arg("var"), py::arg("log_pi"), py::arg("BK") = 4);
    m.def("flash_em_fused", &flash_em_fused,
          "Flash E+M fused (ZERO N×K materialization, returns new params)",
          py::arg("X"), py::arg("mu"), py::arg("var"), py::arg("log_pi"),
          py::arg("BN") = 64, py::arg("BK") = 8);
    m.def("standard_m_step", &standard_m_step,
          "Standard M-step from responsibilities",
          py::arg("X"), py::arg("gamma"));
}

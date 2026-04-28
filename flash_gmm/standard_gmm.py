"""
Standard GMM/EM implementation with explicit IO counting.

This is the baseline implementation that materializes all intermediate matrices
(log-likelihood L ∈ R^{N×K} and responsibility γ ∈ R^{N×K}) to simulate HBM traffic.
Supports diagonal covariance only.
"""

import torch
import time
from dataclasses import dataclass, field


@dataclass
class IOCounter:
    """Tracks simulated HBM read/write bytes per EM iteration."""
    reads: int = 0
    writes: int = 0
    details: dict = field(default_factory=dict)

    def record(self, name: str, tensor: torch.Tensor, op: str):
        nbytes = tensor.numel() * tensor.element_size()
        if op == "read":
            self.reads += nbytes
        else:
            self.writes += nbytes
        key = f"{op}:{name}"
        self.details[key] = self.details.get(key, 0) + nbytes

    def total(self):
        return self.reads + self.writes

    def summary(self):
        lines = [f"Total IO: {self.total() / 1e6:.2f} MB  (R: {self.reads / 1e6:.2f} MB, W: {self.writes / 1e6:.2f} MB)"]
        for k, v in sorted(self.details.items()):
            lines.append(f"  {k}: {v / 1e6:.2f} MB")
        return "\n".join(lines)

    def reset(self):
        self.reads = 0
        self.writes = 0
        self.details.clear()


class StandardGMM:
    """
    Standard GMM with diagonal covariance.

    Parameters:
        K: number of mixture components
        d: feature dimension
        max_iter: maximum EM iterations
        tol: convergence tolerance on log-likelihood
        seed: random seed for initialization
    """

    def __init__(self, K: int, d: int, max_iter: int = 100, tol: float = 1e-4, seed: int = 42):
        self.K = K
        self.d = d
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        # Parameters (initialized in fit)
        self.mu = None       # (K, d) means
        self.var = None      # (K, d) diagonal variances
        self.log_pi = None   # (K,) log mixing weights
        self.io_counter = IOCounter()

    def _init_params(self, X: torch.Tensor):
        """Initialize parameters using random subset + uniform weights."""
        N = X.shape[0]
        gen = torch.Generator().manual_seed(self.seed)
        idx = torch.randperm(N, generator=gen)[:self.K]
        self.mu = X[idx].clone()                          # (K, d)
        self.var = torch.ones(self.K, self.d) * X.var(dim=0, keepdim=True).mean()  # (K, d)
        self.log_pi = torch.full((self.K,), -torch.log(torch.tensor(float(self.K))))  # uniform

    def e_step(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        E-step: compute responsibilities γ ∈ R^{N×K}.

        Standard implementation that explicitly materializes:
          1. log-likelihood matrix L ∈ R^{N×K} → write to HBM
          2. reads L back for log-sum-exp normalization
          3. writes γ ∈ R^{N×K} → write to HBM

        Returns: (gamma, log_likelihood_per_sample)
        """
        io = self.io_counter
        N, d = X.shape
        K = self.K

        # --- Read inputs from HBM ---
        io.record("X", X, "read")
        io.record("mu", self.mu, "read")
        io.record("var", self.var, "read")
        io.record("log_pi", self.log_pi, "read")

        # --- Compute log-likelihood matrix L ∈ R^{N×K} ---
        # log N(x|μ,σ²) = -d/2 log(2π) - 1/2 Σ log(σ²_j) - 1/2 Σ (x_j - μ_j)² / σ²_j
        # Using broadcasting: X is (N,d), mu is (K,d), var is (K,d)
        log_det = self.var.log().sum(dim=1)  # (K,) = Σ_j log(σ²_{k,j})
        diff = X.unsqueeze(1) - self.mu.unsqueeze(0)  # (N, K, d)
        mahal = (diff ** 2 / self.var.unsqueeze(0)).sum(dim=2)  # (N, K)
        L = self.log_pi.unsqueeze(0) - 0.5 * (d * torch.log(torch.tensor(2 * torch.pi)) + log_det.unsqueeze(0) + mahal)
        # L shape: (N, K)

        # --- WRITE L to HBM (materialization!) ---
        io.record("L_matrix", L, "write")

        # --- READ L back for log-sum-exp ---
        io.record("L_matrix", L, "read")

        # --- Row-wise log-sum-exp ---
        log_sum = torch.logsumexp(L, dim=1, keepdim=True)  # (N, 1)
        log_gamma = L - log_sum
        gamma = log_gamma.exp()  # (N, K)

        # --- WRITE γ to HBM ---
        io.record("gamma", gamma, "write")

        # Per-sample log-likelihood
        log_ll = log_sum.squeeze(1)  # (N,)

        return gamma, log_ll

    def m_step(self, X: torch.Tensor, gamma: torch.Tensor):
        """
        M-step: update parameters using responsibilities.

        Standard implementation reads γ from HBM for weighted aggregation.
        """
        io = self.io_counter
        N = X.shape[0]

        # --- Read inputs from HBM ---
        io.record("gamma", gamma, "read")
        io.record("X", X, "read")

        # Effective counts
        n_k = gamma.sum(dim=0)  # (K,)

        # Update means: μ_k = Σ_n γ_nk x_n / n_k
        # This is a GEMM: (K, N) @ (N, d) = (K, d)
        self.mu = (gamma.T @ X) / n_k.unsqueeze(1)

        # Update variances: σ²_k = Σ_n γ_nk (x_n - μ_k)² / n_k
        diff = X.unsqueeze(1) - self.mu.unsqueeze(0)  # (N, K, d)
        self.var = (gamma.unsqueeze(2) * diff ** 2).sum(dim=0) / n_k.unsqueeze(1)
        # Clamp for numerical stability
        self.var = self.var.clamp(min=1e-6)

        # Update mixing weights
        self.log_pi = (n_k / N).log()

        # --- Write updated parameters to HBM ---
        io.record("mu_new", self.mu, "write")
        io.record("var_new", self.var, "write")
        io.record("log_pi_new", self.log_pi, "write")

    def fit(self, X: torch.Tensor, verbose: bool = False):
        """Run EM iterations until convergence."""
        self._init_params(X)
        prev_ll = -float("inf")
        history = []

        for it in range(self.max_iter):
            self.io_counter.reset()
            t0 = time.time()

            gamma, log_ll = self.e_step(X)
            avg_ll = log_ll.mean().item()

            self.m_step(X, gamma)
            elapsed = time.time() - t0

            history.append({
                "iter": it,
                "log_likelihood": avg_ll,
                "time_ms": elapsed * 1000,
                "io_bytes": self.io_counter.total(),
            })

            if verbose and (it % 10 == 0 or it < 5):
                print(f"  iter {it:3d}: LL={avg_ll:.4f}  time={elapsed*1000:.1f}ms  IO={self.io_counter.total()/1e6:.1f}MB")

            if abs(avg_ll - prev_ll) < self.tol:
                if verbose:
                    print(f"  Converged at iter {it}")
                break
            prev_ll = avg_ll

        return history

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return responsibilities γ for given data."""
        self.io_counter.reset()
        gamma, _ = self.e_step(X)
        return gamma

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return hard assignments."""
        return self.predict_proba(X).argmax(dim=1)


def generate_gmm_data(N: int, K: int, d: int, seed: int = 0):
    """Generate synthetic data from a known GMM for testing."""
    gen = torch.Generator().manual_seed(seed)
    # Random means spread out
    true_mu = torch.randn(K, d, generator=gen) * 5.0
    # Random diagonal variances
    true_var = (torch.rand(K, d, generator=gen) * 0.5 + 0.5)  # [0.5, 1.0]
    # Uniform mixing weights
    true_pi = torch.ones(K) / K

    # Sample data
    counts = torch.multinomial(true_pi, N, replacement=True, generator=gen)
    X_parts = []
    labels = []
    for k in range(K):
        mask = (counts == k)
        nk = mask.sum().item()
        if nk > 0:
            noise = torch.randn(nk, d, generator=gen) * true_var[k].sqrt()
            X_parts.append(true_mu[k] + noise)
            labels.extend([k] * nk)

    X = torch.cat(X_parts, dim=0)
    labels = torch.tensor(labels)

    # Shuffle
    perm = torch.randperm(N, generator=gen)
    X = X[perm]
    labels = labels[perm]

    return X, labels, true_mu, true_var, true_pi


if __name__ == "__main__":
    print("=== Standard GMM Baseline ===")
    N, K, d = 4096, 8, 32
    X, labels, _, _, _ = generate_gmm_data(N, K, d)
    print(f"Data: N={N}, K={K}, d={d}")

    model = StandardGMM(K=K, d=d, max_iter=50, tol=1e-4)
    history = model.fit(X, verbose=True)
    print(f"\nFinal IO per iteration: {history[-1]['io_bytes']/1e6:.2f} MB")
    print(f"IO breakdown (last iter):\n{model.io_counter.summary()}")

    # Theoretical IO
    elem_size = 4  # float32
    theoretical_standard = (2 * N * K + 2 * N * K + 2 * N * d + 2 * K * d) * elem_size
    print(f"\nTheoretical standard IO: {theoretical_standard/1e6:.2f} MB")

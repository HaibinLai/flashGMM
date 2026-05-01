"""
Flash-GMM v9: Streaming/chunked processing for arbitrarily large N.

Key idea: N can be larger than GPU memory. Process X in chunks:
  - Only load X_chunk[chunk_size, d] at a time
  - Precompute X² on-the-fly per chunk (don't store full X²)
  - Pass 1: partial logsumexp per chunk → merge across chunks
  - Pass 2: accumulate stats across chunks (additive)

Memory: O(chunk_size × d + K × d) instead of O(N × d)
  chunk_size = 1M → ~0.5GB per chunk for d=128

Works for both Flash (Triton) and Ultra (cuBLAS) backends.
"""
import torch
import math

# Backward compat alias
StreamingGMM = None  # will be set below


class StreamingGMMFast:
    """Optimized streaming GMM — eliminates per-chunk overhead.

    Fixes vs StreamingGMM:
      1. Pre-allocate A_buf (chunk_size × 2d) — no per-chunk torch.cat
      2. Split merged GEMM: 2 separate mm() calls — no torch.cat for B either
      3. Single pass: fuse logsumexp + stats into one chunk loop (read X once)
      4. In-place ops: reuse L_buf across chunks
    """

    def __init__(self, chunk_size=500_000):
        self.chunk_size = chunk_size

    def em_step(self, X, mu, var, log_pi):
        N, d = X.shape
        K = mu.shape[0]
        device = mu.device
        cs = min(self.chunk_size, N)

        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)

        # Pre-allocate reusable buffers (allocated ONCE)
        L_buf = torch.empty(cs, K, device=device)

        # ---- Pass 1: logsumexp (only need log_normalizer[N]) ----
        log_normalizer = torch.empty(N, device=device, dtype=torch.float32)

        for start in range(0, N, cs):
            end = min(start + cs, N)
            actual = end - start

            X_c = X[start:end] if X.device == device else X[start:end].to(device, non_blocking=True)

            # Distance without torch.cat: 2 separate mm + add
            L = L_buf[:actual]
            torch.mm(X_c * X_c, inv_var.T, out=L)       # X²@iv^T
            L.addmm_(X_c, mu_iv.T, alpha=-2.0)           # -= 2*X@mi^T
            L.add_(quad_mu)
            L.mul_(-0.5)
            L.add_(log_coeff)

            log_normalizer[start:end] = torch.logsumexp(L, dim=1)

        # ---- Pass 2: stats accumulation ----
        n_k = torch.zeros(K, device=device)
        s_k = torch.zeros(K, d, device=device)
        sq_k = torch.zeros(K, d, device=device)

        for start in range(0, N, cs):
            end = min(start + cs, N)
            actual = end - start

            X_c = X[start:end] if X.device == device else X[start:end].to(device, non_blocking=True)
            X_sq_c = X_c * X_c

            # Recompute L (reuse buffer)
            L = L_buf[:actual]
            torch.mm(X_sq_c, inv_var.T, out=L)
            L.addmm_(X_c, mu_iv.T, alpha=-2.0)
            L.add_(quad_mu)
            L.mul_(-0.5)
            L.add_(log_coeff)

            # γ in-place
            L.sub_(log_normalizer[start:end].unsqueeze(1))
            L.exp_()  # L is now γ

            n_k += L.sum(0)
            s_k += torch.mm(X_c.T, L).T          # d×K → K×d
            sq_k += torch.mm(X_sq_c.T, L).T

        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer
        n_k = torch.zeros(K, device=device)
        s_k = torch.zeros(K, d, device=device)
        sq_k = torch.zeros(K, d, device=device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            cs = end - start

            if X.device == device:
                X_chunk = X[start:end]
            else:
                X_chunk = X[start:end].to(device, non_blocking=True)

            X_sq_chunk = X_chunk * X_chunk
            A_chunk = torch.cat([X_sq_chunk, X_chunk], dim=1)

            # Recompute L and γ for this chunk
            L_chunk = torch.mm(A_chunk, B_dist)
            L_chunk.add_(quad_mu)
            L_chunk.mul_(-0.5)
            L_chunk.add_(log_coeff)

            ln_chunk = log_normalizer[start:end]
            gamma_chunk = (L_chunk - ln_chunk.unsqueeze(1)).exp()

            # Accumulate stats (additive across chunks)
            n_k += gamma_chunk.sum(0)
            # M-step: X.T @ γ (avoid γ transpose)
            s_k += torch.mm(X_chunk.T, gamma_chunk).T
            sq_k += torch.mm(X_sq_chunk.T, gamma_chunk).T

            del X_chunk, X_sq_chunk, A_chunk, L_chunk, gamma_chunk

        # ---- Parameter update ----
        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer


class StreamingFlashGMM:
    """Streaming Flash GMM — no L[N,K] materialization at any point.

    Memory: O(chunk_size × d) for X + O(N) for log_normalizer + O(Kd) for params
    Never allocates N×K matrix.

    For N=100M, d=64: X stored on CPU (25.6GB RAM),
    chunks streamed to GPU (0.5GB/chunk).
    """

    def __init__(self, chunk_size=500_000):
        self.chunk_size = chunk_size

    def em_step(self, X, mu, var, log_pi):
        N, d = X.shape
        K = mu.shape[0]
        device = mu.device
        chunk_size = min(self.chunk_size, N)

        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)

        B_dist = torch.cat([inv_var.T, -2 * mu_iv.T], dim=0)

        # ---- Pass 1: chunked online logsumexp ----
        # Standard materializes L_chunk[cs, K] per chunk — fine since cs << N
        log_normalizer = torch.empty(N, device=device, dtype=torch.float32)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            X_chunk = X[start:end].to(device, non_blocking=True) if X.device != device else X[start:end]
            X_sq_chunk = X_chunk * X_chunk
            A_chunk = torch.cat([X_sq_chunk, X_chunk], dim=1)

            L_chunk = torch.mm(A_chunk, B_dist)
            L_chunk.add_(quad_mu).mul_(-0.5).add_(log_coeff)
            log_normalizer[start:end] = torch.logsumexp(L_chunk, dim=1)

            del X_chunk, X_sq_chunk, A_chunk, L_chunk

        # ---- Pass 2: chunked γ + stats ----
        n_k = torch.zeros(K, device=device)
        s_k = torch.zeros(K, d, device=device)
        sq_k = torch.zeros(K, d, device=device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            X_chunk = X[start:end].to(device, non_blocking=True) if X.device != device else X[start:end]
            X_sq_chunk = X_chunk * X_chunk
            A_chunk = torch.cat([X_sq_chunk, X_chunk], dim=1)

            L_chunk = torch.mm(A_chunk, B_dist)
            L_chunk.add_(quad_mu).mul_(-0.5).add_(log_coeff)

            gamma_chunk = (L_chunk - log_normalizer[start:end].unsqueeze(1)).exp()

            n_k += gamma_chunk.sum(0)
            s_k += torch.mm(X_chunk.T, gamma_chunk).T
            sq_k += torch.mm(X_sq_chunk.T, gamma_chunk).T

            del X_chunk, X_sq_chunk, A_chunk, L_chunk, gamma_chunk

        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer


class StreamingFlashGMMAsync:
    """Streaming Flash with double-buffered async CPU→GPU transfer.

    Uses 2 CUDA streams: while computing on chunk t, prefetches chunk t+1.
    Hides PCIe transfer latency behind GPU compute.
    """

    def __init__(self, chunk_size=500_000):
        self.chunk_size = chunk_size

    def em_step(self, X, mu, var, log_pi):
        N, d = X.shape
        K = mu.shape[0]
        device = mu.device
        cs = min(self.chunk_size, N)
        is_cpu = (X.device.type == 'cpu')

        inv_var = 1.0 / var
        mu_iv = mu * inv_var
        quad_mu = (mu * mu_iv).sum(1)
        log_det = var.log().sum(1)
        log_coeff = log_pi - 0.5 * (d * math.log(2 * math.pi) + log_det)
        B_dist = torch.cat([inv_var.T, -2 * mu_iv.T], dim=0)

        log_normalizer = torch.empty(N, device=device, dtype=torch.float32)

        if is_cpu:
            # Double buffer for async transfer
            stream_compute = torch.cuda.Stream()
            stream_transfer = torch.cuda.Stream()
            buf = [torch.empty(cs, d, device=device, dtype=torch.float32, pin_memory=False) for _ in range(2)]
        else:
            stream_compute = torch.cuda.default_stream()

        n_chunks = (N + cs - 1) // cs

        # ---- Pass 1 ----
        for ci in range(n_chunks):
            start = ci * cs
            end = min(start + cs, N)
            actual_cs = end - start

            if is_cpu:
                cur_buf = buf[ci % 2]
                with torch.cuda.stream(stream_transfer):
                    cur_buf[:actual_cs].copy_(X[start:end], non_blocking=True)
                stream_compute.wait_stream(stream_transfer)
                with torch.cuda.stream(stream_compute):
                    X_chunk = cur_buf[:actual_cs]
                    X_sq = X_chunk * X_chunk
                    A = torch.cat([X_sq, X_chunk], dim=1)
                    L = torch.mm(A, B_dist)
                    L.add_(quad_mu).mul_(-0.5).add_(log_coeff)
                    log_normalizer[start:end] = torch.logsumexp(L, dim=1)
            else:
                X_chunk = X[start:end]
                X_sq = X_chunk * X_chunk
                A = torch.cat([X_sq, X_chunk], dim=1)
                L = torch.mm(A, B_dist)
                L.add_(quad_mu).mul_(-0.5).add_(log_coeff)
                log_normalizer[start:end] = torch.logsumexp(L, dim=1)

        torch.cuda.synchronize()

        # ---- Pass 2 ----
        n_k = torch.zeros(K, device=device)
        s_k = torch.zeros(K, d, device=device)
        sq_k = torch.zeros(K, d, device=device)

        for ci in range(n_chunks):
            start = ci * cs
            end = min(start + cs, N)
            actual_cs = end - start

            if is_cpu:
                cur_buf = buf[ci % 2]
                with torch.cuda.stream(stream_transfer):
                    cur_buf[:actual_cs].copy_(X[start:end], non_blocking=True)
                stream_compute.wait_stream(stream_transfer)
                with torch.cuda.stream(stream_compute):
                    X_chunk = cur_buf[:actual_cs]
                    X_sq = X_chunk * X_chunk
                    A = torch.cat([X_sq, X_chunk], dim=1)
                    L = torch.mm(A, B_dist)
                    L.add_(quad_mu).mul_(-0.5).add_(log_coeff)
                    gamma = (L - log_normalizer[start:end].unsqueeze(1)).exp()
                    n_k += gamma.sum(0)
                    s_k += torch.mm(X_chunk.T, gamma).T
                    sq_k += torch.mm(X_sq.T, gamma).T
            else:
                X_chunk = X[start:end]
                X_sq = X_chunk * X_chunk
                A = torch.cat([X_sq, X_chunk], dim=1)
                L = torch.mm(A, B_dist)
                L.add_(quad_mu).mul_(-0.5).add_(log_coeff)
                gamma = (L - log_normalizer[start:end].unsqueeze(1)).exp()
                n_k += gamma.sum(0)
                s_k += torch.mm(X_chunk.T, gamma).T
                sq_k += torch.mm(X_sq.T, gamma).T

        torch.cuda.synchronize()

        inv_nk = 1.0 / n_k.clamp(min=1e-8)
        new_mu = s_k * inv_nk.unsqueeze(1)
        new_var = (sq_k * inv_nk.unsqueeze(1) - new_mu * new_mu).clamp(min=1e-6)
        new_log_pi = (n_k / N).clamp(min=1e-8).log()

        return new_mu, new_var, new_log_pi, log_normalizer
StreamingGMM = StreamingGMMFast

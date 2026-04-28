# Flash-GMM: IO-Aware Gaussian Mixture Model on GPU

> Extending the IO-aware fusion ideas from [Flash-KMeans](https://arxiv.org/abs/2603.09229) to GMM/EM with **zero N×K matrix materialization**.

## Highlights

- **Up to 1.56× faster** than standard GPU GMM on A100 (d=128, IO-bound regime)
- **Up to 32× less VRAM** — eliminates all $N \times K$ intermediate matrices
- **Up to 14.4× IO reduction** — theoretical IO savings validated by measurement
- **C++ CPU 2.3–5.6× faster** than PyTorch (ATen/MKL) with OpenMP 8-core
- **Mathematically exact** — same EM result as standard implementation, verified to float precision
- Includes an **IO Complexity Environment** for agents to discover Flash-style optimizations

## Algorithm

Standard GMM EM materializes the $N \times K$ log-likelihood matrix $L$ and responsibility matrix $\gamma$ to HBM, causing $\Theta(NK)$ IO traffic. Flash-GMM eliminates this via:

1. **Pass 1 (Flash Log-Normalizer)**: Stream centroid tiles through shared memory, maintain online log-sum-exp in registers → output `log_normalizer ∈ R^N` (only $O(N)$, not $O(NK)$)
2. **Pass 2 (Flash Accumulate Stats)**: Recompute $\gamma$ on-chip, immediately accumulate sufficient statistics $n_k, s_k, sq_k$ → output $O(Kd)$
3. **Parameter Update**: $\mu = s_k/n_k$, $\sigma^2 = sq_k/n_k - \mu^2$, $\pi = n_k/N$

Total IO: $O(Nd + Kd)$ per iteration — eliminates all $\Theta(NK)$ traffic.

IO saving ≈ $2K/d + 1$, maximum benefit when $K \gg d$.

## GPU Performance (A100 80GB)

CUDA kernels optimized with: pre-computed `inv_var`/`log_coeff` (no `logf` in hot loop), warp-collaborative Mahalanobis distance (lane-striped coalesced access), per-warp accumulators (zero shared memory atomicAdd in hot loop).

| N | K | d | Standard E+M | Flash E+M | Speedup | VRAM (Std) | VRAM (Flash) | VRAM Saving |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 8 | 32 | 0.18 ms | 0.34 ms | 0.52× | 0.8 MB | 0.5 MB | — |
| 16,384 | 64 | 64 | 1.04 ms | 1.60 ms | 0.65× | 12.6 MB | 4.2 MB | 3.0× |
| 16,384 | 128 | 64 | 1.99 ms | 2.75 ms | 0.73× | 21.0 MB | 4.3 MB | 4.9× |
| 65,536 | 64 | 128 | 5.42 ms | 3.58 ms | **1.51×** | 67.2 MB | 33.6 MB | 2.0× |
| 65,536 | 256 | 128 | 21.65 ms | 13.84 ms | **1.56×** | 168.0 MB | 33.8 MB | 5.0× |
| 65,536 | 512 | 128 | 42.54 ms | 27.61 ms | **1.54×** | 302.5 MB | 34.1 MB | **8.9×** |
| 262,144 | 64 | 64 | 9.44 ms | 12.13 ms | 0.78× | 201.4 MB | 67.1 MB | 3.0× |
| 262,144 | 128 | 64 | 18.53 ms | 24.14 ms | 0.77× | 335.6 MB | 67.2 MB | 5.0× |
| 32,768 | 1,024 | 64 | 30.01 ms | 30.61 ms | 0.98× | 277.3 MB | 8.9 MB | **31.2×** |
| 65,536 | 1,024 | 64 | 45.41 ms | 51.67 ms | 0.88× | 554.2 MB | 17.3 MB | **32.0×** |

**Key observations:**
- **d=128 (IO-bound)**: Flash-GMM **1.5× faster** — IO savings dominate, flash advantage realized
- **d=64 (compute-bound)**: Standard still faster — cuBLAS GEMM M-step with Tensor Core is extremely efficient
- **VRAM**: Flash always wins, up to **32× less** memory (K=1024, d=64)
- All configurations: correctness **PASS** (log-normalizer diff < 0.1)

## CPU Performance (C++ OpenMP, 8 cores)

C++ extension with OpenMP achieves near-linear scaling and significantly outperforms PyTorch ATen/MKL.

| N | K | d | PyTorch (8 cores) | C++ Standard (8 cores) | Speedup vs PyTorch |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 8 | 32 | 4.21 ms | **0.75 ms** | **5.6×** |
| 4,096 | 64 | 32 | 18.01 ms | **5.40 ms** | **3.3×** |
| 16,384 | 64 | 64 | 86.52 ms | **37.97 ms** | **2.3×** |
| 16,384 | 128 | 64 | 185.84 ms | **75.05 ms** | **2.5×** |
| 65,536 | 256 | 128 | 2,854.02 ms | **1,148.78 ms** | **2.5×** |

**Parallel scaling efficiency (C++ OpenMP):**

| Implementation | 1→4 cores | 1→8 cores |
|---|---|---|
| PyTorch (ATen/MKL) | 1.8–2.8× | 1.9–3.6× |
| C++ Standard (OpenMP) | **3.6–3.9×** | **6.3–7.9×** |

## IO Theoretical Analysis

| N | K | d | K/d | Standard IO | Flash E+M IO | IO Saving |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 64 | 32 | 2.0 | 5.28 MB | 1.10 MB | **4.80×** |
| 4,096 | 256 | 32 | 8.0 | 17.96 MB | 1.25 MB | **14.39×** |
| 8,192 | 512 | 64 | 8.0 | 71.83 MB | 4.99 MB | **14.40×** |
| 16,384 | 256 | 128 | 2.0 | 84.41 MB | 17.57 MB | **4.81×** |

## Correctness

All implementations verified to float precision:

| Test | Result | Precision |
|---|---|---|
| Flash E-step vs Standard E-step (Python) | **PASS** | max\|Δγ\| < 1.5e-5 |
| Flash E+M vs Standard EM single round (Python) | **PASS** | max\|Δμ\| < 2e-6 |
| Multi-iteration convergence tracking (10 rounds) | **PASS** | all rounds diff < 2e-6 |
| C++ Native correctness (OpenMP) | **PASS** | max\|Δγ\| = 7.27e-6 |
| CUDA GPU convergence (20 iters, 12 configs) | **PASS** | final LL diff = 0.00e+00 |

## Project Structure

```
flash_gmm/                    # Core GMM implementation
├── csrc/
│   ├── flash_gmm_cuda.cu     # CUDA kernels (optimized v2)
│   ├── flash_gmm_cpu.cpp     # C++ CPU kernels (OpenMP)
│   └── binding.cpp            # PyTorch pybind11 bindings
├── standard_gmm.py            # Python baseline + IO counter
├── flash_gmm.py               # Python Flash E+M implementation
├── native_wrapper.py           # C++/CUDA wrapper
├── benchmark_gpu.py            # GPU benchmark
├── benchmark_cpu.py            # CPU benchmark
├── test_correctness.py         # Correctness tests
└── setup.py                    # Build script

io_env/                         # IO Complexity Environment for Agents
├── dsl.py                      # Computation graph DSL
├── calculator.py               # Symbolic IO calculator + roofline model
├── actions.py                  # Design action space
├── agent_loop.py               # Agent optimization loop
└── examples/                   # Pre-built operator examples
```

## Quick Start

```bash
cd flash_gmm/

# 1. Build C++ + CUDA extension
export PATH=/usr/local/cuda/bin:$PATH
FLASH_GMM_CUDA=1 python setup.py build_ext --inplace

# 2. Run correctness tests
python test_correctness.py

# 3. Run GPU benchmark
LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH \
  python benchmark_gpu.py
```

## IO Complexity Environment

An environment for LLM agents to discover IO-efficient GPU operators:

```bash
cd io_env/
python demo.py
```

The agent automatically discovers Flash-style optimizations (online logsumexp, online argmin, online softmax) by analyzing IO complexity of baseline operator graphs.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA)
- CUDA Toolkit 12.0+
- GCC 11+

## References

- [Flash-KMeans: Fast and Memory-Efficient Exact K-Means](https://arxiv.org/abs/2603.09229) (arXiv 2026)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) (NeurIPS 2022)

## License

MIT

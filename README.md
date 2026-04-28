# Flash-GMM: IO-Aware Gaussian Mixture Model on GPU

> Extending the IO-aware fusion ideas from [Flash-KMeans](https://arxiv.org/abs/2603.09229) to GMM/EM with **zero N×K matrix materialization**.

## Highlights

- **1.08–1.50× faster** than standard GPU GMM (cuBLAS + Tensor Core baseline) on A100
- **100–260× less VRAM** — eliminates all $N \times K$ intermediate matrices
- **Mathematically exact** — same EM result as standard implementation, verified to float precision
- Includes an **IO Complexity Environment** for agents to discover Flash-style optimizations

## Algorithm

Standard GMM EM materializes the $N \times K$ log-likelihood matrix $L$ and responsibility matrix $\gamma$ to HBM, causing $\Theta(NK)$ IO traffic. Flash-GMM eliminates this via:

1. **Pass 1 (Flash Log-Normalizer)**: Stream centroid tiles through shared memory, maintain online log-sum-exp in registers → output `log_normalizer ∈ R^N` (only $O(N)$, not $O(NK)$)
2. **Pass 2 (Flash Accumulate Stats)**: Recompute $\gamma$ on-chip, immediately accumulate sufficient statistics $n_k, s_k, sq_k$ → output $O(Kd)$
3. **Parameter Update**: $\mu = s_k/n_k$, $\sigma^2 = sq_k/n_k - \mu^2$, $\pi = n_k/N$

Total IO: $O(Nd + Kd)$ per iteration — same level as Flash-KMeans.

## Performance (A100 80GB)

| N | K | d | Standard E+M | Flash E+M | Speedup | VRAM Saving |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 8 | 32 | 0.13ms | 0.10ms | 1.33× | — |
| 16,384 | 128 | 64 | 1.97ms | 1.69ms | 1.17× | 63× |
| 65,536 | 256 | 128 | 21.66ms | 19.64ms | 1.10× | 144× |
| 65,536 | 512 | 128 | 42.38ms | 39.16ms | 1.08× | 153× |
| 262,144 | 128 | 64 | 18.59ms | 16.38ms | 1.14× | 183× |
| 32,768 | 1,024 | 64 | 29.82ms | 19.87ms | **1.50×** | 159× |
| 65,536 | 1,024 | 64 | 45.27ms | 33.21ms | **1.36×** | 260× |

## Project Structure

```
flash_gmm/                    # Core GMM implementation
├── csrc/
│   ├── flash_gmm_cuda.cu     # CUDA kernels (optimized v3)
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

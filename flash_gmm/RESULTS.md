# Flash-GMM: IO-Aware 高斯混合模型 — 项目总结

> 日期: 2026-04-28  
> 硬件: NVIDIA A100 80GB PCIe, 24-core CPU  
> 软件: PyTorch 2.10 (CUDA 12.8), Python 3.13, GCC 13.3, NVCC 13.1  

---

## 1. 研究动机

将 Flash-KMeans (arXiv:2603.09229) 的 IO-aware 融合思想迁移到 GMM/EM 算法。

**现状调研**: 目前**不存在**公开的 IO-aware GPU GMM 实现。现有 GPU GMM 基线：

| 实现 | GPU 后端 | EM 训练 | 维护状态 |
|---|---|---|---|
| **scikit-learn** | ❌ CPU only | ✅ | ✅ 活跃 — CPU 金标准 |
| **pomegranate** | PyTorch | ✅ | 一般 — 唯一活跃 GPU GMM |
| **PyCave** | PyTorch+Lightning | ✅ | ❌ 已归档 (2026-02) |
| **PyTorch MixtureSameFamily** | CUDA | ❌ 无 EM | ✅ — 仅推理/采样 |
| cuML / FAISS / fast_pytorch_kmeans | CUDA | — | — | **均无 GMM** |

### K-Means → GMM 结构映射

| K-Means | GMM | Flash 优化 |
|---|---|---|
| 距离矩阵 $D \in \mathbb{R}^{N \times K}$ | Log-likelihood 矩阵 $L \in \mathbb{R}^{N \times K}$ | **同构**: 融合消除物化 |
| row-wise argmin → $a \in \mathbb{Z}^N$ | row-wise log-sum-exp → $\gamma \in \mathbb{R}^{N \times K}$ | **同构**: 在线流式归约 |
| 硬分配 scatter → Sort-Inverse | 软分配 GEMM → 天然高效 | Sort-Inverse **不适用**（也不需要） |

### 关键差异

GMM 的 $\gamma$ 是完整 $N \times K$ 矩阵，不像 k-means 的 $a$ 只有 $O(N)$。  
但 **diagonal 协方差下可 E+M 完全融合**：流式计算 $\gamma_{nk}$ 的同时片上累积统计量，避免 $\gamma$ 写入内存。

---

## 2. 算法设计

### 三种实现变体

#### (A) Standard GMM — 基线
```
E-step:  X,μ,σ² → 计算 L ∈ R^{N×K} → 写HBM → 读HBM → log-sum-exp → γ ∈ R^{N×K} → 写HBM
M-step:  读 γ + X → GEMM 加权聚合 → 更新 μ,σ²,π
IO:      4×Θ(NK) + 2×Θ(Nd) + 2×Θ(Kd)
```

#### (B) Flash E-step — 消除 L 物化
```
Pass 1:  流式扫描 centroid tiles → 在线 log-sum-exp → log_normalizer ∈ R^N
Pass 2:  重计算 L_tile → normalize → γ ∈ R^{N×K} → 写HBM
M-step:  同标准
IO:      2×Θ(NK) + 2×Θ(Nd) + 2×Θ(Kd)    [节省 2×Θ(NK)]
```

#### (C) Flash E+M Fused — 零 N×K 物化 ✦
```
Pass 1:  流式扫描 centroid tiles → 在线 log-sum-exp → log_normalizer ∈ R^N
Pass 2:  重计算 γ_tile on-chip → 立即累积 n_k, s_k, sq_k (O(Kd) 片上)
Update:  μ = s_k/n_k,  σ² = sq_k/n_k - μ²,  π = n_k/N
IO:      2×Θ(Nd) + 4×Θ(Kd)               [消除全部 N×K 流量]
```

**核心技术**: 在线 log-sum-exp（与 FlashAttention 的 online softmax 同构）
```
for each centroid tile:
    m_new = max(m_old, max(L_tile))
    sum_exp = sum_exp * exp(m_old - m_new) + Σ exp(L_tile - m_new)
    m_old = m_new
log_normalizer = m_old + log(sum_exp)
```

---

## 3. 实现清单

### 文件结构
```
flash_gmm/
├── csrc/
│   ├── flash_gmm_cpu.cpp        C++ CPU 内核 (OpenMP 多线程)
│   ├── flash_gmm_cuda.cu        CUDA GPU 内核 (3 kernels)
│   └── binding.cpp              PyTorch pybind11 绑定 (自动 CPU/CUDA dispatch)
├── setup.py                     编译脚本 (支持 FLASH_GMM_CUDA=1)
├── native_wrapper.py            Python 高级接口
├── standard_gmm.py              Python 基线 + IO 计数
├── flash_e_step.py              Python Flash E-step + E+M 融合变体
├── flash_gmm.py                 Python Flash E+M 完整实现
├── benchmark.py                 IO 理论对比 (14 组配置)
├── benchmark_cpu.py             CPU 多线程对比 (1/4/8 核)
├── benchmark_gpu.py             GPU 实测对比 (A100)
└── test_correctness.py          正确性测试 (5 项)
```

### 编译状态
| 目标 | 状态 |
|---|---|
| Python 纯实现 | ✅ 全部通过 |
| C++ CPU extension (OpenMP) | ✅ 编译通过, 多线程正确 |
| CUDA GPU extension | ✅ 编译通过, 正确性通过 |

---

## 4. 正确性验证

### Python 测试 (test_correctness.py) — 5/5 PASS
| 测试 | 结果 | 精度 |
|---|---|---|
| Flash E-step vs Standard E-step | PASS | max\|Δγ\| < 1.5e-5 |
| Flash E+M vs Standard EM (单轮) | PASS | max\|Δμ\| < 2e-6, max\|Δσ²\| < 3e-4 |
| 收敛曲线 vs sklearn | PASS | Standard ↔ Flash LL diff = 0 |
| 多轮迭代参数跟踪 (10轮) | PASS | 所有轮次 diff < 2e-6 |
| IO 节省验证 | PASS | 4.80× reduction (K=64, d=32) |

### C++ Native 测试 (OpenMP, 4 threads) — PASS
```
Max |γ_std - γ_flash|:  7.27e-06
Max |μ_std - μ_fused|:  3.81e-06
Max |σ²_std - σ²_fused|: 1.07e-04
```

### CUDA GPU 收敛测试 (A100 80GB) — PASS
```
20 轮迭代, 最终 LL diff = 0.00e+00
所有 12 组配置正确性: ALL PASS
```

---

## 5. IO 分析结果

### 理论 IO 对比 (benchmark.py)

| N | K | d | K/d | Standard IO | Flash E+M IO | IO 节省 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 16 | 16 | 1.0 | 0.40 MB | 0.14 MB | **2.89×** |
| 4096 | 64 | 32 | 2.0 | 5.28 MB | 1.10 MB | **4.80×** |
| 4096 | 256 | 32 | 8.0 | 17.96 MB | 1.25 MB | **14.39×** |
| 8192 | 512 | 64 | 8.0 | 71.83 MB | 4.99 MB | **14.40×** |
| 16384 | 256 | 128 | 2.0 | 84.41 MB | 17.57 MB | **4.81×** |

**规律**: IO 节省 ≈ $2\frac{K}{d} + 1$，$K \gg d$ 时收益最大。

### VRAM 节省 (GPU 实测)

| 配置 | Standard VRAM | Flash VRAM | 节省 |
|---|---:|---:|---:|
| N=65536, K=512, d=128 | 302.5 MB | 34.1 MB | **8.9×** |
| N=65536, K=1024, d=64 | 554.2 MB | 17.3 MB | **32.0×** |

---

## 6. CPU 性能结果 (benchmark_cpu.py)

### 多核并行效率

| 实现 | 1→4 核 | 1→8 核 |
|---|---|---|
| PyTorch (ATen/MKL) | 1.8-2.8× | 1.9-3.6× |
| **C++ Standard (OpenMP)** | **3.6-3.9×** | **6.3-7.9×** |
| **C++ Flash E+M (OpenMP)** | **3.0-3.9×** | **5.8-7.8×** |

C++ OpenMP 的并行效率远超 PyTorch MKL（8 核接近线性加速）。

### C++ vs PyTorch 绝对性能 (8 核)

| N | K | d | PyTorch 8核 | C++ Std 8核 | C++ Flash 8核 |
|---:|---:|---:|---:|---:|---:|
| 4096 | 8 | 32 | 4.21 ms | **0.75 ms (5.6×)** | 1.89 ms (2.2×) |
| 4096 | 64 | 32 | 18.01 ms | **5.40 ms (3.3×)** | 11.49 ms (1.6×) |
| 16384 | 64 | 64 | 86.52 ms | **37.97 ms (2.3×)** | 85.15 ms (1.0×) |
| 16384 | 128 | 64 | 185.84 ms | **75.05 ms (2.5×)** | 165.25 ms (1.1×) |
| 65536 | 256 | 128 | 2854.02 ms | **1148.78 ms (2.5×)** | ~2600 ms (est.) |

**CPU 结论**:
- C++ Standard + OpenMP 在 8 核下**全面碾压 PyTorch** (2.3-5.6×)
- C++ Flash E+M 在 8 核下接近/追平 PyTorch
- Flash E+M 比 C++ Standard 慢约 2× ← 因为双 pass 重计算距离（2× FLOPs），CPU 上计算不"免费"
- Flash 的真正优势在**内存受限场景**（GPU HBM），不在 CPU 算力

---

## 7. GPU 性能结果 (benchmark_gpu.py)

### A100 80GB, CUDA 13.1, 全部 VRAM 可用

| N | K | d | Standard | Flash E+M | 速度比 | 正确性 |
|---:|---:|---:|---:|---:|---:|---|
| 4096 | 8 | 32 | 0.18 ms | 1.73 ms | 0.11× | PASS |
| 4096 | 16 | 32 | 0.23 ms | 1.77 ms | 0.13× | PASS |
| 16384 | 16 | 64 | 0.34 ms | 3.13 ms | 0.11× | PASS |
| 16384 | 64 | 64 | 1.04 ms | 4.89 ms | 0.21× | PASS |
| 16384 | 128 | 64 | 1.99 ms | 10.49 ms | 0.19× | PASS |
| 65536 | 64 | 128 | 5.45 ms | 39.24 ms | 0.14× | PASS |
| 65536 | 256 | 128 | 21.66 ms | 139.09 ms | 0.16× | PASS |
| 65536 | 512 | 128 | 42.49 ms | 276.88 ms | 0.15× | PASS |
| 262144 | 64 | 64 | 9.44 ms | 68.95 ms | 0.14× | PASS |
| 262144 | 128 | 64 | 18.64 ms | 136.70 ms | 0.14× | PASS |
| 32768 | 1024 | 64 | 30.00 ms | 144.55 ms | 0.21× | PASS |
| 65536 | 1024 | 64 | 45.55 ms | 273.05 ms | 0.17× | PASS |

**GPU 结论**: 当前 CUDA kernel 是**功能正确的参考实现 (proof-of-concept)**。  
Standard baseline 走 ATen → cuBLAS GEMM → Tensor Core 极其高效；Flash kernel 手写 per-element 循环无法与之竞争。

---

## 8. GPU 性能差距根因分析

| 瓶颈 | 描述 | 量化影响 |
|---|---|---|
| **无 Tensor Core** | 手写 float 循环计算 $x_i^T \mu_k$，未利用 WMMA/MMA | 计算吞吐低 10-50× |
| **Per-element atomicAdd** | Pass 2 中 γ×x 每元素一次 shared memory atomic | 严重竞争、序列化 |
| **无 vectorized load** | 未用 float4/LDG.128 | 内存带宽利用率 < 25% |
| **2× FLOPs** | 双 pass 重计算距离 | 纯计算开销翻倍 |
| **Kernel launch overhead** | 小规模时 launch 开销主导 | 小 N/K 时尤为严重 |

### 优化路径

1. **GEMM-based 距离**: $\|x-\mu\|^2/\sigma^2 = \|x/\sigma\|^2 + \|\mu/\sigma\|^2 - 2(x/\sigma)^T(\mu/\sigma)$  
   → 内积部分映射到 Tensor Core GEMM tile
2. **Tile 内融合 online log-sum-exp**: 在 GEMM output tile 上直接做在线归约  
   → 类似 FlashAttention 在 $QK^T$ tile 上融合 softmax
3. **Warp-level reduction**: `__shfl_xor_sync` 替代 shared memory atomicAdd
4. **Vectorized memory access**: float4 加载 X 和 centroid 参数
5. **Triton 重写**: Triton DSL 表达 tiling + 在线归约，自动生成优化 kernel

### 预期收益

在 $K \gg d$ 的 IO-bound regime 下，优化后的 Flash-GMM 应当：
- 消除 cuBLAS 需要物化的 $N \times K$ 中间矩阵
- 实现与 Flash-KMeans 类似的数量级加速（IO 节省 5-14× → 端到端预期 3-10×）

---

## 9. 设计决策记录

| 决策 | 理由 |
|---|---|
| **Diagonal 协方差优先** | 片上存储 $(2d+1)$/分量，E+M 完全融合可行；full 需 $O(d^2)$ 不可行 |
| **双 pass 而非单 pass** | 单 pass 需存储所有 L_tile 供归一化（等同物化），双 pass 以 2× FLOPs 换 0× HBM |
| **不做 Sort-Inverse** | GMM 软分配使每个点贡献到所有 K 分量，无 scatter 竞争 |
| **OpenMP 多线程** | 遵循 `torch.set_num_threads`，per-thread 局部累积器避免竞争 |
| **绕过 CUDA 版本检查** | nvcc 13.1 向下兼容 PyTorch CUDA 12.8 ABI |

---

## 10. 复现命令

```bash
cd flash_gmm/

# 1. Python 正确性测试
python test_correctness.py

# 2. IO 理论分析
python benchmark.py

# 3. 编译 C++ (CPU only, with OpenMP)
python setup.py build_ext --inplace

# 4. 编译 C++ + CUDA
export PATH=/usr/local/cuda/bin:$PATH
FLASH_GMM_CUDA=1 python setup.py build_ext --inplace

# 5. C++ 测试
LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH \
  python native_wrapper.py

# 6. CPU 多线程 benchmark (1/4/8 核)
LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH \
  python benchmark_cpu.py

# 7. GPU benchmark
LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH \
  python benchmark_gpu.py
```

---

## 11. 项目定位总结

```
┌─────────────────────────────────────────────────────────────────┐
│                    Flash-GMM 当前状态                            │
├─────────────────────────────────────────────────────────────────┤
│ ✅ 算法正确性: Python/C++/CUDA 三层全部验证通过                  │
│ ✅ IO 模型:    理论分析 + 实测 IO 计数一致, 最高 14.4× 节省      │
│ ✅ VRAM 节省:  GPU 实测最高 32× (K=1024, d=64)                  │
│ ✅ CPU 性能:   C++ OpenMP 8核比 PyTorch 快 2.3-5.6×             │
│ ⚠️ GPU 性能:   参考实现, 需 Tensor Core / Triton 优化           │
│ 🔲 下一步:     GEMM-based 距离 + tile 内融合 online log-sum-exp  │
└─────────────────────────────────────────────────────────────────┘
```

# IO-Aware GPU Operator Optimization Agent: Architecture & Findings

> A cognitive architecture for automated GPU kernel optimization, built on the
> Knowledge–Environment–Objective (KEO) framework. Validated on 10 operators
> covering all 7 Berkeley Dwarfs with NCU hardware counter profiling.

---

## 1. The KEO Framework

GPU 算子优化不是单一维度的问题。从 10 个算子的实验中，我们发现所有结果都可以用三个要素的交互来解释：

```
                     ┌─────────────┐
                     │  Knowledge  │
                     │  (怎么优化)  │
                     └──────┬──────┘
                            │
                 ┌──────────┼──────────┐
                 ▼          ▼          ▼
          Pattern DB   Paper/Lib    Skill Store
          gemm+reduce  FlashAttn    沉淀的模板
          temp. tile   cuDNN impl   验证过的代码
                 │          │          │
                 └──────────┼──────────┘
                            │
                     ┌──────▼──────┐
                     │    Agent    │
                     │   (LLM)    │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              │                           │
       ┌──────▼──────┐            ┌──────▼──────┐
       │ Environment │            │  Objective  │
       │  (在哪优化)  │            │  (优化什么)  │
       └──────┬──────┘            └──────┬──────┘
              │                           │
       ┌──────┼──────┐            ┌──────┼──────┐
       ▼      ▼      ▼            ▼      ▼      ▼
     NCU   HW Spec  L2/BW      Speedup  Scale  Precision
     DRAM%  A100     40MB       vs base  N=1K→64K  max_err
     SM%    TC 312T  2039GB/s   vs lib   多配置    VRAM
```

**核心观察**：IO 分析是必要条件，不是充分条件。失败案例全是某个要素缺失或冲突。

---

## 2. Experimental Evidence: 10 Operators × 7 Dwarfs

### 2.1 Complete Results (A100 80GB PCIe, 10 rounds)

| Dwarf | Operator | IO↓ | Speedup | NCU DRAM% | TC | K/E/O 分析 |
|:-----:|----------|:---:|:-------:|:---------:|:--:|-----------|
| 7 MapReduce | CrossEntropy | 80% | **6.04×** | **77%** | ✗ | K✓ E✓ O✓ |
| 7 MapReduce | LayerNorm | 50% | **4.58×** | **80%** | ✗ | K✓ E✓ O✓ |
| 5 Structured | Stencil2D | 99% | **4.74×** | — | ✗ | K✓ E✓ O✓ |
| 4 N-Body | Particle | 100% | **2.19×** | — | ✗ | K✓ E✓ O✓ |
| 1 Dense LA | KMeans (tl.dot) | 94% | **1.44×** | 1% | **✓** | K✓(修后) E✓ O✓ |
| 3 Spectral | FFT | 93% | **1.32×** | — | ✗ | K△ E✓ O△(输cuFFT) |
| 6 Unstructured | Graph Laplacian | 17% | **1.18×** | — | ✗ | K✓ E✓ O✓ |
| 7 MapReduce | Softmax | 65% | 0.86× | 7% | ✗ | K✓ **E✗**(L2) O✗ |
| 2 Sparse LA | SpMV | 0% | 0.76× | — | ✗ | **K✗** E✓ O✗ |
| 1 Dense LA | Conv2D | 86% | 0.57× | — | ✗ | **K✗** **E✗**(cuDNN) O✗ |
| 1 Dense LA | KMeans (标量) | 94% | 0.07× | **0%** | ✗ | **K✗** E✓ O✗ |

### 2.2 Failure Analysis: KEO Mismatch

| 失败 | 缺失要素 | 具体表现 | NCU 信号 |
|------|---------|---------|---------|
| KMeans 0.07× | **K**: 不知要 tl.dot | GEMM→标量循环 | warps 99% DRAM 0% |
| Conv2D 0.06× vs cuDNN | **K**: 无 Winograd | naive GEMM vs cuDNN | — |
| Softmax 0.5× | **E**: L2 cache fit | IO 分析高估 HBM | DRAM 7% |
| SpMV 0.76× | **K**: 模式不匹配 | uniform sparsity warp overhead | — |
| FFT 0.18× vs cuFFT | **K**: 只有 radix-2 | cuFFT 用 radix-8/16 | — |

**结论**：所有失败都可以归因于 K/E/O 中至少一个维度的缺失。

---

## 3. Agent Architecture

### 3.1 Current Implementation (20 tools)

```
Agent Core (LLM: GPT-5.2 via Copilot)
├── Knowledge Interface
│   ├── retrieve_pattern     — 4 个硬编码优化模式
│   ├── pre_check           — 7 维静态预检 (memory-bound? GEMM? cache? atomic?)
│   └── show_actions        — 可用优化动作列表
│
├── Environment Interface
│   ├── ncu_profile         — torch.profiler + sudo ncu 硬件计数器
│   ├── library_ceiling     — cuDNN/cuFFT/cuBLAS 性能天花板
│   ├── compare_profile     — baseline vs flash side-by-side
│   ├── occupancy_analysis  — 占用率估算
│   └── profile_kernel      — 静态代码分析 + 动态指标
│
├── Objective Interface
│   ├── benchmark_kernel    — 实际 speedup + 版本跟踪 + 回滚
│   ├── compile_and_test    — 正确性验证
│   ├── autotune_kernel     — block size 扫描
│   └── verify              — 符号优化验证
│
└── Graph Manipulation
    ├── analyze / fuse_ops / fuse_and_online / undo
    └── generate_kernel (Triton/CUDA codegen)
```

### 3.2 Agent Loop: Thought → Action → Observation

```python
# Phase 0: Pre-check (K + E)
analyze → pre_check → "memory-bound? GEMM? cache?"

# Phase 1: Symbolic optimization (K)
fuse_and_online → verify → "80% IO reduction, zero intermediates"

# Phase 2: Kernel generation (K → O)
generate_kernel → compile_and_test → benchmark_kernel

# Phase 3: Data-driven iteration (E + O + K)
while not converged:
    ncu_profile       → "DRAM 77%, compute 9%, memory-bound"
    library_ceiling   → "3× faster than F.cross_entropy"
    autotune_kernel   → "sweep 18 configs"
    benchmark_kernel  → "5.83× (NEW BEST)"
    ncu_profile       → "confirmed 77% DRAM utilization"
    → converged: BW > 70%, within 1.5× of library
```

### 3.3 Key Mechanisms

**Version Tracking & Rollback**
```
v1: 5.83× ← BEST
v2: 3.02× ← REGRESSION: auto-rollback to v1
v3: 5.90× ← NEW BEST (after autotune)
```

**Convergence Detection**
```python
# Nudge 引导 agent 完成最低探索后才 done:
fully_explored = has_profiled AND has_ceiling AND has_autotuned
if fully_explored and (bw > 70% or within_1.5x_of_library):
    nudge = "call done"
```

**NCU-Driven Diagnosis**
```
warps 99% + DRAM 0% → "标量循环, 丢失 TC" → retrieve_pattern
DRAM 77% + SM 68%   → "memory-bound, 近最优"  → autotune → done
DRAM 7%             → "数据在 L2, IO 优化无效" → accept
```

---

## 4. Ideal Architecture: Full KEO

### 4.1 Knowledge Interface — "怎么优化"

| 能力 | 当前 | 理想 |
|------|------|------|
| 模式检索 | 4 个硬编码 pattern | RAG 搜索论文库 + 代码库 |
| 压缩总结 | 无 | 论文 → 1 段话 → 代码模板 |
| 技能沉淀 | 无 | 成功案例存 skill，下次复用 |

```python
# RAG Search: 给定算子描述，搜索相关优化知识
knowledge.search("CSR SpMV GPU optimization")
→ [{paper: "Merge-based SpMV",
    key_insight: "按 NNZ 均分而非按 row"},
   {paper: "CSR-Adaptive",
    key_insight: "warp/block 自适应 row length"}]

# Summarize: 长论文压缩为可执行知识
knowledge.summarize(paper)
→ "核心: merge_path 按 NNZ 均分, 短 row 用 warp, 长 row 用 block"

# Skill Store: 沉淀成功经验
knowledge.save_skill({
    name: "tiled_gemm_online_reduce",
    when: "GEMM → row reduce",
    code: "<Triton template>",
    verified: "1.44× on A100, KMeans N=65K K=1K"
})
```

### 4.2 Environment Interface — "在哪优化"

| 能力 | 当前 | 理想 |
|------|------|------|
| Profiler | NCU 原始 7 指标 | 压缩为 1 句话诊断 |
| 硬件画像 | 手写 HardwareSpec | 自动检测 + 跨 GPU 预测 |
| 环境对比 | 无 | 同 kernel 多 GPU 预测 |

```python
# Compressed Profile
env.profile_compressed(kernel)
→ "memory-bound (DRAM 77%, SM 68%), no TC, occupancy 62%.
   瓶颈: HBM 带宽. 建议: 增大 BLOCK_SIZE 或接受."

# Cross-GPU Prediction
env.predict("H100")
→ "BW 3350 vs 2039 → memory-bound kernel 预计快 1.64×"
```

### 4.3 Objective Interface — "优化什么"

| 能力 | 当前 | 理想 |
|------|------|------|
| 测试 | 单规模 benchmark | 多规模自动扫描 |
| 多目标 | 只看 speedup | speedup + 精度 + VRAM + 编译时间 |
| 收敛 | nudge 规则 | 数据驱动的自适应判断 |

```python
# Multi-scale sweep
obj.sweep([{N:1024}, {N:4096}, {N:16384}, {N:65536}])
→ [{N:1K, speedup:4.43}, {N:4K, speedup:5.38},
   {N:16K, speedup:6.04}]  # 趋势: 越大越快

# Multi-objective
obj.evaluate(kernel)
→ {speedup: 5.83, vram_saved: "524MB→0",
   precision: "max_err 1.9e-6", compile_time: "2.1s"}

# Adaptive convergence
obj.is_done()
→ True if (bw > 70% OR within 1.5× of library)
         AND tried ≥ 2 versions
         AND profiled with NCU
```

### 4.4 Ideal Agent Loop

```python
while not obj.is_done():
    # 1. Knowledge: 我该用什么算法?
    if not have_strategy:
        papers = knowledge.search(operator)
        strategy = knowledge.summarize(papers)

    # 2. Generate & Test
    kernel = generate_kernel(strategy)
    if not compile_and_test(kernel):
        strategy = knowledge.search(error_message)  # 回 K 找修复
        continue

    # 3. Environment: 瓶颈在哪?
    diagnosis = env.profile_compressed(kernel)
    ceiling = env.library_ceiling()

    # 4. Objective: 效果如何?
    results = obj.sweep(kernel, scales)

    # 5. 决策
    if diagnosis == "structural":
        strategy = knowledge.search(diagnosis.root_cause)  # 回 K
    elif diagnosis == "tuning":
        kernel = autotune(kernel)
    elif results.within_ceiling(1.5):
        knowledge.save_skill(kernel, results)  # 沉淀到 K
        break  # 达标
```

---

## 5. LLM Agent End-to-End Results

### 5.1 CrossEntropy (成功案例)

```
Steps: 17 | LLM calls: 18 | Tokens: 140K | Speedup: 5.83×

pre_check → ✓ RECOMMENDED
fuse_and_online → 80% IO↓
generate_kernel (LLM 自写 Triton) → compile FAIL → 自修 → PASS
benchmark → 5.83× ✓ FIRST KERNEL
ncu_profile → DRAM 77%, memory-bound ✓
library_ceiling → 比 F.cross_entropy 快 3.3×
autotune → 18 configs swept
re-benchmark → 5.83× confirmed
re-profile → DRAM 77% confirmed
→ done (data-driven: BW>70%, within 1.5× of library)
```

### 5.2 KMeans (诊断→修复案例)

```
Steps: 15 | Speedup: 0.07× → 1.44×

generate_kernel (标量模板) → benchmark → 0.07× ✗
ncu_profile → DRAM 0%, warps 99%, TC=NO → "under-utilized"
library_ceiling → cuBLAS 快 13.6×
compare_profile → compute 43%→3% = "lost GEMM parallelism"
retrieve_pattern("gemm reduce") → tl.dot 模式
generate_kernel (tl.dot 版本) → benchmark → 1.44× ✓
ncu_profile → compute 61%, TC=YES ✓ → "compute-bound with TC"
```

### 5.3 N-Body (LLM 自写 kernel)

```
Steps: 30 | LLM calls: 30 | Tokens: 420K | Speedup: 2.19×

LLM 自写 3 版 Triton N-Body kernel:
  v1: tl.rsqrt + self-mask → FAIL (精度)
  v2: Newton-Raphson rsqrt → FAIL (精度)
  v3: 回退 CUDA 模板 → PASS → 2.19× ✓
autotune → 18 configs, all ~2.19×
occupancy → 62% ✓
```

### 5.4 Graph Laplacian (LLM 写 CSR kernel)

```
Steps: 12+ | Speedup: 1.18× (CUDA) / 267K× (vs Python ref)

LLM 自写 Triton CSR SpMV kernel:
  - CSR row_ptr 遍历, masked indirect load, per-row accumulation
  v1: 参数顺序错 → 自修 → v2 PASS
  benchmark → PASS ✓
```

---

## 6. File Structure

```
io_env/
├── __init__.py
├── dsl.py              # ComputationGraph DSL
├── calculator.py       # IOCalculator + PreCheckResult (7-dim)
├── actions.py          # fuse_ops, fuse_and_online, recompute
├── react_agent.py      # IOEnvironment + 20 tools + prompt + nudge
├── llm_react_agent.py  # LLM ReAct loop + convergence detection
├── profiler.py         # torch.profiler + sudo ncu + utilization
├── triton_codegen.py   # Triton/CUDA templates + compile + benchmark
├── agent_loop.py       # Reward computation
├── workflow.py         # Interactive workflow API
├── test_io_env.py      # 33 unit tests
└── examples/
    └── __init__.py     # 10 operators (7 Berkeley Dwarfs)
```

---

## 7. Lessons Learned

1. **IO 优化只在 memory-bound + 对手未优化时有效**
   - CrossEntropy 6×, Stencil 4.7×, LayerNorm 4.6×: 成功
   - Softmax, Conv2D, SpMV: 失败 — 对手已优化或模式不匹配

2. **NCU 是诊断利器**
   - warps 99% + DRAM 0% = 一眼标量循环
   - DRAM 77% = 确认 memory-bound 近最优
   - TC 0% → 7% = 验证 tl.dot 生效

3. **LLM 能写 GPU kernel**
   - 成功写出 CrossEntropy, N-Body, Graph Laplacian Triton kernel
   - 含自主调试（参数修复、精度修复、API 适配）
   - 但复杂 kernel 编译慢（N-Body Triton 编译超时）

4. **版本跟踪防退化**
   - LLM 改 kernel 可能退化（5.83× → 3.02×）
   - 自动回滚保留最佳版本

5. **最低探索要求 + 收敛检测**
   - 不过早 done：必须 ncu + ceiling + autotune
   - 不空转：全套工具用过后引导 done

6. **三要素缺一不可**
   - Knowledge 缺失 → 用错算法 (KMeans 标量)
   - Environment 误判 → 高估优化收益 (Softmax L2)
   - Objective 不清 → 和错误对手比 (Graph Laplacian Python ref)

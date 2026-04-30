# GPU 算子设计的多维优化指南

> 从 Flash-GMM / BFS / KMeans 的实践中总结的完整决策框架

---

## 1. 问题：IO 复杂度不是唯一指标

IO-aware 优化的核心假设是"减少 HBM 流量 → 加速"。但实践中多次发现这个假设不成立：

| 案例 | IO 减少 | 实测 Speedup | 为什么不匹配 |
|------|:------:|:------:|------|
| Cross-Entropy | 80% | **5.83×** | ✓ 匹配。memory-bound，data >> L2 |
| LayerNorm | 50% | **4.84×** | ✓ 匹配。memory-bound |
| KMeans (模板) | 94% | **0.07×** | ✗ 破坏了 GEMM 并行性，Tensor Core → 标量循环 |
| KMeans (tl.dot) | 94% | **2.41×** | ✓ 恢复了 GEMM 结构后匹配 |
| BFS Flash | 51% | **0.60×** | ✗ 数据在 L2 cache 内，atomic 竞争开销 |
| Softmax | 65% | **1.01×** | ~ 持平。PyTorch softmax 已高度优化 |

**结论：IO 分析是必要条件，不是充分条件。**

---

## 2. 七维决策框架

```
┌─────────────────────────────────────────────────────────────┐
│                GPU 算子优化的 7 个维度                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Dim 1: IO 复杂度 (HBM Traffic)                             │
│    - 中间矩阵物化量（N×K, N×N 等）                           │
│    - 决定是否值得做 Flash 式融合                              │
│    - 工具: IOCalculator.analyze()                           │
│                                                             │
│  Dim 2: 算术强度 (Arithmetic Intensity)                     │
│    - AI = FLOPs / IO_bytes                                  │
│    - AI < 平衡点 → memory-bound → IO 优化有效               │
│    - AI > 平衡点 → compute-bound → IO 优化可能无效           │
│    - 工具: IOReport.arithmetic_intensity                    │
│                                                             │
│  Dim 3: Cache 局部性 (L2/SMEM Hit Rate)                     │
│    - data_size vs L2_cache_size                             │
│    - data << L2 → 连续扫描快于随机 gather，即使总量更大      │
│    - data >> L2 → IO 分析准确                               │
│    - A100: 40MB L2, H200: 50MB L2                           │
│                                                             │
│  Dim 4: 并行模式 (Parallelism Pattern)                      │
│    - GEMM 级并行 (tl.dot/Tensor Core) vs 标量串行循环        │
│    - 优化不能破坏 GEMM 结构！                                │
│    - 正确: tile 内 GEMM + tile 间在线归约                    │
│    - 错误: GEMM → 逐元素标量循环                             │
│                                                             │
│  Dim 5: 原子操作竞争 (Atomic Contention)                     │
│    - atomicAdd/atomicCAS 的热点竞争                          │
│    - 解决: 片上累积 + 稀疏全局写                              │
│    - 或: sort → gather → 分段规约                            │
│                                                             │
│  Dim 6: 占用率 (Occupancy)                                   │
│    - 寄存器用量、共享内存用量 → 每 SM 的活跃 warp 数          │
│    - 过多寄存器 → 低占用率 → SM 空闲                         │
│    - 调优: BLOCK_SIZE, num_warps, num_stages                │
│                                                             │
│  Dim 7: 访存合并 (Memory Coalescing)                         │
│    - 同一 warp 32 线程的访存是否合并为 1-2 次事务             │
│    - 连续地址: 合并 → 高带宽利用率                           │
│    - 随机地址: 不合并 → 带宽利用 < 25%                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 决策流程

```
给定一个 GPU 算子:

Step 1: IO 分析
├─ 有物化中间矩阵？→ 计算 IO 节省量
└─ 无物化 → 不需要 Flash 式优化，转 Step 6

Step 2: Roofline 判断
├─ AI < 平衡点（memory-bound）→ IO 优化有效，继续 Step 3
└─ AI > 平衡点（compute-bound）→ IO 优化可能无效
   └─ 但 VRAM 节省仍然有价值（支持更大数据）

Step 3: Cache 分析
├─ 数据总量 >> L2 cache (40-50MB) → IO 分析可信
└─ 数据总量 << L2 cache → 实际 HBM 流量远小于分析值
   └─ 连续访问模式可能比减少总量更重要

Step 4: 并行模式检查
├─ 优化后保留了 GEMM/tl.dot 结构？→ 好
└─ 退化为标量循环？→ 必须修复
   └─ 正确模式: tile 内 tl.dot + tile 间 online reduce

Step 5: 生成 Kernel → Compile → Benchmark
├─ speedup ≥ 1.0 → 成功
└─ speedup < 1.0 → Profile 诊断
   ├─ 无 Tensor Core 利用 → 加 tl.dot
   ├─ 高 atomic 竞争 → 换为片上累积
   ├─ 低占用率 → 调 BLOCK_SIZE
   └─ 差访存合并 → 改数据布局

Step 6: AutoTune
└─ 扫描 BLOCK_SIZE × num_warps 找最优配置
```

---

## 4. 每个维度的诊断工具

### Dim 1: IO 复杂度

```python
# 已有工具
report = IOCalculator(H200).analyze(graph, params)
print(f"Total IO: {report.total_io/1e6:.1f} MB")
print(f"Materialized: {report.materialized_intermediates}")
```

### Dim 2: 算术强度

```python
# 已有工具
print(f"AI: {report.arithmetic_intensity:.1f} FLOP/Byte")
print(f"Bottleneck: {report.bottleneck}")
# H200 平衡点 ~14 FLOP/Byte
# A100 平衡点 ~10 FLOP/Byte
```

### Dim 3: Cache 分析

```python
# 需要新增
data_size_mb = N * K * 4 / 1e6  # 中间矩阵大小
l2_cache_mb = 40  # A100
if data_size_mb < l2_cache_mb:
    print(f"⚠ Data ({data_size_mb:.0f}MB) fits in L2 ({l2_cache_mb}MB)")
    print(f"  Sequential scan may be faster than random gather")
    print(f"  IO reduction may NOT translate to speedup")
else:
    print(f"✓ Data ({data_size_mb:.0f}MB) >> L2 ({l2_cache_mb}MB)")
    print(f"  IO reduction will translate to speedup")
```

### Dim 4: 并行模式

```python
# 已有 profile_kernel 工具
# 检查 tl.dot / Tensor Core 使用
# 检查串行循环
```

### Dim 5: 原子操作

```python
# 需要新增到 profile_kernel
num_atomics = code.count("atomicAdd") + code.count("atomicCAS")
if num_atomics > 0:
    print(f"⚠ {num_atomics} atomic operations found")
    print(f"  Consider: local accumulation → single global write")
```

### Dim 6: 占用率

```python
# Triton 自动调优
# 或手动: triton.testing.Benchmark 扫描 num_warps, num_stages
```

### Dim 7: 访存合并

```python
# 静态分析: 检查地址计算模式
# row-major 连续访问 → 好
# 间接寻址 (col_idx[i]) → 差
```

---

## 5. 实际案例分析

### Cross-Entropy: IO 优化的完美案例

```
Dim 1: IO    → 524MB exp_logits 物化，80% 可消除        ✓ 值得优化
Dim 2: AI    → 0.2 FLOP/B << 14 平衡点                 ✓ memory-bound
Dim 3: Cache → N×V = 4K×32K = 512MB >> 40MB L2          ✓ IO 分析可信
Dim 4: 并行  → 逐元素 exp/log，无 GEMM 结构可破坏        ✓ 不降级
Dim 5: Atomic → 无                                      ✓
Dim 6: 占用率 → per-row kernel，高占用                   ✓
Dim 7: 合并   → 行连续访问                               ✓
结果: 5.83× ← 所有 7 个维度都绿灯
```

### KMeans (模板): IO 优化的反面教材

```
Dim 1: IO    → 268MB 距离矩阵，94% 可消除               ✓ 值得优化
Dim 2: AI    → 30 FLOP/B > 14 平衡点                    ✗ compute-bound
Dim 3: Cache → N×K = 65K×1K = 256MB >> 40MB             ✓
Dim 4: 并行  → 原本 GEMM(Tensor Core)，优化后标量循环    ✗✗ 致命！
Dim 5: Atomic → 无                                      ✓
Dim 6: 占用率 → 每线程遍历 K=1024，寄存器压力大          ✗
Dim 7: 合并   → 每线程随机读 centroid                    ✗
结果: 0.07× ← Dim 4 致命，直接慢 14×
修复: tl.dot 恢复 GEMM → 2.41×
```

### BFS Flash: 小图的 cache 效应

```
Dim 1: IO    → 51% IO 减少                              ✓
Dim 2: AI    → 极低（BFS 几乎无计算）                    ✓ memory-bound
Dim 3: Cache → CSR 50MB ≈ L2 40MB                       ⚠ 边界情况
Dim 4: 并行  → baseline 是简单并行扫描，flash 用 atomicCAS ✗
Dim 5: Atomic → atomicCAS per neighbor                   ✗
Dim 6: 占用率 → baseline 高（简单逻辑），flash 低（更多逻辑） ✗
Dim 7: 合并   → baseline 连续扫 bitmap ✓，flash 间接寻址 ✗
结果: 0.60× ← Dim 3+5+7 共同导致
```

---

## 6. 对 Agent Prompt 的改进建议

当前 agent prompt 只教了 IO 优化。应该加入：

```
## Pre-optimization Checklist (MUST check before optimizing)

Before applying any Flash-style optimization, verify:

1. Is the operator MEMORY-BOUND? (AI < balance point)
   - If compute-bound, IO optimization may not help.
   
2. Does the data exceed L2 cache? (data_size > 40MB for A100)
   - If data fits in L2, full scans may be cheaper than clever reductions.
   
3. Will the optimization PRESERVE compute parallelism?
   - NEVER replace a GEMM (tl.dot/cuBLAS) with a scalar loop.
   - Use: tiled GEMM + online reduce.
   
4. Does the optimization introduce atomic operations?
   - Atomics on hot addresses serialize execution.
   - Prefer: local accumulation → one global write.

If any check fails, the optimization may slow things down despite IO savings.
```

---

## 7. 未来方向

| 方向 | 描述 | 预期影响 |
|------|------|---------|
| **多维 profile** | profile_kernel 输出 cache hit rate + atomic 竞争 + coalescing | 更准确的诊断 |
| **预测模型** | 不只是 roofline，加入 cache 模型和并行效率 | 减少无效迭代 |
| **硬件感知搜索** | 根据 GPU 型号自动选择优化策略 | 跨硬件泛化 |
| **自适应判断** | agent 先判断是否值得 IO 优化，再决定优化方向 | 避免 KMeans/BFS 式失误 |
| **代码修复循环** | profile → 诊断 → 自动修改 kernel → 重测 | 多轮迭代效率 |

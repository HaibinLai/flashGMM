# IO-Aware Operator Design Environment for Agents

> 设计一个 IO 复杂度感知的环境，让 Agent 自主发现并设计更高效的 GPU 算子

---

## 1. 动机

当前 Agent 辅助算子设计的问题：

| 现状 | 问题 |
|------|------|
| Agent 只看代码正确性 | 不知道生成的 kernel 是 memory-bound 还是 compute-bound |
| 没有 IO 代价信号 | 无法判断"物化中间矩阵"还是"重算"哪个更优 |
| 优化靠经验规则 | 缺乏系统的搜索空间和量化反馈 |
| 验证只跑正确性测试 | 不跑性能 profiling，不知道瓶颈在哪 |

**核心想法**：给 Agent 一个 **IO Complexity Environment**，它能：
1. 接受一个计算图描述（算子规格）
2. **解析计算** IO 复杂度（HBM 读写字节数）
3. **模拟执行**并给出性能预估
4. **实际 benchmark** 验证
5. 以上信号作为 reward，引导 Agent 迭代搜索更优算子设计

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Auto Research Loop                    │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  Agent    │───▶│ IO Complexity │───▶│  Correctness  │  │
│  │ (LLM)    │    │ Environment   │    │  Verifier     │  │
│  │          │◀───│              │◀───│               │  │
│  └──────────┘    └──────────────┘    └───────────────┘  │
│       │               │                     │           │
│       │          ┌────┴─────┐               │           │
│       │          │          │               │           │
│       ▼          ▼          ▼               ▼           │
│  ┌─────────┐ ┌────────┐ ┌────────┐  ┌────────────┐     │
│  │ Design  │ │Symbolic│ │Roofline│  │  GPU       │     │
│  │ Action  │ │IO Calc │ │Model   │  │  Benchmark │     │
│  │ Space   │ │        │ │        │  │            │     │
│  └─────────┘ └────────┘ └────────┘  └────────────┘     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 核心组件设计

### 3.1 Computation Graph DSL — 算子描述语言

Agent 用一种结构化 DSL 描述算子的计算流程。环境解析它来计算 IO。

```python
# 示例：标准 GMM E-step 的计算图描述
standard_gmm_estep = ComputationGraph(
    name="standard_gmm_estep",
    inputs={
        "X":      TensorSpec(shape=(N, d), dtype="f32", storage="HBM"),
        "mu":     TensorSpec(shape=(K, d), dtype="f32", storage="HBM"),
        "var":    TensorSpec(shape=(K, d), dtype="f32", storage="HBM"),
        "log_pi": TensorSpec(shape=(K,),   dtype="f32", storage="HBM"),
    },
    operations=[
        # Op 1: 计算 L 矩阵 — 物化到 HBM
        Op("compute_L",
           reads=["X", "mu", "var", "log_pi"],
           computes="log_likelihood",
           flops=lambda N,K,d: N * K * (3*d),  # log_det + mahal + combine
           output=TensorSpec(shape=(N, K), dtype="f32", storage="HBM"),  # 物化!
           output_name="L"),

        # Op 2: logsumexp — 从 HBM 读回 L
        Op("logsumexp",
           reads=["L"],
           computes="row_reduce",
           flops=lambda N,K,d: N * K * 3,
           output=TensorSpec(shape=(N,), dtype="f32", storage="HBM"),
           output_name="log_norm"),

        # Op 3: normalize — 从 HBM 读 L，写 gamma
        Op("normalize",
           reads=["L", "log_norm"],
           computes="elementwise",
           flops=lambda N,K,d: N * K * 2,
           output=TensorSpec(shape=(N, K), dtype="f32", storage="HBM"),  # 物化!
           output_name="gamma"),
    ],
    outputs=["gamma", "log_norm"]
)
```

```python
# 示例：Flash GMM E-step — Agent 发现的优化版本
flash_gmm_estep = ComputationGraph(
    name="flash_gmm_estep",
    inputs={
        "X":      TensorSpec(shape=(N, d), dtype="f32", storage="HBM"),
        "mu":     TensorSpec(shape=(K, d), dtype="f32", storage="HBM"),
        "var":    TensorSpec(shape=(K, d), dtype="f32", storage="HBM"),
        "log_pi": TensorSpec(shape=(K,),   dtype="f32", storage="HBM"),
    },
    operations=[
        # 融合 Op: 流式处理，L 不物化
        FusedOp("flash_log_normalizer",
                reads=["X", "mu", "var", "log_pi"],
                tiling=TilingSpec(dim_N=BN, dim_K=BK),
                online_state=OnlineState(
                    vars={"running_max": "scalar", "running_sum_exp": "scalar"},
                    update_rule="online_logsumexp"
                ),
                flops=lambda N,K,d: N * K * (3*d),
                output=TensorSpec(shape=(N,), dtype="f32", storage="HBM"),  # 仅 O(N)!
                output_name="log_norm"),
    ],
    outputs=["log_norm"]
)
```

### 3.2 Symbolic IO Calculator — 符号化 IO 计算器

给定计算图，**解析计算** HBM 读写字节数，无需实际执行。

```python
class IOComplexityCalculator:
    """符号化计算一个 ComputationGraph 的 IO 复杂度"""

    def __init__(self, hardware: HardwareSpec):
        self.hw = hardware  # HBM 带宽、SMEM 大小、L2 大小等

    def analyze(self, graph: ComputationGraph, params: dict) -> IOReport:
        """
        输入:
            graph: 计算图描述
            params: 具体参数 {N: 65536, K: 1024, d: 128, BN: 64, BK: 8}

        输出:
            IOReport 包含:
              - total_hbm_reads:  总 HBM 读字节数
              - total_hbm_writes: 总 HBM 写字节数
              - total_flops:      总浮点操作数
              - arithmetic_intensity: FLOPs / IO_bytes
              - bottleneck:       "memory-bound" | "compute-bound"
              - estimated_time_ms: 基于 roofline 的时间估算
              - per_op_breakdown: 每个 Op 的详细 IO 分解
              - materialized_intermediates: 物化的中间矩阵列表
              - optimization_opportunities: 可消除的物化点
        """
        report = IOReport()

        for op in graph.operations:
            reads = sum(self._tensor_bytes(graph.get_tensor(r), params)
                       for r in op.reads)
            writes = self._tensor_bytes(op.output, params)
            flops = op.flops(**params)

            report.add_op(OpIOStats(
                name=op.name,
                hbm_reads=reads,
                hbm_writes=writes,
                flops=flops,
                is_intermediate=(op.output_name not in graph.outputs),
                storage=op.output.storage,
            ))

        report.compute_summary(self.hw)
        return report

    def compare(self, baseline: ComputationGraph, optimized: ComputationGraph,
                params: dict) -> ComparisonReport:
        """对比两个实现的 IO 效率"""
        r1 = self.analyze(baseline, params)
        r2 = self.analyze(optimized, params)
        return ComparisonReport(
            io_reduction=1 - r2.total_io / r1.total_io,
            flop_increase=r2.total_flops / r1.total_flops - 1,
            estimated_speedup=r1.estimated_time_ms / r2.estimated_time_ms,
            eliminated_intermediates=r1.materialized_intermediates - r2.materialized_intermediates,
        )
```

### 3.3 Design Action Space — Agent 的操作空间

Agent 可以执行的优化变换，每个都有明确的 IO 影响：

```python
class DesignActions:
    """Agent 可用的算子设计变换"""

    # ---- 融合类 ----
    def fuse_ops(self, op1: str, op2: str) -> ComputationGraph:
        """将两个相邻 Op 融合，消除中间矩阵物化
        IO 影响: 消除 intermediate 的 write + read = 2×size(intermediate)"""

    def fuse_producer_consumer(self, producer: str, consumer: str) -> ComputationGraph:
        """生产者-消费者融合：consumer 片上直接消费 producer 的输出"""

    # ---- 重算类 ----
    def replace_materialize_with_recompute(self, tensor_name: str) -> ComputationGraph:
        """将某个中间结果从"物化到HBM"改为"需要时重算"
        IO 影响: 消除 write+read，增加 FLOPs
        适用条件: 该 tensor 的 arithmetic_intensity < hw.balance_point"""

    # ---- 在线算法类 ----
    def apply_online_algorithm(self, reduce_op: str,
                                algorithm: str) -> ComputationGraph:
        """将全量 reduce 替换为在线/流式算法
        可用算法:
          - "online_softmax"    (FlashAttention)
          - "online_logsumexp"  (Flash-GMM)
          - "online_argmin"     (Flash-KMeans)
          - "online_topk"
          - "online_welford"    (在线方差)
        IO 影响: 输出从 O(N×K) 降为 O(N) 或 O(K)"""

    # ---- 分块类 ----
    def apply_tiling(self, op: str, tile_dims: dict) -> ComputationGraph:
        """在指定维度上分块，使 tile 适配片上存储
        参数: tile_dims = {"N": BN, "K": BK}
        约束: tile_size * dtype_bytes <= hw.smem_size"""

    # ---- 规约优化类 ----
    def replace_scatter_with_sort_gather(self, scatter_op: str) -> ComputationGraph:
        """将 scatter atomicAdd 替换为 sort + gather + 分段规约
        IO 影响: 原子操作从 O(Nd) 降为 O((K+N/B)d)"""

    def use_local_accumulator(self, reduce_op: str) -> ComputationGraph:
        """使用片上/线程局部累积器替代全局原子操作"""

    # ---- 流水线类 ----
    def apply_double_buffering(self, op: str, buffer_dim: str) -> ComputationGraph:
        """双缓冲：加载下一个 tile 与计算当前 tile 重叠"""

    def apply_stream_overlap(self, h2d_op: str, compute_op: str) -> ComputationGraph:
        """CUDA Stream 重叠：H2D 传输与计算并行"""
```

### 3.4 Correctness Verifier — 正确性验证器

```python
class CorrectnessVerifier:
    """验证优化后的算子与 baseline 数学等价"""

    def verify(self, baseline_graph: ComputationGraph,
               optimized_graph: ComputationGraph,
               test_cases: list[dict]) -> VerifyResult:
        """
        1. 生成随机测试数据
        2. 分别用 baseline 和 optimized 执行
        3. 比较输出的数值误差
        4. 返回 pass/fail + 最大误差
        """
        for params in test_cases:
            X, mu, var, log_pi = generate_random_inputs(**params)
            out_baseline = self.execute(baseline_graph, X, mu, var, log_pi)
            out_optimized = self.execute(optimized_graph, X, mu, var, log_pi)

            max_err = max_relative_error(out_baseline, out_optimized)
            if max_err > self.tolerance:
                return VerifyResult(passed=False, max_error=max_err,
                                    failing_case=params)

        return VerifyResult(passed=True, max_error=max_err)
```

### 3.5 GPU Benchmark Oracle — 实际性能测量

```python
class BenchmarkOracle:
    """将计算图编译为实际 CUDA kernel 并 benchmark"""

    def benchmark(self, graph: ComputationGraph, params: dict,
                  n_warmup: int = 10, n_iter: int = 100) -> BenchmarkResult:
        """
        1. 从计算图生成 Triton/CUDA 代码（或调用预编译的 kernel）
        2. GPU 预热
        3. 测量实际耗时
        4. 返回: 实际时间、实际带宽利用率、与 roofline 预测的偏差
        """
        kernel = self.codegen(graph, params)
        # warmup
        for _ in range(n_warmup):
            kernel.run()
        torch.cuda.synchronize()

        # benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            kernel.run()
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / n_iter
        return BenchmarkResult(
            time_ms=elapsed_ms,
            achieved_bandwidth_gbps=graph.total_io(params) / elapsed_ms / 1e6,
            bandwidth_utilization=...,  # vs peak HBM BW
            roofline_gap=...,           # 实际 vs 理论预测
        )
```

---

## 4. Auto Research Loop — Agent 自主研究循环

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Step 1: PROBLEM SPECIFICATION                     │
│   ┌───────────────────────────────────────────────┐ │
│   │ 用户给出算子规格:                              │ │
│   │   - 输入/输出 tensor shapes                    │ │
│   │   - 数学定义（如 GMM E-step 公式）             │ │
│   │   - 目标硬件 (e.g. H200)                      │ │
│   │   - baseline 实现（可选）                      │ │
│   └───────────────────────────────────────────────┘ │
│                        │                            │
│                        ▼                            │
│   Step 2: BASELINE ANALYSIS                         │
│   ┌───────────────────────────────────────────────┐ │
│   │ Agent 构建 baseline ComputationGraph           │ │
│   │ → IOCalculator.analyze() → 得到 IO report      │ │
│   │ → 识别 bottleneck + materialized intermediates │ │
│   └───────────────────────────────────────────────┘ │
│                        │                            │
│                        ▼                            │
│   Step 3: HYPOTHESIS GENERATION                     │
│   ┌───────────────────────────────────────────────┐ │
│   │ Agent 基于 IO report 提出优化假设:             │ │
│   │   "L矩阵(256MB)是最大的物化点，                │ │
│   │    可以用 online_logsumexp 消除"               │ │
│   │ → 选择 DesignAction                           │ │
│   │ → 生成新的 ComputationGraph                    │ │
│   └───────────────────────────────────────────────┘ │
│                        │                            │
│                        ▼                            │
│   Step 4: SYMBOLIC EVALUATION                       │
│   ┌───────────────────────────────────────────────┐ │
│   │ IOCalculator.compare(baseline, optimized)      │ │
│   │ → IO reduction: 85%                           │ │
│   │ → FLOP increase: 100%                         │ │
│   │ → Estimated speedup: 4.2×                     │ │
│   │                                               │ │
│   │ 如果 estimated_speedup < 1.0 → 回到 Step 3    │ │
│   └───────────────────────────────────────────────┘ │
│                        │                            │
│                        ▼                            │
│   Step 5: CORRECTNESS CHECK                         │
│   ┌───────────────────────────────────────────────┐ │
│   │ CorrectnessVerifier.verify()                   │ │
│   │ → 多组随机数据对比 baseline vs optimized       │ │
│   │                                               │ │
│   │ 如果 failed → 回到 Step 3 (修改设计)           │ │
│   └───────────────────────────────────────────────┘ │
│                        │                            │
│                        ▼                            │
│   Step 6: CODE GENERATION                           │
│   ┌───────────────────────────────────────────────┐ │
│   │ Agent 生成实际 CUDA/Triton kernel 代码         │ │
│   │ → 编译验证                                    │ │
│   └───────────────────────────────────────────────┘ │
│                        │                            │
│                        ▼                            │
│   Step 7: REAL BENCHMARK                            │
│   ┌───────────────────────────────────────────────┐ │
│   │ BenchmarkOracle.benchmark()                    │ │
│   │ → 实际 GPU 耗时                               │ │
│   │ → 与 roofline 预测对比                         │ │
│   │                                               │ │
│   │ 如果 实际加速 < 预测加速 × 0.7:                │ │
│   │   → 诊断: 实际瓶颈在哪?                       │ │
│   │   → 回到 Step 3                               │ │
│   └───────────────────────────────────────────────┘ │
│                        │                            │
│                        ▼                            │
│   Step 8: REPORT                                    │
│   ┌───────────────────────────────────────────────┐ │
│   │ 输出:                                         │ │
│   │   - 优化后的 kernel 代码                       │ │
│   │   - IO 对比分析                               │ │
│   │   - 实际 benchmark 结果                        │ │
│   │   - 设计决策的推理链                           │ │
│   └───────────────────────────────────────────────┘ │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 5. Reward Signal 设计

Agent 在每轮迭代中获得多维 reward：

```python
def compute_reward(baseline_report, optimized_report, verify_result, benchmark_result):
    """
    多维 reward 信号，引导 Agent 搜索 IO 高效的算子设计
    """
    reward = 0.0

    # R1: IO reduction (主要信号, 权重最高)
    io_reduction = 1 - optimized_report.total_io / baseline_report.total_io
    reward += 5.0 * io_reduction  # 范围 [0, 5]

    # R2: 正确性 (硬约束)
    if not verify_result.passed:
        return -10.0  # 严重惩罚

    # R3: 实际加速比 (ground truth 信号)
    if benchmark_result:
        speedup = baseline_report.time_ms / benchmark_result.time_ms
        reward += 3.0 * math.log2(max(speedup, 0.5))  # log scale

    # R4: 预测准确度 (校准信号)
    if benchmark_result:
        prediction_error = abs(optimized_report.estimated_time_ms - benchmark_result.time_ms)
        prediction_accuracy = 1 - prediction_error / benchmark_result.time_ms
        reward += 1.0 * max(prediction_accuracy, 0)

    # R5: 消除的中间矩阵数量 (结构化信号)
    eliminated = len(baseline_report.materialized_intermediates) - \
                 len(optimized_report.materialized_intermediates)
    reward += 0.5 * eliminated

    # R6: FLOP 增长惩罚 (防止无限重算)
    flop_ratio = optimized_report.total_flops / baseline_report.total_flops
    if flop_ratio > 4.0:
        reward -= 1.0 * (flop_ratio - 4.0)  # 超过 4× 重算开始惩罚

    return reward
```

---

## 6. 具体实验方案

### Phase 1: IO Calculator Prototype

实现最小可用的 IO 计算器，验证核心概念。

```
目标算子: GMM E-step, KMeans Assignment, Softmax, LayerNorm
输入: 计算图 DSL 描述
输出: IO report (HBM bytes, FLOPs, roofline 预测)
验证: 对已知优化 (FlashAttention, Flash-GMM) 能正确识别优化点
```

**成功标准**：给定标准 softmax 计算图，IO Calculator 能自动标注 "中间矩阵 S 是最大物化点，建议用 online_softmax 消除"。

### Phase 2: Agent + IO Env 交互

让 LLM Agent 在 IO Environment 中交互式设计算子。

```
实验设置:
  - Agent: GPT-4 / Claude 系列
  - 环境: IO Calculator + Correctness Verifier
  - 任务: 给定 baseline 算子，产出 IO-optimized 版本
  - 评测: 与人工设计的 Flash 版本对比

测试算子集:
  1. GMM E-step        → 期望: 发现 online logsumexp
  2. KMeans Assignment → 期望: 发现 online argmin
  3. Softmax           → 期望: 发现 online softmax (FlashAttention)
  4. BatchNorm         → 期望: 发现 Welford online 算法
  5. TopK + Softmax    → 期望: 发现融合 + online TopK
  6. Cross-Entropy     → 期望: 发现 logsumexp 融合
```

**评测指标**：

| 指标 | 说明 |
|------|------|
| IO-Discovery Rate | Agent 能否识别出主要的中间矩阵物化点 |
| Algorithm Match | Agent 提出的在线算法是否匹配已知最优 |
| IO Reduction | 符号化 IO 降低比例 |
| Correctness | 优化后算子是否数学等价 |
| Novel Discovery | 是否发现人类未设计的新优化 |

### Phase 3: End-to-End Kernel Generation

Agent 不仅设计计算图，还生成可编译运行的 CUDA/Triton 代码。

```
输入: 算子数学定义 + 目标硬件规格
Agent workflow:
  1. 构建 baseline 计算图 → IO 分析
  2. 提出优化 → 符号化验证
  3. 生成 Triton kernel 代码
  4. 编译 + 正确性测试
  5. GPU benchmark
  6. 如果不满意 → 迭代优化

输出: 优化后的 kernel + 完整 IO 分析报告
```

---

## 7. 与 Auto Research 的结合

将上述 IO Environment 嵌入自动化研究框架：

```
┌─────────────────────────────────────────────────────────────┐
│                    Auto Research Pipeline                    │
│                                                             │
│  ┌───────────┐   ┌──────────────┐   ┌───────────────────┐  │
│  │ Literature │   │ IO Complexity │   │ Experiment        │  │
│  │ Survey     │   │ Environment   │   │ Runner            │  │
│  │            │   │              │   │                   │  │
│  │ 论文库中找  │──▶│ 用 IO 视角   │──▶│ 自动生成 kernel  │  │
│  │ memory-    │   │ 分析 baseline │   │ + benchmark       │  │
│  │ bound 算子 │   │ + 搜索优化   │   │ + 对比 SOTA       │  │
│  └───────────┘   └──────────────┘   └───────────────────┘  │
│                         │                     │             │
│                         ▼                     ▼             │
│                  ┌─────────────┐      ┌─────────────┐      │
│                  │ 发现新优化   │      │ 论文/报告    │      │
│                  │ 机会         │      │ 自动生成     │      │
│                  └─────────────┘      └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 具体 Auto Research 任务

**任务 1: 系统性扫描 PyTorch 算子**
```
对 torch.nn.functional 中的每个算子:
  1. 构建 baseline 计算图
  2. IO Calculator 分析
  3. 标注 memory-bound 算子 + 物化瓶颈
  4. 对每个瓶颈尝试 DesignActions
  5. 输出: "Flash-X 可行性报告"
```

**任务 2: 新算法 IO-aware 实现**
```
给定一个新的 ML 算法 (如 Mamba, RWKV):
  1. 解析其计算流程为 ComputationGraph
  2. IO 分析 → 识别瓶颈
  3. Agent 搜索融合/在线化方案
  4. 生成优化 kernel
  5. Benchmark 对比 naive 实现
```

**任务 3: 跨算子融合搜索**
```
给定一个模型的完整 forward pass:
  1. 提取所有相邻算子对
  2. IO Calculator 评估融合收益
  3. 贪心/搜索最优融合方案
  4. 这本质上是 torch.compile 做的事，但用 Agent 做更灵活
```

---

## 8. 实现路线图

```
Phase 1 (MVP):  IO Calculator + DSL
  ├── ComputationGraph DSL 定义
  ├── Symbolic IO 计算 (读/写字节数)
  ├── Roofline model 预测
  ├── 对 GMM / KMeans / Softmax 验证
  └── 交付: Python 包 io_complexity_env

Phase 2 (Agent Loop):  Agent 交互
  ├── DesignAction 空间实现
  ├── Correctness Verifier
  ├── Agent prompt 设计 (如何呈现 IO report)
  ├── 对 6 个测试算子跑 Agent 实验
  └── 交付: 实验结果 + 分析

Phase 3 (Codegen):  端到端代码生成
  ├── ComputationGraph → Triton 代码生成
  ├── 自动编译 + benchmark pipeline
  ├── Agent 迭代优化循环
  └── 交付: 自动化 kernel 优化工具

Phase 4 (Auto Research):  规模化
  ├── PyTorch 算子批量扫描
  ├── 新模型架构自动分析
  ├── 报告/论文自动生成
  └── 交付: 完整的 auto-research 系统
```

---

## 9. 关键洞察与预期结果

### 为什么 Agent 能做好这件事

1. **IO 分析是符号化的** — 不需要真正执行，Agent 可以快速评估多种方案
2. **DesignAction 空间有限且可枚举** — 融合、重算、在线化、分块、排序，组合数可控
3. **有明确的 reward signal** — IO 字节数是精确可计算的，不像"代码质量"那样模糊
4. **在线算法是已知的** — online softmax、online argmin 等是已发表的技术，Agent 从知识库中检索即可
5. **正确性可自动验证** — 数值对比即可，无需人工审核

### 预期能发现什么

| 场景 | 预期 Agent 发现 | 是否已知 |
|------|---------------|---------|
| Softmax | online softmax + 融合 | 已知 (FlashAttention) |
| GMM E-step | online logsumexp + 重算 γ | 已知 (本项目) |
| KMeans | online argmin | 已知 (Flash-KMeans) |
| BatchNorm forward+backward | Welford + 融合 | 已知 |
| Mixture of Experts gating | online TopK + softmax 融合 | **部分新** |
| Contrastive loss (N×N) | 分块在线 + 重算 | **可能新** |
| Graph attention | 稀疏 FlashAttention | **活跃研究** |

**最有价值的情况**：Agent 在新出现的、尚未被人工优化的算子上发现 Flash 式优化机会。

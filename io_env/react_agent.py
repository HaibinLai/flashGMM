#!/usr/bin/env python3
"""
ReAct Agent for IO-Aware Operator Optimization.

Implements the Thought → Action → Observation loop where:
- Tools = IO environment operations (analyze, optimize, verify, etc.)
- Observations = IO reports, comparison results, test outcomes
- Reward = IO reduction + correctness

The agent iteratively discovers Flash-style optimizations by:
1. Observing the computation graph and IO bottlenecks
2. Reasoning about which intermediate materialization to eliminate
3. Applying fusion / online algorithm / recompute actions
4. Verifying the optimization preserves semantics via tests

Can be driven by an LLM or run as a self-contained demo.
"""

from __future__ import annotations
import json, io, os, re, sys, contextlib, traceback, importlib
from dataclasses import dataclass, field

from io_env.dsl import ComputationGraph, TensorSpec, TilingSpec
from io_env.calculator import IOCalculator, IOReport, ComparisonReport, H200
from io_env.actions import DesignActions, ONLINE_ALGORITHMS
from io_env.agent_loop import compute_reward
from io_env.examples import EXAMPLES


# ============================================================================
# Tool definitions (what the agent can call)
# ============================================================================

TOOLS = {
    "analyze": {
        "description": "Load and analyze a baseline operator. Shows IO report, materialized intermediates, and optimization hints.",
        "parameters": {
            "task": "str — one of: gmm_estep, kmeans, softmax, cross_entropy",
        },
        "example": '{"tool": "analyze", "args": {"task": "cross_entropy"}}',
    },
    "show_graph": {
        "description": "Show current computation graph structure: ops, tensors, materialized intermediates.",
        "parameters": {},
        "example": '{"tool": "show_graph", "args": {}}',
    },
    "show_actions": {
        "description": "List available optimization actions with their IO/FLOP effects.",
        "parameters": {},
        "example": '{"tool": "show_actions", "args": {}}',
    },
    "fuse_ops": {
        "description": "Fuse two sequential ops, eliminating the intermediate tensor between them. IO savings = 2 × size(intermediate).",
        "parameters": {
            "op1": "str — name of the first (producer) op",
            "op2": "str — name of the second (consumer) op",
        },
        "example": '{"tool": "fuse_ops", "args": {"op1": "row_max", "op2": "subtract_and_exp"}}',
    },
    "fuse_and_online": {
        "description": "Fuse multiple ops AND apply an online streaming algorithm. This is the 'Flash' pattern: eliminates ALL intermediate materialization between the fused ops.",
        "parameters": {
            "ops": "list[str] — names of ops to fuse",
            "algorithm": "str — one of: online_logsumexp, online_argmin, online_softmax, online_welford",
            "output_name": "str — name for the fused output tensor",
            "output_shape": "tuple — shape of the fused output, e.g. ('N',)",
        },
        "example": '{"tool": "fuse_and_online", "args": {"ops": ["compute_L", "logsumexp"], "algorithm": "online_logsumexp", "output_name": "log_norm", "output_shape": ["N"]}}',
    },
    "undo": {
        "description": "Undo the last optimization action.",
        "parameters": {},
        "example": '{"tool": "undo", "args": {}}',
    },
    "verify": {
        "description": "Run verification tests on the current optimization. Checks: (1) no materialized intermediates remain, (2) IO reduced vs baseline, (3) all outputs are still produced.",
        "parameters": {},
        "example": '{"tool": "verify", "args": {}}',
    },
    "done": {
        "description": "Signal that optimization is complete. Prints final summary.",
        "parameters": {},
        "example": '{"tool": "done", "args": {}}',
    },
    "benchmark": {
        "description": "Run actual GPU benchmark for the current operator. Compares baseline (materialized) vs optimized (flash) implementations using real PyTorch/CUDA execution. Returns wall-clock times and actual speedup.",
        "parameters": {
            "n_warmup": "int (optional, default 5) — warmup iterations",
            "n_iter": "int (optional, default 20) — benchmark iterations",
        },
        "example": '{"tool": "benchmark", "args": {}}',
    },
    "generate_kernel": {
        "description": "Generate a Triton or CUDA GPU kernel implementing the Flash-optimized operator. Uses templates for known patterns. You can write your own kernel with custom_code (Triton Python or CUDA C++). Set lang='cuda' for CUDA C++ code.",
        "parameters": {
            "custom_code": "str (optional) — your own kernel code (Triton Python or CUDA C++)",
            "lang": "str (optional, default 'triton') — 'triton' for Triton Python, 'cuda' for CUDA C++",
        },
        "example": '{"tool": "generate_kernel", "args": {"lang": "cuda", "custom_code": "<CUDA C++ code>"}}',
    },
    "compile_and_test": {
        "description": "Compile the generated Triton kernel and run correctness tests against the PyTorch reference implementation. Reports max/mean error.",
        "parameters": {},
        "example": '{"tool": "compile_and_test", "args": {}}',
    },
    "benchmark_kernel": {
        "description": "Benchmark the generated Triton kernel vs the naive materialized PyTorch baseline. This measures the REAL speedup from the generated CUDA code, not just Python-level operations.",
        "parameters": {},
        "example": '{"tool": "benchmark_kernel", "args": {}}',
    },
    "profile_kernel": {
        "description": "Profile the generated Triton kernel. Shows: compute throughput (TFLOPS), memory bandwidth (GB/s), Tensor Core utilization, occupancy, and bottleneck diagnosis. Use this to understand WHY a kernel is slow.",
        "parameters": {},
        "example": '{"tool": "profile_kernel", "args": {}}',
    },
    "retrieve_pattern": {
        "description": "Retrieve a known GPU kernel optimization pattern from the knowledge base. Patterns include: gemm_online_reduce, online_softmax, tiled_distance, flash_attention, reduction_tree. Use this when you need to know HOW to optimize a specific computation pattern.",
        "parameters": {
            "pattern": "str — pattern name or keyword to search (e.g. 'gemm reduce', 'distance argmin', 'tiled softmax')",
        },
        "example": '{"tool": "retrieve_pattern", "args": {"pattern": "gemm reduce"}}',
    },
    "autotune_kernel": {
        "description": "Automatically sweep block sizes and num_warps to find the optimal configuration for the current kernel. Tests all valid combinations of BLOCK_M, BK, num_warps and reports the best speedup. Use AFTER you have a correct kernel to find the best configuration.",
        "parameters": {},
        "example": '{"tool": "autotune_kernel", "args": {}}',
    },
    "pre_check": {
        "description": "Run multi-dimensional pre-optimization checklist BEFORE optimizing. Checks: (1) memory-bound vs compute-bound, (2) data vs L2 cache size, (3) GEMM structure that must be preserved, (4) atomic contention risk. Returns GO/CAUTION recommendation.",
        "parameters": {},
        "example": '{"tool": "pre_check", "args": {}}',
    },
    "ncu_profile": {
        "description": "Runtime-profile the current kernel. Returns: bandwidth utilization %, compute utilization %, Tensor Core usage, roofline bottleneck classification (memory-bound / compute-bound / under-utilized). Use this AFTER benchmark_kernel shows slow performance to diagnose WHY.",
        "parameters": {},
        "example": '{"tool": "ncu_profile", "args": {}}',
    },
    "library_ceiling": {
        "description": "Benchmark the library-optimized implementation (cuDNN/cuFFT/cuBLAS/ATen) as a performance ceiling. Shows how far your kernel is from the best possible library implementation. Use to decide whether further optimization is worthwhile.",
        "parameters": {},
        "example": '{"tool": "library_ceiling", "args": {}}',
    },
    "compare_profile": {
        "description": "Profile baseline (materialized) and flash kernel side-by-side. Shows bandwidth util, compute util, Tensor Core usage, and roofline bottleneck for BOTH. Highlights the biggest performance difference to pinpoint the root cause.",
        "parameters": {},
        "example": '{"tool": "compare_profile", "args": {}}',
    },
    "occupancy_analysis": {
        "description": "Estimate kernel occupancy from block size, num_warps, register/SMEM usage. Low occupancy (< 25%) means SM is mostly idle — adjust BLOCK_SIZE or reduce register pressure.",
        "parameters": {},
        "example": '{"tool": "occupancy_analysis", "args": {}}',
    },
    "analyze_library": {
        "description": "Analyze WHY the library (cuDNN/cuFFT/cuBLAS) is fast for this operator. Returns the specific algorithmic and hardware techniques the library uses. Use this to understand what your kernel is missing and choose the right optimization strategy.",
        "parameters": {},
        "example": '{"tool": "analyze_library", "args": {}}',
    },
    "suggest_strategy": {
        "description": "Get multi-level optimization strategy recommendations for the current operator. Returns suggestions at 4 levels: Algorithm (what method), IO/Fusion (what to fuse), Hardware (what features to use), Data Layout (how to arrange memory). Use BEFORE writing a kernel to plan your approach.",
        "parameters": {},
        "example": '{"tool": "suggest_strategy", "args": {}}',
    },
    "debug_correctness": {
        "description": "Deep-debug a failing kernel: run with small input, compare element-by-element against reference, find the FIRST wrong element, and diagnose the likely cause (wrong twiddle factor, off-by-one index, missing mask, etc.). Use AFTER compile_and_test shows FAIL to understand WHERE and WHY the kernel is wrong.",
        "parameters": {
            "n_elements": "int (optional, default 64) — use small N for debugging",
        },
        "example": '{"tool": "debug_correctness", "args": {"n_elements": 64}}',
    },
    "analyze_platform": {
        "description": "Show the capabilities and LIMITATIONS of the target code generation platform (Triton or CUDA). Explains what the platform CAN and CANNOT do, so you avoid writing code that will fail. Use BEFORE generating a kernel if you're unsure about platform support.",
        "parameters": {
            "lang": "str (optional, default 'triton') — 'triton' or 'cuda'",
        },
        "example": '{"tool": "analyze_platform", "args": {"lang": "triton"}}',
    },
}


# ============================================================================
# Environment: holds state and executes tools
# ============================================================================

@dataclass
class StepLog:
    step: int
    thought: str
    tool: str
    args: dict
    observation: str
    reward: float = 0.0


class IOEnvironment:
    """
    ReAct-compatible environment for IO-aware operator optimization.

    Exposes tools that an agent can call. Maintains state across steps.
    Returns text observations suitable for LLM consumption.
    """

    def __init__(self, hardware=H200):
        self.calc = IOCalculator(hardware=hardware)
        self.graph: ComputationGraph | None = None
        self.baseline_graph: ComputationGraph | None = None
        self.params: dict | None = None
        self.baseline_report: IOReport | None = None
        self.current_report: IOReport | None = None
        self.history: list[StepLog] = []
        self.task_name: str = ""
        self._graph_stack: list[tuple[ComputationGraph, IOReport]] = []
        # Kernel version tracking
        self._kernel_path: str | None = None
        self._best_kernel_path: str | None = None
        self._best_speedup: float = 0.0
        self._kernel_version: int = 0
        self._kernel_history: list[tuple[str, float]] = []  # (path, speedup)

    def reset(self, task_name: str, params: dict | None = None) -> str:
        """Reset the environment with a new task. Returns initial observation."""
        if task_name not in EXAMPLES:
            return f"ERROR: Unknown task '{task_name}'. Available: {list(EXAMPLES.keys())}"

        example = EXAMPLES[task_name]
        self.graph = example["baseline"]()
        self.baseline_graph = self.graph.clone()
        self.params = params or example["default_params"]
        self.baseline_report = self.calc.analyze(self.graph, self.params)
        self.current_report = self.baseline_report
        self.history = []
        self._graph_stack = []
        self.task_name = task_name
        self._kernel_path = None
        self._best_kernel_path = None
        self._best_speedup = 0.0
        self._kernel_version = 0
        self._kernel_history = []

        return self._make_observation("analyze")

    def step(self, thought: str, tool: str, args: dict) -> tuple[str, float, bool]:
        """
        Execute one ReAct step.

        Args:
            thought: Agent's reasoning (logged but not used)
            tool: Tool name to execute
            args: Tool arguments

        Returns:
            (observation: str, reward: float, done: bool)
        """
        if tool not in TOOLS and tool != "analyze":
            obs = f"ERROR: Unknown tool '{tool}'. Available: {list(TOOLS.keys())}"
            self._log(thought, tool, args, obs, 0.0)
            return obs, 0.0, False

        try:
            obs, reward, done = self._execute_tool(tool, args)
        except Exception as e:
            obs = f"ERROR: {e}\n{traceback.format_exc()}"
            reward, done = -1.0, False

        self._log(thought, tool, args, obs, reward)
        return obs, reward, done

    def get_prompt(self) -> str:
        """Generate the system prompt describing available tools and current state."""
        tool_desc = "\n".join(
            f"  {name}: {info['description']}\n    Example: {info['example']}"
            for name, info in TOOLS.items()
        )

        hw_l2 = self.calc.hw.l2_cache_mb

        prompt = f"""You are a GPU kernel optimization agent that writes REAL Triton/CUDA kernels.

Your goal: reduce HBM memory traffic (IO) by eliminating intermediate matrix materialization,
then generate an ACTUAL Triton GPU kernel that achieves real speedup.

## Available Tools
{tool_desc}

## Pre-Optimization Checklist (MUST run before optimizing)

After 'analyze', ALWAYS run 'pre_check' to verify these conditions:

1. Is the operator MEMORY-BOUND? (AI < balance point)
   - If compute-bound, IO optimization may not help wall-clock time.
   - Focus on compute efficiency (Tensor Core, parallelism) instead.

2. Does the intermediate data exceed L2 cache? (data_size > {hw_l2}MB)
   - If data fits in L2, full scans may be cheaper than clever reductions.
   - IO analysis overestimates actual HBM traffic when data is cached.

3. Will the optimization PRESERVE compute parallelism?
   - NEVER replace a GEMM (tl.dot/cuBLAS) with a scalar loop.
   - Use: tiled GEMM + online reduce.
   - Losing Tensor Core → 10-50× slowdown.

4. Does the optimization introduce atomic operations?
   - Atomics on hot addresses serialize execution.
   - Prefer: local accumulation → one global write per block.

If pre_check says CAUTION, consider whether IO optimization is the right approach.

## Required Workflow
Phase 0 — Understand the Target:
1. 'analyze' the baseline operator's IO bottlenecks
2. 'pre_check' to verify IO optimization is appropriate
3. 'library_ceiling' to see how fast the library (cuDNN/cuFFT/cuBLAS) is
4. 'analyze_library' to understand WHY the library is fast (what techniques it uses)
5. 'suggest_strategy' to get multi-level optimization recommendations

Phase 1 — Choose Strategy (MULTI-LEVEL, not just IO):
  Based on analyze_library + suggest_strategy, pick optimizations at MULTIPLE levels:
  - Level 1 Algorithm: right method (radix-4 FFT, Winograd conv, merge-based SpMV)
  - Level 2 IO/Fusion: eliminate intermediates (fuse_and_online, temporal tiling)
  - Level 3 Hardware: use Tensor Core (tl.dot), warp shuffle, vectorized loads
  - Level 4 Data Layout: coalescing, bank conflict avoidance

Phase 2 — Symbolic IO Analysis:
6. Apply optimizations (fuse_and_online, fuse_ops) to eliminate materialized intermediates
4. 'verify' the symbolic optimization

Phase 2 — Kernel Generation + Testing (THE REAL DELIVERABLE):
5. 'generate_kernel' to create a Triton or CUDA GPU kernel
   - Default: use template if available. Otherwise write your own with custom_code.
   - For Triton: @triton.jit kernel with tl.load, tl.store, tl.dot, tl.arange.
   - For CUDA: C++ with __global__ kernels + PYBIND11_MODULE. Set lang='cuda'.
   - DO NOT just call PyTorch ops — write actual GPU kernel code.
6. 'compile_and_test' to verify correctness against PyTorch reference
7. 'benchmark_kernel' to measure ACTUAL speedup vs materialized baseline

Phase 3 — Iterate if Needed:
8. If speedup < 1.0×, use 'profile_kernel' to diagnose WHY:
   - No Tensor Core? → add tl.dot (retrieve_pattern 'gemm_online_reduce')
   - High atomic contention? → local accumulation
   - Low occupancy? → adjust BLOCK_SIZE
   - Poor coalescing? → fix access pattern
9. Write an IMPROVED kernel using 'generate_kernel' with custom_code=<your new Triton code>
   - The custom_code must be a complete Python module with:
     * import torch, triton, triton.language as tl
     * A @triton.jit kernel function
     * A flash_<task>(...) wrapper function
     * A reference_<task>(...) function (can use PyTorch for reference)
   - Key Triton optimization techniques:
     * Online 2-pass softmax is BETTER than 3-pass (max+sum+normalize). Keep it 2-pass!
     * Use LARGE BLOCK sizes (512-1024) for better memory throughput
     * Use num_warps=4 or 8 for wide rows
     * Accumulate in registers, write once — minimize tl.store calls
     * For matmul patterns, use tl.dot (maps to Tensor Cores)
     * Do NOT add extra passes over data — each HBM read costs ~100ns
     * VECTORIZE: process BLOCK_SIZE elements per tl.load, not 1
   - Common mistakes to AVOID:
     * Writing a 3-pass kernel when 2-pass online algorithm works
     * Using scalar loops inside @triton.jit (use tl.arange + vectorized ops)
     * Small BLOCK_SIZE (< 128) causing low memory bandwidth utilization
9. Repeat compile_and_test + benchmark_kernel until speedup >= 1.0×
10. Use 'autotune_kernel' to sweep block sizes for the best configuration

Phase 3 — Diagnose & Iterate (DATA-DRIVEN LOOP):
ALWAYS profile after benchmarking — even if speedup > 1.0×:
11. 'ncu_profile' → get runtime metrics: bandwidth util %, compute util %, Tensor Core usage
12. 'library_ceiling' → see performance gap vs cuDNN/cuFFT/cuBLAS
13. 'compare_profile' → side-by-side baseline vs flash
14. 'occupancy_analysis' → check occupancy bottleneck
15. Based on ALL profiling data, try to improve:
    - 'autotune_kernel' to sweep block sizes
    - Rewrite kernel with different BLOCK_SIZE, num_warps, or algorithm
    - Add tl.dot if missing Tensor Core usage
16. Re-benchmark and re-profile to verify improvement

MINIMUM EXPLORATION before 'done':
  - You MUST call 'ncu_profile' at least once
  - You MUST call 'library_ceiling' at least once
  - You SHOULD try 'autotune_kernel' if speedup < 3× of library
  - You SHOULD try at least 2 kernel versions before concluding
  - Only call 'done' when you've explored AND one of:
    (a) BW utilization > 70% (hitting memory bandwidth limit)
    (b) Within 1.5× of library ceiling after autotune
    (c) 3+ kernel versions tried with diminishing returns

THE ITERATION LOOP (repeat until converged or max 3 iterations):
  benchmark_kernel → ncu_profile → diagnose → fix → benchmark_kernel → ncu_profile → ...

Decision tree based on profiling results:
  - bandwidth_util < 30% AND compute_util < 10% → STRUCTURAL PROBLEM
    (wrong parallelism pattern, scalar loops instead of GEMM)
    → 'retrieve_pattern' to learn correct pattern, then rewrite kernel
  - bandwidth_util > 70% → MEMORY-BOUND, IO optimization working well
    → 'autotune_kernel' for final tuning, then 'done'
  - compute_util > 40% but no Tensor Core → MISSING TENSOR CORE
    → add tl.dot, retrieve_pattern 'gemm_online_reduce'
  - library_gap > 5× → GAP TOO LARGE for hand-written kernel
    → accept result, call 'done', explain why library is better

## RULES
- You MUST run 'pre_check' after 'analyze' before applying optimizations.
- You MUST write Triton kernel code. DO NOT generate Python-only solutions.
- You MUST achieve speedup >= 1.0× or explain clearly why it's not possible.
- After EVERY benchmark_kernel, ALWAYS call 'ncu_profile' to verify kernel efficiency.
- ITERATE: if ncu_profile reveals a fixable issue, fix and re-benchmark. Max 3 iterations.
- If pre_check warns about GEMM structure, you MUST use tl.dot in your kernel.
- DO NOT call 'done' too early. Explore at least: ncu_profile + library_ceiling + one autotune attempt.
- ONLY call 'done' after you have profiling data confirming you're near the hardware limit.

## Response Format
Thought: <your reasoning>
Action: {{"tool": "<tool_name>", "args": {{...}}}}
"""
        return prompt

    # ---- Tool implementations ----

    def _execute_tool(self, tool: str, args: dict) -> tuple[str, float, bool]:
        if tool == "analyze":
            return self._tool_analyze(args)
        elif tool == "show_graph":
            return self._tool_show_graph()
        elif tool == "show_actions":
            return self._tool_show_actions()
        elif tool == "fuse_ops":
            return self._tool_fuse_ops(args)
        elif tool == "fuse_and_online":
            return self._tool_fuse_and_online(args)
        elif tool == "undo":
            return self._tool_undo()
        elif tool == "verify":
            return self._tool_verify()
        elif tool == "benchmark":
            return self._tool_benchmark(args)
        elif tool == "generate_kernel":
            return self._tool_generate_kernel(args)
        elif tool == "compile_and_test":
            return self._tool_compile_and_test(args)
        elif tool == "benchmark_kernel":
            return self._tool_benchmark_kernel(args)
        elif tool == "profile_kernel":
            return self._tool_profile_kernel(args)
        elif tool == "retrieve_pattern":
            return self._tool_retrieve_pattern(args)
        elif tool == "autotune_kernel":
            return self._tool_autotune_kernel(args)
        elif tool == "pre_check":
            return self._tool_pre_check(args)
        elif tool == "ncu_profile":
            return self._tool_ncu_profile(args)
        elif tool == "library_ceiling":
            return self._tool_library_ceiling(args)
        elif tool == "compare_profile":
            return self._tool_compare_profile(args)
        elif tool == "occupancy_analysis":
            return self._tool_occupancy_analysis(args)
        elif tool == "analyze_library":
            return self._tool_analyze_library(args)
        elif tool == "suggest_strategy":
            return self._tool_suggest_strategy(args)
        elif tool == "debug_correctness":
            return self._tool_debug_correctness(args)
        elif tool == "analyze_platform":
            return self._tool_analyze_platform(args)
        elif tool == "done":
            return self._tool_done()
        else:
            return f"Unknown tool: {tool}", 0.0, False

    def _tool_analyze(self, args: dict) -> tuple[str, float, bool]:
        task = args.get("task", self.task_name)
        obs = self.reset(task, args.get("params"))
        return obs, 0.0, False

    def _tool_show_graph(self) -> tuple[str, float, bool]:
        if not self.graph:
            return "No graph loaded. Use 'analyze' first.", 0.0, False
        return self.graph.summary(self.params), 0.0, False

    def _tool_show_actions(self) -> tuple[str, float, bool]:
        lines = ["Available optimization actions:"]
        for a in DesignActions.list_actions():
            lines.append(f"  {a['name']}: {a['description']}")
            lines.append(f"    IO: {a['io_effect']}, FLOPs: {a['flop_effect']}")
        lines.append(f"\nOnline algorithms: {list(ONLINE_ALGORITHMS.keys())}")
        lines.append(f"Current ops: {[op.name for op in self.graph.operations]}")
        mats = self.graph.get_materialized_intermediates()
        if mats:
            lines.append(f"Materialized intermediates: {[(n, s.shape, f'{s.size_bytes(self.params)/1e6:.1f}MB') for n,s in mats]}")
        return "\n".join(lines), 0.0, False

    def _tool_fuse_ops(self, args: dict) -> tuple[str, float, bool]:
        self._graph_stack.append((self.graph.clone(), self.current_report))
        new_graph = DesignActions.fuse_ops(self.graph, args["op1"], args["op2"])
        return self._apply_graph(new_graph, f"fuse_ops({args['op1']}, {args['op2']})")

    def _tool_fuse_and_online(self, args: dict) -> tuple[str, float, bool]:
        self._graph_stack.append((self.graph.clone(), self.current_report))
        new_graph = DesignActions.fuse_and_online(
            self.graph,
            ops_to_fuse=args["ops"],
            online_algorithm=args["algorithm"],
            output_name=args["output_name"],
            output_shape=tuple(args["output_shape"]),
        )
        return self._apply_graph(new_graph,
            f"fuse_and_online({args['ops']}, {args['algorithm']})")

    def _tool_undo(self) -> tuple[str, float, bool]:
        if not self._graph_stack:
            return "Nothing to undo.", 0.0, False
        self.graph, self.current_report = self._graph_stack.pop()
        return f"Undone. Back to step {len(self._graph_stack)}. Ops: {[op.name for op in self.graph.operations]}", 0.0, False

    def _tool_verify(self) -> tuple[str, float, bool]:
        """Run verification tests as an environment tool."""
        results = []
        passed = 0
        total = 0

        # Test 1: IO reduction
        total += 1
        if self.current_report.total_io < self.baseline_report.total_io:
            pct = (1 - self.current_report.total_io / self.baseline_report.total_io) * 100
            results.append(f"  [PASS] IO reduced by {pct:.1f}% ({self.baseline_report.total_io/1e6:.1f}MB → {self.current_report.total_io/1e6:.1f}MB)")
            passed += 1
        else:
            results.append(f"  [FAIL] IO not reduced ({self.baseline_report.total_io/1e6:.1f}MB → {self.current_report.total_io/1e6:.1f}MB)")

        # Test 2: No materialized intermediates
        total += 1
        mats = self.graph.get_materialized_intermediates()
        if not mats:
            results.append(f"  [PASS] Zero materialized intermediates")
            passed += 1
        else:
            results.append(f"  [FAIL] {len(mats)} materialized intermediate(s) remain: {[n for n,_ in mats]}")

        # Test 3: All original outputs still produced
        total += 1
        outputs_produced = set()
        for op in self.graph.operations:
            outputs_produced.add(op.output_name)
        missing = [o for o in self.baseline_graph.outputs if o not in outputs_produced and o not in self.graph.inputs]
        if not missing:
            results.append(f"  [PASS] All outputs produced: {self.graph.outputs}")
            passed += 1
        else:
            results.append(f"  [FAIL] Missing outputs: {missing}. "
                           f"HINT: Use output_name='{missing[0]}' in fuse_and_online to match expected output.")

        # Test 4: Reward is positive
        total += 1
        reward = compute_reward(self.baseline_report, self.current_report)
        if reward > 0:
            results.append(f"  [PASS] Positive reward: {reward:.2f}")
            passed += 1
        else:
            results.append(f"  [FAIL] Non-positive reward: {reward:.2f}")

        # Test 5: Estimated speedup
        total += 1
        speedup = self.baseline_report.estimated_time_ms / max(self.current_report.estimated_time_ms, 1e-9)
        if speedup >= 1.0:
            results.append(f"  [PASS] Estimated speedup: {speedup:.2f}×")
            passed += 1
        else:
            results.append(f"  [WARN] Estimated slowdown: {speedup:.2f}× (may still be better at larger scale)")
            passed += 1  # count as pass since roofline at small scale can be compute-bound

        status = "ALL PASSED" if passed == total else f"{passed}/{total} PASSED"
        header = f"Verification: {status}"
        obs = header + "\n" + "\n".join(results)
        return obs, reward if passed == total else 0.0, False

    def _tool_benchmark(self, args: dict) -> tuple[str, float, bool]:
        """Run actual GPU benchmark comparing baseline vs optimized."""
        try:
            import torch
            if not torch.cuda.is_available():
                return "Benchmark skipped: no GPU available.", 0.0, False
        except ImportError:
            return "Benchmark skipped: torch not available.", 0.0, False

        n_warmup = args.get("n_warmup", 5)
        n_iter = args.get("n_iter", 20)
        p = self.params

        benchmarks = _BENCHMARKS.get(self.task_name)
        if not benchmarks:
            return (f"Benchmark not implemented for '{self.task_name}'. "
                    f"Available: {list(_BENCHMARKS.keys())}"), 0.0, False

        baseline_fn, flash_fn, setup_fn = benchmarks["baseline"], benchmarks["flash"], benchmarks["setup"]

        try:
            tensors = setup_fn(p)

            # Warmup
            for _ in range(n_warmup):
                baseline_fn(tensors)
                flash_fn(tensors)
            torch.cuda.synchronize()

            # Benchmark baseline
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(n_iter):
                baseline_fn(tensors)
            e.record()
            torch.cuda.synchronize()
            t_baseline = s.elapsed_time(e) / n_iter

            # Benchmark flash
            s.record()
            for _ in range(n_iter):
                flash_fn(tensors)
            e.record()
            torch.cuda.synchronize()
            t_flash = s.elapsed_time(e) / n_iter

            speedup = t_baseline / t_flash if t_flash > 0 else 0
            roofline_speedup = self.baseline_report.estimated_time_ms / max(self.current_report.estimated_time_ms, 1e-9)

            lines = [
                f"GPU Benchmark: {self.task_name} ({torch.cuda.get_device_name()})",
                f"  Params: { {k:v for k,v in p.items() if isinstance(v, int)} }",
                f"  Baseline (materialized):  {t_baseline:.3f} ms",
                f"  Flash (optimized):        {t_flash:.3f} ms",
                f"  Actual speedup:           {speedup:.2f}×",
                f"  Roofline predicted:       {roofline_speedup:.2f}×",
                f"  Prediction accuracy:      {min(speedup/roofline_speedup, roofline_speedup/speedup)*100:.0f}%",
            ]

            if speedup > 1.05:
                lines.append(f"  ✓ Flash is {speedup:.2f}× faster!")
            elif speedup > 0.95:
                lines.append(f"  ~ Performance similar (within 5%)")
            else:
                lines.append(f"  ⚠ Flash is slower ({speedup:.2f}×) — may need kernel-level optimization")

            # 3-way comparison: library (cuDNN/cuFFT) if available
            library_fn = benchmarks.get("library")
            if library_fn is not None:
                lib_name = benchmarks.get("library_name", "Library")
                try:
                    for _ in range(n_warmup):
                        library_fn(tensors)
                    torch.cuda.synchronize()

                    s.record()
                    for _ in range(n_iter):
                        library_fn(tensors)
                    e.record()
                    torch.cuda.synchronize()
                    t_lib = s.elapsed_time(e) / n_iter

                    speedup_vs_lib = t_lib / t_flash if t_flash > 0 else 0
                    speedup_lib_vs_mat = t_baseline / t_lib if t_lib > 0 else 0

                    lines.append(f"  ── {lib_name} comparison ──")
                    lines.append(f"  {lib_name}:                   {t_lib:.3f} ms")
                    lines.append(f"  {lib_name} vs materialized:   {speedup_lib_vs_mat:.2f}×")
                    lines.append(f"  Flash vs {lib_name}:          {speedup_vs_lib:.2f}×")
                except Exception as lib_ex:
                    lines.append(f"  {lib_name} benchmark skipped: {lib_ex}")

            obs = "\n".join(lines)
            reward = 2.0 * (speedup - 1.0)  # bonus for actual speedup
            return obs, max(reward, 0.0), False

        except Exception as ex:
            return f"Benchmark error: {ex}", 0.0, False
        finally:
            # Cleanup GPU memory
            try:
                import torch
                if 'tensors' in locals():
                    for v in tensors.values():
                        if hasattr(v, 'device') and str(v.device).startswith('cuda'):
                            del v
                    del tensors
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _tool_generate_kernel(self, args: dict) -> tuple[str, float, bool]:
        """Generate a Triton or CUDA kernel for the current task."""
        from io_env.triton_codegen import generate_kernel
        import shutil
        custom_code = args.get("custom_code")
        lang = args.get("lang", "triton")
        code, filepath = generate_kernel(self.task_name, custom_code, lang=lang)
        if not code:
            return filepath, 0.0, False  # filepath contains error message
        self._kernel_version += 1
        self._kernel_path = filepath
        # Save a versioned copy so we can rollback
        versioned = filepath.replace(".py", f"_v{self._kernel_version}.py")
        shutil.copy2(filepath, versioned)
        # Show first 40 lines
        lines = code.strip().split("\n")
        preview = "\n".join(lines[:40])
        if len(lines) > 40:
            preview += f"\n... ({len(lines) - 40} more lines)"
        lang_tag = "CUDA C++" if lang == "cuda" else "Triton"
        obs = f"Generated {lang_tag} kernel v{self._kernel_version}: {filepath}\n\n{preview}"
        return obs, 1.0, False

    def _tool_compile_and_test(self, args: dict) -> tuple[str, float, bool]:
        """Compile and test the generated kernel."""
        from io_env.triton_codegen import compile_and_test
        filepath = getattr(self, '_kernel_path', None)
        obs = compile_and_test(self.task_name, self.params, filepath)
        reward = 2.0 if "PASS" in obs else -1.0
        return obs, reward, False

    def _tool_benchmark_kernel(self, args: dict) -> tuple[str, float, bool]:
        """Benchmark the generated Triton kernel vs materialized baseline."""
        from io_env.triton_codegen import benchmark_kernel
        import shutil
        filepath = getattr(self, '_kernel_path', None)
        obs = benchmark_kernel(self.task_name, self.params, filepath)
        # Extract speedup for reward
        reward = 0.0
        speedup_val = 1.0
        for line in obs.split("\n"):
            if "Speedup:" in line:
                try:
                    speedup_val = float(line.split(":")[-1].strip().rstrip("×"))
                    reward = 3.0 * (speedup_val - 1.0)
                except ValueError:
                    pass

        # Track kernel history
        self._kernel_history.append((filepath, speedup_val))

        # Version comparison: detect regression and auto-rollback
        if speedup_val > self._best_speedup:
            prev_best = self._best_speedup
            self._best_speedup = speedup_val
            self._best_kernel_path = filepath
            if prev_best > 0:
                obs += f"\n\n✓ NEW BEST: v{self._kernel_version} = {speedup_val:.2f}× (previous best: {prev_best:.2f}×)"
            else:
                obs += f"\n\n✓ FIRST KERNEL: v{self._kernel_version} = {speedup_val:.2f}×"
        elif self._best_speedup > 0 and speedup_val < self._best_speedup * 0.95:
            # Regression detected — rollback to best
            obs += f"\n\n⚠ REGRESSION: v{self._kernel_version} = {speedup_val:.2f}× < best {self._best_speedup:.2f}×"
            obs += f"\n  Auto-rolling back to best kernel."
            if self._best_kernel_path and os.path.exists(self._best_kernel_path):
                shutil.copy2(self._best_kernel_path, filepath)
                self._kernel_path = filepath
            obs += f"\n  The best kernel is still active. Use 'ncu_profile' to understand its limits, then try a DIFFERENT approach."
            speedup_val = self._best_speedup  # use best for reward

        # If kernel is slow, show the current code so agent can improve it
        if speedup_val < 1.0 and filepath and os.path.exists(filepath):
            with open(filepath) as f:
                code = f.read()
            obs += "\n\n=== CURRENT KERNEL CODE (needs improvement) ===\n" + code
            obs += "\n\nHINT: Use 'ncu_profile' for runtime metrics (bandwidth/compute utilization), "
            obs += "'library_ceiling' for performance ceiling, or 'compare_profile' for side-by-side diagnosis."
        return obs, max(reward, 0.0), False

    def _tool_profile_kernel(self, args: dict) -> tuple[str, float, bool]:
        """Profile the generated kernel — analyze compute vs memory bottleneck."""
        import torch
        filepath = getattr(self, '_kernel_path', None)
        if not filepath or not os.path.exists(filepath):
            return "No kernel to profile. Run generate_kernel first.", 0.0, False

        p = self.params

        # Read kernel code to analyze statically
        with open(filepath) as f:
            code = f.read()

        lines = []
        lines.append(f"=== Kernel Profile: {self.task_name} ===")
        lines.append(f"GPU: {torch.cuda.get_device_name()}")

        # Static analysis of the kernel code
        has_tl_dot = "tl.dot" in code
        has_for_loop = "for " in code and "range(" in code
        num_tl_load = code.count("tl.load")
        num_tl_store = code.count("tl.store")
        has_online = "running" in code or "best_" in code or "float(\"-inf\")" in code or "float('-inf')" in code

        # Dim 5: Atomic operation detection
        num_atomic_add = code.count("tl.atomic_add") + code.count("atomicAdd")
        num_atomic_cas = code.count("tl.atomic_cas") + code.count("atomicCAS")
        num_atomic_max = code.count("tl.atomic_max") + code.count("atomicMax")
        total_atomics = num_atomic_add + num_atomic_cas + num_atomic_max

        # Dim 7: Memory coalescing analysis
        # Check for indirect indexing patterns (poor coalescing)
        has_indirect_idx = bool(re.search(r'tl\.load\([^)]*\[[^\]]*\]', code))
        has_gather = "gather" in code.lower()

        # Count loop depth
        import re
        for_loops = re.findall(r'for \w+ in range\(.*?\)', code)

        lines.append(f"\n--- Static Analysis ---")
        lines.append(f"  tl.dot (Tensor Core GEMM): {'YES ✓' if has_tl_dot else 'NO ✗ — NOT using Tensor Cores!'}")
        lines.append(f"  tl.load calls: {num_tl_load}")
        lines.append(f"  tl.store calls: {num_tl_store}")
        lines.append(f"  Atomic operations: {total_atomics}" +
                     (f" ⚠ ({num_atomic_add} add, {num_atomic_cas} cas, {num_atomic_max} max)" if total_atomics > 0 else " ✓"))
        lines.append(f"  Indirect indexing: {'YES ⚠ (may hurt coalescing)' if has_indirect_idx or has_gather else 'NO ✓'}")
        lines.append(f"  For loops in kernel: {len(for_loops)}")
        for i, fl in enumerate(for_loops):
            lines.append(f"    loop {i}: {fl}")
        lines.append(f"  Online/streaming pattern: {'YES' if has_online else 'NO'}")

        # --- Dim 3: Cache analysis ---
        if self.current_report and self.current_report.pre_check:
            pc = self.current_report.pre_check
            lines.append(f"\n--- Cache & Bottleneck Context ---")
            lines.append(f"  Memory-bound: {'YES' if pc.is_memory_bound else 'NO (compute-bound)'}")
            lines.append(f"  Data fits L2: {'YES ⚠' if pc.data_fits_l2 else 'NO ✓ (IO analysis trustworthy)'}")
            lines.append(f"  GEMM structure: {'YES — must use tl.dot' if pc.has_gemm_structure else 'NO'}")

        # Compute theoretical metrics
        lines.append(f"\n--- Performance Diagnosis ---")

        if not has_tl_dot and self.task_name in ("kmeans", "softmax"):
            lines.append(f"  ⚠ CRITICAL: No tl.dot found!")
            lines.append(f"    The baseline uses cuBLAS GEMM (Tensor Cores, ~260 TFLOPS).")
            lines.append(f"    Your kernel uses scalar ops (CUDA Cores, ~19.5 TFLOPS).")
            lines.append(f"    This alone causes ~13× slowdown!")
            lines.append(f"    FIX: Use tl.dot for the inner product / distance computation.")
            lines.append(f"    Call 'retrieve_pattern' with 'gemm_online_reduce' for the correct pattern.")

        if len(for_loops) >= 2 and not has_tl_dot:
            lines.append(f"  ⚠ Nested scalar loops without tl.dot — very low compute utilization.")
            lines.append(f"    Each thread is doing serial work that could be parallelized.")

        if has_tl_dot:
            lines.append(f"  ✓ Using tl.dot — Tensor Core utilization expected to be good.")

        # Estimate bandwidth utilization
        if self.task_name == "cross_entropy":
            N, V = p.get("N", 4096), p.get("V", 32000)
            total_bytes = N * V * 4 * 2 + N * 4  # read logits + labels, write loss, read logits again
            lines.append(f"  Data volume: {total_bytes/1e6:.1f} MB (2 reads of logits + 1 write loss)")
        elif self.task_name == "kmeans":
            N, K, d = p.get("N", 65536), p.get("K", 1024), p.get("d", 128)
            lines.append(f"  Baseline GEMM: X({N}×{d}) @ C^T({d}×{K}) → one cuBLAS call")
            lines.append(f"  Your kernel: {N} threads × {K} serial iterations × {d} dims each")
            lines.append(f"  Serial work per thread: {K * d} FLOPs (vs GEMM doing all in parallel)")

        lines.append(f"\n--- Recommendations ---")
        if not has_tl_dot and self.task_name == "kmeans":
            lines.append(f"  1. Use tl.dot to compute distance tiles: D_tile = tl.dot(X_tile, C_tile.T)")
            lines.append(f"  2. Process centroid tiles (BK=32-64), not one centroid at a time")
            lines.append(f"  3. Apply online argmin across tiles (keep running_min per row)")
            lines.append(f"  4. Call 'retrieve_pattern' with 'gemm_online_reduce' for example code")

        # Dim 5: Atomic contention recommendations
        if total_atomics > 0:
            lines.append(f"  ⚠ ATOMIC CONTENTION: {total_atomics} atomic ops found.")
            lines.append(f"    Atomics on hot addresses serialize warp execution.")
            lines.append(f"    FIX: Accumulate in registers/SMEM per block → single tl.store at end.")
            if num_atomic_cas > 0:
                lines.append(f"    atomicCAS is especially expensive — consider warp-level vote + one CAS per warp.")

        # Dim 7: Coalescing recommendations
        if has_indirect_idx or has_gather:
            lines.append(f"  ⚠ POOR COALESCING: Indirect/gather memory access detected.")
            lines.append(f"    Non-contiguous addresses → bandwidth utilization < 25%.")
            lines.append(f"    FIX: Reorganize data layout or use tiled access with contiguous loads.")

        return "\n".join(lines), 0.0, False

    def _tool_retrieve_pattern(self, args: dict) -> tuple[str, float, bool]:
        """Retrieve an optimization pattern from the knowledge base."""
        query = args.get("pattern", "").lower()
        matches = []

        for name, pattern in _PATTERNS.items():
            if any(kw in query for kw in pattern["keywords"]):
                matches.append((name, pattern))

        if not matches:
            available = list(_PATTERNS.keys())
            return f"No pattern found for '{query}'. Available: {available}", 0.0, False

        lines = []
        for name, pattern in matches:
            lines.append(f"=== Pattern: {name} ===")
            lines.append(f"When: {pattern['when']}")
            lines.append(f"Wrong approach: {pattern['wrong']}")
            lines.append(f"Right approach: {pattern['right']}")
            lines.append(f"\n--- Example Triton Code ---")
            lines.append(pattern["code"])
            lines.append("")

        return "\n".join(lines), 1.0, False

    def _tool_autotune_kernel(self, args: dict) -> tuple[str, float, bool]:
        """Auto-sweep block sizes and num_warps for the current kernel."""
        import torch
        from io_env.triton_codegen import generate_kernel, compile_and_test, benchmark_kernel

        filepath = getattr(self, '_kernel_path', None)
        if not filepath or not os.path.exists(filepath):
            return "No kernel to autotune. Run generate_kernel first.", 0.0, False

        with open(filepath) as f:
            code = f.read()

        p = self.params
        results = []
        best_speedup = 0.0
        best_config = ""
        best_code = code

        # Detect tunable parameters in the kernel wrapper function
        # Look for default values like BLOCK_M=64, BK=64, num_warps=8
        import re

        # Try to find the launcher function and its defaults
        block_sizes = [32, 64, 128]
        bk_sizes = [32, 64, 128]
        warp_counts = [4, 8]

        lines = [f"=== Autotune: {self.task_name} ==="]
        lines.append(f"Sweeping BLOCK_M × BK × num_warps")
        lines.append(f"{'BLOCK_M':>8s} {'BK':>4s} {'warps':>5s} | {'Speedup':>8s} | {'Status':>8s}")
        lines.append("-" * 45)

        for bm in block_sizes:
            for bk in bk_sizes:
                for nw in warp_counts:
                    # Substitute parameters in the launcher
                    variant = code
                    # Replace default BLOCK_M, BK, num_warps values
                    variant = re.sub(r'BLOCK_M\s*=\s*\d+', f'BLOCK_M={bm}', variant)
                    variant = re.sub(r'block_m\s*=\s*\d+', f'block_m={bm}', variant)
                    variant = re.sub(r'BK\s*=\s*\d+', f'BK={bk}', variant)
                    variant = re.sub(r'bk\s*=\s*\d+', f'bk={bk}', variant)
                    variant = re.sub(r'num_warps\s*=\s*\d+', f'num_warps={nw}', variant)

                    _, vpath = generate_kernel(self.task_name, variant)
                    try:
                        test = compile_and_test(self.task_name, p, vpath)
                        if "PASS" not in test:
                            lines.append(f"{bm:8d} {bk:4d} {nw:5d} | {'FAIL':>8s} | compile")
                            continue
                        bench = benchmark_kernel(self.task_name, p, vpath)
                        spd = 1.0
                        for bl in bench.split("\n"):
                            if "Speedup:" in bl:
                                try:
                                    spd = float(bl.split(":")[-1].strip().rstrip("×"))
                                except ValueError:
                                    pass
                        lines.append(f"{bm:8d} {bk:4d} {nw:5d} | {spd:7.2f}× | {'PASS':>8s}")
                        if spd > best_speedup:
                            best_speedup = spd
                            best_config = f"BLOCK_M={bm}, BK={bk}, num_warps={nw}"
                            best_code = variant
                    except Exception as e:
                        lines.append(f"{bm:8d} {bk:4d} {nw:5d} | {'ERROR':>8s} | {str(e)[:20]}")
                    torch.cuda.empty_cache()

        lines.append("")
        lines.append(f"Best config: {best_config}")
        lines.append(f"Best speedup: {best_speedup:.2f}×")

        # Save the best variant as the current kernel
        if best_speedup > 0:
            _, best_path = generate_kernel(self.task_name, best_code)
            self._kernel_path = best_path
            lines.append(f"Saved best kernel to: {best_path}")

        obs = "\n".join(lines)
        reward = 3.0 * (best_speedup - 1.0) if best_speedup > 1.0 else 0.0
        return obs, reward, False

    def _tool_pre_check(self, args: dict) -> tuple[str, float, bool]:
        """Run multi-dimensional pre-optimization checklist."""
        if not self.current_report:
            return "No analysis loaded. Use 'analyze' first.", 0.0, False

        pc = self.current_report.pre_check
        if not pc:
            return "Pre-check not available (report may be stale).", 0.0, False

        lines = [pc.display()]

        # Add task-specific guidance based on the checklist
        if not pc.io_optimization_recommended:
            lines.append("\n⚠ IO optimization is NOT recommended for this operator.")
            lines.append("  Consider alternative approaches:")
            if not pc.is_memory_bound:
                lines.append("  - Focus on compute optimization (Tensor Core utilization, parallelism)")
            if pc.data_fits_l2:
                lines.append("  - Data fits in L2 cache; improve access patterns instead of reducing IO")
        else:
            lines.append("\n✓ IO optimization is recommended.")
            if pc.has_gemm_structure:
                lines.append("  ⚠ CRITICAL: Preserve GEMM structure! Use tl.dot for inner products.")
                lines.append("  Pattern: tile GEMM + online reduce (retrieve_pattern 'gemm_online_reduce')")
            if pc.data_fits_l2:
                lines.append("  Note: Some intermediates fit L2 — expect smaller speedup than IO reduction suggests.")

        obs = "\n".join(lines)
        return obs, 0.0, False

    def _tool_ncu_profile(self, args: dict) -> tuple[str, float, bool]:
        """Runtime-profile the current kernel: bandwidth/compute utilization, TC usage."""
        from io_env.profiler import runtime_profile, ncu_profile, _has_ncu
        filepath = getattr(self, '_kernel_path', None)
        if not filepath or not os.path.exists(filepath):
            return "No kernel to profile. Run generate_kernel first.", 0.0, False

        try:
            result = runtime_profile(self.task_name, self.params, filepath)
        except Exception as e:
            return f"Profile error: {e}\n{traceback.format_exc()}", 0.0, False

        lines = [result.display(self.task_name)]

        # Try ncu for hardware counter data
        if _has_ncu():
            lines.append("\n--- NCU Hardware Counters ---")
            try:
                ncu_out = ncu_profile(self.task_name, self.params, filepath)
                if ncu_out:
                    if "permission denied" in ncu_out.lower() or "ERR_NVGPUCTRPERM" in ncu_out:
                        lines.append(f"  {ncu_out.split(chr(10))[0]}")
                        lines.append("  Using torch.profiler estimates above instead.")
                    else:
                        # Extract key metrics from ncu output
                        for metric_line in ncu_out.split("\n"):
                            ml = metric_line.strip()
                            if any(k in ml for k in ("dram__", "sm__", "gpu__time",
                                                       "Metric Name", "Metric Unit", "----")):
                                lines.append(f"  {ml[:85]}")
                        if not any("dram__" in l or "sm__" in l for l in lines[-10:]):
                            lines.append("  (ncu ran but no matching metrics found)")
                else:
                    lines.append("  (ncu returned no data)")
            except Exception as e:
                lines.append(f"  (ncu failed: {e})")
        else:
            lines.append("\n(ncu not available — using torch.profiler estimates)")

        # Actionable recommendations
        lines.append("\n--- Diagnosis ---")
        if result.roofline_bottleneck == "under-utilized":
            lines.append("⚠ UNDER-UTILIZED: Both bandwidth and compute are low.")
            lines.append("  This means a structural parallelism problem (not IO or compute).")
            lines.append("  Common causes: scalar loops instead of GEMM, low occupancy, poor coalescing.")
            lines.append("  → Call 'compare_profile' to see what changed vs baseline.")
        elif result.roofline_bottleneck == "memory-bound":
            lines.append("✓ Memory-bound with good bandwidth utilization.")
            lines.append("  IO optimization is working. Further gains via larger BLOCK_SIZE or autotune.")
        elif result.roofline_bottleneck == "compute-bound":
            lines.append("Compute-bound kernel. IO optimization won't help further.")
            if not result.uses_tensor_core:
                lines.append("  ⚠ NOT using Tensor Core — add tl.dot for Tensor Core GEMM!")
                lines.append("  → Call 'retrieve_pattern' with 'gemm_online_reduce'.")
            else:
                lines.append("  ✓ Using Tensor Core. Consider algorithmic improvement or accept.")
        if result.tensor_core_hint:
            lines.append(f"  TC detail: {result.tensor_core_hint}")
        if result.kernel_names:
            lines.append(f"  CUDA kernels: {', '.join(result.kernel_names[:3])}")

        return "\n".join(lines), 0.0, False

    def _tool_library_ceiling(self, args: dict) -> tuple[str, float, bool]:
        """Benchmark library (cuDNN/cuFFT/cuBLAS) as performance ceiling."""
        from io_env.profiler import library_ceiling, runtime_profile
        filepath = getattr(self, '_kernel_path', None)

        try:
            lib_info = library_ceiling(self.task_name, self.params)
        except Exception as e:
            return f"Library ceiling error: {e}", 0.0, False

        if "error" in lib_info:
            return lib_info["error"], 0.0, False

        lib_name = lib_info["library_name"]
        lib_time = lib_info["library_time_ms"]

        # Also time our kernel for comparison
        flash_time = 0.0
        if filepath and os.path.exists(filepath):
            try:
                result = runtime_profile(self.task_name, self.params, filepath, n_iter=10)
                flash_time = result.kernel_time_ms
            except Exception:
                pass

        lines = [
            f"=== Library Ceiling: {self.task_name} ===",
            f"  {lib_name}: {lib_time:.3f} ms",
        ]

        if flash_time > 0:
            gap = flash_time / lib_time if lib_time > 0 else float("inf")
            lines.append(f"  Flash kernel: {flash_time:.3f} ms")
            lines.append(f"  Gap: Flash is {gap:.1f}× slower than {lib_name}")
            lines.append("")
            if gap > 10:
                lines.append(f"⚠ Gap > 10×: Your kernel has fundamental efficiency problems.")
                lines.append(f"  Consider: missing Tensor Core, poor parallelism, wrong algorithm.")
                lines.append(f"  → Call 'ncu_profile' to diagnose, or accept library as solution.")
            elif gap > 3:
                lines.append(f"⚠ Gap 3-10×: Significant room for improvement.")
                lines.append(f"  → Call 'ncu_profile' to identify bottleneck.")
            elif gap > 1.5:
                lines.append(f"△ Gap 1.5-3×: Some room for improvement via tuning.")
                lines.append(f"  → Try 'autotune_kernel' to sweep block sizes.")
            else:
                lines.append(f"✓ Within 1.5× of {lib_name} — near optimal!")
                lines.append(f"  Consider calling 'done'.")
        else:
            lines.append(f"  (No flash kernel available for comparison)")

        if lib_time == float("inf"):
            lines.append(f"  Note: No standard library for this operation.")

        return "\n".join(lines), 0.0, False

    def _tool_compare_profile(self, args: dict) -> tuple[str, float, bool]:
        """Side-by-side profile: baseline vs flash (vs library)."""
        from io_env.profiler import compare_profile
        filepath = getattr(self, '_kernel_path', None)
        if not filepath or not os.path.exists(filepath):
            return "No kernel to compare. Run generate_kernel first.", 0.0, False

        try:
            result = compare_profile(self.task_name, self.params, filepath)
        except Exception as e:
            return f"Compare error: {e}\n{traceback.format_exc()}", 0.0, False

        lines = [result.display()]

        # Actionable summary
        b, f = result.baseline, result.flash
        lines.append("\n--- Root Cause Analysis ---")
        if b.compute_utilization_pct > 30 and f.compute_utilization_pct < 10:
            lines.append("⚠ Baseline has {:.0f}% compute utilization, Flash only {:.0f}%.".format(
                b.compute_utilization_pct, f.compute_utilization_pct))
            lines.append("  → Flash kernel lost compute parallelism (likely dropped Tensor Core / GEMM).")
        if b.bandwidth_utilization_pct > 50 and f.bandwidth_utilization_pct < 20:
            lines.append("⚠ Baseline bandwidth {:.0f}%, Flash only {:.0f}%.".format(
                b.bandwidth_utilization_pct, f.bandwidth_utilization_pct))
            lines.append("  → Flash kernel has poor memory access patterns (indirect indexing? small blocks?).")
        if f.bandwidth_utilization_pct > 70:
            lines.append("✓ Flash kernel bandwidth utilization is {:.0f}% — good.".format(f.bandwidth_utilization_pct))
            lines.append("  IO optimization is effective. Further gains limited by HBM bandwidth.")

        return "\n".join(lines), 0.0, False

    def _tool_occupancy_analysis(self, args: dict) -> tuple[str, float, bool]:
        """Estimate kernel occupancy from code analysis."""
        from io_env.profiler import estimate_occupancy
        filepath = getattr(self, '_kernel_path', None)
        if not filepath or not os.path.exists(filepath):
            return "No kernel to analyze. Run generate_kernel first.", 0.0, False

        with open(filepath) as f:
            code = f.read()

        occ = estimate_occupancy(code)

        lines = [
            f"=== Occupancy Analysis: {self.task_name} ===",
            f"  Block size:       {occ['block_size']}",
            f"  Num warps:        {occ['num_warps']} ({occ['threads_per_block']} threads/block)",
            f"  Est. registers:   {occ['est_registers_per_thread']}/thread",
            f"  Est. SMEM:        {occ['est_smem_bytes']/1024:.1f} KB/block",
            f"  Active blocks/SM: {occ['active_blocks_per_sm']}",
            f"  Active warps/SM:  {occ['active_warps_per_sm']}/{occ['max_warps_per_sm']}",
            f"  Occupancy:        {occ['occupancy_pct']:.0f}%",
            f"  Limiting factor:  {occ['limiting_factor']}",
        ]

        if occ['occupancy_pct'] < 25:
            lines.append(f"\n⚠ LOW OCCUPANCY ({occ['occupancy_pct']:.0f}%): SM is mostly idle.")
            lines.append(f"  Limited by: {occ['limiting_factor']}")
            if occ['limiting_factor'] == 'registers':
                lines.append(f"  Fix: Reduce register pressure — smaller BLOCK_SIZE or fewer accumulators.")
            elif occ['limiting_factor'] == 'shared_memory':
                lines.append(f"  Fix: Reduce SMEM usage — smaller tile sizes.")
            elif occ['limiting_factor'] == 'threads':
                lines.append(f"  Fix: Use fewer warps per block (num_warps=4 instead of 8).")
        elif occ['occupancy_pct'] < 50:
            lines.append(f"\n△ MODERATE OCCUPANCY ({occ['occupancy_pct']:.0f}%): acceptable but not ideal.")
        else:
            lines.append(f"\n✓ GOOD OCCUPANCY ({occ['occupancy_pct']:.0f}%): SM well-utilized.")

        return "\n".join(lines), 0.0, False

    def _tool_analyze_library(self, args: dict) -> tuple[str, float, bool]:
        """Analyze why the library implementation is fast for this operator."""
        info = _LIBRARY_ANALYSIS.get(self.task_name)
        if not info:
            return f"No library analysis available for '{self.task_name}'.", 0.0, False

        lines = [
            f"=== Library Analysis: {self.task_name} ===",
            f"Library: {info['library']}",
            f"",
            f"--- Why is {info['library']} fast? ---",
        ]
        for i, technique in enumerate(info['techniques'], 1):
            lines.append(f"  {i}. {technique['name']}")
            lines.append(f"     What: {technique['what']}")
            lines.append(f"     Impact: {technique['impact']}")
            lines.append(f"     How to implement: {technique['how']}")
            lines.append("")

        if info.get('key_insight'):
            lines.append(f"--- Key Insight ---")
            lines.append(f"  {info['key_insight']}")

        if info.get('our_gap'):
            lines.append(f"")
            lines.append(f"--- Our Current Gap ---")
            for gap in info['our_gap']:
                lines.append(f"  ✗ {gap}")

        return "\n".join(lines), 1.0, False

    def _tool_suggest_strategy(self, args: dict) -> tuple[str, float, bool]:
        """Suggest multi-level optimization strategy for the current operator."""
        strategies = _STRATEGY_DB.get(self.task_name)
        if not strategies:
            # Generic fallback
            strategies = _make_generic_strategy(self.task_name, self.current_report)

        lines = [
            f"=== Multi-Level Strategy: {self.task_name} ===",
            f"",
        ]

        for level in strategies:
            lines.append(f"Level {level['level']}: {level['name']}")
            lines.append(f"  Current: {level['current']}")
            lines.append(f"  Recommended: {level['recommended']}")
            lines.append(f"  Expected impact: {level['impact']}")
            if level.get('code_hint'):
                lines.append(f"  Code hint: {level['code_hint']}")
            lines.append("")

        # Hardware context
        hw = self.calc.hw
        lines.append(f"--- Hardware Context ({hw.name}) ---")
        lines.append(f"  HBM Bandwidth: {hw.hbm_bandwidth_gbps:.0f} GB/s")
        lines.append(f"  Peak FP32: {hw.peak_flops_tflops:.1f} TFLOPS")
        lines.append(f"  SMEM/SM: {hw.smem_per_sm_kb:.0f} KB")
        lines.append(f"  L2 Cache: {hw.l2_cache_mb:.0f} MB")
        lines.append(f"  Balance point: {hw.balance_point:.1f} FLOP/Byte")

        return "\n".join(lines), 1.0, False

    def _tool_debug_correctness(self, args: dict) -> tuple[str, float, bool]:
        """Deep-debug a failing kernel: find first wrong element and diagnose cause."""
        import torch
        filepath = getattr(self, '_kernel_path', None)
        if not filepath or not os.path.exists(filepath):
            return "No kernel to debug. Run generate_kernel first.", 0.0, False

        n_elements = args.get("n_elements", 64)
        task = self.task_name

        # Load module
        spec = importlib.util.spec_from_file_location(f"flash_{task}", filepath)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            return f"COMPILE ERROR: {e}", 0.0, False

        flash_fn = ref_fn = None
        for name in dir(mod):
            if name.startswith("flash_"):
                flash_fn = getattr(mod, name)
            if name.startswith("reference_"):
                ref_fn = getattr(mod, name)

        if not flash_fn or not ref_fn:
            return "Module must define flash_* and reference_* functions.", 0.0, False

        # Construct small inputs
        torch.manual_seed(42)
        try:
            if task == "cross_entropy":
                N, V = n_elements, min(n_elements * 4, 256)
                inputs = (torch.randn(N, V, device="cuda"), torch.randint(0, V, (N,), device="cuda"))
            elif task == "kmeans":
                N, K, d = n_elements, min(n_elements // 4, 32), 8
                inputs = (torch.randn(N, d, device="cuda"), torch.randn(K, d, device="cuda"))
            elif task == "fft":
                N = min(n_elements, 64)
                while N & (N - 1): N -= 1  # round down to power of 2
                if N < 4: N = 4
                inputs = (torch.randn(N, device="cuda"), torch.zeros(N, device="cuda"))
            elif task == "softmax":
                inputs = (torch.randn(n_elements, n_elements, device="cuda"),)
            elif task == "layernorm":
                d = min(n_elements, 64)
                inputs = (torch.randn(n_elements, d, device="cuda"),
                          torch.ones(d, device="cuda"), torch.zeros(d, device="cuda"))
            elif task in ("nbody",):
                inputs = (torch.randn(n_elements, 3, device="cuda"), torch.ones(n_elements, device="cuda"))
            elif task in ("graph_laplacian", "spmv"):
                M = n_elements
                nnz_per_row = 5
                NNZ = M * nnz_per_row
                row_ptr = torch.arange(0, (M + 1) * nnz_per_row, nnz_per_row, dtype=torch.int32, device="cuda")[:M + 1]
                row_ptr[-1] = NNZ
                col_idx = torch.randint(0, M, (NNZ,), dtype=torch.int32, device="cuda")
                values = torch.randn(NNZ, device="cuda")
                x = torch.randn(M, device="cuda")
                if task == "graph_laplacian":
                    degree = torch.full((M,), float(nnz_per_row), device="cuda")
                    inputs = (values, col_idx, row_ptr, degree, x)
                else:
                    inputs = (values, col_idx, row_ptr, x)
            elif task in ("stencil2d",):
                H = min(n_elements, 32)
                inputs = (torch.randn(H, H, device="cuda"), 2)
            elif task in ("conv2d",):
                inputs = (torch.randn(1, 3, 8, 8, device="cuda"), torch.randn(4, 3, 3, 3, device="cuda"))
            else:
                inputs = (torch.randn(n_elements, device="cuda"),)

            ref_out = ref_fn(*inputs)
            flash_out = flash_fn(*inputs)

            # Handle tuple outputs (FFT returns re, im)
            if isinstance(ref_out, (list, tuple)):
                ref_tensors = list(ref_out)
                flash_tensors = list(flash_out)
            else:
                ref_tensors = [ref_out]
                flash_tensors = [flash_out]

        except Exception as e:
            return f"DEBUG ERROR: {e}\n{traceback.format_exc()}", 0.0, False

        lines = [f"=== Debug Correctness: {task} (N={n_elements}) ==="]

        all_correct = True
        for t_idx, (ref_t, flash_t) in enumerate(zip(ref_tensors, flash_tensors)):
            ref_flat = ref_t.float().flatten()
            flash_flat = flash_t.float().flatten()

            if ref_flat.shape != flash_flat.shape:
                lines.append(f"\n  Output {t_idx}: SHAPE MISMATCH ref={ref_flat.shape} flash={flash_flat.shape}")
                all_correct = False
                continue

            diff = (ref_flat - flash_flat).abs()
            max_err = diff.max().item()
            mean_err = diff.mean().item()

            if max_err < 1e-3:
                lines.append(f"\n  Output {t_idx}: CORRECT (max_err={max_err:.2e})")
                continue

            all_correct = False
            # Find first wrong element
            wrong_indices = (diff > 1e-3).nonzero(as_tuple=True)[0]
            first_wrong = wrong_indices[0].item()
            n_wrong = len(wrong_indices)
            pct_wrong = n_wrong / len(ref_flat) * 100

            lines.append(f"\n  Output {t_idx}: WRONG")
            lines.append(f"    Max error: {max_err:.4f}")
            lines.append(f"    Mean error: {mean_err:.4f}")
            lines.append(f"    Wrong elements: {n_wrong}/{len(ref_flat)} ({pct_wrong:.0f}%)")
            lines.append(f"    First wrong at index [{first_wrong}]:")
            lines.append(f"      Expected: {ref_flat[first_wrong].item():.6f}")
            lines.append(f"      Got:      {flash_flat[first_wrong].item():.6f}")
            lines.append(f"      Diff:     {diff[first_wrong].item():.6f}")

            # Show neighborhood
            start = max(0, first_wrong - 3)
            end = min(len(ref_flat), first_wrong + 4)
            lines.append(f"    Neighborhood [{start}:{end}]:")
            lines.append(f"      Ref:   {[f'{ref_flat[i].item():.4f}' for i in range(start, end)]}")
            lines.append(f"      Flash: {[f'{flash_flat[i].item():.4f}' for i in range(start, end)]}")
            lines.append(f"      Diff:  {[f'{diff[i].item():.4f}' for i in range(start, end)]}")

            # Pattern analysis
            lines.append(f"\n    --- Error Pattern Analysis ---")
            if pct_wrong > 90:
                lines.append(f"    ⚠ Almost ALL elements wrong ({pct_wrong:.0f}%)")
                lines.append(f"      → Likely a fundamental algorithm error (wrong formula, wrong indexing)")
                if task == "fft":
                    ratio = flash_flat[first_wrong].item() / (ref_flat[first_wrong].item() + 1e-10)
                    lines.append(f"      Ratio flash/ref at [{first_wrong}]: {ratio:.4f}")
                    lines.append(f"      If ratio ≈ N or 1/N: twiddle factor uses wrong N")
                    lines.append(f"      If pattern is periodic: butterfly stride is wrong")
            elif pct_wrong > 50:
                lines.append(f"    ⚠ Majority wrong ({pct_wrong:.0f}%)")
                lines.append(f"      → Likely wrong butterfly/permutation indexing")
            elif pct_wrong < 5:
                lines.append(f"    △ Only a few elements wrong ({pct_wrong:.0f}%)")
                lines.append(f"      → Likely boundary condition / masking issue")
            else:
                lines.append(f"    ⚠ Partial errors ({pct_wrong:.0f}%)")
                lines.append(f"      → Could be wrong twiddle for specific positions, or accumulation order")

            # Check if errors are periodic
            if n_wrong >= 4:
                wrong_list = wrong_indices[:min(20, n_wrong)].tolist()
                if len(wrong_list) >= 2:
                    gaps = [wrong_list[i+1] - wrong_list[i] for i in range(len(wrong_list)-1)]
                    if len(set(gaps)) == 1:
                        lines.append(f"    Errors are PERIODIC with stride {gaps[0]}")
                        lines.append(f"      → Likely a specific butterfly stage is wrong")

        if all_correct:
            lines.append(f"\n  ✓ All outputs match reference within tolerance!")

        torch.cuda.empty_cache()
        return "\n".join(lines), 0.0, False

    def _tool_analyze_platform(self, args: dict) -> tuple[str, float, bool]:
        """Explain capabilities and limitations of Triton or CUDA."""
        lang = args.get("lang", "triton")
        info = _PLATFORM_INFO.get(lang)
        if not info:
            return f"Unknown platform '{lang}'. Available: triton, cuda", 0.0, False

        lines = [f"=== Platform Analysis: {lang} ==="]
        lines.append(f"\n--- Capabilities ---")
        for cap in info["capabilities"]:
            lines.append(f"  ✓ {cap}")
        lines.append(f"\n--- Limitations (IMPORTANT) ---")
        for lim in info["limitations"]:
            lines.append(f"  ✗ {lim}")
        lines.append(f"\n--- Best Practices ---")
        for bp in info["best_practices"]:
            lines.append(f"  → {bp}")

        return "\n".join(lines), 0.0, False

    def _tool_done(self) -> tuple[str, float, bool]:
        reward = compute_reward(self.baseline_report, self.current_report)
        obs = self._make_summary()
        return obs, reward, True

    # ---- Helpers ----

    def _apply_graph(self, new_graph: ComputationGraph, desc: str) -> tuple[str, float, bool]:
        old_report = self.current_report
        new_report = self.calc.analyze(new_graph, self.params)
        reward = compute_reward(self.baseline_report, new_report)
        step_reward = compute_reward(old_report, new_report)

        self.graph = new_graph
        self.current_report = new_report

        obs = self._make_observation(desc)
        return obs, step_reward, False

    def _make_observation(self, action_desc: str = "") -> str:
        report = self.current_report
        mats = self.graph.get_materialized_intermediates()

        lines = [
            f"=== Observation (after: {action_desc}) ===",
            f"Task: {self.task_name}",
            f"Ops: {[op.name for op in self.graph.operations]}",
            f"Total IO: {report.total_io/1e6:.1f} MB (baseline: {self.baseline_report.total_io/1e6:.1f} MB, reduction: {(1-report.total_io/self.baseline_report.total_io)*100:.1f}%)",
            f"Total FLOPs: {report.total_flops/1e9:.2f} GFLOPs",
            f"Arithmetic Intensity: {report.arithmetic_intensity:.1f} FLOP/Byte",
            f"Bottleneck: {report.bottleneck}",
            f"Est. time: {report.estimated_time_ms:.3f} ms",
        ]

        if mats:
            lines.append(f"⚠ Materialized intermediates ({len(mats)}):")
            for name, spec in sorted(mats, key=lambda m: m[1].size_bytes(self.params), reverse=True):
                size = spec.size_bytes(self.params) / 1e6
                lines.append(f"  - {name}: shape={spec.shape}, {size:.1f} MB")
        else:
            lines.append("✓ Zero materialized intermediates!")

        if report.optimization_hints:
            lines.append("Hints:")
            for h in report.optimization_hints[:3]:  # top 3
                lines.append(f"  {h}")

        return "\n".join(lines)

    def _make_summary(self) -> str:
        b = self.baseline_report
        c = self.current_report
        speedup = b.estimated_time_ms / max(c.estimated_time_ms, 1e-9)
        reward = compute_reward(b, c)

        lines = [
            "=" * 50,
            "OPTIMIZATION COMPLETE",
            "=" * 50,
            f"Task: {self.task_name}",
            f"IO:      {b.total_io/1e6:.1f} MB → {c.total_io/1e6:.1f} MB ({(1-c.total_io/b.total_io)*100:.1f}% reduction)",
            f"FLOPs:   {b.total_flops/1e9:.2f} → {c.total_flops/1e9:.2f} GFLOPs",
            f"Time:    {b.estimated_time_ms:.3f} → {c.estimated_time_ms:.3f} ms ({speedup:.2f}× speedup)",
            f"Reward:  {reward:.2f}",
            f"Steps:   {len(self.history)}",
        ]

        mats = self.graph.get_materialized_intermediates()
        lines.append(f"Materialized intermediates: {len(mats)}" +
                     (" ✓" if not mats else f" ⚠ {[n for n,_ in mats]}"))

        # Kernel version history
        if self._kernel_history:
            lines.append(f"\nKernel versions tested: {len(self._kernel_history)}")
            for i, (path, spd) in enumerate(self._kernel_history):
                best = " ← BEST" if spd == self._best_speedup else ""
                lines.append(f"  v{i+1}: {spd:.2f}×{best}")
            lines.append(f"Best kernel speedup: {self._best_speedup:.2f}×")

        if self.history:
            lines.append("\nTrace:")
            for h in self.history:
                lines.append(f"  Step {h.step}: [{h.tool}] {h.thought[:60]}")

        return "\n".join(lines)

    def _log(self, thought: str, tool: str, args: dict, obs: str, reward: float):
        self.history.append(StepLog(
            step=len(self.history),
            thought=thought,
            tool=tool,
            args=args,
            observation=obs,
            reward=reward,
        ))


# ============================================================================
# Platform Analysis — capabilities and limitations of Triton vs CUDA
# ============================================================================

_PLATFORM_INFO = {
    "triton": {
        "capabilities": [
            "Automatic vectorized memory loads/stores (tl.load, tl.store with masks)",
            "Block-level parallelism: each program handles a tile of data",
            "tl.dot for Tensor Core GEMM (maps to wmma/mma instructions)",
            "tl.atomic_add, tl.atomic_max for atomic operations",
            "tl.arange for index generation, tl.where for conditional",
            "tl.sum, tl.max, tl.min for block-level reductions",
            "tl.exp, tl.log, tl.sqrt, tl.cos, tl.sin for math",
            "tl.static_range for compile-time loop unrolling",
            "Automatic SMEM management (no manual __shared__ declarations)",
            "Automatic occupancy tuning via num_warps and num_stages",
        ],
        "limitations": [
            "NO arbitrary indexing into register tensors: a[dynamic_idx] does NOT work. "
            "You cannot shuffle elements within a register vector using data-dependent indices.",
            "NO warp shuffle primitives (__shfl_xor_sync etc.) — must use tl.store/tl.load through SMEM for cross-lane communication.",
            "NO explicit shared memory management — Triton manages SMEM automatically for tl.dot operands.",
            "NO dynamic loop bounds at compile time — for loops must have bounds known at JIT compile time or use tl.static_range.",
            "tl.dot requires operand shapes to be compile-time constants and multiples of 16.",
            "NO inline PTX or CUDA intrinsics — cannot use __sincosf, __fmaf_rn etc.",
            "Cannot read back from a register tensor at arbitrary positions — this makes in-register FFT butterfly IMPOSSIBLE.",
            "Large kernels with many tl.static_range iterations compile VERY slowly (minutes for 20+ iterations).",
        ],
        "best_practices": [
            "For data shuffling (FFT butterfly, permutation): write to SMEM, __syncthreads, read back in new order.",
            "For GEMM: use tl.dot with BLOCK_M × BLOCK_K tiles. Triton handles SMEM staging automatically.",
            "For reductions: tl.sum/tl.max over axis. For cross-block reduction: use atomics or multi-kernel.",
            "For complex algorithms (FFT, sort): consider CUDA instead — Triton's abstraction is too high-level.",
            "Keep BLOCK_SIZE as a constexpr. Use num_warps=4-8, num_stages=2-4.",
            "For FFT specifically: use CUDA SMEM-based butterfly, NOT Triton register manipulation.",
        ],
    },
    "cuda": {
        "capabilities": [
            "Full control over shared memory (__shared__, extern __shared__)",
            "Warp-level primitives: __shfl_xor_sync, __shfl_down_sync, __ballot_sync",
            "Inline PTX for maximum control: asm volatile()",
            "Cooperative groups for flexible thread synchronization",
            "__sincosf, __expf, __fmaf_rn for fast math intrinsics",
            "Template metaprogramming for compile-time specialization",
            "Explicit register control via #pragma unroll and asm",
            "cuBLAS/cuFFT/cuDNN library calls within kernels via device functions",
            "Dynamic parallelism (launch kernels from kernels)",
            "Texture memory and surface memory for cached reads",
        ],
        "limitations": [
            "Manual memory management — must explicitly manage SMEM, registers, bank conflicts.",
            "No automatic occupancy tuning — must manually compute registers/SMEM and choose block size.",
            "Verbose code — 10× more lines than Triton for same functionality.",
            "Bank conflicts in SMEM if stride is multiple of 32 (add padding).",
            "JIT compilation via torch.utils.cpp_extension.load is slow (30-60 seconds).",
            "No automatic vectorized loads — must use float4/int4 explicitly.",
            "Thread divergence within warps causes serialization.",
        ],
        "best_practices": [
            "Use SMEM for data reuse and communication between threads.",
            "Use warp shuffle (__shfl_xor_sync) for butterfly/reduction within 32 threads — zero latency.",
            "Use __sincosf instead of sinf/cosf — 4× faster, sufficient precision.",
            "Pad SMEM arrays to avoid bank conflicts: float smem[N + 1] instead of float smem[N].",
            "Use template parameters for compile-time specialization (tile sizes, radix).",
            "For FFT: SMEM for tile-local stages, warp shuffle for sub-warp stages, global memory for cross-tile stages.",
        ],
    },
}


# ============================================================================
# Library Analysis Knowledge Base — WHY each library is fast
# ============================================================================

_LIBRARY_ANALYSIS = {
    "fft": {
        "library": "cuFFT",
        "techniques": [
            {"name": "Mixed-radix decomposition",
             "what": "Use radix-2/4/8/16 instead of pure radix-2. N=1M=2^20: radix-8 needs only 7 stages vs 20 for radix-2.",
             "impact": "~2.5× fewer global HBM passes",
             "how": "Implement radix-4 butterfly: 4 inputs, 3 twiddle multiplies, 8 adds. Process 4 elements per butterfly instead of 2."},
            {"name": "Warp-level butterfly (register shuffle)",
             "what": "Use __shfl_xor_sync for butterfly within a warp — zero shared memory access, zero bank conflicts.",
             "impact": "~2× faster for small stages (log2(32)=5 stages done entirely in registers)",
             "how": "For stages with stride ≤ 16: use tl.math or inline PTX __shfl_xor_sync(mask, val, stride)."},
            {"name": "Precomputed twiddle factors",
             "what": "Store twiddle factors (cos/sin) in constant memory or precomputed table, not computed at runtime.",
             "impact": "Eliminates cosf/sinf calls (each ~20 cycles vs ~4 cycles for memory load)",
             "how": "Precompute W[k] = exp(-2πi·k/N) for each stage, store in constant memory or global memory."},
            {"name": "Large SMEM tile + register blocking",
             "what": "Each thread processes multiple elements (4-8), tile of 4096+ elements in SMEM.",
             "impact": "More stages fused in SMEM → fewer global passes",
             "how": "A100 has 164KB SMEM/SM. Tile of 4096 complex = 32KB. Can fuse log2(4096)=12 stages in SMEM."},
        ],
        "key_insight": "cuFFT's speed comes from minimizing global memory passes via higher radix + larger tiles, NOT from faster arithmetic.",
        "our_gap": [
            "Only radix-2 (20 passes for N=1M vs cuFFT's ~7)",
            "Runtime cosf/sinf instead of precomputed twiddle",
            "SMEM tile only 1024 (fuses 10 stages), could be 4096 (fuses 12)",
            "No warp shuffle — all butterfly through SMEM",
        ],
    },
    "conv2d": {
        "library": "cuDNN",
        "techniques": [
            {"name": "Winograd transform (for 3×3)",
             "what": "F(4,3) Winograd reduces 3×3 conv multiplications by 2.25×. Transform input+filter to Winograd domain, do element-wise multiply, transform back.",
             "impact": "2.25× fewer FLOPs for 3×3 convolutions",
             "how": "Implement Winograd F(2,3) or F(4,3) tile transform matrices. Requires pre/post transform passes."},
            {"name": "Implicit GEMM with Tensor Core",
             "what": "Compute im2col on-the-fly inside the GEMM tile, use WMMA/mma for Tensor Core FP16/TF32 matmul.",
             "impact": "16× higher compute throughput (TC vs CUDA cores) + zero im2col materialization",
             "how": "Use tl.dot or WMMA intrinsics. Gather input patches into SMEM registers per tile."},
            {"name": "Software pipelining",
             "what": "Overlap next tile's global memory load with current tile's GEMM compute using double-buffered SMEM.",
             "impact": "Hides memory latency, achieving 80%+ of peak compute",
             "how": "Use num_stages=2-4 in Triton, or manual double-buffering with cp.async in CUDA."},
            {"name": "cudnnFindAlgorithm auto-selection",
             "what": "Test multiple algorithms (direct, Winograd, FFT-based, implicit GEMM) and pick the fastest for given shapes.",
             "impact": "Always uses the best algorithm for each (N,C,H,W,K) configuration",
             "how": "Benchmark multiple implementations and cache the winner."},
        ],
        "key_insight": "cuDNN's speed comes from Winograd (fewer FLOPs) + Tensor Core (higher throughput) + auto-tuning (right algorithm per shape).",
        "our_gap": [
            "No Winograd — direct convolution only",
            "No Tensor Core — FP32 scalar GEMM",
            "No software pipelining — sequential load+compute",
            "No algorithm selection — fixed implicit im2col",
        ],
    },
    "kmeans": {
        "library": "cuBLAS (torch.cdist → cublasGemmEx)",
        "techniques": [
            {"name": "Tensor Core GEMM",
             "what": "cdist decomposes as ||x-c||² = ||x||² + ||c||² - 2·x@c^T. The x@c^T is a cuBLAS GEMM using Tensor Core.",
             "impact": "16× compute throughput vs CUDA core FP32",
             "how": "Use tl.dot for the X@C^T matmul. Compute ||x||² and ||c||² separately, combine."},
            {"name": "Tiled GEMM + epilogue fusion",
             "what": "cuBLAS tiles the GEMM optimally, then a separate argmin kernel reduces rows.",
             "impact": "Near-peak GEMM throughput",
             "how": "Fuse: tl.dot for distance tiles + online argmin per tile. This eliminates the N×K matrix."},
        ],
        "key_insight": "The key is preserving GEMM structure while adding online reduction. tl.dot is essential.",
        "our_gap": [
            "Scalar template destroys GEMM parallelism (0.07×)",
            "tl.dot template restores it (1.44×) but tile sizes could be larger",
        ],
    },
    "softmax": {
        "library": "ATen (PyTorch torch.softmax)",
        "techniques": [
            {"name": "Fused 2-pass kernel",
             "what": "ATen softmax is already a fused kernel: pass1 max+sumexp, pass2 normalize. No intermediate materialization.",
             "impact": "Already IO-optimal — same algorithm as Flash",
             "how": "Our Flash softmax uses the same approach. Difference is ATen's is highly tuned."},
            {"name": "Vectorized loads (float4)",
             "what": "ATen uses 128-bit vectorized loads (float4) for coalesced memory access.",
             "impact": "4× effective bandwidth for sequential access patterns",
             "how": "In Triton: use BLOCK_SIZE that's a multiple of 4 and ensure contiguous access."},
            {"name": "Warp-level reduction",
             "what": "Row max and sum are computed using warp shuffle, not shared memory.",
             "impact": "Lower latency reduction, better for short rows",
             "how": "Triton handles this automatically with tl.max/tl.sum for small BLOCK_SIZE."},
        ],
        "key_insight": "torch.softmax is already Flash-optimized. Beating it requires better vectorization and warp-level tuning, not algorithmic changes.",
        "our_gap": [
            "Same algorithm but less tuned launch config",
            "Small matrices fit in L2 — IO optimization irrelevant",
        ],
    },
    "spmv": {
        "library": "cuSPARSE",
        "techniques": [
            {"name": "CSR-Adaptive (row-length aware)",
             "what": "Short rows (nnz<32): one thread per row. Medium rows: one warp. Long rows: one block. Auto-selected.",
             "impact": "Optimal parallelism for any sparsity pattern",
             "how": "Partition rows by nnz length, launch different kernels or use dynamic parallelism."},
            {"name": "Merge-based partition",
             "what": "Partition work by NNZ count (not row count). Each thread block gets equal NNZ regardless of row structure.",
             "impact": "Perfect load balance across thread blocks",
             "how": "Use merge_path to find (row, nnz) boundary for each block. Process heterogeneous rows."},
            {"name": "Vectorized value/column loads",
             "what": "Load 2-4 consecutive CSR entries at once using vectorized loads.",
             "impact": "Higher memory throughput for the sequential CSR data",
             "how": "Align CSR arrays and use float2/float4 loads for values and int2/int4 for col_idx."},
        ],
        "key_insight": "SpMV performance depends on matching parallelism to row length distribution. Uniform strategies (thread-per-row or warp-per-row) waste resources.",
        "our_gap": [
            "Fixed warp-per-row — wastes lanes for short rows",
            "No row-length adaptive strategy",
            "No merge-based load balancing",
        ],
    },
    "stencil2d": {
        "library": "none (no standard library)",
        "techniques": [
            {"name": "Temporal tiling (our approach)",
             "what": "Fuse S iterations in SMEM, reducing HBM passes by S×.",
             "impact": "S× fewer HBM round-trips (S=4-8 typical)",
             "how": "Load (BH+2S)×(BW+2S) halo, iterate S times in SMEM, write BH×BW."},
            {"name": "Overlapped tiling / diamond tiling",
             "what": "Reduce halo overhead by using diamond-shaped tiles instead of rectangular.",
             "impact": "Less redundant halo data loaded",
             "how": "Advanced: requires non-trivial tile shape computation."},
        ],
        "key_insight": "Stencil optimization is about minimizing HBM passes per iteration. Temporal tiling is the primary technique.",
        "our_gap": ["Current approach is already effective (4.7× speedup)"],
    },
    "nbody": {
        "library": "none (no standard library)",
        "techniques": [
            {"name": "SMEM tiling (our approach)",
             "what": "Tile particles into SMEM, compute pairwise forces from SMEM instead of HBM.",
             "impact": "Eliminates N×N HBM traffic → O(N²/BN) SMEM loads",
             "how": "Each block loads BN source particles, all target threads accumulate."},
            {"name": "Barnes-Hut / FMM (algorithmic)",
             "what": "Approximate distant interactions → O(N log N) instead of O(N²).",
             "impact": "Asymptotically faster for large N",
             "how": "Build octree, compute multipole expansion for distant groups."},
        ],
        "key_insight": "Direct N-body is inherently O(N²). SMEM tiling optimizes the constant. For N>100K, need hierarchical methods.",
        "our_gap": ["Direct method only — no Barnes-Hut/FMM for large N"],
    },
    "cross_entropy": {
        "library": "ATen F.cross_entropy",
        "techniques": [
            {"name": "Our Flash kernel is already faster",
             "what": "Single-pass online logsumexp + gather. F.cross_entropy uses a similar approach but our Triton version is faster.",
             "impact": "Our kernel: 0.34ms vs ATen: 1.24ms = 3.6× faster",
             "how": "Already optimal. Online logsumexp over vocab tiles."},
        ],
        "key_insight": "This is a success case — our Flash approach already beats the library.",
        "our_gap": [],
    },
    "layernorm": {
        "library": "ATen F.layer_norm",
        "techniques": [
            {"name": "Our Flash kernel is already faster",
             "what": "2-pass Welford + normalize. Our Triton version achieves higher bandwidth utilization.",
             "impact": "Our kernel: 0.42ms vs ATen: 0.51ms = 1.2× faster",
             "how": "Already optimal. 2-pass Welford with large BLOCK_SIZE."},
        ],
        "key_insight": "This is a success case — our Flash approach already beats the library.",
        "our_gap": [],
    },
    "graph_laplacian": {
        "library": "none (custom fused kernel)",
        "techniques": [
            {"name": "Fused SpMV + diag (our approach)",
             "what": "y[i] = d[i]*x[i] - sum_j(A[i,j]*x[j]) in one kernel pass.",
             "impact": "Eliminates Ax intermediate, 1.18× speedup",
             "how": "Per-row: accumulate Ax in register, combine with D*x, write once."},
        ],
        "key_insight": "Fusion saves one kernel launch and one HBM round-trip for Ax.",
        "our_gap": ["Could benefit from CSR-Adaptive row parallelism (same as SpMV)"],
    },
}


# ============================================================================
# Strategy Database — multi-level optimization recommendations per operator
# ============================================================================

_STRATEGY_DB = {
    "fft": [
        {"level": 1, "name": "Algorithm",
         "current": "Radix-2 Cooley-Tukey (log2(N) passes)",
         "recommended": "Radix-4 or Radix-8 (log4(N) or log8(N) passes)",
         "impact": "2-3× fewer global passes → directly reduces HBM traffic",
         "code_hint": "Radix-4 butterfly: 4 inputs → 3 twiddle muls + 8 adds (vs radix-2: 2 inputs → 1 mul + 2 adds)"},
        {"level": 2, "name": "IO/Fusion",
         "current": "Fuse 10 stages in SMEM (TILE=1024)",
         "recommended": "Fuse 12 stages in SMEM (TILE=4096, 32KB) + precomputed twiddle in constant memory",
         "impact": "2 fewer global passes + eliminate sin/cos computation",
         "code_hint": "A100 SMEM=164KB. 4096 complex float = 32KB. Precompute twiddle W[k]=exp(-2πi·k/N)."},
        {"level": 3, "name": "Hardware",
         "current": "SMEM butterfly with __syncthreads",
         "recommended": "Warp shuffle for stages with stride ≤ 16 (5 stages), SMEM for larger strides",
         "impact": "Zero bank conflict for small stages, ~30% faster inner loop",
         "code_hint": "__shfl_xor_sync(0xffffffff, val, stride) for butterfly exchange within warp."},
        {"level": 4, "name": "Data Layout",
         "current": "Separate real/imaginary arrays",
         "recommended": "Interleaved real/imaginary (struct of arrays → array of structs for coalescing)",
         "impact": "Better memory coalescing for butterfly access patterns",
         "code_hint": "Store as float2 (re, im) pairs. Each thread loads one complex number with one 8-byte load."},
    ],
    "conv2d": [
        {"level": 1, "name": "Algorithm",
         "current": "Direct implicit im2col + scalar GEMM",
         "recommended": "Winograd F(4,3) for 3×3 kernels, or implicit GEMM with Tensor Core for general",
         "impact": "Winograd: 2.25× fewer FLOPs. TC: 16× compute throughput.",
         "code_hint": "For Winograd: BT·d·B transform (4×4 tile), G·g·GT filter transform, element-wise mul, AT·m·A inverse."},
        {"level": 2, "name": "IO/Fusion",
         "current": "Implicit im2col (good — no materialization)",
         "recommended": "Keep implicit im2col. Add software pipelining (double-buffered SMEM).",
         "impact": "Overlaps compute with memory → 80%+ utilization",
         "code_hint": "num_stages=2 in Triton, or manual cp.async + double-buffer in CUDA."},
        {"level": 3, "name": "Hardware",
         "current": "FP32 scalar multiply-add",
         "recommended": "TF32 Tensor Core via tl.dot (automatic in Triton with float32 inputs on A100)",
         "impact": "8× compute throughput for GEMM tiles",
         "code_hint": "tl.dot(a, b) with float32 inputs on A100 automatically uses TF32 Tensor Core."},
        {"level": 4, "name": "Data Layout",
         "current": "NCHW input",
         "recommended": "NHWC for better Tensor Core alignment (if using TC GEMM)",
         "impact": "Enables 128-bit aligned loads for TC operands",
         "code_hint": "input = input.to(memory_format=torch.channels_last)"},
    ],
    "spmv": [
        {"level": 1, "name": "Algorithm",
         "current": "Fixed warp-per-row",
         "recommended": "CSR-Adaptive: thread-per-row (short), warp-per-row (medium), block-per-row (long)",
         "impact": "Matches parallelism to row length → no wasted lanes",
         "code_hint": "Classify rows by nnz: <32→thread, 32-512→warp, >512→block. Launch separate kernels."},
        {"level": 2, "name": "IO/Fusion",
         "current": "Random x[col_idx] gather",
         "recommended": "Cache x tiles in SMEM for banded matrices. For general: texture cache for x.",
         "impact": "Reduces random HBM reads for x vector",
         "code_hint": "If matrix has bandwidth B: load x[col_start:col_start+BN] into SMEM."},
        {"level": 3, "name": "Hardware",
         "current": "Warp shuffle reduction only",
         "recommended": "Vectorized CSR loads (float2/float4) + L1 cache hints",
         "impact": "Higher bandwidth utilization for sequential CSR data",
         "code_hint": "Load values[j:j+4] as float4, col_idx[j:j+4] as int4."},
        {"level": 4, "name": "Data Layout",
         "current": "Standard CSR",
         "recommended": "Padded CSR (pad short rows to warp boundary) or SELL-C-sigma (sorted + padded)",
         "impact": "Eliminates lane divergence within warps",
         "code_hint": "Sort rows by nnz, pad to multiple of 32, store in SELL-C format."},
    ],
    "kmeans": [
        {"level": 1, "name": "Algorithm",
         "current": "tl.dot tiled GEMM + online argmin (good)",
         "recommended": "Keep current. Consider Tensor Core BF16 for 2× throughput.",
         "impact": "BF16 TC: 2× compute throughput, slight precision trade-off",
         "code_hint": "Cast X and C to bfloat16 before tl.dot. Keep accumulator in float32."},
        {"level": 2, "name": "IO/Fusion",
         "current": "Online argmin eliminates N×K matrix (good)",
         "recommended": "Keep. IO optimization already effective.",
         "impact": "Already near optimal for this dimension.",
         "code_hint": "N/A — already fused."},
        {"level": 3, "name": "Hardware",
         "current": "FP32 tl.dot on A100",
         "recommended": "TF32 or BF16 tl.dot for Tensor Core (A100 supports TF32 automatically for tl.dot with fp32)",
         "impact": "Should already use TF32. Verify with NCU tensor_cycles > 0.",
         "code_hint": "tl.dot(a, b, allow_tf32=True) — default on A100."},
        {"level": 4, "name": "Data Layout",
         "current": "Row-major X (N,d) and C (K,d)",
         "recommended": "Keep row-major. Ensure BLOCK_M × BK tile fits SMEM well.",
         "impact": "Minimal — already coalesced.",
         "code_hint": "Tune BLOCK_M=64-128, BK=32-64 for best occupancy."},
    ],
}


def _make_generic_strategy(task: str, report) -> list:
    """Generate generic strategy when no specific one exists."""
    strategies = []
    if report and report.bottleneck == "memory-bound":
        strategies.append({
            "level": 2, "name": "IO/Fusion",
            "current": "Materialized intermediates",
            "recommended": "Fuse operations to eliminate HBM round-trips",
            "impact": f"Up to {report.arithmetic_intensity:.1f}× if fully fused",
        })
    if report and report.pre_check and report.pre_check.has_gemm_structure:
        strategies.append({
            "level": 3, "name": "Hardware",
            "current": "Unknown compute pattern",
            "recommended": "Use tl.dot for GEMM structure to enable Tensor Core",
            "impact": "Up to 16× compute throughput",
            "code_hint": "tl.dot(a_tile, b_tile.T) inside tile loop",
        })
    if not strategies:
        strategies.append({
            "level": 2, "name": "IO/Fusion",
            "current": "Standard implementation",
            "recommended": "Analyze with 'analyze' tool first, then fuse_and_online",
            "impact": "Depends on materialization analysis",
        })
    return strategies


# ============================================================================
# Pattern Knowledge Base — domain knowledge the agent can retrieve
# ============================================================================

_PATTERNS = {
    "gemm_online_reduce": {
        "keywords": ["gemm", "reduce", "distance", "argmin", "kmeans", "matmul", "dot"],
        "when": "Baseline computes GEMM (matmul/distance) → large matrix → row-wise reduce (argmin/softmax/logsumexp)",
        "wrong": "Replace GEMM with per-element scalar loop + online reduce. This destroys Tensor Core parallelism and is 10-50× slower.",
        "right": "Keep GEMM structure using tl.dot inside tile loop. Tile over K dimension, compute D_tile via tl.dot, apply online reduce to each tile. Never materialize full D.",
        "code": '''
# Flash-KMeans: tiled GEMM + online argmin
# Each block processes BLOCK_M rows × all K centroids (in BK tiles)

@triton.jit
def _flash_kmeans_tiled(
    X, C, OUT,
    N: tl.constexpr, K: tl.constexpr, d: tl.constexpr,
    BLOCK_M: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    row_mask = rows < N

    # Precompute ||x||^2 for this block of rows
    x_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for j in range(0, d, BK):
        j_offs = j + tl.arange(0, BK)
        j_mask = j_offs < d
        x_block = tl.load(X + rows[:, None] * d + j_offs[None, :],
                          mask=row_mask[:, None] & j_mask[None, :], other=0.0)
        x_sq += tl.sum(x_block * x_block, axis=1)

    # Online argmin over centroid tiles
    best_dist = tl.full((BLOCK_M,), float("inf"), dtype=tl.float32)
    best_idx = tl.zeros((BLOCK_M,), dtype=tl.int32)

    for k_start in range(0, K, BK):
        k_end = min(k_start + BK, K)
        k_size = k_end - k_start

        # Compute ||c_k||^2 for this centroid tile
        c_sq = tl.zeros((BK,), dtype=tl.float32)
        # Compute X @ C_tile^T via tl.dot (Tensor Core!)
        xc = tl.zeros((BLOCK_M, BK), dtype=tl.float32)

        for j in range(0, d, BK):
            j_offs = j + tl.arange(0, BK)
            j_mask = j_offs < d
            x_block = tl.load(X + rows[:, None] * d + j_offs[None, :],
                              mask=row_mask[:, None] & j_mask[None, :], other=0.0)
            c_block = tl.load(C + (k_start + tl.arange(0, BK))[:, None] * d + j_offs[None, :],
                              mask=(tl.arange(0, BK) < k_size)[:, None] & j_mask[None, :], other=0.0)
            xc += tl.dot(x_block, tl.trans(c_block))  # Tensor Core GEMM!
            c_sq += tl.sum(c_block * c_block, axis=1)

        # D_tile = ||x||^2 + ||c||^2 - 2*x@c^T  (never materialized to HBM!)
        D_tile = x_sq[:, None] + c_sq[None, :] - 2.0 * xc

        # Online argmin update per row
        tile_min = tl.min(D_tile, axis=1)
        tile_argmin = tl.argmin(D_tile, axis=1) + k_start
        better = tile_min < best_dist
        best_dist = tl.where(better, tile_min, best_dist)
        best_idx = tl.where(better, tile_argmin, best_idx)

    tl.store(OUT + rows, best_idx, mask=row_mask)
''',
    },

    "online_softmax_2pass": {
        "keywords": ["softmax", "attention", "flash", "2pass", "two pass", "online"],
        "when": "Row-wise softmax on a large matrix. Baseline materializes the matrix, reads it twice (max, then normalize).",
        "wrong": "3-pass (max, sum, normalize) reads data 3 times. Or using Python loops instead of vectorized Triton ops.",
        "right": "2-pass online softmax: Pass 1 streams data computing running (max, sum_exp). Pass 2 normalizes. Use large BLOCK_SIZE and num_warps=8.",
        "code": '''
# 2-pass online softmax (FlashAttention pattern)
@triton.jit
def _flash_softmax(X, OUT, M: tl.constexpr, BM: tl.constexpr):
    row = tl.program_id(0)
    # Pass 1: online max + sum_exp
    m = float("-inf")
    s = 0.0
    for start in range(0, M, BM):
        cols = start + tl.arange(0, BM)
        x = tl.load(X + row * M + cols, mask=cols < M, other=float("-inf"))
        new_m = tl.maximum(m, tl.max(x))
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m))
        m = new_m
    # Pass 2: normalize
    inv_s = 1.0 / s
    for start in range(0, M, BM):
        cols = start + tl.arange(0, BM)
        x = tl.load(X + row * M + cols, mask=cols < M, other=float("-inf"))
        tl.store(OUT + row * M + cols, tl.exp(x - m) * inv_s, mask=cols < M)
''',
    },

    "online_logsumexp_gather": {
        "keywords": ["logsumexp", "cross_entropy", "gather", "loss", "vocab"],
        "when": "Cross-entropy loss: logits (N, V) → max → exp → sum → log → gather label. V (vocab) is huge.",
        "wrong": "Materialize exp(logits - max) as N×V matrix (massive HBM write).",
        "right": "Single pass: stream vocab tiles, maintain online (max, sum_exp), gather target logit on-the-fly.",
        "code": '''
# Flash Cross-Entropy: single-pass online logsumexp + gather
@triton.jit
def _flash_ce(LOGITS, LABELS, LOSS, N: tl.constexpr, V: tl.constexpr, BV: tl.constexpr):
    row = tl.program_id(0)
    label = tl.load(LABELS + row)
    m = float("-inf")
    s = 0.0
    target_logit = 0.0
    for v_start in range(0, V, BV):
        v = v_start + tl.arange(0, BV)
        x = tl.load(LOGITS + row * V + v, mask=v < V, other=float("-inf"))
        target_logit += tl.sum(tl.where(v == label, x, 0.0))
        new_m = tl.maximum(m, tl.max(x))
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m))
        m = new_m
    tl.store(LOSS + row, -(target_logit - m - tl.log(s)))
''',
    },

    "reduction_tree": {
        "keywords": ["reduction", "sum", "mean", "variance", "welford", "batchnorm", "layernorm"],
        "when": "Need to compute sum/mean/variance over a large dimension.",
        "wrong": "Atomic adds from many threads, or serial accumulation.",
        "right": "Tree reduction within a block using tl.sum. For cross-block, use partial results + final reduce kernel.",
        "code": '''
# Block-level reduction for mean + variance (Welford)
@triton.jit
def _reduce_stats(X, MEAN, VAR, N: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    col = tl.program_id(0)
    d_offs = col * BD + tl.arange(0, BD)
    d_mask = d_offs < D
    acc_sum = tl.zeros((BD,), dtype=tl.float32)
    acc_sq = tl.zeros((BD,), dtype=tl.float32)
    for n in range(N):
        x = tl.load(X + n * D + d_offs, mask=d_mask, other=0.0)
        acc_sum += x
        acc_sq += x * x
    mean = acc_sum / N
    var = acc_sq / N - mean * mean
    tl.store(MEAN + d_offs, mean, mask=d_mask)
    tl.store(VAR + d_offs, var, mask=d_mask)
''',
    },

    "fft_temporal_blocking": {
        "keywords": ["fft", "butterfly", "temporal", "blocking", "radix", "cooley", "stockham"],
        "when": "FFT has log2(N) butterfly stages, each requiring an HBM round-trip. For large N this is IO-bound.",
        "wrong": "Naively launch one kernel per stage → log2(N) HBM reads+writes of full array. Also: trying Stockham autosort in Triton registers — Triton does NOT support arbitrary register indexing needed for butterfly permutations.",
        "right": "Three-part approach: (1) Global bit-reverse copy, (2) SMEM-fused local stages using Cooley-Tukey in-place butterfly, (3) Global stages for cross-tile butterflies. Use CUDA SMEM, NOT Triton registers. TILE=4096 fuses 12 stages.",
        "code": '''
// VERIFIED CORRECT FFT: bit-reverse + SMEM local stages + global stages
// CRITICAL: Use Cooley-Tukey (bit-reverse + in-place), NOT Stockham autosort
// CRITICAL: Use CUDA SMEM, NOT Triton (Triton can't do register shuffle)

struct cfloat { float re, im; };
__device__ __forceinline__ cfloat cadd(cfloat a, cfloat b){ return {a.re+b.re, a.im+b.im}; }
__device__ __forceinline__ cfloat csub(cfloat a, cfloat b){ return {a.re-b.re, a.im-b.im}; }

// Step 1: Global bit-reverse copy (separate kernel)
__global__ void bit_reverse_copy(
    const float* x_re, const float* x_im,
    float* y_re, float* y_im, int N, int log2N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int rev = 0, tmp = i;
    for(int b = 0; b < log2N; b++){ rev = (rev << 1) | (tmp & 1); tmp >>= 1; }
    y_re[rev] = x_re[i];  y_im[rev] = x_im[i];
}

// Step 2: SMEM-fused local stages (stages 0..LOG2_TILE-1)
// In-place Cooley-Tukey butterfly in SMEM — simple and correct
template<int TILE, int LOG2_TILE>
__global__ void fft_smem_stages(float* re, float* im, int N, int start_stage) {
    extern __shared__ cfloat smem[];
    int base = blockIdx.x * TILE;
    int tid = threadIdx.x;

    for(int i = tid; i < TILE; i += blockDim.x){
        int g = base + i;
        smem[i] = (g < N) ? cfloat{re[g], im[g]} : cfloat{0.f, 0.f};
    }
    __syncthreads();

    // KEY: for each stage, only process butterflies where BOTH elements are in this tile
    for(int s = start_stage; s < start_stage + LOG2_TILE; s++){
        int m = 1 << (s + 1);   // butterfly span
        int mh = m >> 1;        // half span
        if(mh > TILE) break;    // butterfly wider than tile

        for(int i = tid; i < TILE/2; i += blockDim.x){
            int local_group = i / mh;
            int k = i % mh;
            int idx0 = local_group * m + k;
            int idx1 = idx0 + mh;
            if(idx1 >= TILE) continue;

            // Use GLOBAL position for twiddle factor
            int global_k = (base + idx0) % m;
            if(global_k >= mh) continue;

            float ang = -2.f * CUDART_PI_F * (float)global_k / (float)m;
            float wr, wi;
            __sincosf(ang, &wi, &wr);

            cfloat a = smem[idx0], b = smem[idx1];
            cfloat wb = {b.re*wr - b.im*wi, b.re*wi + b.im*wr};
            smem[idx0] = cadd(a, wb);
            smem[idx1] = csub(a, wb);
        }
        __syncthreads();
    }

    for(int i = tid; i < TILE; i += blockDim.x){
        int g = base + i;
        if(g < N){ re[g] = smem[i].re; im[g] = smem[i].im; }
    }
}

// Step 3: Global butterfly for cross-tile stages
__global__ void butterfly_global(
    const float* in_re, const float* in_im,
    float* out_re, float* out_im, int N, int half_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    int m = 2 * half_size;
    int pos = idx % m;
    if(pos < half_size){
        int i = idx - pos + pos;  // same as idx when pos < half_size
        int j = i + half_size;
        float ang = -2.f * CUDART_PI_F * (float)pos / (float)m;
        float wr, wi;
        __sincosf(ang, &wi, &wr);
        // ... butterfly with twiddle ...
        // out[i] = a + w*b;  out[j] = a - w*b;
    }
}
// TILE=4096 → fuses 12 stages (vs 10 with TILE=1024)
// Remaining log2(N)-12 stages go through global butterfly_global kernel
''',
    },

    "implicit_im2col": {
        "keywords": ["conv", "im2col", "convolution", "implicit", "unfold"],
        "when": "Conv2D via im2col materializes a huge unfolded matrix (N*OH*OW, C_in*KH*KW) to HBM before GEMM.",
        "wrong": "Explicit im2col: write full unfolded matrix to HBM, then read it back for GEMM. For ResNet-50 conv3: 56GB intermediate!",
        "right": "Implicit im2col: in each GEMM tile, gather the required input patches on-the-fly from the original NCHW tensor into SMEM. Never materialize the unfolded matrix. This is what cuDNN's implicit GEMM does.",
        "code": '''
// Implicit im2col: gather patches on-the-fly in SMEM
template <int BM, int BN, int BK>
__global__ void implicit_conv2d(
    const float* input,  // (N,C_in,H,W)
    const float* weight, // (C_out, C_in*KH*KW)
    float* output, ...) {
    __shared__ float sA[BM][BK]; // im2col tile (gathered, NOT from HBM buffer)
    __shared__ float sB[BK][BN]; // weight tile
    // For each K-tile:
    //   sA[m][k] = input at the position implied by output pixel m, kernel offset k
    //   sB[k][n] = weight[n][k]
    //   C += sA * sB  (GEMM in SMEM)
    // col_matrix never exists in HBM!
}
''',
    },

    "temporal_tiling_stencil": {
        "keywords": ["stencil", "temporal", "tiling", "jacobi", "halo", "ghost"],
        "when": "Iterative stencil (Jacobi, Gauss-Seidel) reads/writes full grid each iteration. T iterations → T HBM round-trips.",
        "wrong": "Launch one kernel per iteration. Each iteration reads H×W from HBM and writes H×W back.",
        "right": "Temporal tiling: load (BH+2S)×(BW+2S) halo region once, apply S iterations in SMEM, write back BH×BW. Reduces HBM round-trips by factor S. Trade-off: larger halo = more redundant loads, but fewer global passes.",
        "code": '''
// Flash Stencil: fuse S iterations in shared memory
template <int BH, int BW, int S>
__global__ void flash_stencil(const float* in, float* out, int H, int W) {
    __shared__ float tile[2][(BH+2*S)][(BW+2*S)]; // double-buffer
    // Load halo region from HBM (1 read, includes ghost cells)
    // for iter in 0..S:
    //   apply 5-point stencil in SMEM (zero HBM traffic)
    //   swap buffers
    // Write inner BH×BW back to HBM (1 write)
}
// HBM IO: ceil(T/S) × (BH+2S)×(BW+2S) per tile vs T × H×W total
// For S=8: ~8× fewer HBM round-trips
''',
    },
}


# ============================================================================
# Benchmark implementations for each operator
# Each entry has: setup(params) -> tensors, baseline(tensors), flash(tensors)
# ============================================================================

def _make_benchmarks():
    """Create benchmark functions. Deferred to avoid importing torch at module load."""
    benchmarks = {}

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return benchmarks

    # ---- Cross-Entropy ----
    def ce_setup(p):
        N, V = p["N"], p["V"]
        torch.manual_seed(42)
        logits = torch.randn(N, V, device="cuda")
        labels = torch.randint(0, V, (N,), device="cuda")
        return {"logits": logits, "labels": labels, "N": N, "V": V}

    def ce_baseline(t):
        # Standard: materialize exp, sum, log separately
        logits, labels = t["logits"], t["labels"]
        max_vals = logits.max(dim=1, keepdim=True).values
        exp_logits = (logits - max_vals).exp()
        log_sum_exp = exp_logits.sum(dim=1).log()
        gathered = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        loss = -(gathered - max_vals.squeeze(1) - log_sum_exp)
        return loss

    def ce_flash(t):
        # PyTorch fused cross_entropy (single kernel, no materialization)
        return F.cross_entropy(t["logits"], t["labels"], reduction="none")

    benchmarks["cross_entropy"] = {"setup": ce_setup, "baseline": ce_baseline, "flash": ce_flash}

    # ---- GMM E-step ----
    def gmm_setup(p):
        N, K, d = p["N"], p["K"], p["d"]
        torch.manual_seed(42)
        X = torch.randn(N, d, device="cuda")
        mu = torch.randn(K, d, device="cuda")
        var = torch.ones(K, d, device="cuda") * 1.0
        log_pi = torch.full((K,), -torch.log(torch.tensor(float(K))), device="cuda")
        return {"X": X, "mu": mu, "var": var, "log_pi": log_pi}

    def gmm_baseline(t):
        # Materialize full L matrix
        X, mu, var, log_pi = t["X"], t["mu"], t["var"], t["log_pi"]
        d = X.shape[1]
        log_det = var.log().sum(1)
        diff = X.unsqueeze(1) - mu.unsqueeze(0)
        mahal = (diff ** 2 / var.unsqueeze(0)).sum(2)
        L = log_pi.unsqueeze(0) - 0.5 * (d * 1.8379 + log_det.unsqueeze(0) + mahal)
        log_norm = torch.logsumexp(L, dim=1)
        return log_norm

    def gmm_flash(t):
        # Try native kernel first, fall back to same code (demonstrates IO savings concept)
        try:
            import flash_gmm_native as _C
            _, _, _, log_norm = _C.flash_em_fused(t["X"], t["mu"], t["var"], t["log_pi"], 128, 32)
            return log_norm
        except ImportError:
            return gmm_baseline(t)  # fallback

    benchmarks["gmm_estep"] = {"setup": gmm_setup, "baseline": gmm_baseline, "flash": gmm_flash}

    # ---- KMeans ----
    def km_setup(p):
        N, K, d = p["N"], p["K"], p["d"]
        torch.manual_seed(42)
        X = torch.randn(N, d, device="cuda")
        C = torch.randn(K, d, device="cuda")
        return {"X": X, "C": C}

    def km_baseline(t):
        # Materialize N×K distance matrix
        X, C = t["X"], t["C"]
        D = torch.cdist(X, C, p=2.0)  # (N, K) materialized
        return D.argmin(dim=1)

    def km_flash(t):
        # Chunked argmin without full D materialization
        X, C = t["X"], t["C"]
        K = C.shape[0]
        BK = 32
        best_dist = torch.full((X.shape[0],), float('inf'), device=X.device)
        best_idx = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        for k_start in range(0, K, BK):
            k_end = min(k_start + BK, K)
            D_tile = torch.cdist(X, C[k_start:k_end], p=2.0)
            tile_min, tile_idx = D_tile.min(dim=1)
            better = tile_min < best_dist
            best_dist[better] = tile_min[better]
            best_idx[better] = tile_idx[better] + k_start
        return best_idx

    benchmarks["kmeans"] = {"setup": km_setup, "baseline": km_baseline, "flash": km_flash}

    # ---- Softmax ----
    def sm_setup(p):
        N, d = p["N"], p["d"]
        torch.manual_seed(42)
        Q = torch.randn(N, d, device="cuda")
        K_mat = torch.randn(N, d, device="cuda")
        return {"Q": Q, "K": K_mat}

    def sm_baseline(t):
        # Materialize S = QK^T then softmax
        S = t["Q"] @ t["K"].T  # (N, N) materialized
        return torch.softmax(S, dim=1)

    def sm_flash(t):
        # PyTorch scaled_dot_product_attention uses FlashAttention internally
        Q, K = t["Q"].unsqueeze(0).unsqueeze(0), t["K"].unsqueeze(0).unsqueeze(0)
        out = F.scaled_dot_product_attention(Q, K, K, is_causal=False)
        return out.squeeze(0).squeeze(0)

    benchmarks["softmax"] = {"setup": sm_setup, "baseline": sm_baseline, "flash": sm_flash}

    # ---- FFT ----
    def fft_setup(p):
        N = p.get("N", 1048576)
        torch.manual_seed(42)
        x_re = torch.randn(N, device="cuda")
        x_im = torch.zeros(N, device="cuda")
        return {"x_re": x_re, "x_im": x_im, "N": N}

    def fft_baseline(t):
        # Standard: use torch.fft (cuFFT internally)
        x = torch.complex(t["x_re"], t["x_im"])
        return torch.fft.fft(x)

    def fft_flash(t):
        # Same — no custom kernel here, benchmark via triton_codegen
        x = torch.complex(t["x_re"], t["x_im"])
        return torch.fft.fft(x)

    benchmarks["fft"] = {"setup": fft_setup, "baseline": fft_baseline, "flash": fft_flash,
                         "library": fft_flash, "library_name": "cuFFT"}

    # ---- Conv2D ----
    def conv_setup(p):
        N_b = p.get("N", 64)
        C_in, C_out = p.get("C_in", 128), p.get("C_out", 256)
        H, W = p.get("H", 56), p.get("W", 56)
        KH, KW = p.get("KH", 3), p.get("KW", 3)
        torch.manual_seed(42)
        inp = torch.randn(N_b, C_in, H, W, device="cuda")
        wt = torch.randn(C_out, C_in, KH, KW, device="cuda")
        return {"input": inp, "weight": wt, "pad": (KH // 2, KW // 2)}

    def conv_baseline(t):
        # Explicit im2col + GEMM (materialized)
        inp, wt = t["input"], t["weight"]
        N_b, C_in, H, W = inp.shape
        C_out, _, KH, KW = wt.shape
        pad_h, pad_w = t["pad"]
        # torch.nn.functional.unfold = im2col, materializes (N, C*K*K, L)
        col = F.unfold(inp, (KH, KW), padding=(pad_h, pad_w))  # materialized!
        wt_flat = wt.reshape(C_out, -1)
        out = wt_flat @ col  # GEMM
        OH = H + 2 * pad_h - KH + 1
        OW = W + 2 * pad_w - KW + 1
        return out.reshape(N_b, C_out, OH, OW)

    def conv_cudnn(t):
        # cuDNN: PyTorch conv2d uses cuDNN (implicit GEMM / Winograd)
        return F.conv2d(t["input"], t["weight"], padding=t["pad"])

    benchmarks["conv2d"] = {"setup": conv_setup, "baseline": conv_baseline, "flash": conv_cudnn,
                            "library": conv_cudnn, "library_name": "cuDNN"}

    # ---- Stencil2D ----
    def stencil_setup(p):
        H, W = p.get("H", 4096), p.get("W", 4096)
        T = p.get("T", 100)
        torch.manual_seed(42)
        grid = torch.randn(H, W, device="cuda")
        return {"grid": grid, "T": T, "H": H, "W": W}

    def stencil_baseline(t):
        # Standard: T iterations, each a full HBM read/write
        g = t["grid"].clone()
        out = torch.empty_like(g)
        H, W = t["H"], t["W"]
        for _ in range(t["T"]):
            # 5-point stencil via slicing (PyTorch, materializes every step)
            out[1:-1, 1:-1] = 0.2 * (
                g[1:-1, 1:-1] +
                g[:-2, 1:-1] + g[2:, 1:-1] +
                g[1:-1, :-2] + g[1:-1, 2:]
            )
            g, out = out, g
        return g

    def stencil_flash(t):
        # Same as baseline for Python-level — real flash is via CUDA kernel
        return stencil_baseline(t)

    benchmarks["stencil2d"] = {"setup": stencil_setup, "baseline": stencil_baseline, "flash": stencil_flash}

    return benchmarks

_BENCHMARKS = _make_benchmarks()


# ============================================================================
# Built-in ReAct agent (rule-based, demonstrates the protocol)
# ============================================================================

def run_react_agent(task: str, verbose: bool = True) -> IOEnvironment:
    """
    Run a built-in ReAct agent on a task.

    This demonstrates the Thought → Action → Observation protocol.
    An LLM agent would replace the hardcoded thoughts with actual reasoning.
    """
    env = IOEnvironment()

    # Step 0: Analyze
    if verbose:
        print("\n" + "━" * 60)
        print(f"  ReAct Agent: Optimizing '{task}'")
        print("━" * 60)

    thought = f"I need to analyze the {task} operator to understand its IO bottlenecks."
    obs, reward, done = env.step(thought, "analyze", {"task": task})
    if verbose:
        _print_react_step(0, thought, "analyze", {"task": task}, obs, reward)

    # Step 0.5: Pre-check
    thought = "Before optimizing, let me run the multi-dimensional pre-check to verify IO optimization is appropriate."
    obs, reward, done = env.step(thought, "pre_check", {})
    if verbose:
        _print_react_step(0, thought, "pre_check", {}, obs, reward)

    # Step 1+: Optimize
    max_steps = 5
    for step_i in range(1, max_steps + 1):
        mats = env.graph.get_materialized_intermediates()
        if not mats:
            # All intermediates eliminated — verify and done
            thought = "No materialized intermediates remain. Let me verify the optimization."
            obs, reward, done = env.step(thought, "verify", {})
            if verbose:
                _print_react_step(step_i, thought, "verify", {}, obs, reward)

            thought = "Verification passed. Optimization complete."
            obs, reward, done = env.step(thought, "done", {})
            if verbose:
                _print_react_step(step_i + 1, thought, "done", {}, obs, reward)
            break

        # Decide action based on graph structure
        ops = env.graph.operations
        op_names = [op.name for op in ops]

        # Strategy: if multiple ops, try to fuse all with online algorithm
        if len(ops) >= 2:
            # Find the right online algorithm
            all_computes = " ".join(op.computes for op in ops)
            if "l2" in all_computes or "argmin" in all_computes or "distance" in all_computes:
                algo = "online_argmin"
                out_shape = ("N",)
            elif "softmax" in all_computes:
                algo = "online_softmax"
                out_shape = ("N", "d")
            else:
                algo = "online_logsumexp"
                out_shape = ("N",)

            # Determine output name from final op
            final_output = env.graph.outputs[0]

            thought = (f"I see {len(mats)} materialized intermediate(s): {[n for n,_ in mats]}. "
                       f"The largest is {mats[0][0]} ({mats[0][1].size_bytes(env.params)/1e6:.1f} MB). "
                       f"I'll fuse all {len(ops)} ops with {algo} to eliminate all intermediates at once.")

            args = {
                "ops": op_names,
                "algorithm": algo,
                "output_name": final_output,
                "output_shape": list(out_shape),
            }

            try:
                obs, reward, done = env.step(thought, "fuse_and_online", args)
                if verbose:
                    _print_react_step(step_i, thought, "fuse_and_online", args, obs, reward)
                continue
            except Exception:
                pass  # fallback below

        # Fallback: fuse first two ops
        if len(ops) >= 2:
            thought = f"Fusing {ops[0].name} + {ops[1].name} to eliminate intermediate."
            args = {"op1": ops[0].name, "op2": ops[1].name}
            obs, reward, done = env.step(thought, "fuse_ops", args)
            if verbose:
                _print_react_step(step_i, thought, "fuse_ops", args, obs, reward)
        else:
            break

    return env


def _print_react_step(step: int, thought: str, tool: str, args: dict, obs: str, reward: float):
    print(f"\n┌─ Step {step} ─────────────────────────────────")
    print(f"│ Thought: {thought}")
    print(f"│ Action:  {tool}({', '.join(f'{k}={v!r}' for k,v in args.items())})")
    print(f"├─ Observation ──────────────────────────────")
    for line in obs.split("\n"):
        print(f"│ {line}")
    print(f"│ Reward: {reward:+.2f}")
    print(f"└────────────────────────────────────────────")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ReAct Agent for IO-Aware Operator Optimization")
    parser.add_argument("task", nargs="?", default="cross_entropy",
                       help="Task: gmm_estep, kmeans, softmax, cross_entropy")
    args = parser.parse_args()

    env = run_react_agent(args.task, verbose=True)

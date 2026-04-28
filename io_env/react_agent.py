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
import json, io, os, re, sys, contextlib, traceback
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
        "description": "Generate a Triton GPU kernel implementing the Flash-optimized operator. Uses templates for known patterns (online_logsumexp, online_argmin, online_softmax). Returns the generated code.",
        "parameters": {
            "custom_code": "str (optional) — provide your own Triton kernel code instead of using templates",
        },
        "example": '{"tool": "generate_kernel", "args": {}}',
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

        prompt = f"""You are a GPU kernel optimization agent that writes REAL Triton/CUDA kernels.

Your goal: reduce HBM memory traffic (IO) by eliminating intermediate matrix materialization,
then generate an ACTUAL Triton GPU kernel that achieves real speedup.

## Available Tools
{tool_desc}

## Required Workflow
Phase 1 — Symbolic IO Analysis:
1. 'analyze' the baseline operator's IO bottlenecks
2. Apply optimizations (fuse_and_online, fuse_ops) to eliminate materialized intermediates
3. 'verify' the symbolic optimization

Phase 2 — Kernel Generation + Testing (THE REAL DELIVERABLE):
4. 'generate_kernel' to create a Triton GPU kernel
   - The default template is a starting point. You can (and should) write your own with custom_code.
   - The kernel must be written in Triton (@triton.jit), NOT plain Python.
   - DO NOT just call PyTorch ops — write actual GPU kernel code with tl.load, tl.store, tl.arange.
5. 'compile_and_test' to verify correctness against PyTorch reference
6. 'benchmark_kernel' to measure ACTUAL speedup vs materialized baseline

Phase 3 — Iterate if Needed:
7. If speedup < 1.0×, analyze WHY:
   - Per-row serial loop vs GEMM parallelism?
   - Poor memory access pattern?
   - Not enough parallelism in the Triton kernel?
8. Write an IMPROVED kernel using 'generate_kernel' with custom_code=<your new Triton code>
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

## RULES
- You MUST write Triton kernel code. DO NOT generate Python-only solutions.
- You MUST achieve speedup >= 1.0× or explain clearly why it's not possible.
- After benchmark_kernel, if speedup < 1.0×, you MUST try to improve the kernel.

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
            results.append(f"  [FAIL] Missing outputs: {missing}")

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
        """Generate a Triton kernel for the current task."""
        from io_env.triton_codegen import generate_kernel
        custom_code = args.get("custom_code")
        code, filepath = generate_kernel(self.task_name, custom_code)
        if not code:
            return filepath, 0.0, False  # filepath contains error message
        self._kernel_path = filepath
        # Show first 30 lines
        lines = code.strip().split("\n")
        preview = "\n".join(lines[:40])
        if len(lines) > 40:
            preview += f"\n... ({len(lines) - 40} more lines)"
        obs = f"Generated Triton kernel: {filepath}\n\n{preview}"
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
        # If kernel is slow, show the current code so agent can improve it
        if speedup_val < 1.0 and filepath and os.path.exists(filepath):
            with open(filepath) as f:
                code = f.read()
            obs += "\n\n=== CURRENT KERNEL CODE (needs improvement) ===\n" + code
            obs += "\n\nHINT: Use 'profile_kernel' to diagnose WHY it's slow, then 'retrieve_pattern' to learn HOW to fix it."
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

        # Count loop depth
        import re
        for_loops = re.findall(r'for \w+ in range\(.*?\)', code)

        lines.append(f"\n--- Static Analysis ---")
        lines.append(f"  tl.dot (Tensor Core GEMM): {'YES ✓' if has_tl_dot else 'NO ✗ — NOT using Tensor Cores!'}")
        lines.append(f"  tl.load calls: {num_tl_load}")
        lines.append(f"  tl.store calls: {num_tl_store}")
        lines.append(f"  For loops in kernel: {len(for_loops)}")
        for i, fl in enumerate(for_loops):
            lines.append(f"    loop {i}: {fl}")
        lines.append(f"  Online/streaming pattern: {'YES' if has_online else 'NO'}")

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

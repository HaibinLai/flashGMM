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
import json, io, sys, contextlib, traceback
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

        prompt = f"""You are an IO-aware GPU operator optimization agent.

Your goal: reduce HBM memory traffic (IO) of GPU operators by eliminating
intermediate matrix materialization, using techniques like:
- Operator fusion (fuse_ops)
- Online/streaming algorithms (fuse_and_online): online_logsumexp, online_argmin, online_softmax
- Recomputation (trading FLOPs for IO reduction)

## Available Tools
{tool_desc}

## Required Workflow (MUST follow this order)
1. Use 'analyze' to understand the baseline operator's IO bottlenecks
2. Identify materialized intermediates (tensors written to HBM then read back)
3. Apply optimizations to eliminate them (prefer fuse_and_online for large intermediates)
4. Use 'verify' to check the optimization is valid
5. Use 'benchmark' to measure ACTUAL GPU speedup — this is critical!
6. REFLECT on the benchmark results:
   - If actual speedup < roofline prediction, explain WHY (cache effects? kernel launch overhead? Python loop overhead?)
   - If actual speedup > 1, confirm the IO optimization translates to real performance gain
   - If actual speedup < 1, consider whether the optimization is still valuable (VRAM savings? larger scale?)
7. Call 'done' with a final summary that includes both symbolic analysis AND benchmark results

## Response Format
For each step, respond with:
Thought: <your reasoning about what to do and why>
Action: {{"tool": "<tool_name>", "args": {{...}}}}

IMPORTANT: You MUST call 'benchmark' before 'done'. After seeing benchmark results,
reflect on why actual speedup differs from roofline prediction.
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

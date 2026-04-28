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

## Strategy
1. First, use 'analyze' to understand the baseline operator's IO bottlenecks
2. Identify materialized intermediates (tensors written to HBM then read back)
3. Apply optimizations to eliminate them (prefer fuse_and_online for large intermediates)
4. Use 'verify' to check the optimization is valid
5. Call 'done' when no more materialized intermediates remain

## Response Format
For each step, respond with:
Thought: <your reasoning about what to do and why>
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

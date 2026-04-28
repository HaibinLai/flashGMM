#!/usr/bin/env python3
"""
IO-Aware Operator Optimization Workflow.

Interactive tools for LLM agents (or humans) to analyze and optimize
GPU operators by reducing HBM IO traffic.

Usage (in a Python session or via CLI):

    from io_env.workflow import *

    # Step 1: Analyze a baseline operator
    analyze("cross_entropy")

    # Step 2: See what can be optimized
    show_actions()

    # Step 3: Apply an optimization
    try_action("fuse_ops", op1="row_max", op2="subtract_and_exp")

    # Step 4: Check progress, apply more actions...
    try_action("fuse_and_online",
               ops=["fused_row_max_subtract_and_exp", "sum_and_log", "gather_loss"],
               algorithm="online_logsumexp",
               output_name="loss", output_shape=("N",))

    # Step 5: See the full optimization trace
    summary()

    # Or: define a completely new operator
    analyze_custom(my_graph, my_params)
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from io_env.dsl import ComputationGraph, TensorSpec, Op, FusedOp, TilingSpec, OnlineStateSpec
from io_env.calculator import IOCalculator, IOReport, ComparisonReport, H200, H100, A100
from io_env.actions import DesignActions, ONLINE_ALGORITHMS
from io_env.agent_loop import compute_reward
from io_env.examples import EXAMPLES

# ============================================================================
# Module-level state (the "environment")
# ============================================================================
_state = {
    "graph": None,           # current ComputationGraph
    "baseline_graph": None,  # original baseline (for reward computation)
    "params": None,           # concrete parameter values
    "calc": IOCalculator(hardware=H200),
    "baseline_report": None,
    "current_report": None,
    "history": [],            # list of (action_desc, graph_before, report_before, graph_after, report_after)
    "task_name": None,
}


def _ensure_state():
    if _state["graph"] is None:
        print("⚠ No operator loaded. Run analyze('cross_entropy') or analyze_custom(graph, params) first.")
        return False
    return True


# ============================================================================
# Core API
# ============================================================================

def analyze(task_name: str, params: dict | None = None, hardware: str = "H200"):
    """
    Load and analyze a baseline operator from the example library.

    Available tasks: gmm_estep, gmm_em_fused, kmeans, softmax, cross_entropy
    """
    if task_name not in EXAMPLES:
        print(f"Unknown task '{task_name}'. Available: {list(EXAMPLES.keys())}")
        return

    hw = {"H200": H200, "H100": H100, "A100": A100}.get(hardware, H200)
    _state["calc"] = IOCalculator(hardware=hw)

    example = EXAMPLES[task_name]
    graph = example["baseline"]()
    p = params or example["default_params"]

    _state["graph"] = graph
    _state["baseline_graph"] = graph.clone()
    _state["params"] = p
    _state["history"] = []
    _state["task_name"] = task_name

    report = _state["calc"].analyze(graph, p)
    _state["baseline_report"] = report
    _state["current_report"] = report

    _print_analysis(graph, report, p, title="BASELINE ANALYSIS")


def analyze_custom(graph: ComputationGraph, params: dict, hardware: str = "H200"):
    """Load and analyze a custom computation graph."""
    hw = {"H200": H200, "H100": H100, "A100": A100}.get(hardware, H200)
    _state["calc"] = IOCalculator(hardware=hw)
    _state["graph"] = graph
    _state["baseline_graph"] = graph.clone()
    _state["params"] = params
    _state["history"] = []
    _state["task_name"] = graph.name

    report = _state["calc"].analyze(graph, params)
    _state["baseline_report"] = report
    _state["current_report"] = report

    _print_analysis(graph, report, params, title="BASELINE ANALYSIS")


def status():
    """Show the current state: graph structure + IO report."""
    if not _ensure_state():
        return
    _print_analysis(_state["graph"], _state["current_report"], _state["params"],
                    title=f"CURRENT STATE (step {len(_state['history'])})")


def show_actions():
    """Show available optimization actions and their expected effects."""
    print()
    print("=" * 60)
    print("  AVAILABLE ACTIONS")
    print("=" * 60)

    actions = DesignActions.list_actions()
    for i, a in enumerate(actions, 1):
        print(f"\n  [{i}] {a['name']}")
        print(f"      {a['description']}")
        print(f"      IO:    {a['io_effect']}")
        print(f"      FLOPs: {a['flop_effect']}")
        if 'algorithms' in a:
            print(f"      Algorithms: {a['algorithms']}")

    if _ensure_state():
        mats = _state["graph"].get_materialized_intermediates()
        if mats:
            print(f"\n  Current materialized intermediates:")
            for name, spec in mats:
                size = spec.size_bytes(_state["params"]) / 1e6
                print(f"    - {name}: {spec.shape} ({size:.1f} MB)")

        print(f"\n  Current ops: {[op.name for op in _state['graph'].operations]}")
    print()


def try_action(action: str, **kwargs) -> bool:
    """
    Apply an optimization action to the current graph.

    Actions:
        try_action("fuse_ops", op1="compute_L", op2="logsumexp")
        try_action("fuse_and_online", ops=["op1","op2"], algorithm="online_logsumexp",
                   output_name="log_norm", output_shape=("N",))
        try_action("apply_online", target_op="compute_L", algorithm="online_logsumexp")
        try_action("recompute", tensor="L", recompute_in="normalize")
        try_action("add_pass", name="flash_pass2", reads=[...], computes="...",
                   flops_fn=lambda p: ..., output_shape=(...), output_name="stats")

    Returns True if action succeeded.
    """
    if not _ensure_state():
        return False

    graph = _state["graph"]
    params = _state["params"]
    old_report = _state["current_report"]

    try:
        if action == "fuse_ops":
            new_graph = DesignActions.fuse_ops(graph, kwargs["op1"], kwargs["op2"],
                                               fused_name=kwargs.get("name"))
            desc = f'fuse_ops({kwargs["op1"]}, {kwargs["op2"]})'

        elif action == "fuse_and_online":
            ops = kwargs["ops"]
            algo = kwargs["algorithm"]
            out_name = kwargs["output_name"]
            out_shape = tuple(kwargs["output_shape"])
            tiling = None
            if "tiling" in kwargs:
                tiling = TilingSpec(tiles=kwargs["tiling"])
            new_graph = DesignActions.fuse_and_online(
                graph, ops_to_fuse=ops, online_algorithm=algo,
                output_name=out_name, output_shape=out_shape, tiling=tiling)
            desc = f'fuse_and_online({ops}, {algo})'

        elif action == "apply_online":
            new_graph = DesignActions.apply_online_algorithm(
                graph, kwargs["target_op"], kwargs["algorithm"])
            desc = f'apply_online({kwargs["target_op"]}, {kwargs["algorithm"]})'

        elif action == "recompute":
            new_graph = DesignActions.replace_with_recompute(
                graph, kwargs["tensor"], kwargs["recompute_in"])
            desc = f'recompute({kwargs["tensor"]})'

        elif action == "add_pass":
            out_spec = TensorSpec(shape=tuple(kwargs["output_shape"]), dtype="f32", storage="HBM")
            new_graph = DesignActions.add_recompute_pass(
                graph, pass_name=kwargs["name"], reads=kwargs["reads"],
                computes=kwargs.get("computes", "recompute+accumulate"),
                flops_fn=kwargs["flops_fn"], output=out_spec,
                output_name=kwargs["output_name"],
                notes=kwargs.get("notes", ""))
            desc = f'add_pass({kwargs["name"]})'

        else:
            print(f"⚠ Unknown action '{action}'. Run show_actions() to see available actions.")
            return False

    except Exception as e:
        print(f"✗ Action failed: {e}")
        return False

    # Analyze the new graph
    new_report = _state["calc"].analyze(new_graph, params)
    reward = compute_reward(_state["baseline_report"], new_report)
    step_reward = compute_reward(old_report, new_report)

    # Record history
    _state["history"].append({
        "step": len(_state["history"]),
        "action": desc,
        "graph_before": graph,
        "report_before": old_report,
        "graph_after": new_graph,
        "report_after": new_report,
        "reward_cumulative": reward,
        "reward_step": step_reward,
    })

    # Update state
    _state["graph"] = new_graph
    _state["current_report"] = new_report

    # Print comparison
    _print_step_result(desc, old_report, new_report, reward, step_reward)
    return True


def undo():
    """Undo the last action."""
    if not _state["history"]:
        print("⚠ Nothing to undo.")
        return

    last = _state["history"].pop()
    _state["graph"] = last["graph_before"]
    _state["current_report"] = last["report_before"]
    print(f"↩ Undid: {last['action']}")
    print(f"  Back to step {len(_state['history'])}")


def summary():
    """Print the complete optimization trace."""
    if not _ensure_state():
        return

    baseline = _state["baseline_report"]
    current = _state["current_report"]
    params = _state["params"]

    print()
    print("█" * 60)
    print("█  OPTIMIZATION SUMMARY")
    print("█" * 60)
    print(f"  Task:     {_state['task_name']}")
    print(f"  Hardware: {_state['calc'].hw.name}")
    int_params = {k: v for k, v in params.items() if isinstance(v, int)}
    print(f"  Params:   {int_params}")
    print(f"  Steps:    {len(_state['history'])}")
    print()

    # Before/After
    print(f"  {'':30s} {'Baseline':>12s} {'Optimized':>12s} {'Change':>10s}")
    print(f"  {'─'*66}")
    print(f"  {'Total IO':30s} {baseline.total_io/1e6:>10.1f}MB {current.total_io/1e6:>10.1f}MB "
          f"{(1-current.total_io/baseline.total_io)*100:>+8.1f}%")
    print(f"  {'  Reads':30s} {baseline.total_hbm_reads/1e6:>10.1f}MB {current.total_hbm_reads/1e6:>10.1f}MB")
    print(f"  {'  Writes':30s} {baseline.total_hbm_writes/1e6:>10.1f}MB {current.total_hbm_writes/1e6:>10.1f}MB")
    print(f"  {'Total FLOPs':30s} {baseline.total_flops/1e9:>10.2f}GF {current.total_flops/1e9:>10.2f}GF "
          f"{(current.total_flops/baseline.total_flops-1)*100:>+8.1f}%")
    print(f"  {'Arithmetic Intensity':30s} {baseline.arithmetic_intensity:>10.1f}   {current.arithmetic_intensity:>10.1f}")
    print(f"  {'Bottleneck':30s} {baseline.bottleneck:>12s} {current.bottleneck:>12s}")
    print(f"  {'Est. Time':30s} {baseline.estimated_time_ms:>9.3f}ms {current.estimated_time_ms:>9.3f}ms "
          f"{'':>2s}{baseline.estimated_time_ms/max(current.estimated_time_ms,1e-9):.2f}×")
    print(f"  {'Materialized intermediates':30s} "
          f"{len(baseline.materialized_intermediates):>12d} {len(current.materialized_intermediates):>12d}")
    print()

    # Step-by-step
    if _state["history"]:
        print(f"  Step-by-step trace:")
        print(f"  {'─'*66}")
        for h in _state["history"]:
            io_before = h["report_before"].total_io / 1e6
            io_after = h["report_after"].total_io / 1e6
            pct = (1 - h["report_after"].total_io / h["report_before"].total_io) * 100
            print(f"  Step {h['step']}: {h['action']}")
            print(f"    IO: {io_before:.1f}MB → {io_after:.1f}MB ({pct:+.1f}%)  "
                  f"reward: {h['reward_step']:+.2f} (cum: {h['reward_cumulative']:.2f})")
        print()

    # Final graph
    print(f"  Final graph ops: {[op.name for op in _state['graph'].operations]}")
    mats = _state["graph"].get_materialized_intermediates()
    if mats:
        print(f"  ⚠ Remaining materialized: {[n for n,_ in mats]}")
    else:
        print(f"  ✓ Zero materialized intermediates!")
    print("█" * 60)
    print()


# ============================================================================
# Printing helpers
# ============================================================================

def _print_analysis(graph, report, params, title="ANALYSIS"):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(graph.summary(params))
    print()
    print(report.display())

    mats = graph.get_materialized_intermediates()
    if mats:
        print(f"\n  🎯 Optimization targets (materialized intermediates):")
        for name, spec in sorted(mats, key=lambda m: m[1].size_bytes(params), reverse=True):
            size = spec.size_bytes(params) / 1e6
            print(f"     {name}: shape={spec.shape}, {size:.1f} MB in HBM")

    if report.optimization_hints:
        print(f"\n  💡 Hints:")
        for hint in report.optimization_hints:
            print(f"     {hint}")
    print()


def _print_step_result(desc, old_report, new_report, reward_cum, reward_step):
    step = len(_state["history"])
    io_before = old_report.total_io / 1e6
    io_after = new_report.total_io / 1e6
    pct = (1 - new_report.total_io / old_report.total_io) * 100

    print()
    print(f"  ✓ Step {step}: {desc}")
    print(f"    IO:      {io_before:.1f} MB → {io_after:.1f} MB ({pct:+.1f}%)")
    print(f"    FLOPs:   {old_report.total_flops/1e9:.2f} → {new_report.total_flops/1e9:.2f} GF")
    print(f"    Time:    {old_report.estimated_time_ms:.3f} → {new_report.estimated_time_ms:.3f} ms")
    print(f"    Reward:  step={reward_step:+.2f}  cumulative={reward_cum:.2f}")

    mats = _state["graph"].get_materialized_intermediates()
    if mats:
        print(f"    Remaining materialized: {[n for n,_ in mats]}")
    else:
        print(f"    ✓ Zero materialized intermediates!")

    print(f"    Ops: {[op.name for op in _state['graph'].operations]}")
    print()


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IO-Aware Operator Optimization Workflow")
    parser.add_argument("task", nargs="?", default="cross_entropy",
                       help="Task name: gmm_estep, kmeans, softmax, cross_entropy")
    parser.add_argument("--hardware", default="H200", help="GPU: H200, H100, A100")
    args = parser.parse_args()

    analyze(args.task, hardware=args.hardware)
    print("Ready. Use show_actions(), try_action(...), undo(), summary().")
    print("Or run: python -i workflow.py cross_entropy")
